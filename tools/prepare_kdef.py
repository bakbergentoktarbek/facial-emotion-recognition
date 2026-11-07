"""Validate original KDEF and split by person, keeping both sessions together.

No model is trained and no existing model or FER-2013 artifact is modified.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from zipfile import ZipFile

from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import EMOTIONS_EN

PATTERN = re.compile(r"([AB])([FM])(\d{2})(AF|AN|DI|HA|NE|SA|SU)(FL|FR|HL|HR|S)\.JPG", re.I)
EMOTIONS = {"AF": "fear", "AN": "angry", "DI": "disgust", "HA": "happy",
            "NE": "neutral", "SA": "sad", "SU": "surprise"}
DATA = ROOT / "data/kdef"
REPORTS = ROOT / "reports/kdef"


def write_new_or_identical(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite a different file: {path}")
    else:
        path.write_bytes(content)


def prepare(archive, seed):
    print("Checking archive integrity...", flush=True)
    with ZipFile(archive) as z:
        for info in z.infolist():
            relative = PurePosixPath(info.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or any(":" in p for p in relative.parts):
                raise ValueError(f"Unsafe archive path: {info.filename}")
        bad = z.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP member: {bad}")
        prefix = "KDEF_and_AKDEF/KDEF/"
        names = sorted(n for n in z.namelist() if n.startswith(prefix) and n.lower().endswith(".jpg"))
        if len(names) != 4900:
            raise ValueError(f"Expected original KDEF with 4900 photos, found {len(names)}")

        def validate_and_extract(name):
            relative = PurePosixPath(name[len(prefix):])
            content = z.read(name)
            with Image.open(io.BytesIO(content)) as image:
                image.load()  # Decode every image, not just its header.
                dimensions, mode = list(image.size), image.mode
                pixels_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            destination = DATA / "original" / Path(*relative.parts)
            write_new_or_identical(destination, content)
            match = PATTERN.fullmatch(relative.name)
            record = {"path": relative.as_posix(), "size": dimensions, "mode": mode,
                      "sha256": hashlib.sha256(content).hexdigest(), "pixels_sha256": pixels_hash}
            if match is None:
                return record, "Filename does not encode a reliable emotion and angle"
            session, gender, number, code, angle = (part.upper() for part in match.groups())
            if relative.parent.name.upper() != session + gender + number:
                raise ValueError(f"Filename/folder mismatch: {relative}")
            record.update(subject=gender + number, session=session, gender=gender,
                          emotion=EMOTIONS[code], label=EMOTIONS_EN.index(EMOTIONS[code]), angle=angle)
            return record, None

        records, excluded = [], []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(validate_and_extract, name) for name in names]
            for count, future in enumerate(as_completed(futures), 1):
                record, reason = future.result()
                if reason:
                    excluded.append({**record, "reason": reason})
                else:
                    records.append(record)
                if count % 500 == 0 or count == len(names):
                    print(f"Decoded and checked {count}/{len(names)} photos", flush=True)
        for name in ("KDEF_and_AKDEF/ReadThis.TXT", "KDEF_and_AKDEF/References.txt",
                     "KDEF_and_AKDEF/ReadThis/KDEF.DOC"):
            write_new_or_identical(DATA / "documentation" / PurePosixPath(name).name, z.read(name))

    # Exact duplicate images must not connect different people across splits.
    by_pixels = defaultdict(list)
    for record in records:
        by_pixels[record["pixels_sha256"]].append(record)
    ambiguous = {key for key, group in by_pixels.items() if len({r["subject"] for r in group}) > 1}
    if ambiguous:
        excluded.extend({**r, "reason": "Identical decoded image assigned to multiple subjects"}
                        for r in records if r["pixels_sha256"] in ambiguous)
        records = [r for r in records if r["pixels_sha256"] not in ambiguous]
    records.sort(key=lambda r: r["path"])
    subjects = sorted({r["subject"] for r in records})
    if len(subjects) != 70:
        raise ValueError(f"Expected 70 subjects, found {len(subjects)}")
    train_val, test_subjects = train_test_split(
        subjects, test_size=11, random_state=seed, stratify=[s[0] for s in subjects])
    train_subjects, val_subjects = train_test_split(
        train_val, test_size=10, random_state=seed, stratify=[s[0] for s in train_val])
    groups = {"train": sorted(train_subjects), "validation": sorted(val_subjects), "test": sorted(test_subjects)}
    split_records = {name: [r for r in records if r["subject"] in people] for name, people in groups.items()}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(groups[first]) & set(groups[second]):
            raise ValueError("Subject leakage")
        if {r["pixels_sha256"] for r in split_records[first]} & {r["pixels_sha256"] for r in split_records[second]}:
            raise ValueError("Duplicate image leakage")
    for name, values in split_records.items():
        if set(r["emotion"] for r in values) != set(EMOTIONS_EN):
            raise ValueError(f"Missing classes in {name}")

    with archive.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = {"dataset": "KDEF original", "archive_name": archive.name,
               "archive_sha256": archive_hash, "seed": seed,
               "image_root": "data/kdef/original", "class_names": EMOTIONS_EN,
               "split_method": "Subject-disjoint; sessions A/B of a person are kept together; gender-stratified",
               "angles": ["FL", "FR", "HL", "HR", "S"],
               "source_photos": len(names), "included_photos": len(records),
               "excluded": sorted(excluded, key=lambda r: r["path"]),
               "subjects": groups,
               "counts": {name: {"subjects": len(groups[name]), "images": len(values),
                                 "classes": dict(sorted(Counter(r["emotion"] for r in values).items()))}
                          for name, values in split_records.items()},
               "image_sizes": dict(Counter(f"{r['size'][0]}x{r['size'][1]}" for r in records)),
               "note": "AKDEF and contact sheets are excluded. No images are published. Test is reserved for final evaluation."}
    manifest = {**summary, "records": split_records}
    for name, obj in ((f"data_split_seed{seed}.json", manifest), ("preparation_summary.json", summary)):
        write_new_or_identical(REPORTS / name, json.dumps(obj, indent=2).encode("utf-8"))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Ready: {REPORTS / f'data_split_seed{seed}.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.archive.resolve(strict=True), args.seed)
