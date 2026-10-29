"""Fetch large wheels from official PyPI in resumable parts, then verify SHA-256.

Usage: python tools/fetch_wheels.py package==version [...]
Requires only Python and pip; does not install or execute downloaded packages.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1] / ".wheels"
CHUNK = 256 * 1024


def log(message):
    ROOT.mkdir(exist_ok=True)
    with (ROOT / "download.log").open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")
    try:
        print(message, flush=True)
    except OSError:
        # Keep downloads running if the chat's output pipe is interrupted.
        pass


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fetch(spec):
    name, version = spec.split("==")
    with urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
        metadata = json.load(response)
    priorities = {tag: i for i, tag in enumerate(sys_tags())}
    candidates = []
    for item in metadata["urls"]:
        if item["packagetype"] == "bdist_wheel":
            tags = parse_wheel_filename(item["filename"])[3]
            matching = tags & priorities.keys()
            if matching:
                candidates.append((min(priorities[t] for t in matching), item))
    if not candidates:
        raise RuntimeError(f"No compatible wheel: {spec}")
    item = min(candidates, key=lambda pair: pair[0])[1]
    filename, size, expected = item["filename"], item["size"], item["digests"]["sha256"]
    if Path(filename).name != filename or not item["url"].startswith("https://files.pythonhosted.org/"):
        raise ValueError("Unexpected PyPI artifact path")
    ROOT.mkdir(exist_ok=True)
    target = ROOT / filename
    if target.exists() and digest(target) == expected:
        log(f"Verified cached: {filename}")
        return
    parts_dir = ROOT / ".parts" / filename
    parts_dir.mkdir(parents=True, exist_ok=True)
    count = (size + CHUNK - 1) // CHUNK
    log(f"Fetching {filename} ({size / 1024**2:.1f} MiB)")

    def part(index):
        start, end = index * CHUNK, min(size, (index + 1) * CHUNK) - 1
        path = parts_dir / f"{index:06d}"
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(3):
            try:
                request = Request(item["url"], headers={"Range": f"bytes={start}-{end}"})
                with urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("Server did not honor byte range")
                    block = response.read(end - start + 2)
                if len(block) != end - start + 1:
                    raise RuntimeError("Incomplete byte range")
                path.write_bytes(block)
                return
            except Exception:
                if attempt == 2:
                    raise

    started = last_log = time.monotonic()
    with ThreadPoolExecutor(max_workers=32) as pool:
        pending = [pool.submit(part, i) for i in range(count)]
        for done, future in enumerate(as_completed(pending), 1):
            future.result()
            now = time.monotonic()
            if now - last_log >= 10:
                log(f"  {done}/{count} parts, {now - started:.0f}s")
                last_log = now
    temporary = target.with_suffix(".download")
    with temporary.open("wb") as output:
        for index in range(count):
            with (parts_dir / f"{index:06d}").open("rb") as stream:
                for block in iter(lambda: stream.read(CHUNK), b""):
                    output.write(block)
    if temporary.stat().st_size != size or digest(temporary) != expected:
        raise RuntimeError(f"SHA-256 mismatch: {filename}; file will not be installed")
    temporary.replace(target)
    log(f"SHA-256 OK: {filename}")


if __name__ == "__main__":
    if not sys.argv[1:]:
        raise SystemExit("Usage: python tools/fetch_wheels.py package==version [...]")
    for specification in sys.argv[1:]:
        fetch(specification)
