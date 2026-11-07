"""Load the fixed, person-disjoint KDEF split without using FER-2013 paths."""
import hashlib
import json
from pathlib import Path

from config import BASE_DIR, EMOTIONS_EN

ROOT = Path(BASE_DIR)
MANIFEST = ROOT / "reports/kdef/data_split_seed42.json"
PREPROCESSING = "RGB JPEG; trim top/bottom 8% with fixed crop; resize; CNN grayscale, MobileNet RGB; values 0..255"


def read_manifest(path=MANIFEST):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["class_names"] != EMOTIONS_EN:
        raise ValueError("KDEF class order does not match config.py")
    root = (ROOT / manifest["image_root"]).resolve()
    groups, hashes, files = set(), set(), set()
    for split in ("train", "validation", "test"):
        records = manifest["records"][split]
        subjects = {r["subject"] for r in records}
        pixels = {r["pixels_sha256"] for r in records}
        names = {r["path"] for r in records}
        if subjects != set(manifest["subjects"][split]) or len(names) != len(records):
            raise ValueError(f"Invalid KDEF split: {split}")
        if groups & subjects or hashes & pixels or files & names:
            raise ValueError("KDEF splits overlap")
        for record in records:
            target = (root / record["path"]).resolve()
            if not target.is_relative_to(root):
                raise ValueError("KDEF path escapes dataset directory")
            if record["label"] != EMOTIONS_EN.index(record["emotion"]):
                raise ValueError("KDEF label mismatch")
        if {r["label"] for r in records} != set(range(len(EMOTIONS_EN))):
            raise ValueError(f"Missing KDEF classes in {split}")
        groups.update(subjects)
        hashes.update(pixels)
        files.update(names)
    return manifest, root, hashlib.sha256(path.read_bytes()).hexdigest()


def make_dataset(manifest, root, split, size, channels, batch=32, training=False):
    import tensorflow as tf

    if training and split != "train":
        raise ValueError("Only the train split can be used for training")
    records = manifest["records"][split]
    paths = [str(root / r["path"]) for r in records]
    labels = [r["label"] for r in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode(path, label):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        height = tf.shape(image)[0]
        trim = tf.cast(tf.cast(height, tf.float32) * 0.08, tf.int32)
        image = image[trim:height - trim, :, :]
        if channels == 1:
            image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, (size, size), antialias=True)
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
        image.set_shape((size, size, channels))
        return image, label

    # Cache uint8 images, rather than large float32 RGB tensors.
    dataset = dataset.map(decode, num_parallel_calls=4).cache()
    if training:
        dataset = dataset.shuffle(len(records), seed=42, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    options.threading.max_intra_op_parallelism = 1
    return dataset.batch(batch).prefetch(2).with_options(options)
