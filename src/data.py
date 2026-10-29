# src/data.py
import os
import numpy as np
from config import DATA_DIR, IMG_SIZE, EMOTIONS_EN, EMOTIONS_KZ, REPORTS_DIR

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def validate_dataset():
    missing = [os.path.join(DATA_DIR, split, emotion)
               for split in ("train", "test") for emotion in EMOTIONS_EN
               if not os.path.isdir(os.path.join(DATA_DIR, split, emotion))]
    if missing:
        raise FileNotFoundError(
            "FER-2013 not found. See README.md: put train/ and test/ in "
            f"{DATA_DIR}. Missing: {missing[0]}")
    for split in ("train", "test"):
        for emotion in EMOTIONS_EN:
            folder = os.path.join(DATA_DIR, split, emotion)
            if not any(f.lower().endswith(IMAGE_EXTENSIONS) for f in os.listdir(folder)):
                raise ValueError(f"No images in {folder}")


def load_datasets(batch_size=64, val_split=0.15, seed=42,
                  include_test=True, return_metadata=False):
    """train/val/test tf.data жиындарын қайтарады. Суреттер: (48, 48, 1), мәндер 0–255."""
    import tensorflow as tf
    AUTOTUNE = tf.data.AUTOTUNE
    validate_dataset()
    common = dict(
        labels="inferred",          # белгі = папка атауы
        label_mode="int",           # 0..6 бүтін сандар (sparse_categorical_crossentropy үшін)
        class_names=EMOTIONS_EN,    # класс ретін анық бекітеміз
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None,            # батчты кейін, араластырғаннан соң жасаймыз
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"), validation_split=val_split,
        subset="training", seed=seed, shuffle=True, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"), validation_split=val_split,
        subset="validation", seed=seed, shuffle=True, **common)  # seed бірдей → жиындар қиылыспайды
    metadata = {
        "seed": seed, "validation_fraction": val_split, "class_names": EMOTIONS_EN,
        "train": [os.path.relpath(p, DATA_DIR).replace("\\", "/") for p in train_ds.file_paths],
        "validation": [os.path.relpath(p, DATA_DIR).replace("\\", "/") for p in val_ds.file_paths],
    }
    test_ds = None
    if include_test:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(DATA_DIR, "test"), shuffle=False, **common)

    # cache: суреттерді жадта сақтау (FER шағын, ~100 МБ); shuffle: әр эпохада араластыру;
    # prefetch: GPU оқып жатқанда келесі батчты дайындау
    train_ds = train_ds.cache().shuffle(10_000, seed=seed).batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().batch(batch_size).prefetch(AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.cache().batch(batch_size).prefetch(AUTOTUNE)
    if return_metadata:
        return train_ds, val_ds, test_ds, metadata
    return train_ds, val_ds, test_ds


def class_counts():
    validate_dataset()
    return np.array([sum(f.lower().endswith(IMAGE_EXTENSIONS)
                         for f in os.listdir(os.path.join(DATA_DIR, "train", e)))
                     for e in EMOTIONS_EN])


def compute_class_weights(training_paths=None):
    """Сирек класқа үлкен салмақ. Түбір (sqrt) салмақты тым асыра сілтемеуге көмектеседі:
    disgust үшін толық салмақ ≈ 9.4, ал sqrt-тен кейін ≈ 3.1."""
    if training_paths is None:
        counts = class_counts()
    else:
        from collections import Counter
        from pathlib import PurePosixPath
        labels = Counter(PurePosixPath(p).parent.name for p in training_paths)
        counts = np.array([labels[name] for name in EMOTIONS_EN])
    if np.any(counts == 0):
        raise ValueError("Each emotion needs at least one training image")
    weights = np.sqrt(counts.sum() / (len(counts) * counts))
    return {i: float(w) for i, w in enumerate(weights)}


if __name__ == "__main__":
    # 1-апта: деректерді зерттеу (EDA)
    import matplotlib.pyplot as plt
    os.makedirs(REPORTS_DIR, exist_ok=True)
    names = [EMOTIONS_KZ[e] for e in EMOTIONS_EN]
    counts = class_counts()
    for n, c in zip(names, counts):
        print(f"{n:10s} {c:5d}")

    plt.figure(figsize=(8, 4))
    plt.bar(names, counts, color="#3498db")
    plt.title("FER-2013 train: класс бойынша сурет саны")
    plt.ylabel("Сурет саны")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "class_distribution.png"), dpi=150)

    # Әр класстан 5 мысал
    from pathlib import Path
    from PIL import Image
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(7, 5, figsize=(7, 10))
    for cls in range(7):
        files = sorted(p for p in (Path(DATA_DIR) / "train" / EMOTIONS_EN[cls]).iterdir()
                       if p.suffix.lower() in IMAGE_EXTENSIONS)
        selected = rng.choice(len(files), size=min(5, len(files)), replace=False)
        for j, ax in enumerate(axes[cls]):
            ax.axis("off")
            if j < len(selected):
                with Image.open(files[selected[j]]) as image:
                    ax.imshow(image.convert("L"), cmap="gray")
            if j == 0:
                ax.set_title(names[cls], fontsize=9, loc="left")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "samples.png"), dpi=150)
    plt.show()
