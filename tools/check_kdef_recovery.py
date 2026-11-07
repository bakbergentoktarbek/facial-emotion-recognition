"""Check interrupted CNN recovery without training or accessing test images."""
import csv
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports/.mplconfig"))
os.environ.setdefault("KERAS_HOME", str(ROOT / "reports/.keras"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from train_kdef import (Checkpoint, MetricsCSV, Progress, RestoreCallbackState,
                        file_hash, load_cnn_resume, plot_history)


def main():
    directory = ROOT / "reports/kdef/cnn"
    best = ROOT / "models/emotion_cnn_kdef.keras"
    latest = ROOT / "models/kdef_checkpoints/cnn_adaptation_latest.keras"
    protected = [best, latest, ROOT / "models/emotion_cnn.keras", ROOT / "models/emotion_mobilenet.keras"]
    hashes = {path: file_hash(path) for path in protected}
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model, history = load_cnn_resume(directory, best, config)
    completed = len(history["loss"])
    assert completed > 0
    probabilities = model(np.zeros((1, 96, 96, 1), np.float32), training=False).numpy()
    assert probabilities.shape == (1, 7) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(), 1, atol=1e-5)
    check_dir = ROOT / "reports/kdef/recovery_checks"
    check_dir.mkdir(exist_ok=True)
    assert plot_history(history, check_dir / "history.png")
    checkpoint = Checkpoint(best, latest, max(history["val_accuracy"]))
    progress = Progress(check_dir, "check", 60, 0, checkpoint)
    progress.on_epoch_begin(0)
    logs = {key: values[-1] for key, values in history.items()}
    with patch("matplotlib.figure.Figure.savefig", side_effect=OSError(22, "simulated PNG write failure")):
        progress.on_epoch_end(0, logs)
    status = json.loads((check_dir / "status.json").read_text())
    assert status["state"] == "epoch_complete"
    assert (check_dir / "history_check.json").exists()
    assert not plt.get_fignums()

    stopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True)
    scheduler = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-6)
    stopping.set_model(model)
    scheduler.set_model(model)
    stopping.on_train_begin()
    scheduler.on_train_begin()
    restore = RestoreCallbackState(history, stopping, scheduler, best)
    restore.set_model(model)
    original_lr = float(model.optimizer.learning_rate.numpy())
    restore.on_train_begin()
    assert stopping.best == max(history["val_accuracy"])
    assert stopping.best_epoch == int(np.argmax(history["val_accuracy"]))
    assert stopping.wait == completed - 1 - stopping.best_epoch
    assert scheduler.wait >= 0
    assert float(model.optimizer.learning_rate.numpy()) == original_lr

    csv_path = check_dir / "append.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])
        writer.writerow([0, 0.5, 1.0, 0.4, 1.2])
    logger = MetricsCSV(str(csv_path), append=True)
    logger.set_model(model)
    logger.on_train_begin()
    logger.on_epoch_end(1, logs)
    logger.on_train_end()
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and None not in rows[-1] and rows[-1]["epoch"] == "1"
    assert all(file_hash(path) == value for path, value in hashes.items())
    print(f"RECOVERY OK: {completed} epochs, Adam step {int(model.optimizer.iterations.numpy())}; "
          "callback counters restored; PNG failure nonfatal; CSV schema preserved; checkpoints unchanged.")


if __name__ == "__main__":
    main()
