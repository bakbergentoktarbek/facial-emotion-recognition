"""Train independent KDEF models, selecting checkpoints only by validation."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import BASE_DIR, CNN_PATH
from data_kdef import PREPROCESSING, make_dataset, read_manifest
from model_kdef import (CNN_SIZE, MOBILENET_SIZE, IMAGENET_WEIGHTS,
                        build_cnn, build_mobilenet, unfreeze_mobilenet)

ROOT = Path(BASE_DIR)
REPORTS = ROOT / "reports/kdef"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(path)


def compile_model(model, learning_rate):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


class Checkpoint(keras.callbacks.Callback):
    """Save a complete latest checkpoint atomically, and retain the global best."""
    def __init__(self, best_path, latest_path, previous_best=0.0):
        super().__init__()
        self.best_path, self.latest_path = best_path, latest_path
        self.best = previous_best

    def on_epoch_end(self, epoch, logs=None):
        accuracy = float(logs["val_accuracy"])
        if not all(np.isfinite(float(v)) for v in logs.values()):
            raise ValueError("Non-finite training metrics; checkpoint not saved")
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.latest_path.with_name(self.latest_path.stem + ".tmp.keras")
        self.model.save(temporary)
        temporary.replace(self.latest_path)
        if accuracy > self.best:
            self.best = accuracy
            temporary = self.best_path.with_name(self.best_path.stem + ".tmp.keras")
            shutil.copyfile(self.latest_path, temporary)
            temporary.replace(self.best_path)
            print(f"Best validation accuracy: {accuracy:.4%}; saved {self.best_path.name}", flush=True)


class MetricsCSV(keras.callbacks.CSVLogger):
    def on_epoch_end(self, epoch, logs=None):
        # Keep the original CSV schema when appending a resumed run.
        # Learning rate is already retained in the JSON history.
        super().on_epoch_end(epoch, {key: value for key, value in (logs or {}).items()
                                     if key != "learning_rate"})


class Progress(keras.callbacks.Callback):
    def __init__(self, directory, phase, limit, offset, checkpoint, history=None):
        super().__init__()
        self.directory, self.phase = directory, phase
        self.limit, self.offset, self.checkpoint = limit, offset, checkpoint
        self.history = {key: list(values) for key, values in (history or {}).items()}
        self.started = time.monotonic()

    def status(self, **values):
        save_json(self.directory / "status.json", {
            "phase": self.phase, "phase_epochs_limit": self.limit,
            "best_val_accuracy": self.checkpoint.best,
            "elapsed_seconds": round(time.monotonic() - self.started, 1), **values})

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.status(state="training", epoch=self.epoch, total_epoch=self.offset + self.epoch, batch=0)

    def on_train_batch_end(self, batch, logs=None):
        if batch == 0 or (batch + 1) % 10 == 0:
            self.status(state="training", epoch=self.epoch, total_epoch=self.offset + self.epoch,
                        batch=batch + 1, training_metrics={k: float(v) for k, v in logs.items()})

    def on_epoch_end(self, epoch, logs=None):
        metrics = {key: float(value) for key, value in logs.items()}
        for key, value in metrics.items():
            self.history.setdefault(key, []).append(value)
        save_json(self.directory / f"history_{self.phase}.json", self.history)
        plot_history(self.history, self.directory / f"history_{self.phase}.png")
        self.status(state="epoch_complete", epoch=epoch + 1, total_epoch=self.offset + epoch + 1,
                    metrics=metrics)

    def on_train_end(self, logs=None):
        self.status(state="phase_finished", epochs_completed=len(self.history.get("loss", [])))


def plot_history(history, path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    try:
        for key, axis in zip(("loss", "accuracy"), axes):
            axis.plot(history[key], label="train (augmented)")
            axis.plot(history["val_" + key], label="validation")
            axis.set(title=key, xlabel="Epoch")
            axis.legend()
            axis.grid(alpha=0.3)
        figure.tight_layout()
        # Render in memory, then replace the image atomically. An editor may
        # temporarily hold the old PNG open on Windows; plotting is optional.
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=130)
        path = Path(path)
        temporary = path.with_name(path.stem + ".tmp.png")
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(path)
        return True
    except OSError as error:
        print(f"Warning: graph update skipped ({error}); checkpoints/JSON remain saved.", flush=True)
        return False
    finally:
        plt.close(figure)


class RestoreCallbackState(keras.callbacks.Callback):
    def __init__(self, history, stopping, scheduler, best_path):
        super().__init__()
        self.history, self.stopping, self.scheduler = history, stopping, scheduler
        self.best_path = best_path

    def on_train_begin(self, logs=None):
        if not self.history:
            return
        stopping, scheduler = self.stopping, self.scheduler
        stopping._set_monitor_op()
        stopping.best_epoch = int(np.argmax(self.history["val_accuracy"]))
        stopping.best = max(self.history["val_accuracy"])
        stopping.wait = len(self.history["loss"]) - 1 - stopping.best_epoch
        stopping.best_weights = keras.models.load_model(self.best_path, compile=False).get_weights()
        # Replay scheduler counters without changing the restored optimizer LR.
        for loss, rate in zip(self.history["val_loss"], self.history["learning_rate"]):
            if scheduler.monitor_op(loss, scheduler.best):
                scheduler.best, scheduler.wait = loss, 0
            else:
                scheduler.wait += 1
                if scheduler.wait >= scheduler.patience and rate > np.float32(scheduler.min_lr):
                    scheduler.wait = 0


def fit_stage(model, name, phase, epochs, train, validation, directory, best_path,
              previous_best=0.0, offset=0, previous_history=None):
    latest = ROOT / "models/kdef_checkpoints" / f"{name}_{phase}_latest.keras"
    checkpoint = Checkpoint(best_path, latest, previous_best)
    progress = Progress(directory, phase, epochs, offset, checkpoint, previous_history)
    initial_epoch = len((previous_history or {}).get("loss", []))
    if initial_epoch >= epochs:
        return previous_history, previous_best
    stopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                             patience=12, restore_best_weights=True)
    scheduler = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5,
                                                 factor=0.5, min_lr=1e-6, verbose=1)
    model.fit(train, validation_data=validation, initial_epoch=initial_epoch, epochs=epochs, verbose=2,
              callbacks=[keras.callbacks.TerminateOnNaN(), stopping, scheduler,
                         RestoreCallbackState(previous_history, stopping, scheduler, best_path),
                         checkpoint, MetricsCSV(str(directory / f"training_{phase}.csv"),
                                                append=bool(previous_history)), progress])
    return progress.history, checkpoint.best


def load_cnn_resume(directory, best_path, config):
    history = json.loads((directory / "history_adaptation.json").read_text(encoding="utf-8"))
    count = len(history["loss"])
    if not count or any(len(values) != count for values in history.values()):
        raise ValueError("Incomplete saved CNN history")
    with (directory / "training_adaptation.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in rows] != list(range(count)):
        raise ValueError("CSV/history epochs disagree; refusing an ambiguous resume")
    latest = ROOT / "models/kdef_checkpoints/cnn_adaptation_latest.keras"
    model = keras.models.load_model(latest)  # Restore Adam state and LR as well as weights.
    expected_steps = count * ((config["train_samples"] + config["batch"] - 1) // config["batch"])
    if int(model.optimizer.iterations.numpy()) != expected_steps:
        raise ValueError("Latest checkpoint does not match the saved epoch history")
    if not best_path.is_file() or list(model.input_shape[1:]) != config["input_shape"]:
        raise ValueError("Missing best checkpoint or incompatible model input")
    print(f"Resuming CNN after {count} complete epochs; optimizer step {expected_steps}; "
          f"LR {float(model.optimizer.learning_rate.numpy()):.8g}", flush=True)
    return model, history


def smoke_check(manifest, root):
    """Verify training, both fine-tuning stages, and serialization on train only."""
    for name in ("cnn", "mobilenet"):
        keras.backend.clear_session()
        keras.utils.set_random_seed(42)
        size, channels = (CNN_SIZE, 1) if name == "cnn" else (MOBILENET_SIZE, 3)
        dataset = make_dataset(manifest, root, "train", size, channels, batch=2)
        # take(1) is for a smoke check only; the training process caches all data.
        x, y = next(iter(dataset))
        if name == "cnn":
            model = build_cnn()
        else:
            model, base = build_mobilenet()
        compile_model(model, 1e-4)
        metrics = model.train_on_batch(x, y, return_dict=True)
        if not all(np.isfinite(float(value)) for value in metrics.values()):
            raise ValueError("Smoke check produced non-finite metrics")
        if name == "mobilenet":
            unfreeze_mobilenet(base)
            assert all(not layer.trainable for layer in base.layers if isinstance(layer, keras.layers.BatchNormalization))
            compile_model(model, 5e-5)
            metrics = model.train_on_batch(x, y, return_dict=True)
            if not all(np.isfinite(float(value)) for value in metrics.values()):
                raise ValueError("Fine-tuning smoke check failed")
        probabilities = model(x, training=False).numpy()
        path = REPORTS / f"smoke_{name}.keras"
        model.save(path)
        restored = keras.models.load_model(path, compile=False)
        np.testing.assert_allclose(restored(x, training=False).numpy(), probabilities, atol=1e-5)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        print(f"SMOKE OK: {name}; input {x.shape}; training and save/reload verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "mobilenet"], default="cnn")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--frozen-epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--smoke-check", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted CNN KDEF run")
    args = parser.parse_args()
    if min(args.epochs, args.frozen_epochs, args.batch, args.threads) < 1:
        parser.error("Epochs, batch and threads must be positive")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(42)
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest, root, manifest_hash = read_manifest()
    if args.smoke_check:
        smoke_check(manifest, root)
        return
    best_path = ROOT / "models" / f"emotion_{args.model}_kdef.keras"
    directory = REPORTS / args.model
    if args.resume and args.model != "cnn":
        parser.error("This resume option currently restores the CNN stage only")
    if not args.resume and (best_path.exists() or (directory / "config.json").exists()):
        raise FileExistsError("KDEF model/run already exists. Inspect it before starting a new experiment.")
    directory.mkdir(parents=True, exist_ok=True)
    size, channels = (CNN_SIZE, 1) if args.model == "cnn" else (MOBILENET_SIZE, 3)
    source = Path(CNN_PATH) if args.model == "cnn" else IMAGENET_WEIGHTS
    config = {"dataset": "KDEF original", "model": args.model, "seed": 42,
              "manifest_sha256": manifest_hash, "input_shape": [size, size, channels],
              "preprocessing": PREPROCESSING, "initial_weights": str(source.relative_to(ROOT)),
              "initial_weights_sha256": file_hash(source), "batch": args.batch,
              "epochs_limit": args.epochs, "frozen_epochs_limit": args.frozen_epochs if args.model == "mobilenet" else 0,
              "class_weights": None, "selection_metric": "val_accuracy", "test_loaded": False,
              "train_samples": len(manifest["records"]["train"]),
              "validation_samples": len(manifest["records"]["validation"]),
              "tensorflow": tf.__version__, "keras": keras.__version__,
              "code_sha256": {name: file_hash(ROOT / "src" / name) for name in
                              ("train_kdef.py", "data_kdef.py", "model_kdef.py", "model_cnn.py")}}
    if args.resume:
        original = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        for key in ("dataset", "model", "seed", "manifest_sha256", "input_shape", "preprocessing",
                    "initial_weights_sha256", "batch", "epochs_limit", "train_samples", "validation_samples"):
            if original[key] != config[key]:
                raise ValueError(f"Resume settings differ: {key}")
        events_path = directory / "resume_events.json"
        events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.exists() else []
        events.append({"utc": datetime.now(timezone.utc).isoformat(), "code_sha256": config["code_sha256"]})
        save_json(events_path, events)
        config = original
    else:
        save_json(directory / "config.json", config)
    print(json.dumps(config, indent=2), flush=True)
    print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)
    train = make_dataset(manifest, root, "train", size, channels, args.batch, training=True)
    validation = make_dataset(manifest, root, "validation", size, channels, args.batch)
    if args.model == "cnn":
        previous_history = None
        if args.resume:
            model, previous_history = load_cnn_resume(directory, best_path, config)
        else:
            model = build_cnn()
            compile_model(model, 1e-4)
        print("CNN: adapting a copy of FER-2013 weights to KDEF", flush=True)
        history, best = fit_stage(model, "cnn", "adaptation", args.epochs, train, validation, directory, best_path,
                                  previous_best=max(previous_history["val_accuracy"]) if previous_history else 0.0,
                                  previous_history=previous_history)
    else:
        model, base = build_mobilenet()
        compile_model(model, 1e-3)
        print("MobileNetV2: frozen ImageNet backbone", flush=True)
        first, best = fit_stage(model, "mobilenet", "frozen", args.frozen_epochs,
                                train, validation, directory, best_path)
        # Start the second stage from the best frozen checkpoint.
        model = keras.models.load_model(best_path, compile=False)
        base = next(layer for layer in model.layers if isinstance(layer, keras.Model)
                    and layer.name.startswith("mobilenetv2"))
        unfreeze_mobilenet(base)
        compile_model(model, 5e-5)
        print("MobileNetV2: fine-tuning top 100 layers; BatchNorm stays frozen", flush=True)
        second, best = fit_stage(model, "mobilenet", "finetuning", args.epochs,
                                 train, validation, directory, best_path,
                                 previous_best=best, offset=len(first["loss"]))
        history = {key: first[key] + second[key] for key in first}
    if file_hash(source) != config["initial_weights_sha256"]:
        raise RuntimeError("Initial checkpoint unexpectedly changed")
    _, _, current_hash = read_manifest()
    if current_hash != manifest_hash:
        raise RuntimeError("KDEF split changed during training")
    save_json(directory / "history.json", history)
    plot_history(history, directory / "history.png")
    save_json(directory / "result.json", {"state": "finished", "best_val_accuracy": best,
              "epochs_completed": len(history["loss"]), "model": best_path.name,
              "model_sha256": file_hash(best_path), "manifest_sha256": manifest_hash,
              "test_evaluated": False})
    save_json(directory / "status.json", {"state": "finished", "best_val_accuracy": best,
                                           "epochs_completed": len(history["loss"])})
    print(f"Training finished: best validation accuracy {best:.4%}. Test not evaluated.", flush=True)


if __name__ == "__main__":
    main()
