# src/train.py
import argparse
import os
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from config import CNN_PATH, MOBILENET_PATH, MODELS_DIR, REPORTS_DIR
from data import load_datasets, compute_class_weights
from model_cnn import build_cnn
from model_transfer import build_mobilenet, unfreeze_top


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


class TrainingReporter(keras.callbacks.Callback):
    """Keep progress, metrics and graphs even while a long CPU run is active."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.history = {}
        self.started = time.time()
        self.status_path = Path(REPORTS_DIR) / f"status_{name}.json"

    def write_status(self, **values):
        values["elapsed_seconds"] = round(time.time() - self.started, 1)
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
        temporary.replace(self.status_path)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.write_status(state="training", epoch=self.epoch, batch=0)

    def on_train_batch_end(self, batch, logs=None):
        if batch == 0 or (batch + 1) % 50 == 0:
            self.write_status(state="training", epoch=self.epoch, batch=batch + 1,
                              training_metrics={k: float(v) for k, v in (logs or {}).items()})

    def on_epoch_end(self, epoch, logs=None):
        metrics = {k: float(v) for k, v in (logs or {}).items()}
        for key, value in metrics.items():
            self.history.setdefault(key, []).append(value)
        Path(REPORTS_DIR, f"history_{self.name}.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8")
        plot_history(self.history, self.name)
        self.write_status(state="epoch_complete", epoch=epoch + 1, metrics=metrics,
                          best_val_accuracy=max(self.history["val_accuracy"]))

    def on_train_end(self, logs=None):
        self.write_status(state="finished", epochs_completed=len(self.history.get("loss", [])),
                          best_val_accuracy=max(self.history.get("val_accuracy", [0.0])))


def get_callbacks(path, previous_best=None, name=None):
    return [
        keras.callbacks.ModelCheckpoint(path, monitor="val_accuracy", save_best_only=True,
                                        initial_value_threshold=previous_best, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(os.path.join(REPORTS_DIR, f"training_{name or Path(path).stem}.csv")),
        TrainingReporter(name or Path(path).stem),
    ]


def plot_history(history, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, ax in [("loss", ax1), ("accuracy", ax2)]:
        ax.plot(history[key], label="train")
        ax.plot(history["val_" + key], label="validation")
        ax.set_title(key); ax.set_xlabel("эпоха"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"history_{name}.png"), dpi=150)
    plt.close(fig)


def merge(*histories):
    out = {}
    for h in histories:
        for k, v in h.history.items():
            out.setdefault(k, []).extend(v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "mobilenet"], default="cnn")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = p.parse_args()
    if min(args.epochs, args.batch, args.threads) < 1:
        p.error("epochs, batch and threads must be positive")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(REPORTS_DIR, exist_ok=True)
    keras.utils.set_random_seed(42)  # нәтижені қайталауға болатындай
    print("GPU:", tf.config.list_physical_devices("GPU"))

    train_ds, val_ds, _, split = load_datasets(
        batch_size=args.batch, include_test=False, return_metadata=True)
    if set(split["train"]) & set(split["validation"]):
        raise RuntimeError("Training and validation sets overlap")
    Path(REPORTS_DIR, "data_split_seed42.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8")
    class_weight = compute_class_weights(split["train"])
    Path(REPORTS_DIR, "class_weights.json").write_text(
        json.dumps(class_weight, indent=2), encoding="utf-8")
    print(f"Train: {len(split['train'])}; validation: {len(split['validation'])}; test: not loaded")
    print("Class weights:", class_weight)

    if args.model == "cnn":
        path = CNN_PATH
        model = build_cnn()
        compile_model(model, 1e-3)
        model.summary()
        h = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                      class_weight=class_weight, callbacks=get_callbacks(path, name="cnn"), verbose=2)
        history = merge(h)
    else:
        path = MOBILENET_PATH
        model, base = build_mobilenet()
        print("1-кезең: frozen base")
        compile_model(model, 1e-3)
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=10,
                       class_weight=class_weight, callbacks=get_callbacks(path))
        print("2-кезең: fine-tuning")
        unfreeze_top(base, 30)
        compile_model(model, 1e-5)  # trainable өзгерген соң қайта compile міндетті!
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                       class_weight=class_weight,
                       callbacks=get_callbacks(path, max(h1.history["val_accuracy"])))
        history = merge(h1, h2)

    plot_history(history, args.model)
    # ModelCheckpoint already saved the best model. Do not overwrite it or use
    # test results while choosing settings. Run evaluate.py for the final report.
    print(f"Saved checkpoint: {path}. Final test evaluation: python evaluate.py --model {path}")


if __name__ == "__main__":
    main()
