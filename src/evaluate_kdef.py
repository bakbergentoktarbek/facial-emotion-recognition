"""One final test evaluation of a KDEF checkpoint; never modifies model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

from config import BASE_DIR, EMOTIONS_EN
from data_kdef import make_dataset, read_manifest

ROOT = Path(BASE_DIR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "mobilenet"], required=True)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    directory = ROOT / "reports/kdef" / args.model
    if (directory / "metrics.json").exists():
        raise FileExistsError("Final KDEF evaluation already exists; refusing to overwrite it")
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    manifest, root, manifest_hash = read_manifest()
    if result["manifest_sha256"] != manifest_hash:
        raise ValueError("Evaluation split differs from the training split")
    model_path = ROOT / "models" / result["model"]
    with model_path.open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if result["model_sha256"] != model_hash:
        raise ValueError("Checkpoint changed since training finished")
    model = keras.models.load_model(model_path, compile=False)
    size, width, channels = model.input_shape[1:]
    if size != width or channels not in (1, 3) or list(model.input_shape[1:]) != config["input_shape"]:
        raise ValueError("Unsupported model input shape")
    dataset = make_dataset(manifest, root, "test", size, channels, batch=32)
    truth = np.array([r["label"] for r in manifest["records"]["test"]])
    print(f"Final KDEF test: {len(truth)} photos, {len(manifest['subjects']['test'])} unseen people", flush=True)
    probabilities = model.predict(dataset, verbose=2)
    if probabilities.shape != (len(truth), 7) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid model probabilities")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-4)
    prediction = probabilities.argmax(axis=1)
    report = classification_report(truth, prediction, labels=list(range(7)), target_names=EMOTIONS_EN,
                                   output_dict=True, zero_division=0)
    text = classification_report(truth, prediction, labels=list(range(7)), target_names=EMOTIONS_EN,
                                 digits=4, zero_division=0)
    confusion = confusion_matrix(truth, prediction, labels=list(range(7)))
    per_angle = {}
    for angle in manifest["angles"]:
        selected = np.array([r["angle"] == angle for r in manifest["records"]["test"]])
        per_angle[angle] = {"samples": int(selected.sum()),
                            "accuracy": float(accuracy_score(truth[selected], prediction[selected]))}
    accuracy = float(accuracy_score(truth, prediction))
    metrics = {"dataset": "KDEF original", "model": model_path.name, "model_sha256": model_hash,
               "manifest_sha256": manifest_hash, "test_samples": len(truth),
               "test_subjects": manifest["subjects"]["test"], "class_names": EMOTIONS_EN,
               "accuracy": accuracy, "macro_f1": report["macro avg"]["f1-score"],
               "weighted_f1": report["weighted avg"]["f1-score"],
               "classification_report": report, "confusion_matrix": confusion.tolist(),
               "per_angle": per_angle, "above_70_percent": accuracy > 0.70}
    (directory / "classification_report.txt").write_text(text, encoding="utf-8")
    np.savez_compressed(directory / "predictions.npz", y_true=truth, y_pred=prediction, probabilities=probabilities)
    # Save image identifiers for analysis, not copyrighted KDEF face images.
    errors = [{"path": r["path"], "subject": r["subject"], "angle": r["angle"],
               "true": EMOTIONS_EN[int(truth[i])], "predicted": EMOTIONS_EN[int(prediction[i])],
               "confidence": float(probabilities[i].max())}
              for i, r in enumerate(manifest["records"]["test"]) if truth[i] != prediction[i]]
    (directory / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    normalized = confusion / confusion.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    heatmap = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set(xticks=range(7), yticks=range(7), xticklabels=EMOTIONS_EN, yticklabels=EMOTIONS_EN,
             xlabel="Predicted", ylabel="True", title=f"KDEF — {args.model}")
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right")
    for i in range(7):
        for j in range(7):
            axis.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                      color="white" if normalized[i, j] > 0.5 else "black")
    figure.colorbar(heatmap, ax=axis)
    figure.tight_layout()
    figure.savefig(directory / "confusion.png", dpi=140)
    plt.close(figure)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    result["test_evaluated"] = True
    result["test_metrics"] = f"reports/kdef/{args.model}/metrics.json"
    (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(text, flush=True)
    print(f"KDEF test accuracy: {accuracy:.4%}; macro F1: {metrics['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
