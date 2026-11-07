"""Start the two KDEF runs in one persistent background job, or execute its worker."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/kdef"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_status(**values):
    values["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = REPORTS / "run_status.tmp"
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(REPORTS / "run_status.json")


def worker(resume=False):
    protected = [ROOT / "models/emotion_cnn.keras", ROOT / "models/emotion_mobilenet.keras",
                 ROOT / "reports/data_split_seed42.json", REPORTS / "data_split_seed42.json"]
    initial = {str(path.relative_to(ROOT)): digest(path) for path in protected}

    def verify_originals():
        if any(digest(ROOT / name) != value for name, value in initial.items()):
            raise RuntimeError("An original checkpoint or data split changed")

    for name in ("cnn", "mobilenet"):
        if resume and (REPORTS / name / "result.json").exists():
            result = json.loads((REPORTS / name / "result.json").read_text(encoding="utf-8"))
            if digest(ROOT / "models" / result["model"]) != result["model_sha256"]:
                raise RuntimeError(f"Completed {name} checkpoint has changed")
            print(f"Keeping completed {name} run", flush=True)
            continue
        save_status(state="training", model=name, protected_sha256=initial)
        command = [sys.executable, "-u", str(ROOT / "src/train_kdef.py"), "--model", name,
                   "--epochs", "60", "--frozen-epochs", "8", "--batch", "32", "--threads", "8"]
        if resume and (REPORTS / name / "config.json").exists():
            if name != "cnn":
                raise RuntimeError("Interrupted MobileNet run requires stage-specific recovery")
            command.append("--resume")
        print(f"Starting KDEF training: {name}", flush=True)
        with (REPORTS / f"training_{name}.log").open("a" if resume else "w", encoding="utf-8") as log:
            if resume:
                log.write(f"\n=== Recovery {datetime.now(timezone.utc).isoformat()} ===\n")
                log.flush()
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, check=True)
        verify_originals()
    validation = {name: json.loads((REPORTS / name / "result.json").read_text(encoding="utf-8"))
                  for name in ("cnn", "mobilenet")}
    # Avoid using the test set to tune models that are not ready on validation.
    if any(result["best_val_accuracy"] <= 0.70 for result in validation.values()):
        save_status(state="validation_review_required", validation=validation,
                    message="Training finished. At least one model is not above 70% on validation. Test remains unopened.")
        print("Validation review required before final test evaluation.", flush=True)
        return
    comparison = {}
    for name in ("cnn", "mobilenet"):
        save_status(state="evaluating", model=name, validation=validation)
        with (REPORTS / f"evaluation_{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", str(ROOT / "src/evaluate_kdef.py"), "--model", name],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)
        metrics = json.loads((REPORTS / name / "metrics.json").read_text(encoding="utf-8"))
        comparison[name] = {key: metrics[key] for key in
                            ("model", "model_sha256", "test_samples", "accuracy", "macro_f1", "weighted_f1", "above_70_percent")}
        comparison[name]["best_val_accuracy"] = validation[name]["best_val_accuracy"]
    verify_originals()
    (REPORTS / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    save_status(state="finished", comparison=comparison,
                target_met=all(value["above_70_percent"] for value in comparison.values()))
    print(json.dumps(comparison, indent=2), flush=True)


def launch(resume=False):
    REPORTS.mkdir(parents=True, exist_ok=True)
    record_path = REPORTS / "run.json"
    if record_path.exists() and not resume:
        raise FileExistsError("A KDEF run is already recorded. Inspect reports/kdef/run_status.json before restarting.")
    if resume:
        if not record_path.exists():
            raise FileNotFoundError("No existing KDEF run to resume")
        status = json.loads((REPORTS / "run_status.json").read_text(encoding="utf-8"))
        if status["state"] != "failed":
            raise RuntimeError("Resume is allowed only for a confirmed failed run")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        shutil.copyfile(record_path, REPORTS / f"run_before_resume_{stamp}.json")
    else:
        for name in ("cnn", "mobilenet"):
            if (ROOT / "models" / f"emotion_{name}_kdef.keras").exists() or (REPORTS / name / "config.json").exists():
                raise FileExistsError("An existing KDEF run/model would be overwritten")
    environment = os.environ.copy()
    environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
                        "MPLCONFIGDIR": str(ROOT / "reports/.mplconfig"),
                        "KERAS_HOME": str(ROOT / "reports/.keras"), "TF_CPP_MIN_LOG_LEVEL": "2"})
    arguments = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"]
    if resume:
        arguments.append("--resume")
    # Reserve a run record exclusively before launching the background process.
    with record_path.open("w" if resume else "x", encoding="utf-8") as record:
        with (REPORTS / "pipeline.log").open("a" if resume else "w", encoding="utf-8") as log:
            if resume:
                log.write(f"\n=== Recovery {datetime.now(timezone.utc).isoformat()} ===\n")
                log.flush()
            process = subprocess.Popen(
                arguments, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                if os.name == "nt" else 0)
        values = {"pid": process.pid, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "command": arguments, "cnn_epochs_limit": 60,
                  "mobilenet_frozen_epochs_limit": 8, "mobilenet_finetune_epochs_limit": 60,
                  "batch_size": 32, "resumed": resume}
        record.write(json.dumps(values, indent=2))
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help="Resume a failed CNN run without discarding saved epochs")
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.resume)
        except Exception as error:
            save_status(state="failed", error=str(error))
            raise
    else:
        launch(args.resume)
