"""Launch CNN training in a hidden process with a persistent log and PID."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
reports = ROOT / "reports"
reports.mkdir(exist_ok=True)
environment = os.environ.copy()
environment.update({
    "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
    "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(reports / ".mplconfig"),
    "KERAS_HOME": str(reports / ".keras"), "TF_CPP_MIN_LOG_LEVEL": "2",
})
arguments = [sys.executable, "-u", str(ROOT / "src" / "train.py"),
             "--model", "cnn", "--epochs", "60", "--batch", "64"]
log_path = reports / "training_cnn.log"
if (reports / "training_run.json").exists():
    raise SystemExit("A run is already recorded in reports/training_run.json. Check it before starting another.")
with log_path.open("w", encoding="utf-8") as log:
    process = subprocess.Popen(
        arguments, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
        creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        if os.name == "nt" else 0,
        stdin=subprocess.DEVNULL,
    )
record = {"pid": process.pid, "started_utc": datetime.now(timezone.utc).isoformat(),
          "command": arguments, "log": str(log_path), "epochs_limit": 60, "batch_size": 64}
(reports / "training_run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
