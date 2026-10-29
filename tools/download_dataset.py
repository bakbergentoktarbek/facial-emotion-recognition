"""Download the public FER-2013 archive; never read Kaggle credentials."""
from pathlib import Path
import time
import urllib.request
from zipfile import ZipFile, is_zipfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "fer2013"
URL = "https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    archive_path = DESTINATION / "fer2013.zip"
    if not archive_path.exists():
        temporary = archive_path.with_suffix(".download")
        print("Downloading public FER-2013 archive...", flush=True)
        started = time.monotonic()
        with urllib.request.urlopen(URL, timeout=15) as response, temporary.open("wb") as output:
            total = 0
            while True:
                if time.monotonic() - started > 120:
                    raise TimeoutError("Download exceeded 2 minutes. Try the browser link in README.md.")
                block = response.read1(1024 * 1024)
                if not block:
                    break
                output.write(block)
                total += len(block)
        if not is_zipfile(temporary):
            raise RuntimeError("Kaggle returned a page instead of ZIP. Use the browser link in README.md.")
        temporary.replace(archive_path)
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (DESTINATION / member.filename).resolve()
            if not target.is_relative_to(DESTINATION.resolve()):
                raise ValueError(f"Unsafe archive path: {member.filename}")
        print(f"Extracting {len(archive.infolist())} entries...", flush=True)
        # Existing images are preserved if the command is run again.
        for member in archive.infolist():
            if not (DESTINATION / member.filename).exists():
                archive.extract(member, DESTINATION)
    print(f"Dataset ready: {DESTINATION}", flush=True)


if __name__ == "__main__":
    main()
