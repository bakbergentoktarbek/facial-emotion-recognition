"""Extract the teacher's examples, preserving Word line breaks. Never overwrite files."""
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def main():
    document = next(ROOT.glob("*.docx"))
    with ZipFile(document) as archive:
        tree = ET.fromstring(archive.read("word/document.xml"))
    for paragraph in tree.iter(NS + "p"):
        code = "".join(
            node.text or "" if node.tag == NS + "t" else
            "\n" if node.tag in (NS + "br", NS + "cr") else
            "\t" if node.tag == NS + "tab" else ""
            for node in paragraph.iter()
        )
        first = code.splitlines()[0] if code else ""
        if first.startswith("# src/") and first.endswith(".py"):
            name = first.removeprefix("# ")
        elif code.startswith("tensorflow=="):
            name = "requirements.txt"
        else:
            continue
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(code + "\n", encoding="utf-8")
            print(name)
    for name in ("models", "reports", "fonts", "data/fer2013"):
        directory = ROOT / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch()


if __name__ == "__main__":
    main()
