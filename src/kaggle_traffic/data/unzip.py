import zipfile
from pathlib import Path

def unzip_file(zip_path: Path, extract_to: Path) -> None:

    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)