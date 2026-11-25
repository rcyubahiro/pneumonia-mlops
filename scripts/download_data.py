"""Data acquisition: download the Kaggle "Chest X-Ray Images (Pneumonia)" dataset
(paultimothymooney/chest-xray-pneumonia, ~5,860 JPEGs, classes NORMAL / PNEUMONIA).

Needs a Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com -> Settings -> Create New Token).

    python scripts/download_data.py
    python scripts/download_data.py --zip path/to/chest-xray-pneumonia.zip   # if you downloaded it manually

The dataset's official split is kept: its `train` + `val` folders become data/train,
its `test` folder becomes data/test. Raw images are NOT committed to git (see .gitignore).
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

RAW = ROOT / "data" / "_raw"


def copy_split(src: Path, dest: Path):
    for cls in config.CLASS_NAMES:
        (dest / cls).mkdir(parents=True, exist_ok=True)
        for f in (src / cls).glob("*"):
            if f.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                shutil.copy(f, dest / cls / f.name)


def organise():
    # the zip nests folders (chest_xray/chest_xray/...) and may hold a __MACOSX copy: find the real root
    test_dirs = [p for p in RAW.rglob("test") if (p / "NORMAL").is_dir() and "__MACOSX" not in p.parts]
    if not test_dirs:
        raise SystemExit("Could not find train/test/NORMAL folders in the download.")
    root = test_dirs[0].parent
    for split in ("train", "val"):
        if (root / split).exists():
            copy_split(root / split, config.TRAIN_DIR)
    copy_split(root / "test", config.TEST_DIR)
    shutil.rmtree(RAW)
    for name, d in (("train", config.TRAIN_DIR), ("test", config.TEST_DIR)):
        print(name, {c: len(list((d / c).glob("*"))) for c in config.CLASS_NAMES})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="path to an already-downloaded zip")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if args.zip:
        zipfile.ZipFile(args.zip).extractall(RAW)
    else:
        subprocess.run(["kaggle", "datasets", "download", "-d", "paultimothymooney/chest-xray-pneumonia",
                        "-p", str(RAW), "--unzip"], check=True)
    organise()
