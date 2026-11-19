import shutil
from pathlib import Path

import kagglehub

KAGGLE_DATASET = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
FILE_NAME = "IMDB Dataset.csv"
DATA_DIR = Path(_file_).resolve().parent.parent / "data"


def download() -> Path:
    target = DATA_DIR / FILE_NAME
    if target.exists():
        print(f"Plik już istnieje: {target}")
        return target

    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    DATA_DIR.mkdir(exist_ok=True)
    shutil.copy(cache_dir / FILE_NAME, target)
    print(f"Zapisano: {target}")
    return target


if _name_ == "_main_":
    download()
