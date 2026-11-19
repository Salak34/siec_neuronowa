
machta rap fan
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .preprocessing import Vocab, clean_text, encode_texts, tokenize

ROOT = Path(_file_).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "IMDB Dataset.csv"
LABELS = {"negative": 0, "positive": 1}


def load_imdb(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df["sentiment"].map(LABELS).astype(int)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset="review").reset_index(drop=True)


def split_data(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1, seed: int = 42):
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val["label"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class ReviewDataset(Dataset):
    def __init__(self, ids: np.ndarray, labels: np.ndarray):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


def make_loader(ids, labels, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    return DataLoader(ReviewDataset(ids, labels), batch_size=batch_size, shuffle=shuffle)


def prepare_data(
    csv_path: str | Path = DEFAULT_CSV,
    max_len: int = 300,
    vocab_size: int = 20_000,
    min_freq: int = 2,
    seed: int = 42,
) -> dict:
    df = remove_duplicates(load_imdb(csv_path))
    df["clean"] = df["review"].map(clean_text)
    df["tokens"] = df["clean"].map(tokenize)

    train, val, test = split_data(df, seed=seed)
    vocab = Vocab.build(train["tokens"], max_size=vocab_size, min_freq=min_freq)

    out = {"vocab": vocab, "train": train, "val": val, "test": test, "max_len": max_len}
    for name, part in (("train", train), ("val", val), ("test", test)):
        out[f"X_{name}"] = encode_texts(part["tokens"], vocab, max_len)
        out[f"y_{name}"] = part["label"].to_numpy()
    return out
