"""Czyszczenie tekstu, tokenizacja i zamiana słów na liczby (słownik)."""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --- Czyszczenie tekstu -------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_NOT_ALLOWED = re.compile(r"[^a-z0-9' ]+")  # zostawiamy litery, cyfry i apostrofy
_LOOSE_APOSTROPHE = re.compile(r"'(?![a-z])|(?<![a-z])'")  # apostrof nie wewnątrz słowa
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Zamienia surową recenzję na znormalizowany tekst.

    Kroki: małe litery, usunięcie znaczników HTML (np. <br />) i adresów URL,
    usunięcie akcentów, interpunkcji i zbędnych spacji. Apostrofy wewnątrz słów
    zostają, żeby "don't" czy "isn't" nie rozpadły się na osobne tokeny.
    """
    text = text.lower()
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.replace("’", "'").replace("`", "'")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = _NOT_ALLOWED.sub(" ", text)
    text = _LOOSE_APOSTROPHE.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Dzieli oczyszczony tekst na słowa."""
    return text.split()


# --- Stopwords (tylko do analizy danych, model widzi wszystkie słowa) ----------

# Słowa odwracające sens zdania — NIE usuwamy ich, bo "not good" != "good".
NEGATIONS = {
    "not", "no", "nor", "never", "cannot", "none", "nothing", "nobody", "neither",
    "nowhere", "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "weren't",
    "can't", "couldn't", "won't", "wouldn't", "shouldn't", "haven't", "hasn't",
}
_CONTRACTIONS = {
    "it's", "i'm", "i've", "i'd", "i'll", "that's", "there's", "he's", "she's",
    "you're", "you've", "you'll", "they're", "we're", "let's", "what's", "who's",
}
STOPWORDS = (set(ENGLISH_STOP_WORDS) | _CONTRACTIONS) - NEGATIONS


# --- Słownik: słowo <-> liczba --------------------------------------------------

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
PAD_IDX, UNK_IDX = 0, 1


class Vocab:
    """Słownik mapujący słowa na indeksy.

    Indeks 0 to <PAD> (dopełnienie do stałej długości),
    indeks 1 to <UNK> (słowo spoza słownika).
    """

    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {word: idx for idx, word in enumerate(itos)}

    @classmethod
    def build(cls, token_lists, max_size: int = 20_000, min_freq: int = 2) -> "Vocab":
        """Buduje słownik z najczęstszych słów. Wywoływać TYLKO na zbiorze uczącym."""
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        words = [w for w, c in counter.most_common(max_size - 2) if c >= min_freq]
        return cls([PAD_TOKEN, UNK_TOKEN] + words)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        """Zamienia listę słów na listę indeksów o długości dokładnie max_len.

        Za długie recenzje przycinamy od początku — zostaje końcówka, bo tam
        zwykle jest podsumowanie oceny. Za krótkie dopełniamy zerami (<PAD>).
        """
        ids = [self.stoi.get(t, UNK_IDX) for t in tokens][-max_len:]
        return ids + [PAD_IDX] * (max_len - len(ids))

    def decode(self, ids) -> list[str]:
        return [self.itos[i] for i in ids if i != PAD_IDX]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))


def encode_texts(token_lists, vocab: Vocab, max_len: int) -> np.ndarray:
    """Koduje wiele recenzji naraz -> macierz (liczba_recenzji, max_len)."""
    return np.array([vocab.encode(t, max_len) for t in token_lists], dtype=np.int64)
