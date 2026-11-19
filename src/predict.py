"""Klasyfikacja własnych, wpisanych ręcznie wiadomości.

Użycie (z głównego katalogu projektu, po treningu):
    python -m src.predict "This movie was absolutely wonderful!"
"""
import sys

import torch

from .model import MeanEmbeddingClassifier
from .preprocessing import Vocab, clean_text, encode_texts, tokenize
from .train import MODELS_DIR, get_device


@torch.no_grad()
def predict_texts(model, vocab: Vocab, texts: list[str], max_len: int, device=None) -> list[float]:
    """Zwraca prawdopodobieństwo klasy pozytywnej dla każdego tekstu."""
    device = device or next(model.parameters()).device
    tokens = [tokenize(clean_text(t)) for t in texts]
    x = torch.as_tensor(encode_texts(tokens, vocab, max_len)).to(device)
    model.eval()
    return torch.sigmoid(model(x)).cpu().tolist()


def load_model(name: str = "baseline", device=None):
    device = device or get_device()
    checkpoint = torch.load(MODELS_DIR / f"{name}.pt", map_location=device)
    config = checkpoint["config"]
    model = MeanEmbeddingClassifier(
        config["vocab_size"], config["embed_dim"], config["hidden_dim"], config["dropout"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    vocab = Vocab.load(MODELS_DIR / f"{name}_vocab.json")
    return model.to(device), vocab, config["max_len"]


def describe(prob: float) -> str:
    label = "pozytywny" if prob >= 0.5 else "negatywny"
    confidence = prob if prob >= 0.5 else 1 - prob
    return f"{label} ({confidence:.0%})"


if __name__ == "__main__":
    texts = sys.argv[1:] or [input("Wpisz wiadomość: ")]
    model, vocab, max_len = load_model()
    for text, prob in zip(texts, predict_texts(model, vocab, texts, max_len)):
        print(f"{describe(prob):>18}  <-  {text}")
