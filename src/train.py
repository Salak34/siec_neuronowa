"""Pętla treningowa, ewaluacja i uruchomienie treningu modelu bazowego z terminala.

Użycie (z głównego katalogu projektu):
    python -m src.train --epochs 10
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import ROOT, make_loader, prepare_data
from .model import MeanEmbeddingClassifier, count_parameters

MODELS_DIR = ROOT / "models"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")  # GPU w Macach z Apple Silicon
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += ((logits > 0).float() == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += ((logits > 0).float() == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    """Prawdopodobieństwo klasy pozytywnej dla każdej recenzji."""
    model.eval()
    probs = [torch.sigmoid(model(x.to(device))).cpu() for x, _ in loader]
    return torch.cat(probs).numpy()


def fit(model, train_loader, val_loader, epochs: int = 10, lr: float = 1e-3,
        device=None, patience: int | None = 3, verbose: bool = True):
    """Trenuje model i na końcu przywraca wagi z epoki o najlepszej accuracy walidacyjnej.

    patience — wczesne zatrzymanie, gdy accuracy walidacyjna nie rośnie przez tyle epok.
    """
    device = device or get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best_acc, best_state, epochs_without_improvement = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        if verbose:
            print(f"Epoka {epoch:2d} | strata ucz. {train_loss:.4f}  acc ucz. {train_acc:.4f} | "
                  f"strata wal. {val_loss:.4f}  acc wal. {val_acc:.4f} | {time.time() - start:.1f}s")

        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                if verbose:
                    print(f"Wczesne zatrzymanie: brak poprawy przez {patience} epoki.")
                break

    model.load_state_dict(best_state)
    return history


def save_checkpoint(model, vocab, config: dict, history, name: str = "baseline") -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, path)
    vocab.save(MODELS_DIR / f"{name}_vocab.json")
    (MODELS_DIR / f"{name}_history.json").write_text(json.dumps(history, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description="Trening modelu bazowego (etap 1)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--vocab-size", type=int, default=20_000)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    print("Przygotowanie danych...")
    data = prepare_data(max_len=args.max_len, vocab_size=args.vocab_size, seed=args.seed)
    train_loader = make_loader(data["X_train"], data["y_train"], args.batch_size, shuffle=True)
    val_loader = make_loader(data["X_val"], data["y_val"], args.batch_size)

    config = {"vocab_size": len(data["vocab"]), "embed_dim": args.embed_dim,
              "hidden_dim": args.hidden_dim, "dropout": args.dropout, "max_len": args.max_len}
    model = MeanEmbeddingClassifier(**{k: v for k, v in config.items() if k != "max_len"})
    device = get_device()
    print(f"Urządzenie: {device}, parametry modelu: {count_parameters(model):,}")

    history = fit(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)
    path = save_checkpoint(model, data["vocab"], config, history)
    best = max(h["val_acc"] for h in history)
    print(f"Najlepsza accuracy walidacyjna: {best:.4f}. Zapisano model: {path}")


if __name__ == "__main__":
    main()
