# Klasyfikacja sentymentu wiadomości tekstowej

Projekt z przedmiotu **Sieci Neuronowe** — Karol Salamon (280091), Maciej Soroka (284196).

Sieć neuronowa, która dla wiadomości tekstowej (recenzji, opinii) określa, czy jej wydźwięk jest
**pozytywny** czy **negatywny**. Dane: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Uruchomienie

Wymagany Python 3.12 (zalecany [uv](https://docs.astral.sh/uv/)).

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

python -m src.download_data          # pobiera dane z Kaggle do data/ (bez logowania)
python -m src.train                  # trening modelu bazowego, zapis do models/
python -m src.predict "Not bad at all, I enjoyed it."
```

Główny materiał etapu 1 to notatnik [`notebooks/01_etap1_dane_i_model_bazowy.ipynb`](notebooks/01_etap1_dane_i_model_bazowy.ipynb)
(otwórz w VS Code lub `jupyter lab`, kernel z `.venv`). Wykonuje się w ok. 40 s i sam pobiera dane, jeśli ich brakuje.

## Struktura

```
data/                     zbiór IMDB (nie jest w repozytorium)
notebooks/                notatniki z analizą i wynikami dla kolejnych etapów
src/
  download_data.py        pobranie danych z Kaggle
  preprocessing.py        czyszczenie tekstu, tokenizacja, słownik (Vocab)
  data.py                 wczytanie, usunięcie duplikatów, podział, DataLoader
  model.py                architektury sieci
  train.py                pętla treningowa, ewaluacja, trening z terminala
  predict.py              klasyfikacja własnych wiadomości
models/                   zapisane wagi, słownik i historia treningu
```

## Etap 1 — przygotowanie danych i model bazowy

- 49 582 unikalne recenzje (usunięto 418 duplikatów), klasy zbalansowane.
- Czyszczenie: małe litery, usunięcie HTML (`<br />`), URL-i i interpunkcji; negacje (`not`, `don't`, ...) zostają.
- Podział 80/10/10 (uczący / walidacyjny / testowy) z zachowaniem proporcji klas.
- Słownik 20 000 słów budowany tylko na zbiorze uczącym; recenzje przycięte / dopełnione do 300 słów.
- Model bazowy: `Embedding(20000, 64)` → średnia po słowach → `Linear(64, 32)` + ReLU + Dropout → `Linear(32, 1)`.

| Model | Accuracy (walidacja) |
|---|---|
| Losowe zgadywanie | 50,0% |
| TF-IDF + regresja logistyczna (bez sieci) | 89,13% |
| Sieć bazowa (Embedding + średnia + MLP) | 89,11% |

Model bazowy nie widzi kolejności słów, więc np. „Not bad.” ocenia jako negatywne.
W etapie 2 sprawdzimy sieci rekurencyjne (LSTM / GRU), które czytają tekst po kolei.

## Plan dalszych etapów

- **Etap 2:** dwukierunkowy LSTM / GRU, dobór hiperparametrów, regularyzacja, ewentualnie wektory GloVe.
- **Etap 3:** ocena na zbiorze testowym (accuracy, precision, recall, F1, macierz pomyłek), analiza błędów, test na własnych wiadomościach.
- **Etap 4:** raport i prezentacja.
