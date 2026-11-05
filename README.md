# Robustezza e verifica formale di reti neurali su GTSRB

[ 🇬🇧 Read this in English ](ENGLISH_README.md)

Progetto per l'esame di **Introduction to Machine Learning** (A.A. 2025/2026), Università degli Studi di Trieste.

Confronto tra modello standard (ERM) e modello robusto (*adversarial training* con PGD) su 5 classi di segnali stradali GTSRB (STOP, Precedenza, Limite 30, Limite 50, Divieto), valutando la robustezza empirica sotto attacco **PGD** e la certificazione formale esatta con il solver SMT **Marabou**.

## Documenti

- **Relazione tecnica**: [`REPORT.md`](REPORT.md)
- **Presentazione**: [`presentazione.pdf`](presentazione.pdf)

## Setup

Il progetto utilizza [`uv`](https://docs.astral.sh/uv/) per la gestione dell'ambiente virtuale e delle dipendenze dichiarate in `pyproject.toml`:

```bash
uv sync
```

## Utilizzo

L'intera pipeline sperimentale (ispezione dati, training, attacchi PGD, matrici di confusione e verifica formale con Marabou) è implementata in modo autocontenuto e riproducibile nel notebook [`main.ipynb`](main.ipynb):

```bash
uv run jupyter notebook main.ipynb
```
