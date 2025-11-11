# Titanic Survival Prediction

A small Python project skeleton for predicting survival on the Titanic dataset (Kaggle). This repo provides a minimal, well-structured starting point with data loading, simple model training, and a CLI entrypoint.

## Project structure

- `data/` - place raw CSVs (train.csv, test.csv) here (not included)
- `notebooks/` - exploration and analysis notebooks
- `src/` - python package containing data processing and modeling code
- `models/` - saved model artifacts
- `tests/` - unit tests

## Setup

1. Create a virtual environment and activate it.

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Usage

Train a model on `data/train.csv` and save it to `models/model.joblib`:

```powershell
python -m src.predict --train data/train.csv --save models/model.joblib
```

Load a saved model and predict on a CSV with the same features (no target column):

```powershell
python -m src.predict --predict data/new.csv --model models/model.joblib --out predictions.csv
```

## Notes

- This is a starter scaffold. Replace the simple preprocessing with domain-appropriate feature engineering for best results.
