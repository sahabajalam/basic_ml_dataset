"""CLI entrypoint for training/predicting.

Usage examples:
python -m src.predict --train data/train.csv --save models/model.joblib
python -m src.predict --predict data/new.csv --model models/model.joblib --out predictions.csv
"""
import argparse
import os
import pandas as pd
from .data import load_data
from .model import train, save_model, load_model


def main():
    parser = argparse.ArgumentParser(description="Train or predict with a simple Titanic model")
    parser.add_argument("--train", help="Path to training CSV (contains target column 'Survived')")
    parser.add_argument("--save", help="Where to save trained model")
    parser.add_argument("--model", help="Path to saved model for prediction")
    parser.add_argument("--predict", help="Path to CSV to run predictions on (no target column expected)")
    parser.add_argument("--out", help="Output CSV path for predictions")
    args = parser.parse_args()

    if args.train:
        df = load_data(args.train)
        model, acc, _, _ = train(df)
        print(f"Trained model. Holdout accuracy: {acc:.4f}")
        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            save_model(model, args.save)
            print(f"Saved model to {args.save}")

    if args.predict:
        if not args.model:
            raise SystemExit("--model is required for prediction")
        model = load_model(args.model)
        df = load_data(args.predict)
        # minimal preprocessing to match train: rely on model expecting same columns
        # for now, use model.predict on raw/dummified features
        X = pd.get_dummies(df, drop_first=True).fillna(0)
        preds = model.predict(X)
        out_df = df.copy()
        out_df["prediction"] = preds
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            out_df.to_csv(args.out, index=False)
            print(f"Wrote predictions to {args.out}")
        else:
            print(out_df.head())


if __name__ == "__main__":
    main()
