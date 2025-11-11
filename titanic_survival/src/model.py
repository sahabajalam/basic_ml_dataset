from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd


def _simple_preprocess(X: pd.DataFrame) -> pd.DataFrame:
    """A minimal preprocessing step: fillna and get dummies for categorical features.
    Replace with more robust preprocessing for real experiments.
    """
    X = X.copy()
    X = X.fillna(0)
    X = pd.get_dummies(X, drop_first=True)
    return X


def train(df: pd.DataFrame, target: str = "Survived", test_size: float = 0.2, random_state: int = 42):
    """Train a simple logistic regression model.

    Returns (model, accuracy_on_holdout, X_test, y_test)
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    X = df.drop(columns=[target])
    y = df[target]
    X = _simple_preprocess(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc, X_test, y_test


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
