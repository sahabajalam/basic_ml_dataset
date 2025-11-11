import re
from typing import Tuple, Optional

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data from the given path into a DataFrame.

    This is a thin wrapper around `pd.read_csv` kept for compatibility with
    the rest of the project (and tests). It will raise the same errors as
    pandas when the file is not found or is malformed.
    """
    return pd.read_csv(path)


def _extract_title(name: str) -> str:
    """Extract title (Mr, Mrs, Miss, etc.) from a passenger name.

    If the name doesn't match the expected pattern, returns 'Unknown'.
    """
    if not isinstance(name, str):
        return "Unknown"
    m = re.search(r",\s*([^\.]+)\.", name)
    if m:
        return m.group(1).strip()
    return "Unknown"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform common Titanic dataset cleaning operations.

    Operations performed (best-effort; safe to run on minimal synthetic
    DataFrames used in tests):
    - Fill numeric missing values with sensible defaults (median-based)
    - Fill categorical missing values with mode or 'Missing'
    - Drop columns that are generally not useful for this simple model
      (PassengerId, Ticket, Cabin) if they exist
    """
    df = df.copy()

    # Drop obvious identifiers that won't help a generic model
    for c in ("PassengerId", "Ticket", "Cabin"):
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Age and Fare: numeric fills
    if "Age" in df.columns:
        try:
            age_med = df["Age"].median()
            df["Age"] = df["Age"].fillna(age_med)
        except Exception:
            df["Age"] = df["Age"].fillna(0)

    if "Fare" in df.columns:
        try:
            fare_med = df["Fare"].median()
            df["Fare"] = df["Fare"].fillna(fare_med)
        except Exception:
            df["Fare"] = df["Fare"].fillna(0)

    # Embarked and Sex: categorical fills
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0] if not df["Embarked"].mode().empty else "Missing")

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].fillna("missing")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple engineered features commonly used for Titanic models.

    - Extract `Title` from `Name` (if present)
    - Create `IsAlone` from `SibSp` and `Parch`
    - Optionally create `FamilySize`
    """
    df = df.copy()

    # Title
    if "Name" in df.columns:
        df["Title"] = df["Name"].apply(_extract_title)
    else:
        df["Title"] = "Unknown"

    # IsAlone and FamilySize
    if "SibSp" in df.columns or "Parch" in df.columns:
        sibsp = df["SibSp"] if "SibSp" in df.columns else 0
        parch = df["Parch"] if "Parch" in df.columns else 0
        df["FamilySize"] = sibsp + parch + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    else:
        df["FamilySize"] = 1
        df["IsAlone"] = 1

    return df


def encode_features(df: pd.DataFrame, drop_name: bool = True) -> pd.DataFrame:
    """Encode categorical variables and drop unused columns.

    - Convert categorical variables to dummies (Sex, Embarked, Title)
    - Optionally drop the original `Name` column
    - Ensures numeric dtype for ML models
    """
    df = df.copy()

    # Columns we generally want to drop after feature extraction
    for c in ("Name",):
        if drop_name and c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Categorical columns that are safe to one-hot encode if present
    cat_cols = [c for c in ("Sex", "Embarked", "Title") if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Ensure numeric types where possible
    for c in df.columns:
        if df[c].dtype == object:
            # try to coerce to numeric (non-convertible -> NaN); fillna below
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.fillna(0)
    return df


def preprocess(df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
    """Full preprocessing pipeline applied in order.

    If `target` is provided, the target column will be left intact in the
    returned DataFrame (i.e., preprocessing is applied only to feature columns).
    """
    df = df.copy()
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_features(df)
    # keep target column if present and requested
    if target and target in df.columns:
        # ensure target stays as the last column for readability (not required)
        cols = [c for c in df.columns if c != target] + [target]
        df = df[cols]
    return df


def split_features_target(df: pd.DataFrame, target: str = "Survived") -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into X, y by target column.

    Keeps the function signature from the previous implementation so other
    modules/tests depending on it continue to work.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


__all__ = [
    "load_data",
    "clean_data",
    "engineer_features",
    "encode_features",
    "preprocess",
    "split_features_target",
]

