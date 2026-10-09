"""
data_preprocessing.py
---------------------
Handles loading, cleaning, encoding, and splitting the Bank Marketing dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load the Bank Marketing CSV dataset."""
    df = pd.read_csv(filepath, sep=";")
    return df


def encode_features(df: pd.DataFrame, target_col: str = "y"):
    """
    One-hot encode categorical features and label-encode the target.

    Returns
    -------
    X_encoded : pd.DataFrame
    y_encoded : pd.Series  (0 / 1)
    feature_names : list[str]
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].map({"no": 0, "yes": 1})

    X_encoded = pd.get_dummies(X)
    return X_encoded, y, list(X_encoded.columns)


def split_data(X, y, val_size: float = 0.1, test_size: float = 0.1, random_state: int = 42):
    """
    Split into train / validation / test sets.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
