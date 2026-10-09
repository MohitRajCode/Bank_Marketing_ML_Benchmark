"""
train.py
--------
Train Logistic Regression, Random Forest, and XGBoost models on the
Bank Marketing dataset and record training times.
"""

import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_preprocessing import load_data, encode_features, split_data


def train_logistic_regression(X_train, y_train, random_state: int = 42):
    start = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr.fit(X_train, y_train)
    elapsed = round(time.time() - start, 4)
    return lr, elapsed


def train_random_forest(X_train, y_train, n_estimators: int = 100,
                        max_depth: int = 10, random_state: int = 42):
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    elapsed = round(time.time() - start, 4)
    return rf, elapsed


def train_xgboost(X_train, y_train, X_val, y_val, random_state: int = 42):
    start = time.time()
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        use_label_encoder=False,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=random_state,
        verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = round(time.time() - start, 4)
    return xgb, elapsed


if __name__ == "__main__":
    df = load_data("data/bank-additional-full.csv")
    X, y, _ = encode_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    lr, lr_time = train_logistic_regression(X_train, y_train)
    rf, rf_time = train_random_forest(X_train, y_train)
    xgb_model, xgb_time = train_xgboost(X_train, y_train, X_val, y_val)

    print(f"LR training time  : {lr_time}s")
    print(f"RF training time  : {rf_time}s")
    print(f"XGB training time : {xgb_time}s")
