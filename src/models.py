"""
All model definitions, consolidated into reusable functions (instead of the
original three copy-pasted, hardcoded-path notebook scripts).

Every function takes X_train/y_train/X_test/y_test and returns a fitted
estimator + metrics, so this module can be imported by the training
pipeline (src/train_pipeline.py) or by a notebook, without re-executing
side effects on import.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def _report(name, y_test, y_pred, y_prob):
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"ROC-AUC: {auc:.4f}")
    return auc


def train_logistic_regression(X_train, y_train, X_test, y_test, verbose=True):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:, 1]
    y_pred = lr.predict(X_test)
    if verbose:
        _report("Logistic Regression (baseline)", y_test, y_pred, y_prob)
    return lr, y_prob


def train_decision_tree(X_train, y_train, X_test, y_test, verbose=True):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_prob = dt.predict_proba(X_test)[:, 1]
    y_pred = dt.predict(X_test)
    if verbose:
        _report("Decision Tree (baseline)", y_test, y_pred, y_prob)
    return dt, y_prob


def train_weighted_logistic_regression(X_train, y_train, X_test, y_test, verbose=True):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight={0: 1, 1: 10})),
    ])
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    if verbose:
        _report("Weighted Logistic Regression (cost-sensitive)", y_test, y_pred, y_prob)
    return pipeline, y_prob


def train_smote_logistic_regression(X_train, y_train, X_test, y_test, verbose=True):
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_smote, y_train_smote)
    y_prob = lr.predict_proba(X_test)[:, 1]
    y_pred = lr.predict(X_test)
    if verbose:
        print("Before SMOTE:", dict(pd.Series(y_train).value_counts()))
        print("After SMOTE:", dict(pd.Series(y_train_smote).value_counts()))
        _report("SMOTE Logistic Regression", y_test, y_pred, y_prob)
    return lr, y_prob


def train_random_forest(X_train, y_train, X_test, y_test, verbose=True):
    rf = RandomForestClassifier(
        n_estimators=150, class_weight={0: 1, 1: 10}, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    if verbose:
        _report("Random Forest", y_test, y_pred, y_prob)
    return rf, y_prob


def train_xgboost(X_train, y_train, X_test, y_test, verbose=True):
    fraud_count = y_train.sum()
    non_fraud_count = len(y_train) - fraud_count
    scale_pos_weight = non_fraud_count / max(fraud_count, 1)

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    y_pred = xgb.predict(X_test)
    if verbose:
        _report("XGBoost (primary model)", y_test, y_pred, y_prob)
    return xgb, y_prob


def train_isolation_forest(X_train, y_train, X_test, y_test, contamination=0.002, verbose=True):
    X_train_legit = X_train[y_train == 0]
    iso = IsolationForest(n_estimators=150, contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(X_train_legit)
    raw_pred = iso.predict(X_test)
    y_pred = (raw_pred == -1).astype(int)
    if verbose:
        print("\n--- Isolation Forest (unsupervised, novel-pattern detector) ---")
        print(classification_report(y_test, y_pred, zero_division=0))
    return iso, y_pred
