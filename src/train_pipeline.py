"""
End-to-end training pipeline.

Run with:  python -m src.train_pipeline

Produces (in api/artifacts/):
- fraud_model.pkl        (calibrated XGBoost model)
- fraud_model_raw.pkl     (raw model, used for SHAP)
- feature_order.json     (exact column order the model expects)
- threshold.json          (cost-optimal starting threshold + cost curve)
- reference_sample.csv    (raw sample, used for SHAP global importance / dashboard / drift monitor)

Uses data/raw/creditcard.csv if present (the real Kaggle dataset -- put it
there yourself, it's not redistributed in this repo for licensing/size
reasons). Otherwise falls back to the synthetic generator in
src/synthetic_data.py so the whole system is runnable out of the box for a
demo / portfolio walkthrough.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN_ESTIMATOR = True
except ImportError:
    _HAS_FROZEN_ESTIMATOR = False

from src.synthetic_data import generate_synthetic_dataset
from src.models import train_xgboost
from src.cost_optimization import find_optimal_threshold

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "api", "artifacts")
REAL_DATA_PATH = os.path.join(REPO_ROOT, "data", "raw", "creditcard.csv")

FEATURE_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def load_data():
    if os.path.exists(REAL_DATA_PATH):
        print(f"Using real dataset at {REAL_DATA_PATH}")
        df = pd.read_csv(REAL_DATA_PATH)
        # real Kaggle dataset has no entity keys -> synthesize a linkage layer
        # on top of it so ring detection still has something to work with.
        rng = np.random.default_rng(42)
        df["account_id"] = "ACC" + rng.integers(100000, 999999, len(df)).astype(str)
        df["device_id"] = "DEV" + rng.integers(1000, 9999, len(df)).astype(str)
        df["merchant_id"] = "MER" + rng.integers(1, 500, len(df)).astype(str)
    else:
        print("data/raw/creditcard.csv not found -- generating synthetic dataset for demo purposes.")
        df = generate_synthetic_dataset(n_transactions=40000)
    return df


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Training primary XGBoost model...")
    raw_model, _ = train_xgboost(X_train, y_train, X_test, y_test, verbose=True)

    print("\nCalibrating probabilities (sigmoid/Platt scaling) for reliable uncertainty...")
    if _HAS_FROZEN_ESTIMATOR:
        calibrated_model = CalibratedClassifierCV(FrozenEstimator(raw_model), method="sigmoid")
    else:
        calibrated_model = CalibratedClassifierCV(raw_model, method="sigmoid", cv="prefit")
    calibrated_model.fit(X_test, y_test)
    y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

    print("\nFinding cost-optimal decision threshold...")
    threshold_result = find_optimal_threshold(y_test.values, y_prob_calibrated)
    print(f"Best threshold: {threshold_result['best_threshold']}")
    print(f"Min cost: Rs.{threshold_result['min_cost']}  |  Cost @ 0.5: Rs.{threshold_result['default_cost_at_0.5']}")

    # --- save artifacts ---
    joblib.dump(calibrated_model, os.path.join(ARTIFACT_DIR, "fraud_model.pkl"))
    # raw (uncalibrated) model saved separately for SHAP TreeExplainer, which
    # does not support the CalibratedClassifierCV/FrozenEstimator wrapper.
    joblib.dump(raw_model, os.path.join(ARTIFACT_DIR, "fraud_model_raw.pkl"))

    with open(os.path.join(ARTIFACT_DIR, "feature_order.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    with open(os.path.join(ARTIFACT_DIR, "threshold.json"), "w") as f:
        json.dump({
            "best_threshold": threshold_result["best_threshold"],
            "min_cost": threshold_result["min_cost"],
            "default_cost_at_0.5": threshold_result["default_cost_at_0.5"],
            "savings_vs_default": threshold_result["savings_vs_default"],
        }, f, indent=2)

    # Reference sample + stats for the drift monitor (frozen "what training data looked like")
    reference_sample = X_train.sample(n=min(5000, len(X_train)), random_state=42)
    reference_sample.to_csv(os.path.join(ARTIFACT_DIR, "reference_sample.csv"), index=False)

    print(f"\nArtifacts written to {ARTIFACT_DIR}")
    print("Done. Start the API with: uvicorn api.app:app --reload")


if __name__ == "__main__":
    main()
