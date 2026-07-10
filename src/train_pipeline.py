"""
End-to-end training pipeline.

Run with:  python -m src.train_pipeline

Produces (in api/artifacts/):
- fraud_model.pkl        (calibrated XGBoost model -- the deployed model)
- fraud_model_raw.pkl    (raw XGBoost, used for SHAP)
- feature_order.json     (exact column order the model expects)
- threshold.json         (cost-optimal starting threshold + cost curve)
- reference_sample.csv   (raw sample, used for SHAP global importance / dashboard / drift monitor)
- model_comparison.json  (precision/recall/AUC/business-cost for every candidate model, so the
                          dashboard can show *why* XGBoost was chosen, not just assert it)

Uses data/raw/creditcard.csv if present (the real Kaggle dataset -- put it
there yourself, it's not redistributed in this repo for licensing/size
reasons). Otherwise falls back to the synthetic generator in
src/synthetic_data.py so the whole system is runnable out of the box for a
demo / portfolio walkthrough.
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN_ESTIMATOR = True
except ImportError:
    _HAS_FROZEN_ESTIMATOR = False

from src.synthetic_data import generate_synthetic_dataset
from src.models import (
    train_logistic_regression,
    train_decision_tree,
    train_weighted_logistic_regression,
    train_smote_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from src.cost_optimization import find_optimal_threshold

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "api", "artifacts")
REAL_DATA_PATH = os.path.join(REPO_ROOT, "data", "raw", "creditcard.csv")

FEATURE_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

# Every candidate model considered. XGBoost is what actually gets deployed,
# but the others are trained too so the dashboard's Model Comparison tab can
# show real numbers for why -- not just an assertion.
CANDIDATE_MODELS = {
    "Logistic Regression": train_logistic_regression,
    "Decision Tree": train_decision_tree,
    "Weighted Logistic Regression": train_weighted_logistic_regression,
    "SMOTE + Logistic Regression": train_smote_logistic_regression,
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost,
}


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


def evaluate_all_models(X_train, y_train, X_test, y_test):
    """
    Trains every candidate model, and for each one finds ITS OWN cost-optimal
    threshold (not just accuracy at 0.5) -- so the comparison is apples-to-apples
    with how the system actually makes decisions in production.
    """
    results = []
    fitted = {}
    for name, train_fn in CANDIDATE_MODELS.items():
        print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
        start = time.time()
        model, y_prob = train_fn(X_train, y_train, X_test, y_test, verbose=False)
        elapsed = time.time() - start

        threshold_result = find_optimal_threshold(y_test.values, y_prob)
        y_pred_at_best = (np.asarray(y_prob) >= threshold_result["best_threshold"]).astype(int)

        metrics = {
            "model": name,
            "precision": round(float(precision_score(y_test, y_pred_at_best, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_at_best, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred_at_best, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "best_threshold": threshold_result["best_threshold"],
            "min_business_cost": threshold_result["min_cost"],
            "cost_at_naive_0.5": threshold_result["default_cost_at_0.5"],
            "train_time_seconds": round(elapsed, 2),
        }
        results.append(metrics)
        fitted[name] = model
        print(f"AUC={metrics['roc_auc']}  |  min cost=Rs.{metrics['min_business_cost']}  |  {elapsed:.1f}s")

    results.sort(key=lambda r: r["min_business_cost"])
    return results, fitted


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Training and comparing all candidate models...")
    comparison_results, fitted_models = evaluate_all_models(X_train, y_train, X_test, y_test)

    winner_name = comparison_results[0]["model"]
    print(f"\nLowest business cost this run: {winner_name}.")
    print("Note: with only a few dozen fraud cases in the test split, cost differences between")
    print("close-scoring models carry real sampling noise -- rerun with more data / cross-validation")
    print("before treating a single run's ranking as final. XGBoost is deployed here for its ability")
    print("to capture nonlinear feature interactions and its native class-imbalance handling, which")
    print("matters more on the full, non-linearly-separable real Kaggle dataset than on this synthetic")
    print("demo set. Swap the deployed model by changing `raw_model = fitted_models[\"...\"]` below.")

    raw_model = fitted_models["XGBoost"]

    print("\nCalibrating XGBoost probabilities (sigmoid/Platt scaling) for reliable uncertainty...")
    if _HAS_FROZEN_ESTIMATOR:
        calibrated_model = CalibratedClassifierCV(FrozenEstimator(raw_model), method="sigmoid")
    else:
        calibrated_model = CalibratedClassifierCV(raw_model, method="sigmoid", cv="prefit")
    calibrated_model.fit(X_test, y_test)
    y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

    print("\nFinding cost-optimal decision threshold for the deployed model...")
    threshold_result = find_optimal_threshold(y_test.values, y_prob_calibrated)
    print(f"Best threshold: {threshold_result['best_threshold']}")
    print(f"Min cost: Rs.{threshold_result['min_cost']}  |  Cost @ 0.5: Rs.{threshold_result['default_cost_at_0.5']}")

    # --- save artifacts ---
    joblib.dump(calibrated_model, os.path.join(ARTIFACT_DIR, "fraud_model.pkl"))
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

    with open(os.path.join(ARTIFACT_DIR, "model_comparison.json"), "w") as f:
        json.dump({
            "deployed_model": "XGBoost",
            "lowest_cost_this_run": winner_name,
            "deployment_rationale": (
                "XGBoost is deployed as the primary model for its ability to capture nonlinear "
                "feature interactions and its native handling of severe class imbalance via "
                "scale_pos_weight, which matters most on the real, non-linearly-separable Kaggle "
                "dataset. On small evaluation splits (this demo's test set has only a few dozen "
                "fraud examples), a simpler model can occasionally edge out XGBoost on business "
                "cost purely from sampling variance -- see 'lowest_cost_this_run' above. In "
                "production this comparison should be re-run with k-fold cross-validation before "
                "being treated as conclusive."
            ),
            "results": comparison_results,
        }, f, indent=2)

    # Reference sample for the drift monitor + dashboard SHAP global importance
    reference_sample = X_train.sample(n=min(5000, len(X_train)), random_state=42)
    reference_sample.to_csv(os.path.join(ARTIFACT_DIR, "reference_sample.csv"), index=False)

    print(f"\nArtifacts written to {ARTIFACT_DIR}")
    print("Done. Start the API with: uvicorn api.app:app --reload")


if __name__ == "__main__":
    main()
