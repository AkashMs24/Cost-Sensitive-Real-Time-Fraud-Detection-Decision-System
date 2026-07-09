"""
SHAP explainability wrapper.

Wraps a fitted tree model (XGBoost / RandomForest) in a reusable class so
the API can generate a *local* explanation ("why was THIS transaction
flagged") for every single decision, instead of only producing a one-off
global summary plot in a notebook.
"""

import numpy as np
import pandas as pd
import shap


class FraudExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain_instance(self, row: pd.DataFrame, top_k: int = 5):
        """
        row: single-row DataFrame in the exact feature order the model expects.
        Returns a list of {feature, value, shap_value, direction} sorted by
        |shap_value| descending, i.e. "what pushed this decision".
        """
        shap_values = self.explainer.shap_values(row)
        # xgboost binary classifier via TreeExplainer returns a single array
        values = np.array(shap_values).reshape(-1)

        contributions = []
        for feat, val, sv in zip(self.feature_names, row.iloc[0].values, values):
            contributions.append({
                "feature": feat,
                "value": float(val),
                "shap_value": float(sv),
                "direction": "toward_fraud" if sv > 0 else "toward_legit",
            })

        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
        return contributions[:top_k]

    def global_importance(self, X_sample: pd.DataFrame, top_k: int = 10):
        shap_values = self.explainer.shap_values(X_sample)
        mean_abs = np.abs(np.array(shap_values)).mean(axis=0)
        ranked = sorted(zip(self.feature_names, mean_abs), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "mean_abs_shap": float(v)} for f, v in ranked[:top_k]]
