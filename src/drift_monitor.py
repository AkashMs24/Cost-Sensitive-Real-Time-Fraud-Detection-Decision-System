"""
Concept / data drift monitor.

Every "production ML" checklist mentions drift detection, and almost no
student project actually implements it. This module computes the
Population Stability Index (PSI) per feature between a frozen reference
distribution (captured at training time) and a rolling window of live
incoming traffic, flags which features have drifted, and raises a
retrain-recommended flag when drift is severe enough to matter.

PSI interpretation (industry-standard banking/credit-risk thresholds):
    PSI < 0.10           -> no significant change
    0.10 <= PSI < 0.25   -> moderate shift, monitor
    PSI >= 0.25          -> major shift, retrain recommended
"""

from collections import deque
import numpy as np
import pandas as pd


def _psi_for_feature(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        return 0.0  # not enough spread to bin meaningfully

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-4, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


class DriftMonitor:
    def __init__(self, reference_df: pd.DataFrame, feature_columns, window_size: int = 500,
                 moderate_threshold: float = 0.10, severe_threshold: float = 0.25):
        self.reference = {col: reference_df[col].values for col in feature_columns}
        self.feature_columns = list(feature_columns)
        self.window = {col: deque(maxlen=window_size) for col in feature_columns}
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold
        self.window_size = window_size
        self.n_seen = 0

    def ingest(self, row: dict):
        """row: dict of feature_name -> value for one live transaction."""
        self.n_seen += 1
        for col in self.feature_columns:
            if col in row:
                self.window[col].append(row[col])

    def ready(self) -> bool:
        return all(len(self.window[c]) >= min(50, self.window_size) for c in self.feature_columns)

    def report(self, top_k: int = 8):
        if not self.ready():
            return {
                "status": "WARMING_UP",
                "n_seen": self.n_seen,
                "message": f"Need at least {min(50, self.window_size)} live transactions per feature before drift can be assessed.",
            }

        per_feature = []
        for col in self.feature_columns:
            psi = _psi_for_feature(self.reference[col], np.array(self.window[col]))
            if psi >= self.severe_threshold:
                status = "SEVERE"
            elif psi >= self.moderate_threshold:
                status = "MODERATE"
            else:
                status = "STABLE"
            per_feature.append({"feature": col, "psi": round(psi, 4), "status": status})

        per_feature.sort(key=lambda x: x["psi"], reverse=True)
        n_severe = sum(1 for f in per_feature if f["status"] == "SEVERE")
        n_moderate = sum(1 for f in per_feature if f["status"] == "MODERATE")

        overall_status = "SEVERE" if n_severe > 0 else ("MODERATE" if n_moderate > 0 else "STABLE")
        retrain_recommended = n_severe > 0 or n_moderate >= max(3, len(self.feature_columns) // 4)

        return {
            "status": overall_status,
            "n_seen": self.n_seen,
            "n_features_severe": n_severe,
            "n_features_moderate": n_moderate,
            "retrain_recommended": retrain_recommended,
            "top_drifting_features": per_feature[:top_k],
        }
