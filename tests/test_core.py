"""
Unit tests for the core decisioning modules -- the parts of this system
that make real ALLOW/REVIEW/BLOCK calls, so they're the parts most worth
protecting with tests against silent regressions.

Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.cost_optimization import (
    business_cost,
    find_optimal_threshold,
    breakeven_probability,
)
from src.adaptive_threshold import AdaptiveThresholdEngine
from src.graph_rings import FraudRingDetector
from src.drift_monitor import DriftMonitor
from src.synthetic_data import generate_synthetic_dataset


# ---------------------------------------------------------------------------
# cost_optimization
# ---------------------------------------------------------------------------

class TestCostOptimization:
    def test_business_cost_all_correct_is_zero(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        assert business_cost(y_true, y_pred) == 0

    def test_business_cost_missed_fraud_costs_more_than_false_positive(self):
        y_true = [1]
        missed_fraud_cost = business_cost(y_true, [0])   # allowed a fraud through
        y_true2 = [0]
        false_positive_cost = business_cost(y_true2, [1])  # blocked a genuine user
        assert missed_fraud_cost > false_positive_cost

    def test_find_optimal_threshold_beats_naive_default(self):
        rng = np.random.default_rng(0)
        y_true = rng.binomial(1, 0.05, 2000)
        y_prob = np.clip(y_true * 0.5 + rng.normal(0.1, 0.2, 2000), 0, 1)
        result = find_optimal_threshold(y_true, y_prob)
        assert result["min_cost"] <= result["default_cost_at_0.5"]

    def test_breakeven_probability_in_valid_range(self):
        p = breakeven_probability(cost_fn=10000, cost_fp=200)
        assert 0 < p < 1
        # higher false-negative cost relative to false-positive cost -> lower breakeven probability
        p_expensive_fraud = breakeven_probability(cost_fn=100000, cost_fp=200)
        assert p_expensive_fraud < p


# ---------------------------------------------------------------------------
# adaptive_threshold
# ---------------------------------------------------------------------------

class TestAdaptiveThreshold:
    def test_missed_frauds_push_threshold_down(self):
        engine = AdaptiveThresholdEngine(initial_threshold=0.5, base_learning_rate=0.1)
        for _ in range(20):
            engine.record_feedback(fraud_probability=0.3, true_label=1, decision="ALLOW")
        assert engine.current_threshold() < 0.5

    def test_wrongful_blocks_push_threshold_up(self):
        engine = AdaptiveThresholdEngine(initial_threshold=0.5, base_learning_rate=0.1)
        for _ in range(20):
            engine.record_feedback(fraud_probability=0.6, true_label=0, decision="BLOCK")
        assert engine.current_threshold() > 0.5

    def test_correct_decisions_do_not_move_threshold_much(self):
        engine = AdaptiveThresholdEngine(initial_threshold=0.5, base_learning_rate=0.1)
        for _ in range(20):
            engine.record_feedback(fraud_probability=0.1, true_label=0, decision="ALLOW")
        assert abs(engine.current_threshold() - 0.5) < 0.05

    def test_threshold_stays_within_bounds(self):
        engine = AdaptiveThresholdEngine(initial_threshold=0.5, min_threshold=0.02, max_threshold=0.98)
        for _ in range(500):
            engine.record_feedback(fraud_probability=0.9, true_label=0, decision="BLOCK")
        assert engine.current_threshold() <= 0.98

    def test_status_reports_feedback_count(self):
        engine = AdaptiveThresholdEngine(initial_threshold=0.5)
        engine.record_feedback(0.5, 1, "BLOCK")
        engine.record_feedback(0.5, 0, "ALLOW")
        status = engine.status()
        assert status["n_feedback_updates"] == 2


# ---------------------------------------------------------------------------
# graph_rings
# ---------------------------------------------------------------------------

class TestFraudRingDetector:
    def test_detects_planted_ring(self):
        detector = FraudRingDetector(min_community_size=3, risk_score_threshold=0.3)
        for i, acc in enumerate(["ACC1", "ACC2", "ACC3", "ACC4"]):
            detector.ingest(f"txn{i}", acc, "DEV_SHARED", "MER_X", fraud_probability=0.85, decision="BLOCK")
        rings = detector.detect_rings()
        assert len(rings) >= 1
        assert rings[0]["n_devices"] == 1
        assert rings[0]["n_accounts"] == 4

    def test_no_ring_from_unrelated_legit_traffic(self):
        detector = FraudRingDetector(min_community_size=3, risk_score_threshold=0.4)
        for i in range(10):
            detector.ingest(f"txn{i}", f"ACC{i}", f"DEV{i}", f"MER{i}", fraud_probability=0.01, decision="ALLOW")
        rings = detector.detect_rings()
        assert len(rings) == 0

    def test_empty_graph_returns_empty_list(self):
        detector = FraudRingDetector()
        assert detector.detect_rings() == []


# ---------------------------------------------------------------------------
# drift_monitor
# ---------------------------------------------------------------------------

class TestDriftMonitor:
    def test_warming_up_before_enough_data(self):
        rng = np.random.default_rng(0)
        reference = pd.DataFrame({"Amount": rng.normal(50, 10, 200)})
        monitor = DriftMonitor(reference, feature_columns=["Amount"], window_size=500)
        monitor.ingest({"Amount": 55})
        report = monitor.report()
        assert report["status"] == "WARMING_UP"

    def test_stable_when_distribution_unchanged(self):
        rng = np.random.default_rng(0)
        reference = pd.DataFrame({"Amount": rng.normal(50, 10, 1000)})
        monitor = DriftMonitor(reference, feature_columns=["Amount"], window_size=500)
        for v in rng.normal(50, 10, 100):
            monitor.ingest({"Amount": float(v)})
        report = monitor.report()
        assert report["status"] == "STABLE"

    def test_flags_severe_drift_on_shifted_distribution(self):
        rng = np.random.default_rng(0)
        reference = pd.DataFrame({"Amount": rng.normal(50, 10, 1000)})
        monitor = DriftMonitor(reference, feature_columns=["Amount"], window_size=500)
        for v in rng.normal(500, 10, 100):  # wildly different distribution
            monitor.ingest({"Amount": float(v)})
        report = monitor.report()
        assert report["status"] == "SEVERE"
        assert report["retrain_recommended"] is True


# ---------------------------------------------------------------------------
# synthetic_data
# ---------------------------------------------------------------------------

class TestSyntheticData:
    def test_generates_expected_columns(self):
        df = generate_synthetic_dataset(n_transactions=500, n_rings=2)
        expected_cols = {"Time", "Amount", "Class", "account_id", "device_id", "merchant_id", "is_ring_member"}
        assert expected_cols.issubset(set(df.columns))
        assert all(f"V{i}" in df.columns for i in range(1, 29))

    def test_class_imbalance_is_realistic(self):
        df = generate_synthetic_dataset(n_transactions=5000, fraud_rate=0.0017, n_rings=3)
        fraud_rate = df["Class"].mean()
        assert 0 < fraud_rate < 0.05  # rare, but non-zero

    def test_planted_rings_share_a_device(self):
        df = generate_synthetic_dataset(n_transactions=2000, n_rings=5)
        ring_txns = df[df["is_ring_member"] == 1]
        assert len(ring_txns) > 0
        # each ring's transactions should collapse to a small number of shared devices
        assert ring_txns["device_id"].nunique() < ring_txns["account_id"].nunique()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
