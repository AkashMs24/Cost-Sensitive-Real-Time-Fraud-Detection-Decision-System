"""
Adaptive, online-learning decision threshold.

The original project computes ONE cost-optimal threshold offline, on a
static test split, and freezes it forever. In reality the fraud/legit cost
ratio and the population drift over time -- a threshold that was optimal in
January quietly becomes sub-optimal by June.

This module treats threshold selection as a stochastic-approximation
problem (Robbins-Monro): every time a transaction's TRUE label becomes
known (via the /feedback endpoint -- e.g. a customer disputes a charge, or
a human reviewer confirms fraud), we take a small step in the threshold
that would have reduced the ₹ cost of that single decision, with a
learning rate that decays over time for stability. This is the online
analogue of the batch grid-search in cost_optimization.py, and is the
actual research-grade version of "cost-sensitive decisioning."

We keep it intentionally simple (no external RL library) so it's auditable:
a regulator can read this file top to bottom and understand exactly why the
threshold moved.
"""

from collections import deque
import numpy as np

from src.cost_optimization import COST_FN_DEFAULT, COST_FP_DEFAULT, breakeven_probability


class AdaptiveThresholdEngine:
    def __init__(
        self,
        initial_threshold: float,
        cost_fn: float = COST_FN_DEFAULT,
        cost_fp: float = COST_FP_DEFAULT,
        min_threshold: float = 0.02,
        max_threshold: float = 0.98,
        base_learning_rate: float = 0.05,
        history_size: int = 500,
    ):
        self.threshold = float(initial_threshold)
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.base_learning_rate = base_learning_rate
        self.n_updates = 0
        self.history = deque(maxlen=history_size)
        self.history.append({"n_updates": 0, "threshold": self.threshold, "event": "init"})
        self.running_cost = 0.0
        self.running_cost_at_fixed_baseline = 0.0
        self._baseline_threshold = float(initial_threshold)

    def current_threshold(self) -> float:
        return self.threshold

    def record_feedback(self, fraud_probability: float, true_label: int, decision: str):
        """
        true_label: 1 = actually fraud, 0 = actually genuine.
        decision: the decision that was actually made (ALLOW/REVIEW/BLOCK).

        Moves the threshold a small step toward the direction that would
        have minimized cost for THIS transaction, using a decaying learning
        rate (1/sqrt(n)) for Robbins-Monro style convergence.
        """
        self.n_updates += 1
        lr = self.base_learning_rate / np.sqrt(self.n_updates)

        was_allowed = decision == "ALLOW"
        # Realized cost of the decision actually taken:
        if true_label == 1 and was_allowed:
            realized_cost = self.cost_fn      # missed fraud
        elif true_label == 0 and not was_allowed:
            realized_cost = self.cost_fp      # genuine user inconvenienced/blocked
        else:
            realized_cost = 0.0               # correct decision, no cost

        self.running_cost += realized_cost
        baseline_pred_allow = fraud_probability < self._baseline_threshold
        if true_label == 1 and baseline_pred_allow:
            self.running_cost_at_fixed_baseline += self.cost_fn
        elif true_label == 0 and not baseline_pred_allow:
            self.running_cost_at_fixed_baseline += self.cost_fp

        # Gradient signal: if we missed a fraud, the threshold was too high
        # (too permissive) -> move it down. If we blocked a genuine user, the
        # threshold was too low (too strict) -> move it up.
        if true_label == 1 and was_allowed:
            gradient = -1.0
        elif true_label == 0 and not was_allowed:
            gradient = +1.0
        else:
            gradient = 0.0

        self.threshold = float(np.clip(self.threshold + lr * gradient, self.min_threshold, self.max_threshold))

        self.history.append({
            "n_updates": self.n_updates,
            "threshold": self.threshold,
            "realized_cost": realized_cost,
            "event": "feedback",
        })

    def status(self):
        breakeven = breakeven_probability(self.cost_fn, self.cost_fp)
        return {
            "current_threshold": round(self.threshold, 4),
            "baseline_threshold": round(self._baseline_threshold, 4),
            "theoretical_breakeven_probability": round(breakeven, 4),
            "n_feedback_updates": self.n_updates,
            "cumulative_realized_cost": self.running_cost,
            "cumulative_cost_if_frozen_at_baseline": self.running_cost_at_fixed_baseline,
            "savings_vs_frozen_threshold": self.running_cost_at_fixed_baseline - self.running_cost,
            "history": list(self.history)[-100:],
        }
