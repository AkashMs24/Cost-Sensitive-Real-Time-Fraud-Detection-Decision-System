"""
Cost-sensitive threshold optimization.

Given predicted fraud probabilities and known business costs of the two
error types (missed fraud vs. wrongly-blocked genuine customer), find the
decision threshold that minimizes total expected ₹ cost -- not the
threshold that maximizes accuracy or F1.
"""

import numpy as np
from sklearn.metrics import confusion_matrix

COST_FN_DEFAULT = 10000  # fraud missed -> direct loss
COST_FP_DEFAULT = 200    # genuine customer wrongly blocked


def business_cost(y_true, y_pred, cost_fn=COST_FN_DEFAULT, cost_fp=COST_FP_DEFAULT):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fn * cost_fn + fp * cost_fp


def find_optimal_threshold(
    y_true,
    y_prob,
    cost_fn=COST_FN_DEFAULT,
    cost_fp=COST_FP_DEFAULT,
    grid=None,
):
    """
    Grid-searches the threshold that minimizes total expected business cost.

    Returns: dict(best_threshold, min_cost, default_cost_at_0.5, thresholds, costs)
    """
    if grid is None:
        grid = np.arange(0.01, 0.99, 0.01)

    costs = np.array([
        business_cost(y_true, (y_prob >= t).astype(int), cost_fn, cost_fp)
        for t in grid
    ])

    best_idx = int(np.argmin(costs))
    best_threshold = float(grid[best_idx])
    min_cost = float(costs[best_idx])

    default_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    default_cost = float(business_cost(y_true, default_pred, cost_fn, cost_fp))

    return {
        "best_threshold": best_threshold,
        "min_cost": min_cost,
        "default_cost_at_0.5": default_cost,
        "savings_vs_default": default_cost - min_cost,
        "thresholds": grid.tolist(),
        "costs": costs.tolist(),
    }


def expected_marginal_cost(prob, threshold, cost_fn=COST_FN_DEFAULT, cost_fp=COST_FP_DEFAULT):
    """
    Expected cost of ALLOWing a single transaction at a given fraud
    probability, vs. BLOCKing it -- used by the adaptive threshold engine to
    do an online (per-transaction) gradient step instead of a batch grid
    search.

    E[cost | ALLOW] = prob * cost_fn
    E[cost | BLOCK] = (1 - prob) * cost_fp
    The break-even probability is where these are equal:
        p* = cost_fp / (cost_fn + cost_fp)
    """
    allow_cost = prob * cost_fn
    block_cost = (1 - prob) * cost_fp
    return allow_cost, block_cost


def breakeven_probability(cost_fn=COST_FN_DEFAULT, cost_fp=COST_FP_DEFAULT):
    return cost_fp / (cost_fn + cost_fp)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.02, 5000)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0.1, 0.15, 5000), 0, 1)
    result = find_optimal_threshold(y_true, y_prob)
    print("Best threshold:", result["best_threshold"])
    print("Min cost: ₹", result["min_cost"])
    print("Cost @ 0.5: ₹", result["default_cost_at_0.5"])
    print("Break-even probability:", breakeven_probability())
