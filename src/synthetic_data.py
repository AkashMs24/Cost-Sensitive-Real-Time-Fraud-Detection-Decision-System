"""
Synthetic transaction data generator.

The public Kaggle "creditcard.csv" dataset (Time, Amount, V1..V28, Class) is
PCA-anonymized and has NO entity keys (no account/device/merchant id), so it
is impossible to do graph-based fraud-ring detection on it directly.

This module generates a dataset that is statistically similar in shape
(class imbalance ~0.17%, same column names) but ALSO carries synthetic
entity keys, and deliberately plants a handful of "fraud rings" -- clusters
of accounts/devices that collude across several transactions. This lets the
graph-ring detector in src/graph_rings.py be demonstrated end-to-end without
needing a real, sensitive, entity-linked banking dataset (which nobody
publishes for obvious privacy reasons).

If a real creditcard.csv is present at data/raw/creditcard.csv, the training
pipeline will use it for the model itself, and only use this module to
synthesize the *entity linkage layer* on top of it.
"""

import numpy as np
import pandas as pd

N_V_FEATURES = 28


def _random_v_block(n, fraud_like=False, rng=None):
    rng = rng or np.random.default_rng()
    base = rng.normal(loc=0.0, scale=1.0, size=(n, N_V_FEATURES))
    if fraud_like:
        # Fraud transactions in the real dataset tend to be more extreme
        # on a handful of components (V14, V12, V10, V17 in the well-known EDA),
        # but with realistic overlap -- not perfectly separable -- so a
        # cost-sensitive threshold actually has a real tradeoff to make.
        shift = np.zeros(N_V_FEATURES)
        for idx in (9, 11, 13, 16):  # V10, V12, V14, V17 (0-indexed)
            shift[idx] = -1.6
        # per-row random attenuation: some "fraud" transactions look almost
        # legitimate (hard cases), most look moderately anomalous.
        attenuation = rng.beta(2, 2, size=(n, 1))
        base += shift * attenuation
    return base


def generate_synthetic_dataset(
    n_transactions: int = 40000,
    fraud_rate: float = 0.0017,
    n_rings: int = 12,
    ring_size_range=(3, 7),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    Time, Amount, V1..V28, Class, account_id, device_id, merchant_id, is_ring_member
    """
    rng = np.random.default_rng(seed)

    n_fraud = max(int(n_transactions * fraud_rate), n_rings * ring_size_range[0])
    n_legit = n_transactions - n_fraud

    # --- legit transactions ---
    legit_v = _random_v_block(n_legit, fraud_like=False, rng=rng)
    legit_amount = np.round(np.abs(rng.normal(70, 60, n_legit)) + 1, 2)
    legit_time = rng.integers(0, 172800, n_legit)  # 2-day window, seconds
    legit_account = rng.integers(100000, 999999, n_legit).astype(str)
    legit_device = rng.integers(1000, 9999, n_legit).astype(str)
    legit_merchant = rng.integers(1, 500, n_legit).astype(str)

    legit_df = pd.DataFrame(legit_v, columns=[f"V{i}" for i in range(1, N_V_FEATURES + 1)])
    legit_df["Time"] = legit_time
    legit_df["Amount"] = legit_amount
    legit_df["Class"] = 0
    legit_df["account_id"] = "ACC" + legit_account
    legit_df["device_id"] = "DEV" + legit_device
    legit_df["merchant_id"] = "MER" + legit_merchant
    legit_df["is_ring_member"] = 0

    # --- planted fraud rings ---
    ring_rows = []
    fraud_used = 0
    shared_devices_pool = [f"DEV{rng.integers(1000, 9999)}" for _ in range(n_rings)]
    shared_merchants_pool = [f"MER{rng.integers(1, 500)}" for _ in range(n_rings)]

    for ring_idx in range(n_rings):
        ring_size = rng.integers(ring_size_range[0], ring_size_range[1] + 1)
        ring_accounts = [f"ACC{rng.integers(100000, 999999)}" for _ in range(ring_size)]
        shared_device = shared_devices_pool[ring_idx]
        shared_merchant = shared_merchants_pool[ring_idx]

        # each account in the ring fires 1-3 fraudulent transactions,
        # all funneled through the SAME device (classic device-fingerprint fraud ring)
        for acc in ring_accounts:
            n_txn_this_acc = rng.integers(1, 4)
            for _ in range(n_txn_this_acc):
                v = _random_v_block(1, fraud_like=True, rng=rng)[0]
                amount = np.round(np.abs(rng.normal(800, 400)) + 50, 2)
                time = int(rng.integers(0, 172800))
                ring_rows.append(
                    dict(
                        **{f"V{i+1}": v[i] for i in range(N_V_FEATURES)},
                        Time=time,
                        Amount=amount,
                        Class=1,
                        account_id=acc,
                        device_id=shared_device,
                        merchant_id=shared_merchant,
                        is_ring_member=1,
                    )
                )
                fraud_used += 1

    # --- remaining "lone wolf" fraud (not part of an organized ring) ---
    n_lone_fraud = max(n_fraud - fraud_used, 0)
    if n_lone_fraud > 0:
        lone_v = _random_v_block(n_lone_fraud, fraud_like=True, rng=rng)
        lone_amount = np.round(np.abs(rng.normal(500, 300, n_lone_fraud)) + 20, 2)
        lone_time = rng.integers(0, 172800, n_lone_fraud)
        lone_account = rng.integers(100000, 999999, n_lone_fraud).astype(str)
        lone_device = rng.integers(1000, 9999, n_lone_fraud).astype(str)
        lone_merchant = rng.integers(1, 500, n_lone_fraud).astype(str)

        lone_df = pd.DataFrame(lone_v, columns=[f"V{i}" for i in range(1, N_V_FEATURES + 1)])
        lone_df["Time"] = lone_time
        lone_df["Amount"] = lone_amount
        lone_df["Class"] = 1
        lone_df["account_id"] = "ACC" + lone_account
        lone_df["device_id"] = "DEV" + lone_device
        lone_df["merchant_id"] = "MER" + lone_merchant
        lone_df["is_ring_member"] = 0
    else:
        lone_df = pd.DataFrame(columns=legit_df.columns)

    ring_df = pd.DataFrame(ring_rows)

    full = pd.concat([legit_df, ring_df, lone_df], ignore_index=True)
    full = full.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    ordered_cols = (
        ["Time", "Amount"]
        + [f"V{i}" for i in range(1, N_V_FEATURES + 1)]
        + ["Class", "account_id", "device_id", "merchant_id", "is_ring_member"]
    )
    full = full[ordered_cols]

    numeric_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, N_V_FEATURES + 1)] + ["Class", "is_ring_member"]
    for col in numeric_cols:
        full[col] = pd.to_numeric(full[col])

    return full


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    print(df["Class"].value_counts())
    print(df["is_ring_member"].value_counts())
    print(df.head())
