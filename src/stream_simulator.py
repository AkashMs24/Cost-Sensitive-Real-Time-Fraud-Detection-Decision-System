"""
Real-time transaction stream simulator.

Run with:  python -m src.stream_simulator --rate 3 --duration 120

Simulates a live feed of incoming transactions hitting the API's
/predict_fraud endpoint at a configurable rate (transactions/second),
occasionally replaying one of the "ring" account/device combos so you can
watch /fraud_rings light up in real time, and randomly sending back
/feedback (ground truth) for a fraction of past transactions after a short
delay -- simulating how ground truth actually arrives in production (a
chargeback, a confirmed-fraud report, a cleared review), which is what
drives the adaptive threshold engine and the drift monitor.

This is intentionally a lightweight polling/HTTP simulator rather than a
Kafka/Flink pipeline: it demonstrates the real-time DECISIONING logic (the
actual hard part of this project) without requiring the reader to spin up
a message broker just to see the system work end to end.
"""

import argparse
import random
import time
import requests

from src.synthetic_data import generate_synthetic_dataset

API_URL = "http://127.0.0.1:8000"


def run(rate: float, duration: int, api_url: str):
    print(f"Streaming synthetic transactions to {api_url} at ~{rate}/sec for {duration}s...")
    pool = generate_synthetic_dataset(n_transactions=3000, n_rings=6).to_dict("records")
    pending_feedback = []  # (txn_id, true_label, submit_after_ts)

    start = time.time()
    sent = 0
    while time.time() - start < duration:
        row = random.choice(pool)
        payload = {k: row[k] for k in row if k not in ("Class", "is_ring_member")}
        try:
            resp = requests.post(f"{api_url}/predict_fraud", json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            sent += 1
            print(
                f"[{sent:04d}] acct={payload.get('account_id')} amt={payload.get('Amount'):.2f} "
                f"-> prob={result.get('fraud_probability')} decision={result.get('decision')}"
            )
            txn_id = result.get("transaction_id")
            if txn_id:
                pending_feedback.append((txn_id, int(row["Class"]), time.time() + random.uniform(2, 6)))
        except requests.exceptions.ConnectionError:
            print(f"Could not reach {api_url} -- is `uvicorn api.app:app` running?")
            return
        except Exception as e:
            print("Error posting transaction:", e)

        # resolve due feedback (simulates ground truth arriving with a delay)
        still_pending = []
        for txn_id, true_label, due_at in pending_feedback:
            if time.time() >= due_at:
                try:
                    requests.post(f"{api_url}/feedback", json={
                        "transaction_id": txn_id, "true_label": true_label
                    }, timeout=5)
                except Exception:
                    pass
            else:
                still_pending.append((txn_id, true_label, due_at))
        pending_feedback = still_pending

        time.sleep(1.0 / max(rate, 0.1))

    print(f"\nDone. Sent {sent} transactions. Check /fraud_rings, /drift/status, /threshold/status.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a real-time fraud transaction stream.")
    parser.add_argument("--rate", type=float, default=3.0, help="transactions per second")
    parser.add_argument("--duration", type=int, default=120, help="seconds to stream")
    parser.add_argument("--api-url", type=str, default=API_URL)
    args = parser.parse_args()
    run(args.rate, args.duration, args.api_url)
