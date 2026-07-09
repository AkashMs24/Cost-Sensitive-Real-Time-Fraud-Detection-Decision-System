"""
Graph-based fraud ring detection.

Almost every clone of this Kaggle-dataset project scores transactions in
isolation. Real fraud rarely is isolated: mules, stolen-card testing, and
account-takeover rings show up as a *cluster* of accounts/devices/merchants
that share infrastructure across multiple flagged transactions.

This module builds a graph where transactions are linked if they share an
account_id, device_id, or merchant_id, then runs community detection to
surface clusters that are small-world (few devices, many accounts) and
disproportionately flagged as risky -- the signature of an organized ring
rather than coincidence.
"""

from collections import defaultdict
import networkx as nx


class FraudRingDetector:
    def __init__(self, min_community_size: int = 3, risk_score_threshold: float = 0.4):
        self.graph = nx.Graph()
        self.txn_meta = {}  # txn_id -> {account_id, device_id, merchant_id, decision, prob}
        self.min_community_size = min_community_size
        self.risk_score_threshold = risk_score_threshold

    def ingest(self, txn_id, account_id, device_id, merchant_id, fraud_probability, decision):
        """Add one transaction/decision to the rolling entity graph."""
        self.txn_meta[txn_id] = dict(
            account_id=account_id,
            device_id=device_id,
            merchant_id=merchant_id,
            fraud_probability=fraud_probability,
            decision=decision,
        )

        self.graph.add_node(txn_id, type="transaction")
        for key, val in (("account", account_id), ("device", device_id), ("merchant", merchant_id)):
            entity_node = f"{key}:{val}"
            self.graph.add_node(entity_node, type=key)
            self.graph.add_edge(txn_id, entity_node)

    def detect_rings(self, top_k: int = 10):
        """
        Runs community detection over the entity graph and returns the
        highest-risk clusters: groups of transactions that share
        infrastructure (device/merchant) across multiple distinct accounts,
        with an elevated average fraud probability.
        """
        if self.graph.number_of_nodes() == 0:
            return []

        communities = nx.algorithms.community.greedy_modularity_communities(self.graph)

        rings = []
        for community in communities:
            txn_ids = [n for n in community if n in self.txn_meta]
            if len(txn_ids) < self.min_community_size:
                continue

            accounts = {self.txn_meta[t]["account_id"] for t in txn_ids}
            devices = {self.txn_meta[t]["device_id"] for t in txn_ids}
            merchants = {self.txn_meta[t]["merchant_id"] for t in txn_ids}
            probs = [self.txn_meta[t]["fraud_probability"] for t in txn_ids]
            avg_prob = sum(probs) / len(probs)
            blocked_or_review = sum(
                1 for t in txn_ids if self.txn_meta[t]["decision"] in ("BLOCK", "REVIEW")
            )

            # A ring signature: MANY accounts funneled through FEW shared
            # devices/merchants, with high average risk. Fan-out ratio
            # captures "many accounts, one device" which is the classic tell.
            fan_out_ratio = len(accounts) / max(len(devices), 1)
            risk_score = round(
                0.5 * avg_prob
                + 0.3 * (blocked_or_review / len(txn_ids))
                + 0.2 * min(fan_out_ratio / 5.0, 1.0),
                4,
            )

            if risk_score < self.risk_score_threshold:
                continue

            rings.append({
                "ring_id": f"ring_{len(rings)+1}",
                "transactions": txn_ids,
                "n_transactions": len(txn_ids),
                "n_accounts": len(accounts),
                "n_devices": len(devices),
                "n_merchants": len(merchants),
                "shared_devices": list(devices),
                "shared_merchants": list(merchants),
                "avg_fraud_probability": round(avg_prob, 4),
                "fan_out_ratio": round(fan_out_ratio, 2),
                "risk_score": risk_score,
            })

        rings.sort(key=lambda r: r["risk_score"], reverse=True)
        return rings[:top_k]

    def stats(self):
        return {
            "n_transactions_tracked": len(self.txn_meta),
            "n_graph_nodes": self.graph.number_of_nodes(),
            "n_graph_edges": self.graph.number_of_edges(),
        }
