import json
import os
import uuid
from collections import deque
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.adaptive_threshold import AdaptiveThresholdEngine
from src.drift_monitor import DriftMonitor
from src.explainability import FraudExplainer
from src.graph_rings import FraudRingDetector
from src.llm_copilot import generate_investigation_note

app = FastAPI(title="Cost-Sensitive Real-Time Fraud Detection Decision System API", version="2.0")

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "fraud_model.pkl")
RAW_MODEL_PATH = os.path.join(ARTIFACT_DIR, "fraud_model_raw.pkl")
FEATURE_ORDER_PATH = os.path.join(ARTIFACT_DIR, "feature_order.json")
THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "threshold.json")
REFERENCE_SAMPLE_PATH = os.path.join(ARTIFACT_DIR, "reference_sample.csv")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "No trained model found. Run `python -m src.train_pipeline` first "
        "to generate api/artifacts/fraud_model.pkl."
    )

model = joblib.load(MODEL_PATH)
with open(FEATURE_ORDER_PATH) as f:
    FEATURE_ORDER = json.load(f)
with open(THRESHOLD_PATH) as f:
    threshold_meta = json.load(f)

reference_sample = pd.read_csv(REFERENCE_SAMPLE_PATH)

# SHAP needs the raw tree model, not the CalibratedClassifierCV wrapper
# (calibration is used for the probability the user sees; the raw model's
# tree structure is used purely to compute feature attributions).
raw_model_for_shap = joblib.load(RAW_MODEL_PATH) if os.path.exists(RAW_MODEL_PATH) else model
explainer = FraudExplainer(raw_model_for_shap, FEATURE_ORDER)
threshold_engine = AdaptiveThresholdEngine(initial_threshold=threshold_meta["best_threshold"])
drift_monitor = DriftMonitor(reference_sample, feature_columns=FEATURE_ORDER, window_size=500)
ring_detector = FraudRingDetector()

# in-memory transaction log (swap for a real DB in production)
TRANSACTION_LOG = {}
RECENT_FEED = deque(maxlen=200)


class WSManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = WSManager()


class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    account_id: Optional[str] = "UNKNOWN_ACC"
    device_id: Optional[str] = "UNKNOWN_DEV"
    merchant_id: Optional[str] = "UNKNOWN_MER"


class Feedback(BaseModel):
    transaction_id: str
    true_label: int  # 1 = confirmed fraud, 0 = confirmed genuine


def _decide(fraud_probability: float, threshold: float):
    if fraud_probability >= threshold:
        return "BLOCK", "HIGH RISK"
    elif fraud_probability >= threshold * 0.5:
        return "REVIEW", "MEDIUM RISK"
    else:
        return "ALLOW", "LOW RISK"


@app.get("/")
def health_check():
    return {
        "status": "API is running",
        "model_loaded": True,
        "current_threshold": threshold_engine.current_threshold(),
        "n_transactions_processed": len(TRANSACTION_LOG),
    }


@app.post("/predict_fraud")
async def predict_fraud(txn: Transaction):
    row = pd.DataFrame([{field: getattr(txn, field) for field in FEATURE_ORDER}])
    fraud_probability = float(model.predict_proba(row)[:, 1][0])

    threshold = threshold_engine.current_threshold()
    decision, risk_level = _decide(fraud_probability, threshold)

    top_features = explainer.explain_instance(row, top_k=5)

    txn_id = str(uuid.uuid4())
    TRANSACTION_LOG[txn_id] = {
        "transaction_id": txn_id,
        "input": txn.model_dump(),
        "fraud_probability": fraud_probability,
        "decision": decision,
        "risk_level": risk_level,
        "threshold_used": threshold,
        "top_features": top_features,
        "true_label": None,
    }
    RECENT_FEED.append(TRANSACTION_LOG[txn_id])

    ring_detector.ingest(
        txn_id, txn.account_id, txn.device_id, txn.merchant_id, fraud_probability, decision
    )
    drift_monitor.ingest({field: getattr(txn, field) for field in FEATURE_ORDER})

    result = {
        "transaction_id": txn_id,
        "fraud_probability": round(fraud_probability, 4),
        "risk_level": risk_level,
        "decision": decision,
        "threshold_used": round(threshold, 4),
        "top_features": top_features,
    }

    await ws_manager.broadcast({"type": "transaction", "data": result})
    return result


@app.post("/feedback")
async def submit_feedback(feedback: Feedback):
    txn = TRANSACTION_LOG.get(feedback.transaction_id)
    if txn is None:
        raise HTTPException(status_code=404, detail="Unknown transaction_id")

    txn["true_label"] = feedback.true_label
    threshold_engine.record_feedback(
        fraud_probability=txn["fraud_probability"],
        true_label=feedback.true_label,
        decision=txn["decision"],
    )
    await ws_manager.broadcast({
        "type": "threshold_update",
        "data": {"current_threshold": threshold_engine.current_threshold()},
    })
    return {"status": "ok", "updated_threshold": threshold_engine.current_threshold()}


@app.get("/threshold/status")
def threshold_status():
    return threshold_engine.status()


@app.get("/drift/status")
def drift_status():
    return drift_monitor.report()


@app.get("/fraud_rings")
def fraud_rings(top_k: int = 10):
    return {"rings": ring_detector.detect_rings(top_k=top_k), "graph_stats": ring_detector.stats()}


@app.get("/transactions/recent")
def recent_transactions(limit: int = 50):
    items = list(RECENT_FEED)[-limit:]
    return {"transactions": list(reversed(items))}


@app.post("/investigate/{transaction_id}")
def investigate(transaction_id: str):
    txn = TRANSACTION_LOG.get(transaction_id)
    if txn is None:
        raise HTTPException(status_code=404, detail="Unknown transaction_id")

    note = generate_investigation_note(
        transaction=txn["input"],
        top_features=txn["top_features"],
        decision=txn["decision"],
        fraud_probability=txn["fraud_probability"],
    )
    return note


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive; client doesn't need to send anything meaningful
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
