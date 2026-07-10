import io
import json
import os
import uuid
from collections import deque
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
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
MODEL_COMPARISON_PATH = os.path.join(ARTIFACT_DIR, "model_comparison.json")

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

if os.path.exists(MODEL_COMPARISON_PATH):
    with open(MODEL_COMPARISON_PATH) as f:
        MODEL_COMPARISON = json.load(f)
else:
    MODEL_COMPARISON = None

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
RECENT_FEED = deque(maxlen=2000)


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


def _score_one(feature_values: dict, account_id: str, device_id: str, merchant_id: str,
                raw_input: dict, log_and_broadcast: bool = True):
    """
    Shared scoring path used by /predict_fraud (one transaction from the form)
    and /predict_batch (many rows from a CSV) -- guarantees a CSV-uploaded
    transaction is scored, explained, ring-checked, and drift-monitored
    exactly the same way a single live transaction would be.
    """
    row = pd.DataFrame([{field: feature_values.get(field, 0.0) for field in FEATURE_ORDER}])
    fraud_probability = float(model.predict_proba(row)[:, 1][0])

    threshold = threshold_engine.current_threshold()
    decision, risk_level = _decide(fraud_probability, threshold)
    top_features = explainer.explain_instance(row, top_k=5)

    txn_id = str(uuid.uuid4())
    record = {
        "transaction_id": txn_id,
        "input": raw_input,
        "fraud_probability": fraud_probability,
        "decision": decision,
        "risk_level": risk_level,
        "threshold_used": threshold,
        "top_features": top_features,
        "true_label": None,
    }

    if log_and_broadcast:
        TRANSACTION_LOG[txn_id] = record
        RECENT_FEED.append(record)
        ring_detector.ingest(txn_id, account_id, device_id, merchant_id, fraud_probability, decision)
        drift_monitor.ingest({field: feature_values.get(field, 0.0) for field in FEATURE_ORDER})

    return record


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
    feature_values = {field: getattr(txn, field) for field in FEATURE_ORDER}
    record = _score_one(
        feature_values, txn.account_id, txn.device_id, txn.merchant_id,
        raw_input=txn.model_dump(), log_and_broadcast=True,
    )

    result = {
        "transaction_id": record["transaction_id"],
        "fraud_probability": round(record["fraud_probability"], 4),
        "risk_level": record["risk_level"],
        "decision": record["decision"],
        "threshold_used": round(record["threshold_used"], 4),
        "top_features": record["top_features"],
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


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Score an uploaded CSV of transactions in one call. Expected columns:
    Time, Amount, V1..V28 (required), account_id/device_id/merchant_id (optional).
    Returns per-row results as JSON AND logs every row through the same
    ring-detector/drift-monitor pipeline as live traffic, so a batch upload
    can surface fraud rings too.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV is missing required columns: {missing}")

    if len(df) > 5000:
        raise HTTPException(status_code=400, detail="Batch limit is 5000 rows per upload for this demo instance.")

    results = []
    for _, row in df.iterrows():
        feature_values = {field: float(row[field]) for field in FEATURE_ORDER}
        account_id = str(row.get("account_id", "BATCH_UNKNOWN_ACC"))
        device_id = str(row.get("device_id", "BATCH_UNKNOWN_DEV"))
        merchant_id = str(row.get("merchant_id", "BATCH_UNKNOWN_MER"))

        record = _score_one(
            feature_values, account_id, device_id, merchant_id,
            raw_input={**feature_values, "account_id": account_id, "device_id": device_id, "merchant_id": merchant_id},
            log_and_broadcast=True,
        )
        results.append({
            "transaction_id": record["transaction_id"],
            "account_id": account_id,
            "Amount": feature_values["Amount"],
            "fraud_probability": round(record["fraud_probability"], 4),
            "decision": record["decision"],
            "risk_level": record["risk_level"],
        })

    summary = {
        "n_scored": len(results),
        "n_block": sum(1 for r in results if r["decision"] == "BLOCK"),
        "n_review": sum(1 for r in results if r["decision"] == "REVIEW"),
        "n_allow": sum(1 for r in results if r["decision"] == "ALLOW"),
    }

    await ws_manager.broadcast({"type": "batch_scored", "data": summary})
    return {"summary": summary, "results": results}


@app.post("/predict_batch/download")
async def predict_batch_download(file: UploadFile = File(...)):
    """Same as /predict_batch, but returns a downloadable scored CSV instead of JSON."""
    response = await predict_batch(file)
    out_df = pd.DataFrame(response["results"])
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=scored_transactions.csv"},
    )


@app.get("/models/comparison")
def models_comparison():
    if MODEL_COMPARISON is None:
        raise HTTPException(
            status_code=404,
            detail="No model_comparison.json artifact found. Re-run `python -m src.train_pipeline` "
                   "with the current version of src/train_pipeline.py to generate it.",
        )
    return MODEL_COMPARISON


@app.get("/transactions/history")
def transactions_history(
    decision: Optional[str] = None,
    account_id: Optional[str] = None,
    min_probability: float = 0.0,
    limit: int = 200,
):
    """Full filterable transaction history, for the dashboard's History tab."""
    items = list(RECENT_FEED)
    if decision:
        items = [t for t in items if t["decision"] == decision.upper()]
    if account_id:
        items = [t for t in items if account_id.lower() in str(t["input"].get("account_id", "")).lower()]
    if min_probability > 0:
        items = [t for t in items if t["fraud_probability"] >= min_probability]

    items = list(reversed(items))[:limit]
    return {"transactions": items, "total_matching": len(items)}


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
