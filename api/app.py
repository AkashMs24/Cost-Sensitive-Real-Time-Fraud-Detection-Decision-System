from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Cost-Sensitive Real-Time Fraud Detection Decision System API")

# Load the real trained model + threshold ONCE when the server starts
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
THRESHOLD_PATH = os.path.join(os.path.dirname(__file__), "threshold.txt")

model = joblib.load(MODEL_PATH)
with open(THRESHOLD_PATH) as f:
    THRESHOLD = float(f.read().strip())

FEATURE_ORDER = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

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

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict_fraud")
def predict_fraud(txn: Transaction):
    # Build a single-row dataframe in the exact column order the model expects
    row = pd.DataFrame([{field: getattr(txn, field) for field in FEATURE_ORDER}])

    # REAL model prediction (not a guess anymore)
    fraud_probability = float(model.predict_proba(row)[:, 1][0])

    # REAL cost-optimized decision (not a hardcoded 0.7/0.4 split)
    if fraud_probability >= THRESHOLD:
        decision = "BLOCK"
        risk_level = "HIGH RISK"
    elif fraud_probability >= THRESHOLD * 0.5:
        decision = "REVIEW"
        risk_level = "MEDIUM RISK"
    else:
        decision = "ALLOW"
        risk_level = "LOW RISK"

    return {
        "fraud_probability": round(fraud_probability, 4),
        "risk_level": risk_level,
        "decision": decision
    }
