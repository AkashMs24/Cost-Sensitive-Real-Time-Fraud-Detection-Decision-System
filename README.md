<div align="center">

# Cost-Sensitive Real-Time Fraud Detection System

### Because missing one fraud costs ₹10,000. Blocking one genuine user costs ₹200. The threshold matters.

[![FastAPI](https://img.shields.io/badge/Core%20API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fraud-detection-system-2-7ake.onrender.com)
[![Swagger Docs](https://img.shields.io/badge/API%20Docs-Swagger-85EA2D?style=flat-square&logo=swagger&logoColor=black)](https://fraud-detection-system-2-7ake.onrender.com/docs)
[![Streamlit](https://img.shields.io/badge/Analyst%20UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://cost-sensitive-real-time-fraud-detection-decision-system-erhna.streamlit.app/)
![XAI](https://img.shields.io/badge/XAI-SHAP-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)

</div>

---

## This is not a fraud classifier. It is a fraud decision engine.

Most fraud detection projects optimize for accuracy on an imbalanced dataset.  
That is the wrong objective.

Fraud is rare (~0.17% of transactions). A model that predicts "not fraud" every single time achieves **99.83% accuracy** — and is completely useless.

The real problem is a **cost trade-off:**

| Decision error | Business cost |
|---|---|
| Fraud missed (False Negative) | ₹10,000 direct loss |
| Genuine user blocked (False Positive) | ₹200 + customer dissatisfaction |

This system optimizes the **decision threshold** to minimize total expected business cost — not classification error.

---

## How it works
Transaction Input
↓
Feature Processing
↓
Imbalance-Aware ML Models (XGBoost / Random Forest / Logistic Regression)
↓
Fraud Probability Score
↓
Cost-Optimized Decision Threshold  ← the key differentiator
↓
Decision: ALLOW / REVIEW / BLOCK
↓
SHAP Explanation (why this transaction was flagged)
↓
FastAPI Endpoint (deployed on Render)

---

## API — the core system

```bash
POST /predict_fraud
```

**Request:**
```json
{
  "Time": 0,
  "Amount": 52000,
  "V1": 0.01,
  "V2": -0.03,
  "V28": 0.14
}
```

**Response:**
```json
{
  "fraud_probability": 0.87,
  "risk_level": "HIGH RISK",
  "decision": "BLOCK"
}
```

The Streamlit app is an **optional analyst interface** built on top of this API.  
The FastAPI service is the core engine — production-first, UI-optional.

---

## Dataset

- Credit Card Fraud Dataset (UCI / Kaggle)
- 284,807 transactions · 492 fraud cases · 0.17% fraud rate
- PCA-anonymized features for privacy compliance
- Dataset not included in this repo

---

## Models

| Model | Role |
|---|---|
| Logistic Regression | Baseline |
| Weighted Logistic Regression | Cost-sensitive baseline |
| Random Forest | High precision, conservative decisions |
| XGBoost | Best recall–precision balance — primary model |
| Isolation Forest | Unsupervised, novel fraud pattern detection |

---

## Handling class imbalance

- Class weighting on all models
- SMOTE for experimentation
- Imbalance-aware tree configurations
- **Threshold tuning over label tuning** — the decision boundary is moved, not the data

---

## Explainability (SHAP)

- Global explanations — which features drive fraud across the dataset
- Local explanations — why *this specific transaction* was flagged
- Audit-ready output for regulated financial environments

---

## Key design decisions

**API-first, not dashboard-first** — real fraud systems are APIs consumed by banking infrastructure, not Streamlit apps. The UI is a demo layer.

**Cost optimization over accuracy** — the threshold is selected by minimizing ₹ expected loss, not F1 score.

**SHAP for every decision** — a fraud system without explainability cannot be deployed in regulated environments. Every BLOCK decision is justified.

**Decoupled architecture** — training pipeline, inference API, and analyst UI are fully separated.

---

## Stack

`Python` `XGBoost` `scikit-learn` `imbalanced-learn` `SHAP` `FastAPI` `Streamlit` `Render`

---

## Run locally

```bash
git clone https://github.com/AkashMs24/Cost-Sensitive-Real-Time-Fraud-Detection-Decision-System.git
cd Cost-Sensitive-Real-Time-Fraud-Detection-Decision-System
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## What's next

- Real-time velocity features (transaction frequency, device fingerprinting)
- Concept drift detection + automated retraining
- Kafka streaming integration
- Role-based analyst dashboards

---

## Related projects

- [Decision Intelligence System](https://github.com/AkashMs24/Decisioniq-ai-business-intelligence) — ML + LLM business intelligence platform
- [Employee Attrition XAI](https://github.com/AkashMs24/Employee-Attrition-Risk-Assessment-Using-Explainable-Machine-Learning) — SHAP-powered HR risk scoring
- [FarmVoice AI](https://github.com/AkashMs24/FarmVoice-AI) — NLP + SHAP crop advisory

---

<div align="center">

Built by **Akash M S** · Presidency University, Bengaluru  
[LinkedIn](https://www.linkedin.com/in/akash-m-s-414a21297) · [GitHub](https://github.com/AkashMs24) · ms29akash@gmail.com

</div>
