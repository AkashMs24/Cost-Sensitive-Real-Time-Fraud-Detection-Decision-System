🚨Cost-Sensitive-Real-Time-Fraud-Detection-Decision-System

An end-to-end fraud detection decision system that mirrors how real financial institutions detect, evaluate, and act on fraudulent transactions.

This project goes beyond model accuracy and focuses on business-aware decision-making under extreme class imbalance, explainability, and production-ready deployment.

🔗 Live Demo & API
Backend API (FastAPI – deployed on Render):
👉 https://fraud-detection-system-2-7ake.onrender.com

Interactive Swagger Documentation:
👉 https://fraud-detection-system-2-7ake.onrender.com/docs

Streamlit Dashboard (Analyst Demo – Optional UI):
👉 https://cost-sensitive-real-time-fraud-detection-decision-system-erhna.streamlit.app/
(Analyst-facing demonstration layer; not the core system)

⚠️ The FastAPI service is the core fraud decision engine.
The Streamlit app is an optional demo/analyst interface built on top of the API.

🧠 Problem Framing (Why This Project Is Different)

Fraud detection is not a prediction problem — it is a decision problem.

Fraud is extremely rare (~0.17%)

Missing fraud → direct financial loss

Blocking genuine users → customer dissatisfaction

Accuracy alone is misleading and dangerous

🎯 Goal

Minimize total business cost while maintaining customer trust — not maximize accuracy.

📊 Dataset

Credit Card Fraud Dataset (UCI / Kaggle)

284,807 transactions

492 fraud cases (~0.17%)

PCA-anonymized features for privacy compliance

The dataset is not included in this repository.

🏗️ System Architecture
Transaction Input
        ↓
Feature Processing
        ↓
Imbalance-Aware ML Models
        ↓
Fraud Probability
        ↓
Cost-Optimized Threshold
        ↓
Decision (ALLOW / REVIEW / BLOCK)
        ↓
Explainability (SHAP)
        ↓
FastAPI Endpoint (Deployed)

🤖 Models Implemented
Model	Purpose
Logistic Regression	Baseline
Weighted Logistic Regression	Cost-sensitive baseline
Random Forest	Conservative, high precision
XGBoost	Best recall–precision balance
Isolation Forest	Novel / emerging fraud detection
⚖️ Handling Class Imbalance

Techniques used:

Class weighting

SMOTE (for experimentation)

Imbalance-aware tree models

Threshold tuning (instead of label tuning)

💰 Business Cost Optimization

Instead of using a fixed 0.5 threshold, the decision threshold is optimized using business cost:

Outcome	Cost
Fraud missed (False Negative)	₹10,000
Genuine blocked (False Positive)	₹200

The selected threshold minimizes expected total cost, not classification error.

🔍 Explainability (SHAP)

Global explanations: Identify key fraud-driving features

Local explanations: Explain why a specific transaction was flagged

Makes the system suitable for regulated financial environments

⚙️ API Usage (FastAPI)
Endpoint
POST /predict_fraud

Sample Request
{
  "Time": 0,
  "Amount": 52000,
  "V1": 0.01,
  "V2": -0.03,
  "...": "...",
  "V28": 0.14
}

Sample Response
{
  "fraud_probability": 0.87,
  "risk_level": "HIGH RISK",
  "decision": "BLOCK"
}

🌐 Deployment

Backend: FastAPI deployed on Render

Docs: Swagger UI available publicly

UI: Streamlit used as an optional analyst demo

Architecture: API-first, decoupled UI

🧠 Key Design Decisions

API-first design to reflect real fraud systems

Optimized decisions using business cost, not accuracy

Used SHAP for transparency and auditability

Treated UI as optional, not core system logic

🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

XGBoost

Imbalanced-learn

SHAP

FastAPI

Streamlit

Render (deployment)

🔮 Limitations & Future Improvements

Real-time feature generation (velocity, device fingerprinting)

Concept drift detection and automated retraining

Streaming integration (Kafka)

Role-based dashboards for fraud analysts

📌 Key Takeaway

This project demonstrates real-world ML maturity by focusing on:

Decision systems, not just models

Business trade-offs under uncertainty

Explainability and compliance

Deployment-ready engineering

🏷️ Version

v1.0 • Portfolio Demonstration Project
Built by Akash M S




