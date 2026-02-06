# Fraud Detection Decision System

An end-to-end **fraud detection decision system** that detects fraudulent transactions in real time, explains why they are flagged, and optimizes decisions based on **business cost**, not just accuracy.

This project is designed to reflect **real-world fraud systems used in fintech and banking**, not a Kaggle-style notebook.

---

## 🚀 Key Features

- Extreme class imbalance handling (fraud < 1%)
- Multiple ML models:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost (primary model)
  - Isolation Forest (anomaly detection)
- Cost-sensitive threshold optimization
- Explainable AI using SHAP
- Real-time fraud scoring via FastAPI
- Production-ready project structure

---

## 🧠 Problem Framing

Fraud detection is treated as a **decision problem under uncertainty**, not a pure prediction task.

- False negatives (missed fraud) cause large financial loss
- False positives (blocking genuine users) hurt customer trust
- Accuracy is misleading due to extreme class imbalance

The system optimizes decisions using **business cost trade-offs**.

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- SHAP (Explainability)
- FastAPI (API deployment)

---

## 📊 Dataset

**Credit Card Fraud Dataset (European cardholders)**  
- Source: UCI / Kaggle  
- Transactions: ~284,000  
- Fraud rate: ~0.17%  

⚠️ Dataset is not included in the repository due to size and licensing.

---

## 🧪 Project Structure

