import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv(
    r"C:\Users\bumik\OneDrive\Documents\New folder (2)\fraud-detection-system\data\raw\creditcard.csv"
)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Train XGBoost
fraud_count = y_train.sum()
non_fraud_count = len(y_train) - fraud_count
scale_pos_weight = non_fraud_count / fraud_count

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

# Probabilities
y_prob = xgb.predict_proba(X_test)[:, 1]

COST_FN = 10000  # fraud missed
COST_FP = 200    # genuine blocked

def business_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn * COST_FN + fp * COST_FP

thresholds = np.arange(0.01, 0.99, 0.01)
costs = []

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    cost = business_cost(y_test, y_pred_t)
    costs.append(cost)

best_threshold = thresholds[np.argmin(costs)]
min_cost = min(costs)

print("Best Threshold:", best_threshold)
print("Minimum Business Cost: ₹", min_cost)

default_pred = (y_prob >= 0.5).astype(int)
default_cost = business_cost(y_test, default_pred)

print("Cost @ 0.5 threshold: ₹", default_cost)
print("Cost @ optimal threshold: ₹", min_cost)

