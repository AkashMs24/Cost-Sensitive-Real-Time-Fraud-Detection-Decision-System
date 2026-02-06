import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

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

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight={0: 1, 1: 10},
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("Random Forest Results")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

from xgboost import XGBClassifier

fraud_count = y_train.sum()
non_fraud_count = len(y_train) - fraud_count

scale_pos_weight = non_fraud_count / fraud_count
print("scale_pos_weight:", scale_pos_weight)

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

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

print("XGBoost Results")
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

from sklearn.ensemble import IsolationForest

# Train only on non-fraud data
X_train_legit = X_train[y_train == 0]

iso = IsolationForest(
    n_estimators=100,
    contamination=0.002,  # ~fraud rate
    random_state=42,
    n_jobs=-1
)

iso.fit(X_train_legit)

iso_pred = iso.predict(X_test)
iso_pred = (iso_pred == -1).astype(int)

print("Isolation Forest Results")
print(classification_report(y_test, iso_pred))
