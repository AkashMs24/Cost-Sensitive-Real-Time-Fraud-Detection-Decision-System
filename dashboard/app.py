import streamlit as st
import requests

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    layout="centered"
)

st.title("💳 Fraud Detection System")
st.write("Enter transaction details to assess fraud risk in real time.")

# ---------------------------
# Info note (IMPORTANT)
# ---------------------------
st.info(
    "⚠️ Use realistic transaction values. "
    "Random or zeroed PCA inputs may produce extreme risk scores."
)

API_URL = "http://127.0.0.1:8000/predict_fraud"

# ---------------------------
# Input form
# ---------------------------
with st.form("fraud_form"):
    st.subheader("Transaction Details")

    Time = st.number_input("Time", value=0.0, format="%.2f")
    Amount = st.number_input("Amount", value=0.0, format="%.2f")

    st.subheader("PCA Features (V1 – V28)")
    features = {}

    cols = st.columns(4)
    for i in range(1, 29):
        with cols[(i - 1) % 4]:
            features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.4f"
            )

    submit = st.form_submit_button("🔍 Check Fraud Risk")

# ---------------------------
# Prediction logic
# ---------------------------
if submit:
    payload = {
        "Time": Time,
        "Amount": Amount,
        **features
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code != 200:
            st.error("❌ API error. Ensure FastAPI backend is running.")
        else:
            result = response.json()

            risk = result["risk_level"]
            prob = float(result["fraud_probability"])

            st.divider()
            st.subheader("🔍 Fraud Assessment Result")

            # Probability bar (visual intuition)
            st.progress(min(prob, 1.0))

            col1, col2 = st.columns(2)
            col1.metric("Fraud Probability", f"{prob:.2f}")
            col2.metric("Risk Level", risk)

            if risk == "HIGH RISK":
                st.error(
                    "🚨 **HIGH RISK TRANSACTION**\n\n"
                    "Immediate action recommended."
                )
            elif risk == "MEDIUM RISK":
                st.warning(
                    "⚠️ **MEDIUM RISK TRANSACTION**\n\n"
                    "Manual review suggested."
                )
            else:
                st.success(
                    "✅ **LOW RISK TRANSACTION**\n\n"
                    "Transaction appears safe."
                )

            st.caption(
                "Prediction served by a FastAPI backend. "
                "Decision thresholds are optimized based on business cost trade-offs."
            )

    except Exception as e:
        st.error("❌ API call failed.")
        st.write(e)
