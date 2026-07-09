import streamlit as st
import requests
import os

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="FraudShield — Cost-Sensitive Fraud Decisioning",
    page_icon="💳",
    layout="wide",
)

# ==============================
# PREMIUM DARK THEME CSS
# ==============================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --bg:       #080808;
        --surface:  #111111;
        --surface2: #181818;
        --border:   #242424;
        --accent:   #c8ff00;
        --danger:   #ff4444;
        --warn:     #ffb800;
        --ok:       #00e676;
        --text:     #f0f0f0;
        --muted:    #666666;
        --radius:   14px;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    /* Main container */
    .main .block-container {
        background: var(--bg);
        max-width: 1100px;
        padding: clamp(1rem, 4vw, 2.5rem);
    }

    /* Sidebar */
    [data-testid="stSidebar"] { display: none; }

    /* Hero */
    .hero {
        text-align: center;
        padding: clamp(2.5rem, 8vw, 5rem) 1rem clamp(1.5rem, 5vw, 3rem);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse 70% 50% at 50% 0%,
            rgba(200,255,0,0.07) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(200,255,0,0.08);
        border: 1px solid rgba(200,255,0,0.25);
        color: var(--accent);
        font-size: clamp(0.6rem, 1.8vw, 0.72rem);
        font-weight: 500;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        padding: 0.3rem 0.9rem;
        border-radius: 100px;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2rem, 7vw, 3.8rem);
        font-weight: 800;
        line-height: 1.05;
        color: var(--text);
        margin: 0 0 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-title span { color: var(--accent); }
    .hero-sub {
        font-size: clamp(0.85rem, 2.5vw, 1rem);
        color: var(--muted);
        font-weight: 300;
        margin: 0;
    }

    /* Section labels */
    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: clamp(0.65rem, 1.8vw, 0.72rem);
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 2rem 0 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }

    /* Cards */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: clamp(1rem, 3vw, 1.6rem);
        margin-bottom: 1rem;
    }
    .card-danger { border-left: 3px solid var(--danger); background: rgba(255,68,68,0.04); }
    .card-warn   { border-left: 3px solid var(--warn);   background: rgba(255,184,0,0.04); }
    .card-ok     { border-left: 3px solid var(--ok);     background: rgba(0,230,118,0.04); }
    .card-accent { border-left: 3px solid var(--accent); }

    /* Info box override */
    [data-testid="stAlert"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-left: 3px solid var(--warn) !important;
        border-radius: var(--radius) !important;
        color: var(--text) !important;
        font-size: 0.88rem;
    }

    /* Headers */
    h1,h2,h3 {
        font-family: 'Syne', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.01em;
    }
    h2 { font-size: clamp(1.1rem, 3vw, 1.4rem) !important; font-weight: 700 !important; margin-top: 0 !important; }
    h3 { font-size: clamp(0.95rem, 2.5vw, 1.1rem) !important; font-weight: 600 !important; }

    /* Inputs */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.88rem !important;
        transition: border-color 0.2s;
    }
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(200,255,0,0.1) !important;
    }
    [data-testid="stNumberInput"] label,
    [data-testid="stTextInput"] label {
        color: var(--muted) !important;
        font-size: 0.75rem !important;
        font-family: 'DM Mono', monospace !important;
        letter-spacing: 0.04em;
    }

    /* Submit button */
    [data-testid="stFormSubmitButton"] button {
        background: var(--accent) !important;
        color: #000 !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: clamp(0.9rem, 2.5vw, 1rem) !important;
        letter-spacing: 0.04em;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s, transform 0.15s !important;
        cursor: pointer;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem 1.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        color: var(--muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'DM Mono', monospace !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: clamp(1.2rem, 3.5vw, 1.7rem) !important;
        font-weight: 700 !important;
        color: var(--accent) !important;
    }

    /* Progress bar */
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, var(--accent), var(--danger)) !important;
        border-radius: 4px !important;
    }
    [data-testid="stProgressBar"] > div {
        background: var(--surface2) !important;
        border-radius: 4px !important;
        height: 6px !important;
    }

    /* Divider */
    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    /* Caption */
    [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
        font-size: 0.75rem !important;
        font-family: 'DM Mono', monospace !important;
    }

    /* PCA grid label styling */
    .pca-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: var(--muted);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    /* Result block */
    .result-score {
        font-family: 'Syne', sans-serif;
        font-size: clamp(3rem, 10vw, 5rem);
        font-weight: 800;
        line-height: 1;
        text-align: center;
    }
    .result-label {
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        text-align: center;
        margin-top: 0.3rem;
        color: var(--muted);
        font-family: 'DM Mono', monospace;
    }
    .decision-pill {
        display: inline-block;
        font-family: 'Syne', sans-serif;
        font-size: clamp(0.8rem, 2vw, 0.95rem);
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.35rem 1.1rem;
        border-radius: 100px;
    }
    .pill-block  { background: rgba(255,68,68,0.15);  color: #ff6b6b; border: 1px solid rgba(255,68,68,0.3); }
    .pill-review { background: rgba(255,184,0,0.15);  color: #ffcc44; border: 1px solid rgba(255,184,0,0.3); }
    .pill-allow  { background: rgba(0,230,118,0.12);  color: #00e676; border: 1px solid rgba(0,230,118,0.25); }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--surface); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* Mobile */
    @media (max-width: 600px) {
        .main .block-container { padding: 0.75rem !important; }
        .hero { padding: 2rem 0.5rem 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# HERO
# ==============================

st.markdown("""
<div class="hero">
    <div class="hero-badge">Real-Time · Cost-Sensitive · Graph-Aware · Adaptive</div>
    <div class="hero-title">Fraud<span>Shield</span></div>
    <p class="hero-sub">Cost-sensitive decisioning, fraud-ring graphs, drift monitoring, and an LLM investigation copilot</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# API ENDPOINT
# ==============================

API_URL = os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8000")

def api_get(path, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=8)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def api_post(path, payload=None):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload or {}, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

health, health_err = api_get("/")
if health_err:
    st.error(f"⚠️ Can't reach the API at {API_URL}. Start it with `uvicorn api.app:app --reload`, or set FRAUD_API_URL. ({health_err})")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("API Status", "Online")
    c2.metric("Current Threshold", f"{health['current_threshold']:.3f}")
    c3.metric("Transactions Processed", health["n_transactions_processed"])

st.markdown("<br>", unsafe_allow_html=True)

tab_predict, tab_rings, tab_drift, tab_threshold, tab_investigate = st.tabs(
    ["💳 Predict", "🕸️ Fraud Rings", "📉 Drift Monitor", "🎯 Adaptive Threshold", "🧑‍💼 Investigate"]
)

# ==============================
# TAB 1 — PREDICT
# ==============================
with tab_predict:
    st.info("⚠️ Use realistic transaction values. Random or zeroed PCA inputs may produce extreme risk scores.")
    st.markdown("<div class='section-label'>Transaction Details</div>", unsafe_allow_html=True)

    with st.form("fraud_form"):
        col_t, col_a = st.columns(2)
        with col_t:
            Time = st.number_input("Time", value=0.0, format="%.2f")
        with col_a:
            Amount = st.number_input("Amount (₹ / $)", value=0.0, format="%.2f")

        col_acc, col_dev, col_mer = st.columns(3)
        with col_acc:
            account_id = st.text_input("Account ID", value="ACC100001")
        with col_dev:
            device_id = st.text_input("Device ID", value="DEV1001")
        with col_mer:
            merchant_id = st.text_input("Merchant ID", value="MER1")

        st.markdown("<div class='section-label' style='margin-top:1.5rem;'>PCA Features — V1 to V28</div>", unsafe_allow_html=True)

        features = {}
        cols = st.columns(4)
        for i in range(1, 29):
            with cols[(i - 1) % 4]:
                features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f")

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("💳 Analyze Transaction")

    if submit:
        payload = {
            "Time": Time, "Amount": Amount,
            "account_id": account_id, "device_id": device_id, "merchant_id": merchant_id,
            **features,
        }
        result, err = api_post("/predict_fraud", payload)

        if err:
            st.error(f"❌ API call failed: {err}")
        else:
            st.session_state["last_transaction_id"] = result["transaction_id"]
            decision = result["decision"]
            prob = float(result["fraud_probability"])

            st.markdown("<hr>", unsafe_allow_html=True)

            score_color = {"BLOCK": "#ff4444", "REVIEW": "#ffb800", "ALLOW": "#00e676"}.get(decision, "#c8ff00")
            pill_class = {"BLOCK": "pill-block", "REVIEW": "pill-review", "ALLOW": "pill-allow"}.get(decision, "pill-allow")

            st.markdown(f"""
            <div style='text-align:center; padding: 1.5rem 0 1rem;'>
                <div class='result-score' style='color:{score_color};'>{prob:.0%}</div>
                <div class='result-label'>Fraud Probability</div>
                <div style='margin-top:1rem;'>
                    <span class='decision-pill {pill_class}'>{decision}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(prob, 1.0))
            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud Probability", f"{prob:.4f}")
            c2.metric("Risk Level", result["risk_level"])
            c3.metric("Threshold Used", f"{result['threshold_used']:.3f}")

            st.markdown("<div class='section-label'>Top SHAP Drivers</div>", unsafe_allow_html=True)
            for feat in result["top_features"]:
                arrow = "🔺" if feat["direction"] == "toward_fraud" else "🔻"
                st.markdown(
                    f"<div class='card' style='padding:0.7rem 1rem; margin-bottom:0.4rem;'>"
                    f"{arrow} <strong>{feat['feature']}</strong> = {feat['value']:.3f} "
                    f"&nbsp;·&nbsp; impact <code>{feat['shap_value']:+.4f}</code></div>",
                    unsafe_allow_html=True,
                )

            st.caption(
                f"Transaction ID: {result['transaction_id']} — use the Investigate tab for an LLM-generated "
                "case note, or submit ground-truth feedback once the outcome is known."
            )

    st.markdown("<div class='section-label'>Submit Ground-Truth Feedback</div>", unsafe_allow_html=True)
    st.caption("When a transaction's real outcome becomes known (chargeback, confirmed fraud, cleared review), submit it here — it drives the adaptive threshold.")
    fb_col1, fb_col2, fb_col3 = st.columns([2, 1, 1])
    with fb_col1:
        fb_txn_id = st.text_input("Transaction ID", value=st.session_state.get("last_transaction_id", ""))
    with fb_col2:
        fb_label = st.selectbox("True Outcome", ["Confirmed Fraud", "Confirmed Genuine"])
    with fb_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Submit Feedback"):
            true_label = 1 if fb_label == "Confirmed Fraud" else 0
            res, err = api_post("/feedback", {"transaction_id": fb_txn_id, "true_label": true_label})
            if err:
                st.error(f"Failed: {err}")
            else:
                st.success(f"Threshold updated to {res['updated_threshold']:.4f}")
