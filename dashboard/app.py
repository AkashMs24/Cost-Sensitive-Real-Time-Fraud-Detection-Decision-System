import streamlit as st
import requests
import os

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="FraudShield — Real-Time Detection",
    page_icon="💳",
    layout="centered",
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
        max-width: 860px;
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
    <div class="hero-badge">Real-Time · Cost-Sensitive · ML-Powered</div>
    <div class="hero-title">Fraud<span>Shield</span></div>
    <p class="hero-sub">Enter transaction details to assess fraud risk in real time</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# INFO NOTE  (original logic untouched)
# ==============================

st.info("⚠️ Use realistic transaction values. Random or zeroed PCA inputs may produce extreme risk scores.")

# ==============================
# API ENDPOINT  (original logic untouched)
# ==============================

API_URL = "https://fraud-detection-system-2-7ake.onrender.com/predict_fraud"

# ==============================
# INPUT FORM
# ==============================

st.markdown("<div class='section-label'>Transaction Details</div>", unsafe_allow_html=True)

with st.form("fraud_form"):

    col_t, col_a = st.columns(2)
    with col_t:
        Time = st.number_input("Time", value=0.0, format="%.2f")
    with col_a:
        Amount = st.number_input("Amount (₹ / $)", value=0.0, format="%.2f")

    st.markdown("<div class='section-label' style='margin-top:1.5rem;'>PCA Features — V1 to V28</div>", unsafe_allow_html=True)

    features = {}
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[(i - 1) % 4]:
            features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f")

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("💳 Analyze Transaction")

# ==============================
# PREDICTION LOGIC  (original logic untouched)
# ==============================

if submit:
    payload = {"Time": Time, "Amount": Amount, **features}

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code != 200:
            st.error("❌ API error. Ensure FastAPI backend is running.")
        else:
            result = response.json()
            risk   = result["risk_level"]
            prob   = float(result["fraud_probability"])

            # Business decision mapping (original logic)
            if risk == "HIGH RISK":
                decision = "BLOCK"
            elif risk == "MEDIUM RISK":
                decision = "REVIEW"
            else:
                decision = "ALLOW"

            # ── Result UI ─────────────────────────────────────────────────
            st.markdown("<hr>", unsafe_allow_html=True)

            # Score + decision pill
            score_color = {
                "BLOCK":  "#ff4444",
                "REVIEW": "#ffb800",
                "ALLOW":  "#00e676",
            }.get(decision, "#c8ff00")

            pill_class = {
                "BLOCK":  "pill-block",
                "REVIEW": "pill-review",
                "ALLOW":  "pill-allow",
            }.get(decision, "pill-allow")

            st.markdown(f"""
            <div style='text-align:center; padding: 1.5rem 0 1rem;'>
                <div class='result-score' style='color:{score_color};'>{prob:.0%}</div>
                <div class='result-label'>Fraud Probability</div>
                <div style='margin-top:1rem;'>
                    <span class='decision-pill {pill_class}'>{decision}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            st.progress(min(prob, 1.0))

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics row
            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud Probability", f"{prob:.4f}")
            c2.metric("Risk Level",        risk)
            c3.metric("Final Decision",    decision)

            st.markdown("<br>", unsafe_allow_html=True)

            # Decision card
            if decision == "BLOCK":
                st.markdown("""
                <div class='card card-danger'>
                    <div style='font-family:Syne,sans-serif; font-weight:700; font-size:1rem; color:#ff6b6b; margin-bottom:0.4rem;'>
                        🚨 High Risk Transaction
                    </div>
                    <div style='font-size:0.88rem; color:#aaa; line-height:1.6;'>
                        This transaction exhibits strong fraud signals. It should be <strong style='color:#ff6b6b;'>blocked immediately</strong> and flagged for investigation.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif decision == "REVIEW":
                st.markdown("""
                <div class='card card-warn'>
                    <div style='font-family:Syne,sans-serif; font-weight:700; font-size:1rem; color:#ffcc44; margin-bottom:0.4rem;'>
                        ⚠️ Medium Risk — Manual Review Required
                    </div>
                    <div style='font-size:0.88rem; color:#aaa; line-height:1.6;'>
                        Transaction shows moderate fraud signals. Recommend <strong style='color:#ffcc44;'>manual review</strong> or step-up authentication before processing.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class='card card-ok'>
                    <div style='font-family:Syne,sans-serif; font-weight:700; font-size:1rem; color:#00e676; margin-bottom:0.4rem;'>
                        ✅ Low Risk — Transaction Safe
                    </div>
                    <div style='font-size:0.88rem; color:#aaa; line-height:1.6;'>
                        No significant fraud indicators detected. Transaction can be <strong style='color:#00e676;'>safely allowed</strong>.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.caption(
                "Prediction served by a FastAPI backend. "
                "Decision thresholds are optimized based on business cost trade-offs."
            )

    except Exception as e:
        st.error("❌ API call failed.")
        st.write(e)

# ==============================
# FOOTER
# ==============================

st.markdown("""
<hr>
<div style='text-align:center; padding:1rem 0 0.5rem;'>
    <div style='font-family:DM Mono,monospace; font-size:0.7rem; color:#333; letter-spacing:0.1em;'>
        FRAUDSHIELD &nbsp;·&nbsp; v1.0 &nbsp;·&nbsp; Portfolio Demonstration &nbsp;·&nbsp; Built by Akash M S
    </div>
</div>
""", unsafe_allow_html=True)
