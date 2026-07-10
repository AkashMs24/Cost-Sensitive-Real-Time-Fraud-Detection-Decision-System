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

def _get_api_url():
    # Streamlit Cloud: set via the Secrets tab (st.secrets). Render/other hosts: set via
    # a plain OS environment variable. Check both so this works on either platform.
    try:
        if "FRAUD_API_URL" in st.secrets:
            return st.secrets["FRAUD_API_URL"]
    except Exception:
        pass
    return os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8000")

API_URL = _get_api_url()

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

tab_predict, tab_batch, tab_history, tab_rings, tab_drift, tab_threshold, tab_compare, tab_investigate = st.tabs(
    ["💳 Predict", "📁 Batch Scoring", "🕓 History", "🕸️ Fraud Rings", "📉 Drift Monitor",
     "🎯 Adaptive Threshold", "📊 Model Comparison", "🧑‍💼 Investigate"]
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

# ==============================
# TAB — BATCH SCORING
# ==============================
with tab_batch:
    st.markdown("<div class='section-label'>Batch Transaction Scoring</div>", unsafe_allow_html=True)
    st.caption(
        "Upload a CSV of transactions (columns: Time, Amount, V1..V28, optionally account_id/device_id/merchant_id) "
        "to score up to 5,000 transactions at once — the same pipeline as single-transaction scoring, "
        "including SHAP, ring detection, and drift monitoring for every row."
    )

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    if uploaded is not None:
        if st.button("⚡ Score Batch"):
            with st.spinner("Scoring transactions..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                    r = requests.post(f"{API_URL}/predict_batch", files=files, timeout=120)
                    r.raise_for_status()
                    batch_result = r.json()
                    st.session_state["batch_result"] = batch_result
                    st.session_state["batch_file_bytes"] = uploaded.getvalue()
                    st.session_state["batch_file_name"] = uploaded.name
                except Exception as e:
                    st.error(f"Batch scoring failed: {e}")

    if "batch_result" in st.session_state:
        batch_result = st.session_state["batch_result"]
        summary = batch_result["summary"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scored", summary["n_scored"])
        c2.metric("Blocked", summary["n_block"])
        c3.metric("Review", summary["n_review"])
        c4.metric("Allowed", summary["n_allow"])

        if st.button("⬇️ Download Scored CSV"):
            try:
                files = {"file": (st.session_state["batch_file_name"], st.session_state["batch_file_bytes"], "text/csv")}
                r = requests.post(f"{API_URL}/predict_batch/download", files=files, timeout=120)
                r.raise_for_status()
                st.download_button("Save scored_transactions.csv", data=r.content, file_name="scored_transactions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Download failed: {e}")

        st.markdown("<div class='section-label'>Results (highest risk first)</div>", unsafe_allow_html=True)
        try:
            import pandas as pd
            results_df = pd.DataFrame(batch_result["results"]).sort_values("fraud_probability", ascending=False)
            st.dataframe(results_df, width='stretch', height=400)
        except Exception:
            st.json(batch_result["results"][:20])

# ==============================
# TAB — HISTORY
# ==============================
with tab_history:
    st.markdown("<div class='section-label'>Transaction History</div>", unsafe_allow_html=True)
    st.caption("Every transaction scored this session (single + batch), filterable and searchable.")

    hf1, hf2, hf3, hf4 = st.columns([1.2, 1.2, 1, 0.6])
    with hf1:
        hist_decision = st.selectbox("Decision", ["All", "BLOCK", "REVIEW", "ALLOW"])
    with hf2:
        hist_account = st.text_input("Search Account ID", value="")
    with hf3:
        hist_min_prob = st.slider("Min. fraud probability", 0.0, 1.0, 0.0, 0.05)
    with hf4:
        st.markdown("<br>", unsafe_allow_html=True)
        hist_refresh = st.button("🔄 Refresh")

    params = {"limit": 200, "min_probability": hist_min_prob}
    if hist_decision != "All":
        params["decision"] = hist_decision
    if hist_account:
        params["account_id"] = hist_account

    hist_data, err = api_get("/transactions/history", **params)
    if err:
        st.error(f"Couldn't load history: {err}")
    elif not hist_data["transactions"]:
        st.info("No transactions match these filters yet. Score some in the Predict or Batch Scoring tab.")
    else:
        st.caption(f"Showing {len(hist_data['transactions'])} matching transactions")
        try:
            import pandas as pd
            rows = []
            for t in hist_data["transactions"]:
                rows.append({
                    "transaction_id": t["transaction_id"][:8] + "…",
                    "account_id": t["input"].get("account_id", ""),
                    "Amount": t["input"].get("Amount", 0),
                    "fraud_probability": t["fraud_probability"],
                    "decision": t["decision"],
                    "true_label": {1: "FRAUD", 0: "genuine", None: "—"}.get(t["true_label"], "—"),
                })
            hist_df = pd.DataFrame(rows)
            st.dataframe(hist_df, width='stretch', height=450)
            csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Export filtered history as CSV", data=csv_bytes, file_name="transaction_history.csv", mime="text/csv")
        except Exception:
            st.json(hist_data["transactions"][:20])

    # ==============================
# TAB 2 — FRAUD RINGS
# ==============================
with tab_rings:
    st.markdown("<div class='section-label'>Graph-Based Fraud Ring Detection</div>", unsafe_allow_html=True)
    st.caption(
        "Transactions are linked in a graph by shared account/device/merchant IDs. "
        "Community detection surfaces clusters where many accounts funnel through the same device or merchant — "
        "the classic signature of an organized fraud ring rather than isolated incidents."
    )
    if st.button("🔄 Refresh Rings"):
        st.rerun()

    rings_data, err = api_get("/fraud_rings", top_k=10)
    if err:
        st.error(f"Couldn't load rings: {err}")
    else:
        stats = rings_data["graph_stats"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions Tracked", stats["n_transactions_tracked"])
        c2.metric("Graph Nodes", stats["n_graph_nodes"])
        c3.metric("Graph Edges", stats["n_graph_edges"])

        rings = rings_data["rings"]
        if not rings:
            st.info("No high-risk rings detected yet. Run the stream simulator (`python -m src.stream_simulator`) to generate live traffic, including planted rings.")
        else:
            for ring in rings:
                risk_pct = ring["risk_score"]
                pill = "pill-block" if risk_pct > 0.6 else "pill-review"
                st.markdown(f"""
                <div class='card card-danger'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div style='font-family:Syne,sans-serif; font-weight:700;'>{ring['ring_id'].upper()}</div>
                        <span class='decision-pill {pill}'>risk {risk_pct:.0%}</span>
                    </div>
                    <div style='font-size:0.85rem; color:#aaa; margin-top:0.5rem; line-height:1.7;'>
                        {ring['n_transactions']} transactions across <strong>{ring['n_accounts']} accounts</strong>,
                        funneled through <strong>{ring['n_devices']} device(s)</strong> and {ring['n_merchants']} merchant(s)
                        — fan-out ratio {ring['fan_out_ratio']}.<br>
                        Avg fraud probability: {ring['avg_fraud_probability']:.1%}<br>
                        Shared devices: <code>{', '.join(ring['shared_devices'])}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ==============================
# TAB 3 — DRIFT MONITOR
# ==============================
with tab_drift:
    st.markdown("<div class='section-label'>Concept / Data Drift Monitor</div>", unsafe_allow_html=True)
    st.caption(
        "Population Stability Index (PSI) between the training-time reference distribution and a rolling window "
        "of live traffic. PSI < 0.10 = stable, 0.10–0.25 = moderate shift, ≥ 0.25 = severe — retrain recommended."
    )
    if st.button("🔄 Refresh Drift Status"):
        st.rerun()

    drift, err = api_get("/drift/status")
    if err:
        st.error(f"Couldn't load drift status: {err}")
    elif drift.get("status") == "WARMING_UP":
        st.info(drift["message"])
    else:
        status_color = {"STABLE": "#00e676", "MODERATE": "#ffb800", "SEVERE": "#ff4444"}.get(drift["status"], "#c8ff00")
        st.markdown(f"""
        <div style='text-align:center; padding: 1rem 0;'>
            <div class='result-score' style='color:{status_color}; font-size:2.2rem;'>{drift['status']}</div>
            <div class='result-label'>Overall Drift Status</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions Seen", drift["n_seen"])
        c2.metric("Features Moderate", drift["n_features_moderate"])
        c3.metric("Retrain Recommended", "YES" if drift["retrain_recommended"] else "NO")

        st.markdown("<div class='section-label'>Top Drifting Features</div>", unsafe_allow_html=True)
        for feat in drift["top_drifting_features"]:
            badge_color = {"STABLE": "pill-allow", "MODERATE": "pill-review", "SEVERE": "pill-block"}[feat["status"]]
            st.markdown(
                f"<div class='card' style='display:flex; justify-content:space-between; align-items:center; padding:0.7rem 1rem; margin-bottom:0.4rem;'>"
                f"<span><strong>{feat['feature']}</strong> &nbsp; PSI = {feat['psi']}</span>"
                f"<span class='decision-pill {badge_color}'>{feat['status']}</span></div>",
                unsafe_allow_html=True,
            )

# ==============================
# TAB 4 — ADAPTIVE THRESHOLD
# ==============================
with tab_threshold:
    st.markdown("<div class='section-label'>Online-Learning Decision Threshold</div>", unsafe_allow_html=True)
    st.caption(
        "Instead of freezing one cost-optimal threshold forever, the system nudges it toward the direction that "
        "would have minimized cost every time ground-truth feedback arrives (Robbins-Monro stochastic approximation)."
    )
    if st.button("🔄 Refresh Threshold Status"):
        st.rerun()

    thr, err = api_get("/threshold/status")
    if err:
        st.error(f"Couldn't load threshold status: {err}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Threshold", f"{thr['current_threshold']:.4f}")
        c2.metric("Baseline (offline) Threshold", f"{thr['baseline_threshold']:.4f}")
        c3.metric("Feedback Updates", thr["n_feedback_updates"])

        c4, c5 = st.columns(2)
        c4.metric("Cumulative Realized Cost", f"₹{thr['cumulative_realized_cost']:.0f}")
        c5.metric("Savings vs. Frozen Threshold", f"₹{thr['savings_vs_frozen_threshold']:.0f}")

        history = thr.get("history", [])
        if len(history) > 1:
            try:
                import pandas as pd
                hist_df = pd.DataFrame(history)
                st.markdown("<div class='section-label'>Threshold Over Time</div>", unsafe_allow_html=True)
                st.line_chart(hist_df.set_index("n_updates")["threshold"])
            except Exception:
                pass

# ==============================
# TAB — MODEL COMPARISON
# ==============================
with tab_compare:
    st.markdown("<div class='section-label'>Model Comparison</div>", unsafe_allow_html=True)
    st.caption(
        "Every candidate model trained during `python -m src.train_pipeline`, evaluated at ITS OWN "
        "cost-optimal threshold (not just accuracy at 0.5) — so this is an honest comparison of what "
        "each model would actually cost the business in production, not just an AUC leaderboard."
    )

    comp, err = api_get("/models/comparison")
    if err:
        st.error(f"Couldn't load model comparison: {err}")
    else:
        st.markdown(f"""
        <div class='card card-accent'>
            <div style='font-family:Syne,sans-serif; font-weight:700; margin-bottom:0.5rem;'>
                Deployed: {comp['deployed_model']}
            </div>
            <div style='font-size:0.85rem; color:#aaa; line-height:1.7;'>{comp['deployment_rationale']}</div>
        </div>
        """, unsafe_allow_html=True)

        try:
            import pandas as pd
            comp_df = pd.DataFrame(comp["results"])
            comp_df = comp_df.rename(columns={
                "model": "Model", "precision": "Precision", "recall": "Recall",
                "f1_score": "F1", "roc_auc": "ROC-AUC", "best_threshold": "Best Threshold",
                "min_business_cost": "Min Cost (₹)", "cost_at_naive_0.5": "Cost @ 0.5 (₹)",
                "train_time_seconds": "Train Time (s)",
            })
            st.dataframe(comp_df, width='stretch', hide_index=True)

            st.markdown("<div class='section-label'>Business Cost by Model</div>", unsafe_allow_html=True)
            chart_df = pd.DataFrame(comp["results"])[["model", "min_business_cost"]].set_index("model")
            st.bar_chart(chart_df)
        except Exception:
            st.json(comp["results"])

# ==============================
# TAB 5 — INVESTIGATE (LLM COPILOT)
# ==============================
with tab_investigate:
    st.markdown("<div class='section-label'>LLM Fraud Investigation Copilot</div>", unsafe_allow_html=True)
    st.caption(
        "Turns the model's probability + SHAP drivers into a short, human-readable investigation note for an analyst. "
        "Falls back to a template if GROQ_API_KEY (or ANTHROPIC_API_KEY) isn't set on the API server."
    )

    inv_txn_id = st.text_input("Transaction ID to investigate", value=st.session_state.get("last_transaction_id", ""), key="inv_txn_id")
    if st.button("🧑‍💼 Generate Investigation Note"):
        if not inv_txn_id:
            st.warning("Analyze a transaction in the Predict tab first, or paste a transaction ID.")
        else:
            note, err = api_post(f"/investigate/{inv_txn_id}")
            if err:
                st.error(f"Failed: {err}")
            else:
                source_badge = {"groq_api": "🤖 Groq API", "anthropic_api": "🤖 Claude API", "template_fallback": "📋 Template (no API key)", "template_fallback_after_error": "📋 Template (API error)"}.get(note.get("source"), note.get("source"))
                st.markdown(f"""
                <div class='card card-accent'>
                    <div style='font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;'>{source_badge}</div>
                    <div style='font-size:0.92rem; line-height:1.7;'>{note['note']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Recent Transactions</div>", unsafe_allow_html=True)
    recent, err = api_get("/transactions/recent", limit=10)
    if not err and recent["transactions"]:
        for t in recent["transactions"]:
            pill_class = {"BLOCK": "pill-block", "REVIEW": "pill-review", "ALLOW": "pill-allow"}.get(t["decision"], "pill-allow")
            label = "" if t["true_label"] is None else (" · confirmed FRAUD" if t["true_label"] == 1 else " · confirmed genuine")
            st.markdown(
                f"<div class='card' style='display:flex; justify-content:space-between; align-items:center; padding:0.6rem 1rem; margin-bottom:0.3rem;'>"
                f"<span style='font-family:DM Mono,monospace; font-size:0.78rem;'>{t['transaction_id'][:8]}… · ₹{t['input']['Amount']:.2f}{label}</span>"
                f"<span class='decision-pill {pill_class}'>{t['decision']} · {t['fraud_probability']:.0%}</span></div>",
                unsafe_allow_html=True,
            )

# ==============================
# FOOTER
# ==============================

st.markdown("""
<hr>
<div style='text-align:center; padding:1rem 0 0.5rem;'>
    <div style='font-family:DM Mono,monospace; font-size:0.7rem; color:#333; letter-spacing:0.1em;'>
        FRAUDSHIELD &nbsp;·&nbsp; v2.0 &nbsp;·&nbsp; Cost-Sensitive · Graph-Aware · Adaptive · LLM-Explained &nbsp;·&nbsp; Portfolio Demonstration
    </div>
</div>
""", unsafe_allow_html=True)
