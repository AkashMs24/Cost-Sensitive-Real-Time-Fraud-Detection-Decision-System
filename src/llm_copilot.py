"""
LLM fraud-investigation copilot.

Every dashboard in this space shows an analyst a probability score and a
SHAP bar chart, then leaves them to translate that into "should I call the
customer, freeze the card, or escalate to the fraud team?" This module
turns the model's raw output (probability + top SHAP drivers + transaction
context) into a short, human-readable investigation note using an LLM --
the kind of narrative a junior fraud analyst would otherwise spend
5 minutes writing by hand for every flagged case.

Design choices:
- Provider-agnostic: set LLM_PROVIDER to "groq" (default, free tier -- get a
  key at https://console.groq.com/keys) or "anthropic" (paid, get a key at
  https://console.anthropic.com). Both use plain `requests`, no extra SDKs.
- Fully optional: if no API key is set, falls back to a deterministic
  template so the rest of the system (API, dashboard, demo) keeps working
  without any paid or free-tier API calls.

Environment variables:
  LLM_PROVIDER      "groq" (default) or "anthropic"
  GROQ_API_KEY       required if LLM_PROVIDER=groq
  GROQ_MODEL         default: "llama-3.3-70b-versatile"
  ANTHROPIC_API_KEY  required if LLM_PROVIDER=anthropic
  ANTHROPIC_MODEL    default: "claude-sonnet-5"
"""

import os
import requests

PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


def _template_fallback(transaction, top_features, decision, fraud_probability):
    driver_lines = "; ".join(
        f"{f['feature']} ({f['direction'].replace('_', ' ')}, value={f['value']:.2f})"
        for f in top_features[:3]
    )
    return (
        f"[Template note -- set GROQ_API_KEY (or ANTHROPIC_API_KEY) for full LLM narratives]\n"
        f"Decision: {decision} at fraud probability {fraud_probability:.2%}. "
        f"Top signals: {driver_lines}. "
        f"Amount: {transaction.get('Amount', 'n/a')}. "
        f"Recommended action: {'escalate to fraud team immediately' if decision == 'BLOCK' else 'manual review within SLA' if decision == 'REVIEW' else 'no action needed'}."
    )


def _build_prompt(transaction, top_features, decision, fraud_probability):
    driver_summary = "\n".join(
        f"- {f['feature']} = {f['value']:.3f} (pushed {f['direction'].replace('_', ' ')}, "
        f"impact {f['shap_value']:+.4f})"
        for f in top_features
    )
    return f"""You are a fraud-investigation assistant for a bank's risk operations team.
A transaction-scoring model produced the following result. Write a concise (3-4 sentence)
investigation note for a human analyst: state the risk verdict, explain in plain English
WHY the model flagged it using the SHAP drivers below, and give one concrete recommended
next action. Do not repeat raw numbers back verbatim; interpret them.

Decision: {decision}
Fraud probability: {fraud_probability:.2%}
Transaction amount: {transaction.get('Amount', 'unknown')}
Top model drivers (SHAP):
{driver_summary}
"""


def _call_groq(prompt, api_key):
    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_anthropic(prompt, api_key):
    response = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    return "\n".join(text_blocks).strip()


def generate_investigation_note(transaction: dict, top_features: list, decision: str, fraud_probability: float):
    """
    transaction: dict of the raw transaction fields (Amount, Time, account_id, etc.)
    top_features: output of FraudExplainer.explain_instance()
    decision: "ALLOW" | "REVIEW" | "BLOCK"
    fraud_probability: float in [0, 1]
    """
    if PROVIDER == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
    elif PROVIDER == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    else:
        api_key = None

    if not api_key:
        return {
            "note": _template_fallback(transaction, top_features, decision, fraud_probability),
            "source": "template_fallback",
        }

    prompt = _build_prompt(transaction, top_features, decision, fraud_probability)

    try:
        if PROVIDER == "groq":
            note = _call_groq(prompt, api_key)
            model_used = GROQ_MODEL
        else:
            note = _call_anthropic(prompt, api_key)
            model_used = ANTHROPIC_MODEL

        if not note:
            note = _template_fallback(transaction, top_features, decision, fraud_probability)

        return {"note": note, "source": f"{PROVIDER}_api", "model": model_used}
    except Exception as e:
        return {
            "note": _template_fallback(transaction, top_features, decision, fraud_probability),
            "source": "template_fallback_after_error",
            "error": str(e),
        }
