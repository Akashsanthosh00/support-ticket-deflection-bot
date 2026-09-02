"""
Streamlit demo UI for the ticket deflection bot.

Talks to the FastAPI backend over HTTP — start the API first:
    uvicorn app.main:app --reload --port 8000
then run this:
    streamlit run streamlit_app.py

Deploy: push this repo to GitHub, then deploy on Streamlit Community Cloud
pointing at streamlit_app.py. Point API_BASE_URL (in Streamlit secrets or
.env) at wherever the FastAPI backend is hosted.
"""
import os

import requests
import streamlit as st

# set_page_config must be the very first Streamlit command in the script —
# touching st.secrets before this (even inside a try/except) can trip
# Streamlit's "already called a command" check, so it goes first here.
st.set_page_config(page_title="Ticket Deflection Bot", page_icon="🎫", layout="centered")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
try:
    # st.secrets is only used when deployed on Streamlit Community Cloud
    # (secrets.toml set there). Locally, no secrets.toml exists at all, and
    # st.secrets raises rather than returning a default, so this is wrapped.
    API_BASE_URL = st.secrets.get("API_BASE_URL", API_BASE_URL)
except Exception:
    pass

st.title("🎫 Support Ticket Deflection Bot")
st.caption("Type a customer message and see whether the bot answers it or escalates to a human.")

# --- stats row ---
try:
    stats = requests.get(f"{API_BASE_URL}/stats", timeout=5).json()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tickets", stats["total"])
    c2.metric("Answered", stats["answered"])
    c3.metric("Escalated", stats["escalated"])
    c4.metric("Deflection rate", f"{stats['deflection_rate'] * 100:.1f}%")
except requests.exceptions.RequestException:
    st.warning(f"Can't reach the API at {API_BASE_URL}. Is `uvicorn app.main:app` running?")

st.divider()

# --- ticket submission ---
message = st.text_area("Customer message", height=100, placeholder="How do I reset my password?")
submitted = st.button("Submit ticket", type="primary", disabled=not message.strip())

if submitted:
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(f"{API_BASE_URL}/ticket", json={"message": message}, timeout=60)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            result = None

    if result:
        if result["decision"] == "answered":
            st.success("✅ Answered automatically")
            st.markdown(result["answer"])
        else:
            st.warning("🙋 Escalated to a human")
            for reason in result["escalation_reasons"]:
                st.markdown(f"- {reason}")

        with st.expander("Signals & sources"):
            st.json(
                {
                    "answer_confidence": result["answer_confidence"],
                    "retrieval_top_score": result["retrieval_top_score"],
                    "sentiment_score": result["sentiment_score"],
                    "matched_keyword": result["matched_keyword"],
                    "sources": result["sources"],
                }
            )

st.divider()

# --- recent log ---
st.subheader("Recent tickets")
try:
    logs = requests.get(f"{API_BASE_URL}/logs", params={"limit": 20}, timeout=5).json()
    if logs:
        st.dataframe(
            [
                {
                    "time": row["created_at"],
                    "message": row["customer_message"][:80],
                    "decision": row["decision"],
                    "confidence": row["answer_confidence"],
                }
                for row in logs
            ],
            use_container_width=True,
        )
    else:
        st.caption("No tickets logged yet.")
except requests.exceptions.RequestException:
    pass
