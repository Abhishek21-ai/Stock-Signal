"""
Streamlit Dashboard — V1
Section 14.1 + Section 24.2

Panels:
  1. Today's Signals — active signals with ACTIVE/EXPIRED status
  2. Live Performance — rolling Sharpe, drawdown, win rate
  3. Signal History — success rate by stock/regime/sector
  4. Sector Exposure — current allocation vs 30% cap
  5. Pipeline Health — Section 24 system metrics
"""
import streamlit as st

st.set_page_config(
    page_title="Stock Signal Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Stock Signal Platform")
st.caption("Intelligent Multi-Timeframe Signal Generation — NSE/BSE")

st.info(
    "🚧 Dashboard scaffold ready. "
    "Full UI implementation follows in the Dashboard build layer. "
    "DB and pipeline must be running for data to appear.",
    icon="ℹ️",
)

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "View",
    ["Today's Signals", "Performance", "Signal History", "Sector Exposure", "Pipeline Health"],
)

if page == "Today's Signals":
    st.header("Today's Signals")
    st.write("Live signals will appear here post-pipeline run.")

elif page == "Performance":
    st.header("Live Performance")
    st.write("Rolling Sharpe, drawdown curve, and win rate by strategy.")

elif page == "Signal History":
    st.header("Signal History")
    st.write("Success rate by stock, regime, and sector.")

elif page == "Sector Exposure":
    st.header("Sector Exposure")
    st.write("Current allocation vs 30% sector cap.")

elif page == "Pipeline Health":
    st.header("Pipeline Health")
    st.write("System metrics and alert thresholds from Section 24.")
