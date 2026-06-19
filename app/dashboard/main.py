import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from datetime import datetime

# ============================================================
# DATABASE CONFIG
# ============================================================

DB_USER = os.getenv("POSTGRES_USER", "ssp_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "zudamR490227")

# IMPORTANT:
# If Streamlit runs inside Docker use postgres
# If Streamlit runs on host machine use localhost
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")

DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "stock_signals")

DB_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DB_URL)

# ============================================================
# DATABASE DIAGNOSTICS
# ============================================================

st.sidebar.subheader("Database Diagnostics")

st.sidebar.write(f"Host: {DB_HOST}")
st.sidebar.write(f"Database: {DB_NAME}")
st.sidebar.write(f"User: {DB_USER}")

try:
    with engine.connect() as conn:

        daily_count = conn.execute(
            text("SELECT COUNT(*) FROM daily_signals")
        ).scalar()

        history_count = conn.execute(
            text("SELECT COUNT(*) FROM signal_history")
        ).scalar()

        backtest_count = conn.execute(
            text("SELECT COUNT(*) FROM backtest_results")
        ).scalar()

        market_count = conn.execute(
            text("SELECT COUNT(*) FROM market_data")
        ).scalar()

    st.sidebar.success("Database Connected")

    st.sidebar.write(
        f"daily_signals: {daily_count}"
    )

    st.sidebar.write(
        f"signal_history: {history_count}"
    )

    st.sidebar.write(
        f"backtest_results: {backtest_count}"
    )

    st.sidebar.write(
        f"market_data: {market_count}"
    )

except Exception as e:
    st.sidebar.error("Database Connection Failed")
    st.sidebar.exception(e)

# ============================================================
# HELPERS
# ============================================================

@st.cache_data(ttl=60)
def load_daily_signals():
    query = """
    SELECT *
    FROM daily_signals
    ORDER BY date DESC
    """

    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())


@st.cache_data(ttl=60)
def load_signal_history():
    query = """
    SELECT *
    FROM signal_history
    ORDER BY signal_id DESC
    """

    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())

@st.cache_data(ttl=60)
def load_backtests():
    query = """
    SELECT *
    FROM backtest_results
    """

    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())


@st.cache_data(ttl=60)
def load_market_data():
    query = """
    SELECT *
    FROM market_data
    ORDER BY date DESC
    LIMIT 500
    """

    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())
# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select View",
    [
        "Today's Signals",
        "Performance",
        "Signal History",
        "Backtests",
        "Pipeline Health"
    ]
)

auto_refresh = st.sidebar.checkbox(
    "Auto Refresh",
    value=True
)

if auto_refresh:
    st.sidebar.success("Refresh every 60 sec")

# ============================================================
# LOAD DATA
# ============================================================

try:
    signals_df = load_daily_signals()
except Exception as e:
    st.error(f"daily_signals error: {e}")
    signals_df = pd.DataFrame()

try:
    history_df = load_signal_history()
except Exception as e:
    st.error(f"history_signals error: {e}")
    history_df = pd.DataFrame()

try:
    backtest_df = load_backtests()
except Exception as e:
    st.error(f"backtest_signals error: {e}")
    backtest_df = pd.DataFrame()

try:
    market_df = load_market_data()
except Exception as e:
    st.error(f"market data  error: {e}")
    market_df = pd.DataFrame()

# ============================================================
# PAGE 1 - TODAY SIGNALS
# ============================================================

if page == "Today's Signals":

    st.header("📊 Today's Signals")

    if signals_df.empty:
        st.warning("No signals found.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    buy_count = len(
        signals_df[
            signals_df["signal"].isin(
                ["BUY", "STRONG_BUY"]
            )
        ]
    )

    sell_count = len(
        signals_df[
            signals_df["signal"].isin(
                ["SELL", "STRONG_SELL"]
            )
        ]
    )

    hold_count = len(
        signals_df[
            signals_df["signal"] == "HOLD"
        ]
    )

    avg_conf = round(
        signals_df["confidence_pct"].mean(),
        2
    )

    col1.metric("Buy Signals", buy_count)
    col2.metric("Sell Signals", sell_count)
    col3.metric("Hold Signals", hold_count)
    avg_conf = round(
    signals_df["confidence_pct"].mean(),
    1
)

    avg_prob = round(
        signals_df["calibrated_probability"].mean() * 100,
        1
    )

    col4.metric(
        "Avg Confidence",
        f"{avg_conf}%"
    )

    st.metric(
        "Avg Calibrated Probability",
        f"{avg_prob}%"
    )

    st.divider()

    stocks = sorted(
        signals_df["stock"].unique().tolist()
    )

    selected_stock = st.selectbox(
        "Filter Stock",
        ["ALL"] + stocks
    )

    filtered = signals_df.copy()

    if selected_stock != "ALL":
        filtered = filtered[
            filtered["stock"] == selected_stock
        ]

    st.dataframe(
        filtered,
        use_container_width=True
    )

    csv = filtered.to_csv(index=False)

    st.download_button(
        "⬇ Download Signals",
        csv,
        file_name="signals.csv",
        mime="text/csv"
    )

# ============================================================
# PAGE 2 - PERFORMANCE
# ============================================================

elif page == "Performance":

    st.header("📈 Performance")

    if history_df.empty:
        st.warning("No signal history available.")
        st.stop()

    if "return_pct" not in history_df.columns:
        st.warning(
            "return_pct column missing."
        )
        st.stop()

    returns = history_df["return_pct"]

    total_trades = len(history_df)

    wins = len(
        history_df[
            history_df["return_pct"] > 0
        ]
    )

    win_rate = (
        wins / total_trades * 100
        if total_trades > 0
        else 0
    )

    avg_return = round(
        returns.mean(),
        2
    )

    max_return = round(
        returns.max(),
        2
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Trades",
        total_trades
    )

    col2.metric(
        "Win Rate",
        f"{win_rate:.2f}%"
    )

    col3.metric(
        "Avg Return",
        f"{avg_return:.2f}%"
    )

    col4.metric(
        "Best Trade",
        f"{max_return:.2f}%"
    )

    st.divider()

    st.subheader("Return Distribution")

    st.bar_chart(
        history_df["return_pct"]
    )

# ============================================================
# PAGE 3 - SIGNAL HISTORY
# ============================================================

elif page == "Signal History":

    st.header("📚 Signal History")

    if history_df.empty:
        st.warning("No history available.")
        st.stop()

    st.dataframe(
        history_df,
        use_container_width=True
    )

    if (
        "regime_at_signal"
        in history_df.columns
    ):

        st.subheader(
            "Trades by Regime"
        )

        regime_counts = (
            history_df[
                "regime_at_signal"
            ]
            .value_counts()
        )

        st.bar_chart(
            regime_counts
        )

# ============================================================
# PAGE 4 - BACKTESTS
# ============================================================

elif page == "Backtests":

    st.header("🧪 Backtest Results")

    if backtest_df.empty:
        st.warning(
            "No backtest results found."
        )
        st.stop()

    st.dataframe(
        backtest_df,
        use_container_width=True
    )

    numeric_cols = [
        c
        for c in backtest_df.columns
        if pd.api.types.is_numeric_dtype(
            backtest_df[c]
        )
    ]

    if numeric_cols:

        metric = st.selectbox(
            "Metric",
            numeric_cols
        )

        st.bar_chart(
            backtest_df[metric]
        )

# ============================================================
# PAGE 5 - PIPELINE HEALTH
# ============================================================

elif page == "Pipeline Health":

    st.header("⚙ Pipeline Health")

    col1, col2 = st.columns(2)

    col1.metric(
        "Last Refresh",
        datetime.now().strftime(
            "%H:%M:%S"
        )
    )

    col2.metric(
        "Database",
        "Connected"
    )

    st.divider()

    if market_df.empty:
        st.warning(
            "Market data unavailable."
        )
    else:
        st.success(
            f"{len(market_df)} market records loaded."
        )

    st.subheader("Dataset Summary")

    summary = pd.DataFrame(
        {
            "Table": [
                "daily_signals",
                "signal_history",
                "backtest_results",
                "market_data"
            ],
            "Rows": [
                len(signals_df),
                len(history_df),
                len(backtest_df),
                len(market_df)
            ]
        }
    )

    st.dataframe(
        summary,
        use_container_width=True
    )