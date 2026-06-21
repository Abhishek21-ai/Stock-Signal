import os
import pandas as pd
import numpy as np
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

st.set_page_config(page_title="Stock Signal Platform", layout="wide")

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

    st.sidebar.write(f"daily_signals: {daily_count}")
    st.sidebar.write(f"signal_history: {history_count}")
    st.sidebar.write(f"backtest_results: {backtest_count}")
    st.sidebar.write(f"market_data: {market_count}")

except Exception as e:
    st.sidebar.error("Database Connection Failed")
    st.sidebar.exception(e)

# ============================================================
# HELPERS — DATA LOADERS
# ============================================================

@st.cache_data(ttl=60)
def load_daily_signals():
    query = "SELECT * FROM daily_signals ORDER BY date DESC"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())


@st.cache_data(ttl=60)
def load_signal_history():
    query = "SELECT * FROM signal_history ORDER BY signal_id DESC"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())


@st.cache_data(ttl=60)
def load_backtests():
    query = "SELECT * FROM backtest_results"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())


@st.cache_data(ttl=60)
def load_market_data():
    query = "SELECT * FROM market_data ORDER BY date DESC LIMIT 500"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())


@st.cache_data(ttl=60)
def load_trades():
    query = "SELECT * FROM trades ORDER BY signal_date DESC"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())


@st.cache_data(ttl=60)
def load_health_alerts():
    query = "SELECT * FROM health_alerts ORDER BY created_at DESC LIMIT 200"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())


@st.cache_data(ttl=60)
def load_pipeline_runs():
    query = "SELECT * FROM pipeline_runs ORDER BY run_date DESC LIMIT 30"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())


@st.cache_data(ttl=60)
def load_regime_snapshots():
    query = "SELECT * FROM regime_snapshots ORDER BY date DESC LIMIT 90"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return pd.DataFrame(result.fetchall(), columns=result.keys())


# Sector mapping — used for Sector Exposure panel
try:
    from config.watchlist import WATCHLIST_WITH_SECTORS
except Exception:
    WATCHLIST_WITH_SECTORS = {}

SECTOR_CAP_PCT = 0.30   # Section 21.2

# ============================================================
# SIDEBAR — NAVIGATION
# ============================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select View",
    [
        "Today's Signals",
        "Live Performance",          # NEW — Section 24.2
        "Signal History",
        "Sector Exposure",           # NEW — Section 24.2
        "Open Positions",            # NEW — Section 22
        "Backtests",
        "Health & Alerts",           # NEW — Section 24.1
        "Pipeline Health",
    ]
)

auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
if auto_refresh:
    st.sidebar.success("Refresh every 60 sec")

# ============================================================
# LOAD DATA
# ============================================================

def safe_load(loader, label):
    try:
        return loader()
    except Exception as e:
        st.error(f"{label} error: {e}")
        return pd.DataFrame()

signals_df   = safe_load(load_daily_signals,    "daily_signals")
history_df   = safe_load(load_signal_history,   "signal_history")
backtest_df  = safe_load(load_backtests,        "backtest_results")
market_df    = safe_load(load_market_data,      "market_data")
trades_df    = safe_load(load_trades,           "trades")
alerts_df    = safe_load(load_health_alerts,    "health_alerts")
runs_df      = safe_load(load_pipeline_runs,    "pipeline_runs")
regime_df    = safe_load(load_regime_snapshots, "regime_snapshots")

# ============================================================
# PAGE: TODAY'S SIGNALS
# ============================================================

if page == "Today's Signals":

    st.header("📊 Today's Signals")

    if signals_df.empty:
        st.warning("No signals found.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    buy_count  = len(signals_df[signals_df["signal"].isin(["BUY", "STRONG_BUY"])])
    sell_count = len(signals_df[signals_df["signal"].isin(["SELL", "STRONG_SELL"])])
    hold_count = len(signals_df[signals_df["signal"] == "HOLD"])

    col1.metric("Buy Signals", buy_count)
    col2.metric("Sell Signals", sell_count)
    col3.metric("Hold Signals", hold_count)

    avg_conf = round(signals_df["confidence_pct"].mean(), 1)
    col4.metric("Avg Confidence", f"{avg_conf}%")

    if "calibrated_probability" in signals_df.columns and signals_df["calibrated_probability"].notna().any():
        avg_prob = round(signals_df["calibrated_probability"].mean() * 100, 1)
        st.metric("Avg Calibrated Probability", f"{avg_prob}%")
    else:
        st.caption("Confidence (uncalibrated) — Section 25 calibration not yet active "
                   "(requires 3+ months / 100+ resolved signals)")

    st.divider()

    stocks = sorted(signals_df["stock"].unique().tolist())
    selected_stock = st.selectbox("Filter Stock", ["ALL"] + stocks)

    filtered = signals_df.copy()
    if selected_stock != "ALL":
        filtered = filtered[filtered["stock"] == selected_stock]

# TRIAL BLOCK starts here
    display_cols = [
        "date",
        "stock",
        "signal",
        "confidence_pct",
        "entry_price_theoretical",
        "exit_target_theoretical",
        "stop_loss_theoretical",
        "regime",
        "valid_until",
        "created_at",
    ]

    filtered = filtered[display_cols]

# TRIAL BLOCK ends here

    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False)
    st.download_button("⬇ Download Signals", csv, file_name="signals.csv", mime="text/csv")

# ============================================================
# PAGE: LIVE PERFORMANCE  [NEW — Section 24.2]
# Rolling Sharpe, drawdown curve, win rate by strategy,
# portfolio P&L over time.
# ============================================================

elif page == "Live Performance":

    st.header("📈 Live Performance")
    st.caption("Section 24.2 — Rolling Sharpe, drawdown, win rate by strategy, portfolio P&L")

    closed = trades_df[trades_df["status"] == "CLOSED"].copy() if not trades_df.empty else pd.DataFrame()

    if closed.empty:
        st.warning("No closed trades yet — performance metrics will populate as trades complete.")
        st.stop()

    closed["exit_date"] = pd.to_datetime(closed["exit_date"])
    closed = closed.sort_values("exit_date")

    # ── Top-line metrics ────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    total_trades = len(closed)
    wins = len(closed[closed["pnl_pct"] > 0])
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_pnl = closed["pnl_inr"].sum()
    avg_pnl_pct = closed["pnl_pct"].mean()

    col1.metric("Closed Trades", total_trades)
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("Total P&L", f"₹{total_pnl:,.0f}")
    col4.metric("Avg Return/Trade", f"{avg_pnl_pct:.2f}%")

    st.divider()

    # ── Portfolio P&L over time (cumulative) ────────────────
    st.subheader("Portfolio P&L Over Time")
    closed["cumulative_pnl"] = closed["pnl_inr"].cumsum()
    pnl_series = closed.set_index("exit_date")["cumulative_pnl"]
    st.line_chart(pnl_series)

    # ── Drawdown curve ───────────────────────────────────────
    st.subheader("Drawdown Curve")
    running_max = closed["cumulative_pnl"].cummax()
    drawdown = closed["cumulative_pnl"] - running_max
    dd_series = pd.Series(drawdown.values, index=closed["exit_date"])
    st.area_chart(dd_series)

    max_dd = drawdown.min()
    st.caption(f"Max drawdown: ₹{max_dd:,.0f}")

    # ── Rolling Sharpe (20-trade window) ────────────────────
    st.subheader("Rolling Sharpe Ratio (20-trade window)")
    returns = closed["pnl_pct"] / 100
    window = min(20, max(2, len(returns) // 2))
    rolling_mean = returns.rolling(window).mean()
    rolling_std  = returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    sharpe_series = pd.Series(rolling_sharpe.values, index=closed["exit_date"])
    st.line_chart(sharpe_series.dropna())

    overall_sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    st.caption(f"Overall annualized Sharpe: {overall_sharpe:.2f}")

    st.divider()

    # ── Win rate by strategy ─────────────────────────────────
    st.subheader("Win Rate by Strategy")
    st.caption("Approximation: trades where each strategy's score was positive at signal time")

    if not signals_df.empty:
        merged = closed.merge(
            signals_df[["stock", "date", "trend_score", "momentum_score",
                       "reversion_score", "breakout_score", "volume_score"]],
            left_on=["stock", "signal_date"], right_on=["stock", "date"],
            how="left",
        )
        strat_cols = ["trend_score", "momentum_score", "reversion_score",
                      "breakout_score", "volume_score"]
        strat_stats = []
        for col in strat_cols:
            strat_name = col.replace("_score", "")
            contributed = merged[merged[col] > 0]
            if len(contributed) > 0:
                won = len(contributed[contributed["pnl_pct"] > 0])
                strat_stats.append({
                    "strategy": strat_name,
                    "trades": len(contributed),
                    "win_rate_pct": round(won / len(contributed) * 100, 1),
                })
        if strat_stats:
            strat_df = pd.DataFrame(strat_stats).set_index("strategy")
            st.bar_chart(strat_df["win_rate_pct"])
            st.dataframe(strat_df, use_container_width=True)
        else:
            st.info("Not enough strategy-attributed trades yet.")
    else:
        st.info("No signal data to cross-reference strategy contribution.")

# ============================================================
# PAGE: SIGNAL HISTORY (extended — Section 24.2)
# success rate by stock/regime/sector; expired vs acted-on ratio
# ============================================================

elif page == "Signal History":

    st.header("📚 Signal History")

    if history_df.empty:
        st.warning("No history available.")
        st.stop()

    st.dataframe(history_df, use_container_width=True)

    if "regime_at_signal" in history_df.columns:
        st.subheader("Trades by Regime")
        regime_counts = history_df["regime_at_signal"].value_counts()
        st.bar_chart(regime_counts)

    st.divider()

    # ── Success rate by stock ────────────────────────────────
    if not trades_df.empty:
        closed = trades_df[trades_df["status"] == "CLOSED"].copy()
        if not closed.empty:
            st.subheader("Success Rate by Stock")
            by_stock = closed.groupby("stock").agg(
                trades=("pnl_pct", "count"),
                win_rate=("pnl_pct", lambda x: round((x > 0).mean() * 100, 1)),
            ).sort_values("win_rate", ascending=False)
            st.dataframe(by_stock, use_container_width=True)
            st.bar_chart(by_stock["win_rate"])

            # ── Success rate by regime ───────────────────────
            if "regime_at_entry" in closed.columns:
                st.subheader("Success Rate by Regime")
                by_regime = closed.groupby("regime_at_entry").agg(
                    trades=("pnl_pct", "count"),
                    win_rate=("pnl_pct", lambda x: round((x > 0).mean() * 100, 1)),
                )
                st.dataframe(by_regime, use_container_width=True)
                st.bar_chart(by_regime["win_rate"])

            # ── Success rate by sector ────────────────────────
            if WATCHLIST_WITH_SECTORS:
                st.subheader("Success Rate by Sector")
                closed["sector"] = closed["stock"].map(WATCHLIST_WITH_SECTORS).fillna("Unknown")
                by_sector = closed.groupby("sector").agg(
                    trades=("pnl_pct", "count"),
                    win_rate=("pnl_pct", lambda x: round((x > 0).mean() * 100, 1)),
                ).sort_values("win_rate", ascending=False)
                st.dataframe(by_sector, use_container_width=True)
                st.bar_chart(by_sector["win_rate"])

            # ── Expired vs acted-on ratio ─────────────────────
            st.subheader("Expired vs Acted-On Signals")
            exit_counts = trades_df["exit_reason"].value_counts(dropna=False)
            total = len(trades_df)
            expired = exit_counts.get("EXPIRY", 0)
            acted_on = total - expired
            colA, colB, colC = st.columns(3)
            colA.metric("Total Trade Signals", total)
            colB.metric("Acted On (Activated)", acted_on)
            colC.metric("Expired (Never Filled)", expired)
            if total > 0:
                st.progress(acted_on / total, text=f"{acted_on/total:.0%} of signals were activated")
        else:
            st.info("No closed trades yet for success-rate breakdowns.")
    else:
        st.info("No trades table data available.")

# ============================================================
# PAGE: SECTOR EXPOSURE  [NEW — Section 24.2]
# current sector allocation vs 30% cap; heat map of concentration
# ============================================================

elif page == "Sector Exposure":

    st.header("🏭 Sector Exposure")
    st.caption(f"Section 21.2 — Max sector cap: {SECTOR_CAP_PCT:.0%} of portfolio")

    if trades_df.empty or not WATCHLIST_WITH_SECTORS:
        st.warning("No trade data or sector mapping available.")
        st.stop()

    open_trades = trades_df[trades_df["status"].isin(["PENDING", "ACTIVE"])].copy()

    if open_trades.empty:
        st.info("No open positions currently — sector exposure is 0% across all sectors.")
        st.stop()

    open_trades["sector"] = open_trades["stock"].map(WATCHLIST_WITH_SECTORS).fillna("Unknown")

    try:
        from config.settings import settings
        portfolio_value = settings.portfolio_value_inr
    except Exception:
        portfolio_value = 1_000_000.0

    sector_exposure = open_trades.groupby("sector")["position_value_inr"].sum().fillna(0)
    sector_pct = (sector_exposure / portfolio_value * 100).round(2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Allocation")
        alloc_df = pd.DataFrame({
            "Sector":          sector_pct.index,
            "Exposure (%)":    sector_pct.values,
            "Cap (%)":         SECTOR_CAP_PCT * 100,
            "Headroom (%)":    (SECTOR_CAP_PCT * 100 - sector_pct.values).round(2),
        }).sort_values("Exposure (%)", ascending=False)
        st.dataframe(alloc_df, use_container_width=True)

    with col2:
        st.subheader("Allocation vs Cap")
        st.bar_chart(sector_pct)

    # ── Heat map style table — position concentration ───────
    st.subheader("Position Concentration Heat Map")
    pivot = open_trades.pivot_table(
        index="sector", columns="stock", values="position_value_inr",
        aggfunc="sum", fill_value=0,
    )
    st.dataframe(
        pivot.style.background_gradient(cmap="Reds", axis=None),
        use_container_width=True,
    )

    # ── Cap breach warnings ───────────────────────────────────
    breaches = alloc_df[alloc_df["Exposure (%)"] > SECTOR_CAP_PCT * 100]
    if not breaches.empty:
        st.error(f"⚠️ {len(breaches)} sector(s) exceeding the {SECTOR_CAP_PCT:.0%} cap:")
        st.dataframe(breaches, use_container_width=True)
    else:
        st.success("All sectors within allocation cap.")

# ============================================================
# PAGE: OPEN POSITIONS  [NEW — Section 22 Trade Lifecycle]
# ============================================================

elif page == "Open Positions":

    st.header("💼 Open Positions")

    if trades_df.empty:
        st.warning("No trades found.")
        st.stop()

    pending = trades_df[trades_df["status"] == "PENDING"]
    active  = trades_df[trades_df["status"] == "ACTIVE"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Pending (awaiting entry)", len(pending))
    col2.metric("Active (filled)", len(active))
    col3.metric("Total Open", len(pending) + len(active))

    st.divider()

    if not active.empty:
        st.subheader("Active Positions")
        display_cols = [c for c in [
            "stock", "entry_price_actual", "stop_loss_realistic",
            "target_price", "position_size_shares", "position_value_inr",
            "regime_at_entry", "entry_date",
        ] if c in active.columns]
        st.dataframe(active[display_cols], use_container_width=True)

    if not pending.empty:
        st.subheader("Pending Entries")
        display_cols = [c for c in [
            "stock", "entry_price_realistic", "stop_loss_realistic",
            "target_price", "position_size_shares", "signal_date",
        ] if c in pending.columns]
        st.dataframe(pending[display_cols], use_container_width=True)

    if active.empty and pending.empty:
        st.info("No open positions currently.")

# ============================================================
# PAGE: BACKTESTS
# ============================================================

elif page == "Backtests":

    st.header("🧪 Backtest Results")

    if backtest_df.empty:
        st.warning("No backtest results found.")
        st.stop()

    st.dataframe(backtest_df, use_container_width=True)

    numeric_cols = [c for c in backtest_df.columns if pd.api.types.is_numeric_dtype(backtest_df[c])]

    if numeric_cols:
        metric = st.selectbox("Metric", numeric_cols)
        st.bar_chart(backtest_df[metric])

# ============================================================
# PAGE: HEALTH & ALERTS  [NEW — Section 24.1]
# ============================================================

elif page == "Health & Alerts":

    st.header("🚨 Health & Alerts")
    st.caption("Section 24.1 — Ingestion SLA, LLM failure rate, pipeline latency, strategy win rate")

    if alerts_df.empty:
        st.success("✅ No health alerts recorded — all systems healthy.")
    else:
        critical = alerts_df[alerts_df["severity"] == "CRITICAL"]
        warning  = alerts_df[alerts_df["severity"] == "WARNING"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Alerts (recent)", len(alerts_df))
        col2.metric("Critical", len(critical))
        col3.metric("Warning", len(warning))

        st.divider()

        if not critical.empty:
            st.subheader("🔴 Critical Alerts")
            st.dataframe(
                critical[["run_date", "metric", "message", "action", "created_at"]],
                use_container_width=True,
            )

        if not warning.empty:
            st.subheader("🟡 Warnings")
            st.dataframe(
                warning[["run_date", "metric", "message", "action", "created_at"]],
                use_container_width=True,
            )

        st.divider()
        st.subheader("Alerts by Metric")
        metric_counts = alerts_df["metric"].value_counts()
        st.bar_chart(metric_counts)

# ============================================================
# PAGE: PIPELINE HEALTH
# ============================================================

elif page == "Pipeline Health":

    st.header("⚙ Pipeline Health")

    col1, col2 = st.columns(2)
    col1.metric("Last Refresh", datetime.now().strftime("%H:%M:%S"))
    col2.metric("Database", "Connected")

    st.divider()

    if market_df.empty:
        st.warning("Market data unavailable.")
    else:
        st.success(f"{len(market_df)} market records loaded.")

    st.subheader("Dataset Summary")

    summary = pd.DataFrame({
        "Table": ["daily_signals", "signal_history", "backtest_results",
                  "market_data", "trades", "health_alerts"],
        "Rows": [len(signals_df), len(history_df), len(backtest_df),
                 len(market_df), len(trades_df), len(alerts_df)],
    })
    st.dataframe(summary, use_container_width=True)

    st.divider()

    # ── Recent pipeline runs ─────────────────────────────────
    if not runs_df.empty:
        st.subheader("Recent Pipeline Runs")
        display_cols = [c for c in [
            "run_date", "status", "total_ms", "stocks_processed",
            "stocks_skipped", "signals_generated", "llm_overrides",
        ] if c in runs_df.columns]
        st.dataframe(runs_df[display_cols], use_container_width=True)

        if "total_ms" in runs_df.columns:
            st.subheader("Pipeline Latency Trend (ms)")
            latency_series = runs_df.set_index("run_date")["total_ms"].sort_index()
            st.line_chart(latency_series)

    # ── Regime history ────────────────────────────────────────
    if not regime_df.empty:
        st.subheader("Regime History (last 90 days)")
        st.dataframe(
            regime_df[["date", "regime", "nifty_close", "nifty_adx", "regime_confidence"]],
            use_container_width=True,
        )
        regime_counts = regime_df["regime"].value_counts()
        st.bar_chart(regime_counts)