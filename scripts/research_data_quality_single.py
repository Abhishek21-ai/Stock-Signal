"""
Data Quality Diagnostic — Single Stock Deep History Check
Section: Pre-work for §25/§28 single-stock proof of concept.

Purpose:
  1. Fetch MAX available history for RELIANCE via yfinance
  2. Inspect for data quality issues: gaps, zero-volume days, suspicious
     price jumps (unadjusted splits/bonuses), missing OHLC
  3. Recommend a "clean data start date" — the point after which the
     series is reliable enough to use for calibration/meta-model training
  4. Save the full raw history to disk for the next step (backtest run)

Usage:
    python scripts/research_data_quality.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import date

SYMBOL = "RELIANCE"
OUTPUT_DIR = "data_research"


def fetch_max_history(symbol: str) -> pd.DataFrame:
    """Fetch the absolute maximum history yfinance has for this stock."""
    import yfinance as yf
    from app.data.ingestor import to_nse_ticker

    ticker = to_nse_ticker(symbol)
    print(f"Fetching MAX history for {symbol} ({ticker})...")

    stock = yf.Ticker(ticker)
    df = stock.history(period="max", interval="1d", auto_adjust=True)

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def detect_quality_issues(df: pd.DataFrame) -> dict:
    """
    Scan for common data quality red flags:
      - Daily gaps > 5 trading days (missing data)
      - Zero or near-zero volume days
      - Single-day price jumps > 15% (possible unadjusted corporate action)
      - Duplicate dates
    """
    issues = {
        "total_rows": len(df),
        "date_range": (df.index.min().date(), df.index.max().date()),
        "large_gaps": [],
        "zero_volume_days": 0,
        "suspicious_jumps": [],
        "duplicate_dates": df.index.duplicated().sum(),
    }

    # ── Gap detection ──────────────────────────────────────────
    date_diffs = df.index.to_series().diff().dt.days
    large_gaps = date_diffs[date_diffs > 7]   # > 1 week gap (holidays excluded loosely)
    for dt, gap in large_gaps.items():
        issues["large_gaps"].append((str(dt.date()), int(gap)))

    # ── Zero volume ─────────────────────────────────────────────
    issues["zero_volume_days"] = int((df["Volume"] == 0).sum())

    # ── Suspicious single-day jumps (possible bad split adjustment) ──
    daily_return = df["Close"].pct_change()
    jumps = daily_return[abs(daily_return) > 0.15]
    for dt, ret in jumps.items():
        issues["suspicious_jumps"].append((str(dt.date()), round(float(ret) * 100, 1)))

    return issues


def recommend_clean_start(df: pd.DataFrame, issues: dict) -> str:
    """
    Heuristic: find the last suspicious jump or major gap in the early
    history, recommend starting clean data a buffer period after it.
    If issues cluster in early years, recommend skipping to ~2010+,
    consistent with known NSE/yfinance data quality patterns.
    """
    all_flagged_dates = (
        [d for d, _ in issues["suspicious_jumps"]] +
        [d for d, _ in issues["large_gaps"]]
    )

    if not all_flagged_dates:
        return str(df.index.min().date())

    flagged_dates = sorted(pd.to_datetime(all_flagged_dates))
    cutoff_candidates = [d for d in flagged_dates if d.year < 2012]

    if cutoff_candidates:
        last_early_issue = max(cutoff_candidates)
        recommended = (last_early_issue + pd.Timedelta(days=180)).date()
        return str(recommended)

    return str(df.index.min().date())


def main():
    print(f"\n{'='*60}")
    print(f"  Data Quality Diagnostic — {SYMBOL}")
    print(f"{'='*60}\n")

    df = fetch_max_history(SYMBOL)
    print(f"✅ Fetched {len(df)} rows | {df.index.min().date()} → {df.index.max().date()}")
    print(f"   Total span: {(df.index.max() - df.index.min()).days / 365.25:.1f} years\n")

    issues = detect_quality_issues(df)

    print(f"── Data Quality Report ──────────────────────────────")
    print(f"  Total rows:        {issues['total_rows']}")
    print(f"  Date range:        {issues['date_range'][0]} → {issues['date_range'][1]}")
    print(f"  Duplicate dates:   {issues['duplicate_dates']}")
    print(f"  Zero-volume days:  {issues['zero_volume_days']}")
    print(f"  Large gaps (>7d):  {len(issues['large_gaps'])}")
    if issues["large_gaps"][:10]:
        for d, gap in issues["large_gaps"][:10]:
            print(f"      {d}: {gap}-day gap")
        if len(issues["large_gaps"]) > 10:
            print(f"      ... and {len(issues['large_gaps']) - 10} more")

    print(f"  Suspicious jumps (>15% single day): {len(issues['suspicious_jumps'])}")
    if issues["suspicious_jumps"]:
        for d, pct in issues["suspicious_jumps"][:15]:
            print(f"      {d}: {pct:+.1f}%")
        if len(issues["suspicious_jumps"]) > 15:
            print(f"      ... and {len(issues['suspicious_jumps']) - 15} more")

    recommended_start = recommend_clean_start(df, issues)
    print(f"\n── Recommendation ───────────────────────────────────")
    print(f"  Suggested clean-data start: {recommended_start}")
    clean_df = df[df.index >= recommended_start]
    print(f"  Rows after cutoff: {len(clean_df)} "
          f"({len(clean_df) / len(df) * 100:.0f}% of total history retained)")
    print(f"  Years of clean data: {(clean_df.index.max() - clean_df.index.min()).days / 365.25:.1f}")

    # ── Save to disk for next step ───────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_path   = os.path.join(OUTPUT_DIR, f"{SYMBOL}_full_history.csv")
    clean_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}_clean_history.csv")

    df.to_csv(raw_path)
    clean_df.to_csv(clean_path)

    print(f"\n  💾 Saved full history  → {raw_path}")
    print(f"  💾 Saved clean history → {clean_path}")

    print(f"\n{'='*60}")
    print(f"  ✅ Diagnostic complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()