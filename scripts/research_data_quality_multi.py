"""
Data Quality Diagnostic — Multi-Stock Deep History Check

Purpose:
  1. Fetch MAX available history for multiple NSE stocks via yfinance
  2. Inspect for data quality issues:
       - large gaps
       - zero-volume days
       - suspicious price jumps
       - duplicate dates
  3. Recommend a clean start date for each stock
  4. Save raw and clean history CSVs
  5. Produce a cross-stock comparison summary

Usage:
    python scripts/research_data_quality.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

OUTPUT_DIR = "data_research"

SYMBOLS = [
    "RELIANCE",
    "INFY",
    "TCS",
    "HDFCBANK",
    "SBIN",
    "ITC",
    "ONGC",
    "WIPRO",
    "YESBANK",
    "SUZLON"
]


def fetch_max_history(symbol: str) -> pd.DataFrame:
    """
    Fetch maximum available history from yfinance.
    """
    import yfinance as yf
    from app.data.ingestor import to_nse_ticker

    ticker = to_nse_ticker(symbol)

    print(f"\nFetching MAX history for {symbol} ({ticker})...")

    stock = yf.Ticker(ticker)
    df = stock.history(
        period="max",
        interval="1d",
        auto_adjust=True
    )

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    df.index = pd.to_datetime(df.index).tz_localize(None)

    return df


def detect_quality_issues(df: pd.DataFrame) -> dict:
    """
    Scan for common data quality red flags.
    """

    issues = {
        "total_rows": len(df),
        "date_range": (
            df.index.min().date(),
            df.index.max().date()
        ),
        "large_gaps": [],
        "zero_volume_days": 0,
        "suspicious_jumps": [],
        "duplicate_dates": int(df.index.duplicated().sum())
    }

    # ----------------------------------------------------
    # Gap Detection
    # ----------------------------------------------------
    date_diffs = df.index.to_series().diff().dt.days

    large_gaps = date_diffs[date_diffs > 7]

    for dt, gap in large_gaps.items():
        issues["large_gaps"].append(
            (str(dt.date()), int(gap))
        )

    # ----------------------------------------------------
    # Zero Volume
    # ----------------------------------------------------
    issues["zero_volume_days"] = int(
        (df["Volume"] == 0).sum()
    )

    # ----------------------------------------------------
    # Suspicious Jumps
    # Increased threshold from 15% to 30%
    # to reduce market-volatility noise.
    # ----------------------------------------------------
    daily_return = df["Close"].pct_change()

    jumps = daily_return[abs(daily_return) > 0.30]

    for dt, ret in jumps.items():
        issues["suspicious_jumps"].append(
            (
                str(dt.date()),
                round(float(ret) * 100, 1)
            )
        )

    return issues


def recommend_clean_start(
    df: pd.DataFrame,
    issues: dict
) -> str:
    """
    Heuristic recommendation.
    """

    all_flagged_dates = (
        [d for d, _ in issues["suspicious_jumps"]]
        +
        [d for d, _ in issues["large_gaps"]]
    )

    if not all_flagged_dates:
        return str(df.index.min().date())

    flagged_dates = sorted(
        pd.to_datetime(all_flagged_dates)
    )

    cutoff_candidates = [
        d for d in flagged_dates
        if d.year < 2012
    ]

    if cutoff_candidates:
        last_early_issue = max(cutoff_candidates)

        recommended = (
            last_early_issue +
            pd.Timedelta(days=180)
        ).date()

        return str(recommended)

    return str(df.index.min().date())


def process_stock(symbol: str) -> dict:

    print(f"\n{'=' * 70}")
    print(f"Data Quality Diagnostic - {symbol}")
    print(f"{'=' * 70}")

    df = fetch_max_history(symbol)

    print(
        f"Fetched {len(df)} rows | "
        f"{df.index.min().date()} -> "
        f"{df.index.max().date()}"
    )

    years = (
        df.index.max() - df.index.min()
    ).days / 365.25

    print(f"History Span: {years:.1f} years")

    issues = detect_quality_issues(df)

    print("\nData Quality Report")
    print("-" * 40)

    print(f"Duplicate dates: {issues['duplicate_dates']}")
    print(f"Zero volume days: {issues['zero_volume_days']}")
    print(f"Large gaps: {len(issues['large_gaps'])}")
    print(
        f"Suspicious jumps: "
        f"{len(issues['suspicious_jumps'])}"
    )

    recommended_start = recommend_clean_start(
        df,
        issues
    )

    clean_df = df[
        df.index >= recommended_start
    ]

    clean_years = (
        clean_df.index.max() -
        clean_df.index.min()
    ).days / 365.25

    retention_pct = (
        len(clean_df) /
        len(df)
    ) * 100

    print("\nRecommendation")
    print("-" * 40)
    print(f"Suggested clean start: {recommended_start}")
    print(f"Rows retained: {retention_pct:.1f}%")
    print(f"Clean years: {clean_years:.1f}")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    full_path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_full_history.csv"
    )

    clean_path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_clean_history.csv"
    )

    df.to_csv(full_path)
    clean_df.to_csv(clean_path)

    print(f"\nSaved: {full_path}")
    print(f"Saved: {clean_path}")

    return {
        "symbol": symbol,
        "first_date": df.index.min().date(),
        "last_date": df.index.max().date(),
        "total_rows": len(df),
        "years_history": round(years, 1),
        "duplicate_dates": issues["duplicate_dates"],
        "zero_volume_days": issues["zero_volume_days"],
        "large_gaps": len(issues["large_gaps"]),
        "suspicious_jumps": len(
            issues["suspicious_jumps"]
        ),
        "recommended_start": recommended_start,
        "clean_rows": len(clean_df),
        "retention_pct": round(
            retention_pct,
            2
        ),
        "clean_years": round(
            clean_years,
            1
        )
    }


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    summary_rows = []

    print("\n")
    print("=" * 70)
    print("MULTI-STOCK DATA QUALITY DIAGNOSTIC")
    print("=" * 70)

    for symbol in SYMBOLS:

        try:
            result = process_stock(symbol)
            summary_rows.append(result)

        except Exception as e:

            print(
                f"\nFAILED: {symbol}"
            )

            print(
                f"Reason: {str(e)}"
            )

            summary_rows.append({
                "symbol": symbol,
                "error": str(e)
            })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "multi_stock_quality_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(summary_df)

    print(
        f"\nSaved summary -> {summary_path}"
    )

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()