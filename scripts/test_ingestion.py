"""
Standalone ingestion test — run this directly to verify data layer.

Usage (from project root):
    pip install -r requirements.txt
    python scripts/test_ingestion.py

Or inside docker:
    docker compose exec app python scripts/test_ingestion.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from app.data.ingestor import DataIngestor, fetch_nifty50, to_nse_ticker
from app.data.validator import get_data_coverage_report
from config.settings import settings

# Test with small subset first
TEST_STOCKS = ["RELIANCE", "TCS", "HDFCBANK"]

def main():
    print(f"\n{'='*55}")
    print("  Stock Signal Platform — Data Ingestion Test")
    print(f"{'='*55}\n")

    # 1. Test yfinance fetch
    print("1. Testing yfinance fetch (RELIANCE)...")
    import yfinance as yf
    df = yf.Ticker("RELIANCE.NS").history(period="5d", auto_adjust=True)
    if df.empty:
        print("   ❌ yfinance fetch failed — check internet connection")
        sys.exit(1)
    print(f"   ✅ yfinance working | last close: ₹{df['Close'].iloc[-1]:.2f}")

    # 2. Test Nifty50 fetch
    print("\n2. Fetching Nifty50 index...")
    nifty = fetch_nifty50(period="1mo")
    print(f"   ✅ Nifty50 | rows={len(nifty)} | last={nifty.index[-1].date()} | close={nifty['Close'].iloc[-1]:.2f}")

    # 3. Test DB connection
    print("\n3. Testing DB connection...")
    try:
        from app.data.validator import get_data_coverage_report
        report = get_data_coverage_report(date.today())
        print(f"   ✅ DB connected | existing stocks in market_data: {len(report)}")
    except Exception as e:
        print(f"   ❌ DB connection failed: {e}")
        print("   → Is postgres container running? Run: docker compose up -d postgres")
        sys.exit(1)

    # 4. Run ingestion for test stocks
    print(f"\n4. Running ingestion for {TEST_STOCKS}...")
    ingestor = DataIngestor(stocks=TEST_STOCKS, run_date=date.today())
    data = ingestor.run()

    print(f"\n   Results:")
    for symbol, df in data.items():
        print(f"   ✅ {symbol}: {len(df)} rows | "
              f"from {df.index[0].date()} to {df.index[-1].date()} | "
              f"last close: ₹{df['Close'].iloc[-1]:.2f}")

    if ingestor.sla_breaches:
        print(f"\n   ⚠️  SLA breaches: {ingestor.sla_breaches}")
    if ingestor.skipped:
        print(f"   ⚠️  Skipped: {ingestor.skipped}")

    # 5. Verify DB coverage
    print("\n5. DB coverage after ingestion:")
    report = get_data_coverage_report(date.today())
    for symbol, info in report.items():
        status = "✅" if info["is_current"] else "⚠️ "
        print(f"   {status} {symbol}: {info['total_rows']} rows | "
              f"{info['first_date']} → {info['last_date']}")

    print(f"\n{'='*55}")
    print("  ✅ Data ingestion layer verified successfully")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
