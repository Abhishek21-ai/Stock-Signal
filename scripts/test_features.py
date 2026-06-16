"""
Feature Engineering test — run after test_ingestion.py passes.

Usage:
    python scripts/test_features.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from app.data.validator import get_latest_ohlcv
from app.features.engineer import build_dataframe, compute_features, extract_latest_features

TEST_STOCKS = ["RELIANCE", "TCS", "HDFCBANK"]

def main():
    print(f"\n{'='*55}")
    print("  Stock Signal Platform — Feature Engineering Test")
    print(f"{'='*55}\n")

    for symbol in TEST_STOCKS:
        print(f"Testing {symbol}...")

        rows = get_latest_ohlcv(symbol, n=300)
        print(f"  Rows from DB: {len(rows)}")

        df = build_dataframe(rows)
        df = compute_features(df)
        f  = extract_latest_features(df, symbol)

        print(f"  ✅ Features computed ({len(f)} fields)")
        print(f"     Close:        ₹{f['close']}")
        print(f"     RSI(14):      {f['rsi_14']:.1f}  [{f['rsi_zone']}]")
        print(f"     MACD hist:    {f['macd_hist']:.2f}  cross={f['macd_cross']}")
        print(f"     ADX(14):      {f['adx_14']:.1f}  [{f['adx_trend']}]")
        print(f"     EMA align:    {f['ema_bull_alignment']}/3")
        print(f"     BB %B:        {f['bb_pct_b']:.2f}")
        print(f"     Vol ratio:    {f['volume_ratio']:.2f}x")
        print(f"     ATR(14):      ₹{f['atr_14']:.2f}  ({f['atr_pct']:.2f}%)")
        print(f"     Price/VWAP:   {f['price_vs_vwap']:+.2%}")
        print(f"     52w high:     {f['pct_from_52w_high']:.1f}%")
        print(f"     Return 20d:   {f['return_20d']:+.2%}")
        print(f"     ATR stop 1x:  ₹{f['atr_stop_1x']}")
        print(f"     ATR target 2x:₹{f['atr_target_2x']}")
        print()

    print(f"{'='*55}")
    print("  ✅ Feature engineering layer verified")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
