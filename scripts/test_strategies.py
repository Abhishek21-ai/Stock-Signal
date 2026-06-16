"""
Strategy engine test — runs all 5 strategies on live DB data.

Usage:
    python scripts/test_strategies.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.validator import get_latest_ohlcv
from app.features.engineer import build_dataframe, compute_features, extract_latest_features
from app.strategies.runner import StrategyRunner

TEST_STOCKS = ["RELIANCE", "TCS", "HDFCBANK"]
REGIME = "UNCERTAIN"   # will be replaced by real regime detector

def main():
    print(f"\n{'='*65}")
    print("  Stock Signal Platform — Strategy Engine Test")
    print(f"{'='*65}\n")

    runner = StrategyRunner()

    for symbol in TEST_STOCKS:
        rows     = get_latest_ohlcv(symbol, n=300)
        df       = build_dataframe(rows)
        df       = compute_features(df)
        features = extract_latest_features(df, symbol)

        results  = runner.run(features, regime=REGIME)

        print(f"{'─'*65}")
        print(f"  {symbol}  |  Close: ₹{features['close']}  |  Regime: {REGIME}")
        print(f"{'─'*65}")

        for r in results:
            bar = "█" * int(abs(r.score) / 5) + "░" * (20 - int(abs(r.score) / 5))
            direction = "▲" if r.score > 0 else "▼" if r.score < 0 else "─"
            print(f"  {r.strategy_id:<12} {direction} {r.signal:<12} "
                  f"score={r.score:+6.1f}  [{bar}]")
            for reason in r.reasons[:2]:   # show top 2 reasons
                print(f"               → {reason}")

        # Net consensus
        avg_score = sum(r.score for r in results) / len(results)
        from app.strategies.base import score_to_signal
        consensus = score_to_signal(avg_score)
        print(f"\n  Consensus: {consensus}  (avg score={avg_score:+.1f})\n")

    print(f"{'='*65}")
    print("  ✅ Strategy engine test complete")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()
