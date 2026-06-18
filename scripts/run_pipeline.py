"""
Manual pipeline trigger — runs the full daily pipeline now.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --stocks RELIANCE TCS HDFCBANK
    python scripts/run_pipeline.py --date 2026-06-17
"""
import sys, os, asyncio, argparse
from datetime import date
from app.cache import get_redis
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Run daily signal pipeline")
    parser.add_argument("--stocks", nargs="+", help="Override watchlist")
    parser.add_argument("--date",   type=str,  help="Run for specific date (YYYY-MM-DD)")
    return parser.parse_args()


async def main():
    args     = parse_args()
    run_date = date.fromisoformat(args.date) if args.date else date.today()

    from app.pipeline import DailyPipeline
    pipeline = DailyPipeline(
        stocks=args.stocks or None,
        run_date=run_date,
    )

    print(f"\n{'='*65}")
    print(f"  Stock Signal Platform — Daily Pipeline")
    print(f"  Date: {run_date} | Stocks: {len(pipeline.ctx.stocks)}")
    print(f"{'='*65}\n")

    ctx = await pipeline.run()

    print(f"\n{'='*65}")
    print(f"  Pipeline Complete")
    print(f"{'='*65}")
    print(f"  Status:    {('SUCCESS' if not ctx.errors else 'FAILED')}")
    print(f"  Signals:   {ctx.signals_generated}")
    print(f"  Regime:    {ctx.regime.regime if ctx.regime else 'N/A'}")

    print(f"\n  Stage timings:")
    for stage, ms in ctx.stage_timings.items():
        bar = "█" * (ms // 200)
        print(f"    {stage:<20} {ms:>5}ms  {bar}")

    if ctx.final_signals or ctx.fused_signals:
        signals = ctx.fused_signals
        print(f"\n  Signals ({len(signals)}):")
        print(f"  {'Symbol':<12} {'Signal':<14} {'Score':>6}  {'Conf':>5}  Entry")
        print(f"  {'-'*55}")
        for s in sorted(signals, key=lambda x: x.fused_score, reverse=True):
            if s.signal != "HOLD":
                entry = f"₹{s.entry_price:.0f}" if s.entry_price else "N/A"
                print(f"  {s.symbol:<12} {s.signal:<14} {s.fused_score:>+6.1f}  {s.confidence:>4.0f}%  {entry}")

    if ctx.errors:
        print(f"\n  Errors:")
        for e in ctx.errors:
            print(f"    ❌ {e}")

    print(f"\n{'='*65}\n")

async def run():
    try:
        await main()
    finally:
        try:
            await get_redis().aclose()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(run())
