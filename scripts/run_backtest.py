"""
Walk-forward backtest runner.

Usage:
    python scripts/run_backtest.py                          # full watchlist
    python scripts/run_backtest.py --stocks RELIANCE TCS   # subset
    python scripts/run_backtest.py --stocks RELIANCE --train 2 --test 3
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backtest.engine import BacktestEngine


def parse_args():
    p = argparse.ArgumentParser(description="Run walk-forward backtest")
    p.add_argument("--stocks", nargs="+", help="Stocks to backtest (default: full watchlist)")
    p.add_argument("--train",  type=int, default=2, help="Train window years (default: 2)")
    p.add_argument("--test",   type=int, default=3, help="Test window months (default: 3)")
    p.add_argument("--no-db",  action="store_true", help="Don't save results to DB")
    return p.parse_args()


def print_result_table(title: str, results: dict) -> None:
    if not results:
        return
    print(f"\n  ── {title} ─────────────────────────────────────")
    print(f"  {'Segment':<18} {'Trades':>7} {'WinRate':>8} {'Sharpe':>7} "
          f"{'MaxDD':>7} {'AnnRet':>8} {'Status':>6}")
    print(f"  {'-'*65}")
    for name, r in sorted(results.items()):
        status = "✅" if r.meets_acceptance_criteria else "❌"
        print(f"  {name:<18} {r.total_trades:>7} "
              f"{r.win_rate_pct:>7.1f}% {r.sharpe_ratio:>7.2f} "
              f"{r.max_drawdown_pct:>6.1f}% {r.annualized_return_pct:>7.1f}% "
              f"  {status}")


def main():
    args = parse_args()

    print(f"\n{'='*65}")
    print(f"  Stock Signal Platform — Walk-Forward Backtest")
    print(f"  Train: {args.train}y | Test: {args.test}m | "
          f"Stocks: {len(args.stocks) if args.stocks else 'all'}")
    print(f"  Acceptance: Sharpe≥1.0 | MaxDD≤20% | WinRate≥45%")
    print(f"{'='*65}\n")

    engine = BacktestEngine(
        stocks=args.stocks,
        train_years=args.train,
        test_months=args.test,
        verbose=True,
        save_to_db=not args.no_db,
    )
    summary = engine.run()

    if not summary.get("aggregate"):
        print("  ❌ No trades generated — check data coverage")
        return

    agg = summary["aggregate"]

    # ── Aggregate ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  AGGREGATE RESULTS  ({agg.total_trades} total trades)")
    print(f"{'='*65}")
    status = "✅ PASS" if agg.meets_acceptance_criteria else "❌ FAIL"
    print(f"  {status}")
    print(f"  Sharpe (gross):     {agg.sharpe_ratio:.3f}  "
          f"(realistic: {agg.sharpe_realistic:.3f})")
    print(f"  Max drawdown:       {agg.max_drawdown_pct:.1f}%")
    print(f"  Win rate:           {agg.win_rate_pct:.1f}%  "
          f"(realistic: {agg.win_rate_realistic:.1f}%)")
    print(f"  Ann. return:        {agg.annualized_return_pct:.1f}%  "
          f"(realistic: {agg.annualized_return_realistic:.1f}%)")
    print(f"  Profit factor:      {agg.profit_factor:.2f}")
    print(f"  Avg win/loss:       +{agg.avg_win_pct:.2f}% / {agg.avg_loss_pct:.2f}%")
    if agg.notes:
        print(f"  Failing criteria:   {agg.notes}")

    print_result_table("BY STRATEGY",  summary["results_by_strategy"])
    print_result_table("BY REGIME",    summary["results_by_regime"])
    print_result_table("BY SECTOR",    summary["results_by_sector"])

    # ── Trade sample ──────────────────────────────────────────
    trades = summary["all_trades"]
    if trades:
        print(f"\n  ── Sample Trades (last 10) ──────────────────────────")
        print(f"  {'Symbol':<10} {'Entry':>10} {'Exit':>10} "
              f"{'Return':>8} {'Exit reason':<10}")
        print(f"  {'-'*55}")
        for t in trades[-10:]:
            print(f"  {t.symbol:<10} {t.entry_date!s:>10} {t.exit_date!s:>10} "
                  f"  {t.return_pct:>+6.2f}%  {t.exit_reason:<10}")

    print(f"\n{'='*65}")
    overall = "✅ PASSES acceptance criteria" if summary["passes_acceptance"] \
              else "❌ FAILS acceptance criteria"
    print(f"  {overall}")
    print(f"  run_id={summary['run_id']}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
