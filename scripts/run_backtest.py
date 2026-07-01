"""
Walk-forward backtest runner.

Usage:
    python scripts/run_backtest.py                          # full watchlist
    python scripts/run_backtest.py --stocks RELIANCE TCS   # subset
    python scripts/run_backtest.py --stocks RELIANCE --train 2 --test 3
"""
import sys, os, argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backtest.engine import BacktestEngine

DEFAULT_OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output.txt",
)


def parse_args():
    p = argparse.ArgumentParser(description="Run walk-forward backtest")
    p.add_argument("--stocks", nargs="+", help="Stocks to backtest (default: full watchlist)")
    p.add_argument("--train",  type=int, default=2, help="Train window years (default: 2)")
    p.add_argument("--test",   type=int, default=3, help="Test window months (default: 3)")
    p.add_argument("--no-db",  action="store_true", help="Don't save results to DB")
    p.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE,
                   help="File path for the report output (default: output.txt)")
    return p.parse_args()


def init_output_file(output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("")


def log_line(message: str, lines: list, output_path: str | None = None) -> None:
    print(message)
    lines.append(message)
    if output_path:
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(message + "\n")


def print_result_table(title: str, results: dict, lines: list, output_path: str | None = None) -> None:
    if not results:
        return
    log_line(f"\n  ── {title} ─────────────────────────────────────", lines, output_path)
    log_line(f"  {'Segment':<18} {'Trades':>7} {'WinRate':>8} {'Sharpe':>7} "
             f"{'MaxDD':>7} {'AnnRet':>8} {'Status':>6}", lines, output_path)
    log_line(f"  {'-'*65}", lines, output_path)
    for name, r in sorted(results.items()):
        status = "✅" if r.meets_acceptance_criteria else "❌"
        log_line(f"  {name:<18} {r.total_trades:>7} "
                 f"{r.win_rate_pct:>7.1f}% {r.sharpe_ratio:>7.2f} "
                 f"{r.max_drawdown_pct:>6.1f}% {r.annualized_return_pct:>7.1f}% "
                 f"  {status}", lines, output_path)


def summarize_trade_counts(candidate_counts: dict, executed_counts: dict) -> list[tuple]:
    symbols = sorted(set(candidate_counts) | set(executed_counts))
    return [(symbol, candidate_counts.get(symbol, 0), executed_counts.get(symbol, 0)) for symbol in symbols]


def build_best_worst_trade_rows(trades: list) -> list[tuple]:
    grouped = defaultdict(list)
    for trade in trades:
        grouped[trade.symbol].append(trade)

    rows = []
    for symbol in sorted(grouped):
        best_trade = max(grouped[symbol], key=lambda t: t.return_pct)
        worst_trade = min(grouped[symbol], key=lambda t: t.return_pct)
        rows.append((symbol, best_trade, worst_trade))
    return rows


def main():
    args = parse_args()
    init_output_file(args.output_file)

    lines = []

    log_line(f"\n{'='*65}", lines, args.output_file)
    log_line(f"  Stock Signal Platform — Walk-Forward Backtest", lines, args.output_file)
    log_line(f"  Train: {args.train}y | Test: {args.test}m | "
             f"Stocks: {len(args.stocks) if args.stocks else 'all'}", lines, args.output_file)
    log_line(f"  Acceptance: Sharpe≥1.0 | MaxDD≤20% | WinRate≥45%", lines, args.output_file)
    log_line(f"{'='*65}\n", lines, args.output_file)

    engine = BacktestEngine(
        stocks=args.stocks,
        train_years=args.train,
        test_months=args.test,
        verbose=True,
        save_to_db=not args.no_db,
    )
    summary = engine.run()

    if not summary.get("aggregate"):
        log_line("  ❌ No trades generated — check data coverage", lines, args.output_file)
        return

    agg = summary["aggregate"]

    # ── Aggregate ─────────────────────────────────────────────
    log_line(f"\n{'='*65}", lines, args.output_file)
    log_line(f"  AGGREGATE RESULTS  ({agg.total_trades} total trades)", lines, args.output_file)
    log_line(f"{'='*65}", lines, args.output_file)
    status = "✅ PASS" if agg.meets_acceptance_criteria else "❌ FAIL"
    log_line(f"  {status}", lines, args.output_file)
    log_line(f"  Sharpe (gross):     {agg.sharpe_ratio:.3f}  "
             f"(realistic: {agg.sharpe_realistic:.3f})", lines, args.output_file)
    log_line(f"  Max drawdown:       {agg.max_drawdown_pct:.1f}%", lines, args.output_file)
    log_line(f"  Win rate:           {agg.win_rate_pct:.1f}%  "
             f"(realistic: {agg.win_rate_realistic:.1f}%)", lines, args.output_file)
    log_line(f"  Ann. return:        {agg.annualized_return_pct:.1f}%  "
             f"(realistic: {agg.annualized_return_realistic:.1f}%)", lines, args.output_file)
    log_line(f"  Profit factor:      {agg.profit_factor:.2f}", lines, args.output_file)
    log_line(f"  Avg win/loss:       +{agg.avg_win_pct:.2f}% / {agg.avg_loss_pct:.2f}%", lines, args.output_file)
    if agg.notes:
        log_line(f"  Failing criteria:   {agg.notes}", lines, args.output_file)

    print_result_table("BY STRATEGY",  summary["results_by_strategy"], lines, args.output_file)
    print_result_table("BY REGIME",    summary["results_by_regime"], lines, args.output_file)
    print_result_table("BY SECTOR",    summary["results_by_sector"], lines, args.output_file)

    # ── Stock trade counts ───────────────────────────────────
    all_candidates = summary.get("all_candidates", [])
    executed_trades = summary.get("all_trades", [])
    candidate_counts = defaultdict(int)
    for candidate in all_candidates:
        candidate_counts[candidate.symbol] += 1
    executed_counts = defaultdict(int)
    for trade in executed_trades:
        executed_counts[trade.symbol] += 1

    log_line(f"\n  ── Trade Counts by Symbol ─────────────────────────", lines, args.output_file)
    log_line(f"  {'Symbol':<10} {'Generated':>10} {'Executed':>10}", lines, args.output_file)
    log_line(f"  {'-'*35}", lines, args.output_file)
    for symbol, generated, executed in summarize_trade_counts(candidate_counts, executed_counts):
        log_line(f"  {symbol:<10} {generated:>10} {executed:>10}", lines, args.output_file)

    # ── Best / worst trades per symbol ─────────────────────
    best_worst_rows = build_best_worst_trade_rows(executed_trades)
    if best_worst_rows:
        log_line(f"\n  ── Best / Worst Executed Trades by Symbol ─────────", lines, args.output_file)
        log_line(f"  {'Symbol':<10} {'Best':>10} {'Worst':>10}", lines, args.output_file)
        log_line(f"  {'-'*35}", lines, args.output_file)
        for symbol, best_trade, worst_trade in best_worst_rows:
            log_line(f"  {symbol:<10} {best_trade.return_pct:>+9.2f}% {worst_trade.return_pct:>+9.2f}%",
                     lines, args.output_file)

    # ── Executed trade details ──────────────────────────────
    if executed_trades:
        log_line(f"\n  ── Executed Trades (all) ───────────────────────────", lines, args.output_file)
        log_line(f"  {'Symbol':<10} {'Entry':>10} {'Exit':>10} {'Return':>8} {'Exit reason':<10}", lines, args.output_file)
        log_line(f"  {'-'*60}", lines, args.output_file)
        for t in executed_trades:
            log_line(f"  {t.symbol:<10} {t.entry_date!s:>10} {t.exit_date!s:>10} "
                     f"  {t.return_pct:>+6.2f}%  {t.exit_reason:<10}", lines, args.output_file)

    log_line(f"\n{'='*65}", lines, args.output_file)
    overall = "✅ PASSES acceptance criteria" if summary["passes_acceptance"] \
              else "❌ FAILS acceptance criteria"
    log_line(f"  {overall}", lines, args.output_file)
    log_line(f"  run_id={summary['run_id']}", lines, args.output_file)
    log_line(f"{'='*65}\n", lines, args.output_file)


if __name__ == "__main__":
    main()
