"""
System Stress Testing Harness — Production-Grade Institutional Version
Evaluates the unified engine under parameter sensitivity shifts, correlation shocks,
and realistic capital-at-risk path dependencies.
"""
import sys
import os
import random
import copy
import numpy as np
import pandas as pd
from datetime import date, timedelta

# Ensure project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backtest.engine import BacktestEngine, calculate_metrics, TradeRecord, STT_SELL_RATE, BROKERAGE_PER_TRADE
from app.portfolio.acceptance import CROSS_STOCK_MAX_PENALTY
from config.settings import settings
from contextlib import contextmanager
from app.portfolio import acceptance


@contextmanager
def override_acceptance_config(**kwargs):
    """
    Temporarily override acceptance.py module constants.

    Example:
        with override_acceptance_config(
            RISK_PER_TRADE_PCT=0.01,
            MAX_OPEN_POSITIONS=4,
        ):
            ...
    """
    originals = {}

    try:
        for key, value in kwargs.items():
            if not hasattr(acceptance, key):
                raise AttributeError(
                    f"acceptance.py has no constant '{key}'"
                )

            originals[key] = getattr(acceptance, key)
            setattr(acceptance, key, value)

        yield

    finally:
        for key, value in originals.items():
            setattr(acceptance, key, value)

def get_aggregate_metrics(summary: dict):
    """
    BacktestEngine may expose aggregate metrics under different keys
    depending on the version.
    """
    metrics = (
        summary.get("aggregate_metrics")
        or summary.get("aggregate_results")
        or summary.get("metrics")
    )

    if metrics is not None:
        return metrics

    raise KeyError(
        "Could not locate aggregate metrics. "
        f"Available keys: {list(summary.keys())}"
    )

def print_banner(title: str):
    print(f"\n" + "="*75)
    print(f" 🔥 STRESS TEST: {title}")
    print("="*75)

def run_historical_slice_test(all_trades: list, start_date: date, end_date: date, name: str):
    print_banner(name)
    slice_trades = [t for t in all_trades if t.entry_date >= start_date and t.entry_date <= end_date]
    
    if not slice_trades:
        print(f"⚠️  No trades found between {start_date} and {end_date} in the historical dataset.")
        return
        
    res = calculate_metrics(slice_trades, "combined", start_date, end_date)
    if res:
        print(f"  Total Trades Run:  {res.total_trades}")
        print(f"  Win Rate:          {res.win_rate_pct}% (realistic: {res.win_rate_realistic}%)")
        print(f"  Sharpe Ratio:      {res.sharpe_ratio} (realistic: {res.sharpe_realistic})")
        print(f"  Max Drawdown:      {res.max_drawdown_pct}%")
        print(f"  Annualized Return: {res.annualized_return_pct}%")

def run_transaction_sensitivity(all_trades: list, start_date: date, end_date: date):
    print_banner("EXECUTION SLIPPAGE & SPREAD DETERIORATION STRESS (+50% Fee + 0.5% Slippage)")
    
    sensitive_trades = []
    for t in all_trades:
        is_long = "BUY" in t.signal
        adjusted_exit = round(t.exit_price * 0.995, 2) if is_long else round(t.exit_price * 1.005, 2)
        
        gross_pnl = (adjusted_exit - t.entry_price) * t.shares if is_long else (t.entry_price - adjusted_exit) * t.shares
        inflated_stt = adjusted_exit * t.shares * (STT_SELL_RATE * 1.5)
        inflated_brokerage = (BROKERAGE_PER_TRADE * 2) * 1.5
        net_pnl_adjusted = gross_pnl - inflated_stt - inflated_brokerage
        return_pct = (adjusted_exit - t.entry_price) / t.entry_price * 100 if is_long else (t.entry_price - adjusted_exit) / t.entry_price * 100
        
        # FIXED: Added all 4 missing positional arguments
        sensitive_trades.append(TradeRecord(
            window_id=t.window_id, symbol=t.symbol, sector=t.sector, signal=t.signal,
            regime=t.regime, strategy_id=t.strategy_id, entry_date=t.entry_date, exit_date=t.exit_date,
            entry_price=t.entry_price, exit_price=adjusted_exit, stop_loss=t.stop_loss,
            target_price=t.target_price, shares=t.shares, gross_pnl=gross_pnl,
            net_pnl=net_pnl_adjusted, return_pct=return_pct,
            hit_target=t.hit_target, hit_stop=t.hit_stop, timed_out=t.timed_out, exit_reason=t.exit_reason
        ))
        
    res = calculate_metrics(sensitive_trades, "combined", start_date, end_date)
    if res:
        print(f"  Adjusted Sharpe (Realistic): {res.sharpe_realistic}")
        print(f"  Adjusted Return (Realistic): {res.annualized_return_realistic}%")
        print(f"  Max Adjusted Drawdown:      {res.max_drawdown_pct}%")

def run_sector_concentration_stress(all_trades: list, start_date: date, end_date: date):
    print_banner("SECTOR CONCENTRATION SHOCK")
    
    sector_counts = {}
    for t in all_trades:
        sector_counts[t.sector] = sector_counts.get(t.sector, 0) + 1
    if not sector_counts: return
        
    dominant_sector = max(sector_counts, key=sector_counts.get)
    print(f"  Highest exposure vector detected: '{dominant_sector}' ({sector_counts[dominant_sector]} trades)")
    
    shocked_trades = []
    for t in all_trades:
        net_pnl = t.net_pnl; return_pct = t.return_pct; exit_price = t.exit_price
        
        if t.sector == dominant_sector:
            is_long = "BUY" in t.signal
            exit_price = round(t.exit_price * 0.95, 2) if is_long else round(t.exit_price * 1.05, 2)
            gross_pnl = (exit_price - t.entry_price) * t.shares if is_long else (t.entry_price - exit_price) * t.shares
            net_pnl = gross_pnl - (exit_price * t.shares * STT_SELL_RATE) - (BROKERAGE_PER_TRADE * 2)
            return_pct = (exit_price - t.entry_price) / t.entry_price * 100 if is_long else (t.entry_price - exit_price) / t.entry_price * 100
            
        # FIXED: Added all 4 missing positional arguments
        shocked_trades.append(TradeRecord(
            window_id=t.window_id, symbol=t.symbol, sector=t.sector, signal=t.signal,
            regime=t.regime, strategy_id=t.strategy_id, entry_date=t.entry_date, exit_date=t.exit_date,
            entry_price=t.entry_price, exit_price=exit_price, stop_loss=t.stop_loss,
            target_price=t.target_price, shares=t.shares, gross_pnl=t.gross_pnl,
            net_pnl=net_pnl, return_pct=return_pct,
            hit_target=t.hit_target, hit_stop=t.hit_stop, timed_out=t.timed_out, exit_reason=t.exit_reason
        ))
        
    res = calculate_metrics(shocked_trades, "combined", start_date, end_date)
    if res:
        print(f"  Shocked Sharpe (Realistic): {res.sharpe_realistic}")
        print(f"  Shocked Max Drawdown:       {res.max_drawdown_pct}%")

def run_monte_carlo_reshuffle(all_trades: list, iterations: int = 5000):
    print_banner(f"MONTE CARLO RISK-PROPORTIONAL RESHUFFLING ({iterations} Iterations)")
    
    net_returns = [t.net_pnl / (t.entry_price * t.shares) * 100 for t in all_trades if t.shares > 0]
    if len(net_returns) < 10: return
        
    failures = 0
    simulated_drawdowns = []
    portfolio_initial = settings.portfolio_value_inr
    risk_pct = settings.risk_per_trade_pct 
    
    for _ in range(iterations):
        sampled_path = np.random.choice(net_returns, size=len(net_returns), replace=True)
        equity = portfolio_initial
        equity_history = []
        
        for r in sampled_path:
            risked_capital = equity * risk_pct
            pnl = risked_capital * (r / 100)
            equity += pnl
            equity_history.append(equity)
            
        equity_curve = np.array(equity_history)
        peak = np.maximum.accumulate(equity_curve)
        dd_pct = (equity_curve - peak) / peak * 100
        max_dd = abs(np.min(dd_pct))
        
        simulated_drawdowns.append(max_dd)
        if max_dd > 20.0: failures += 1
            
    print(f"  Mean Simulated Drawdown:      {np.mean(simulated_drawdowns):.2f}%")
    print(f"  95th Percentile Drawdown:     {np.percentile(simulated_drawdowns, 95):.2f}%")
    print(f"  99th Percentile Drawdown:     {np.percentile(simulated_drawdowns, 99):.2f}%")
    print(f"  Probability of breaking MaxDD (>20%): {(failures / iterations) * 100:.2f}%")

def run_regime_collapse_stress(all_trades: list, start_date: date, end_date: date):
    print_banner("REGIME COLLAPSE SHOCK: BULL FLATLINING")
    
    collapsed_trades = []
    for t in all_trades:
        return_pct = t.return_pct; net_pnl = t.net_pnl
        if t.regime == "BULL":
            return_pct = 0.0
            net_pnl = 0.0
            
        collapsed_trades.append(TradeRecord(
            window_id=t.window_id, symbol=t.symbol, sector=t.sector, signal=t.signal,
            regime=t.regime, strategy_id=t.strategy_id, entry_date=t.entry_date, exit_date=t.exit_date,
            entry_price=t.entry_price, exit_price=t.exit_price, stop_loss=t.stop_loss,
            target_price=t.target_price, shares=t.shares, gross_pnl=t.gross_pnl,
            net_pnl=net_pnl, return_pct=return_pct,
            hit_target=t.hit_target, hit_stop=t.hit_stop, timed_out=t.timed_out, exit_reason=t.exit_reason
        ))
    res = calculate_metrics(collapsed_trades, "combined", start_date, end_date)
    if res:
        print(f"  Collapsed Sharpe (Realistic): {res.sharpe_realistic}")
        print(f"  Collapsed Return (Realistic): {res.annualized_return_realistic}%")

def run_parameter_sensitivity():
    print_banner("ROBUSTNESS MATRIX: PARAMETER SENSITIVITY")

    scenarios = [
        ("Tight Matrix", 0.010),
        ("Expanded Matrix", 0.020),
    ]

    original = settings.risk_per_trade_pct

    try:
        for name, risk in scenarios:
            print(
                f" ⚙️ Testing configurations for: "
                f"{name} ({risk:.1%} Risk)..."
            )

            settings.risk_per_trade_pct = risk

            engine = BacktestEngine(
                verbose=False,
                save_to_db=False,
            )

            summary = engine.run()

            try:
                metrics = get_aggregate_metrics(summary)
            except KeyError as e:
                print(f"    ❌ {e}")
                continue

            print(
                f"    ↳ Result: "
                f"Sharpe={metrics.sharpe_realistic} | "
                f"MaxDD={metrics.max_drawdown_pct}% | "
                f"Return={metrics.annualized_return_realistic}%"
            )

    finally:
        settings.risk_per_trade_pct = original


def run_correlation_stress():
    print_banner(
        "CONCENTRATION PROXIMITY: "
        "CROSS-STOCK CORRELATION SHOCK"
    )

    for penalty in [0.30, 0.50]:

        print(
            f" ⚙️ Simulating "
            f"CROSS_STOCK_MAX_PENALTY={penalty:.0%}"
        )

        with override_acceptance_config(
            CROSS_STOCK_MAX_PENALTY=penalty
        ):
            engine = BacktestEngine(
                verbose=False,
                save_to_db=False,
            )

            summary = engine.run()

            try:
                metrics = get_aggregate_metrics(summary)
            except KeyError as e:
                print(f"    ❌ {e}")
                continue

        print(
            f"    ↳ Result at {penalty:.0%}: "
            f"Sharpe={metrics.sharpe_realistic} | "
            f"Ret={metrics.annualized_return_realistic}% | "
            f"Trades Count={metrics.total_trades}"
        )

def main():
    print("=" * 75)
    print(" STOCK SIGNAL PLATFORM — PRODUCTION INSTITUTIONAL STRESS TEST MATRIX")
    print("=" * 75)

    print("🔄 Initializing system baseline data harvest...")

    engine = BacktestEngine(
        verbose=False,
        save_to_db=False,
    )

    summary = engine.run()
    print(f"Available summary keys: {list(summary.keys())}")
    all_trades = summary.get("all_trades", [])

    if not all_trades:
        print(
            "❌ Harvest error: "
            "Ensure market data table is properly populated."
        )
        return

    p_start = min(t.entry_date for t in all_trades)
    p_end = max(t.exit_date for t in all_trades)

    print(
        f"✅ Harvest successful. "
        f"Mapped {len(all_trades)} trades "
        f"across unified architecture templates."
    )

    #
    # Historical crisis windows
    #
    if p_start <= date(2020, 6, 30):
        run_historical_slice_test(
            all_trades,
            date(2020, 1, 1),
            date(2020, 6, 30),
            "2020 COVID CRASH MARGIN SKEW",
        )
    else:
        print_banner("2020 COVID CRASH MARGIN SKEW")
        print(
            f"⚠️ Dataset starts on {p_start}. "
            "No 2020 data available."
        )

    if p_start <= date(2022, 12, 31):
        run_historical_slice_test(
            all_trades,
            date(2022, 1, 1),
            date(2022, 12, 31),
            "2022 BEAR MARKET SKEW",
        )
    else:
        print_banner("2022 BEAR MARKET SKEW")
        print(
            f"⚠️ Dataset starts on {p_start}. "
            "No 2022 data available."
        )

    #
    # Stress matrix
    #
    run_sector_concentration_stress(
        all_trades,
        p_start,
        p_end,
    )



    # otherwise keep:
    run_transaction_sensitivity(
        all_trades,
        p_start,
        p_end,
    )

    run_regime_collapse_stress(
        all_trades,
        p_start,
        p_end,
    )

    run_monte_carlo_reshuffle(all_trades)

    #
    # Parameter sensitivity
    #
    run_parameter_sensitivity()

    #
    # Correlation stress
    #
    run_correlation_stress()

    # or if your function name is still:
    # run_correlation_penalty_stress()

    print("\n" + "=" * 75)
    print(" ✅ STRESS TEST MATRIX COMPLETE")
    print(f" Dataset Window : {p_start} → {p_end}")
    print(f" Total Trades   : {len(all_trades)}")
    print("=" * 75)



if __name__ == "__main__":
    main()