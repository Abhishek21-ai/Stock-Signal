"""
Walk-Forward Backtesting Engine — Section 12 + 26
Updated with explicit trading session tracking and cross-stock historical correlation.
"""
from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.data.execution_realism import apply_execution_realism, get_slippage_factor
from app.db import get_sync_db
from app.features.engineer import build_dataframe, compute_features, extract_latest_features
from app.fusion.engine import FusedSignal, fuse
from app.logger import get_logger
from app.portfolio.acceptance import (
    SECTOR_QUARANTINE_SESSIONS,
    AcceptanceCandidate,
    PortfolioAcceptanceEngine,
    PortfolioLedger,
    PositionSize,
)
from app.regime.detector import REGIME_WEIGHTS, RegimeResult, classify_regime
from app.strategies.runner import StrategyRunner
from config.settings import settings
from config.watchlist import WATCHLIST_WITH_SECTORS

logger = get_logger("backtest")

TRADING_DAYS_PER_YEAR = 252
HOLD_DAYS             = 10     
BROKERAGE_PER_TRADE   = 20.0  
STT_SELL_RATE         = 0.001  

@dataclass
class WalkForwardWindow:
    window_id:    int
    train_start:  date
    train_end:    date
    test_start:   date
    test_end:     date

@dataclass
class TradeRecord:
    window_id:     int
    symbol:        str
    sector:        str
    signal:        str           
    regime:        str
    strategy_id:   str           
    entry_date:    date
    exit_date:     date
    entry_price:   float
    exit_price:    float
    stop_loss:     float
    target_price:  float
    shares:        int
    gross_pnl:     float         
    net_pnl:       float         
    return_pct:    float         
    hit_target:    bool
    hit_stop:      bool
    timed_out:     bool          
    exit_reason:   str           

@dataclass
class BacktestResult:
    strategy_id:              str
    period_start:             date
    period_end:               date
    regime:                   Optional[str]
    sector:                   Optional[str]
    total_trades:             int
    winning_trades:           int
    losing_trades:            int
    win_rate_pct:             float
    avg_return_pct:           float
    total_return_pct:         float
    annualized_return_pct:    float
    sharpe_ratio:             float
    max_drawdown_pct:         float
    avg_win_pct:              float
    avg_loss_pct:             float
    profit_factor:            float           
    sharpe_realistic:         float
    win_rate_realistic:       float
    annualized_return_realistic: float
    meets_acceptance_criteria: bool
    notes:                    str = ""


def generate_windows(data_start: date, data_end: date, train_years: int = 2, test_months: int = 3) -> List[WalkForwardWindow]:
    windows  = []
    win_id   = 1
    train_days = int(train_years * TRADING_DAYS_PER_YEAR * (365/252))
    test_days  = int(test_months * 30.5)

    test_start = data_start + timedelta(days=train_days)
    while test_start + timedelta(days=test_days) <= data_end:
        test_end   = test_start + timedelta(days=test_days)
        train_end  = test_start - timedelta(days=1)
        train_start = train_end - timedelta(days=train_days)

        windows.append(WalkForwardWindow(
            window_id=win_id, train_start=max(train_start, data_start), train_end=train_end,
            test_start=test_start, test_end=min(test_end, data_end),
        ))
        test_start += timedelta(days=test_days)
        win_id += 1
    return windows


def simulate_signals_for_window(symbol: str, df: pd.DataFrame, window: WalkForwardWindow, regime: str) -> List[FusedSignal]:
    runner  = StrategyRunner()
    signals = []
    test_df = df[(df.index.date >= window.test_start) & (df.index.date <= window.test_end)]

    for ts, _ in test_df.iterrows():
        signal_date = ts.date()
        history_df = df[df.index < ts].tail(300)
        if len(history_df) < 210: continue

        try:
            feat_df  = compute_features(history_df.copy())
            features = extract_latest_features(feat_df, symbol)
            strategy_results = runner.run(features, regime=regime)

            if regime in ("BULL", "UNCERTAIN"):
                for r in strategy_results:
                    if r.strategy_id == "reversion":
                        r.score = 0.0; r.confidence = 0.0; r.signal = "HOLD"

            regime_result = RegimeResult(regime=regime, regime_confidence="NORMAL", fusion_weights=dict(REGIME_WEIGHTS[regime]))
            fused = fuse(symbol, strategy_results, regime_result, signal_date, features, save_to_db=False)
            fused._signal_date = signal_date
            fused._features    = features
            signals.append(fused)
        except Exception:
            pass
    return signals


@dataclass
class _Candidate:
    window_id: int; symbol: str; sector: str; signal: str; regime: str; strategy_id: str; fused_score: float
    entry_date: date; exit_date: date; entry_price: float; exit_price: float; stop_loss: float; target_price: float
    is_long: bool; slip: float; shares: int; gross_pnl: float; net_pnl: float; return_pct: float
    hit_target: bool; hit_stop: bool; timed_out: bool; exit_reason: str; adv_estimate: Optional[float] = None


def _build_candidates(symbol: str, df: pd.DataFrame, signals: List[FusedSignal], window: WalkForwardWindow, regime: str) -> List[_Candidate]:
    candidates = []
    sector = WATCHLIST_WITH_SECTORS.get(symbol, "Unknown")
    slip = get_slippage_factor(symbol)
    cooldown_until: Optional[date] = None

    for signal in signals:
        if signal.signal not in ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL") or not signal.entry_price: continue
        sig_date  = signal._signal_date
        features  = signal._features
        close     = features.get("close", signal.entry_price)
        atr       = features.get("atr_14", close * 0.02)
        adx       = features.get("adx_14", 0)
        vol_ratio = features.get("volume_ratio", 1.0) or 1.0
        conf      = signal.confidence

        regime_min_conf = {"BULL": 35, "UNCERTAIN": 38, "SIDEWAYS": 28, "BEAR": 100}
        if conf < regime_min_conf.get(regime, 38) or vol_ratio < 1.0: continue

        top_strategy = max(signal.strategy_scores, key=signal.strategy_scores.get) if signal.strategy_scores else "combined"
        if top_strategy == "trend" and adx < 20: continue
        if cooldown_until and sig_date <= cooldown_until: continue

        future_rows = df[df.index.date > sig_date]
        if future_rows.empty: continue

        next_open  = float(future_rows.iloc[0]["Open"]) if "Open" in future_rows.columns else close
        entry_date = future_rows.index[0].date()
        is_long = "BUY" in signal.signal
        gap_pct = (next_open - close) / close
        if (is_long and gap_pct < -0.01) or (not is_long and gap_pct > 0.01): continue

        entry_price = round(next_open * (1 + slip * 0.5), 2)
        stop_mult = 2.0
        if is_long:
            stop_loss = max(round(entry_price - stop_mult * atr, 2), round(entry_price * 0.93, 2))
        else:
            stop_loss = min(round(entry_price + stop_mult * atr, 2), round(entry_price * 1.02, 2))

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0: continue

        risk_inr = settings.portfolio_value_inr * settings.risk_per_trade_pct
        shares = min(max(1, int(risk_inr / risk_per_share)), int(settings.portfolio_value_inr * settings.max_single_stock_pct / entry_price))
        target = round(entry_price + 2.5 * risk_per_share, 2) if is_long else round(entry_price - 2.5 * risk_per_share, 2)

        hit_target = hit_stop = timed_out = False
        exit_price, exit_date = entry_price, entry_date
        holding_rows = future_rows.iloc[1:HOLD_DAYS + 1]

        if not is_long:
            first_row = holding_rows.iloc[0] if not holding_rows.empty else None
            if first_row is not None:
                day_high, day_low = float(first_row.get("High", first_row["Close"])), float(first_row.get("Low", first_row["Close"]))
                if day_high >= stop_loss:
                    exit_price = round(stop_loss * (1 + slip), 2); exit_date = first_row.name.date(); hit_stop = True
                elif day_low <= target:
                    exit_price = round(target * (1 + slip), 2); exit_date = first_row.name.date(); hit_target = True
                else:
                    exit_price = round(float(first_row["Close"]) * (1 - slip * 0.5), 2); exit_date = first_row.name.date(); timed_out = True
            else: timed_out = True
        else:
            for _, (exit_ts, row) in enumerate(holding_rows.iterrows()):
                day_high, day_low = float(row.get("High", row["Close"])), float(row.get("Low", row["Close"]))
                if day_low <= stop_loss:
                    exit_price = round(stop_loss * (1 - slip), 2); exit_date = exit_ts.date(); hit_stop = True; break
                if day_high >= target:
                    exit_price = round(target * (1 - slip), 2); exit_date = exit_ts.date(); hit_target = True; break
            else:
                if not holding_rows.empty:
                    exit_price = round(float(holding_rows.iloc[-1]["Close"]) * (1 - slip * 0.5), 2)
                    exit_date = holding_rows.index[-1].date()
                timed_out = True

        if hit_stop: cooldown_until = exit_date + timedelta(days=5)

        gross_pnl = (exit_price - entry_price) * shares if is_long else (entry_price - exit_price) * shares
        return_pct = (exit_price - entry_price) / entry_price * 100 if is_long else (entry_price - exit_price) / entry_price * 100
        net_pnl = gross_pnl - (exit_price * shares * STT_SELL_RATE) - BROKERAGE_PER_TRADE * 2
        exit_reason = "TARGET" if hit_target else "STOP" if hit_stop else "TIMEOUT"
        vol_sma20 = features.get("volume_sma_20", 0)

        candidates.append(_Candidate(
            window_id=window.window_id, symbol=symbol, sector=sector, signal=signal.signal, regime=regime, strategy_id=top_strategy,
            fused_score=signal.fused_score, entry_date=entry_date, exit_date=exit_date, entry_price=entry_price, exit_price=exit_price,
            stop_loss=stop_loss, target_price=target, is_long=is_long, slip=slip, shares=shares, gross_pnl=gross_pnl, net_pnl=net_pnl,
            return_pct=return_pct, hit_target=hit_target, hit_stop=hit_stop, timed_out=timed_out, exit_reason=exit_reason,
            adv_estimate=float(vol_sma20) * entry_price if vol_sma20 and vol_sma20 > 0 else None
        ))
    return candidates


class BacktestPortfolioSimulator:
    def __init__(self, trading_dates: List[date], returns_cache: Dict[str, pd.Series]):
        self.trading_dates = sorted(set(trading_dates))
        self.ledger = PortfolioLedger()
        self.engine = PortfolioAcceptanceEngine(self.ledger)
        self.returns_cache = returns_cache
        self.engine.correlation_fn = self._backtest_correlation_fn

    def _backtest_correlation_fn(self, stock_a: str, stock_b: str, as_of_date: date) -> Optional[float]:
        ra, rb = self.returns_cache.get(stock_a), self.returns_cache.get(stock_b)
        if ra is None or rb is None: return None
        window_start = as_of_date - timedelta(days=60)
        slice_a, slice_b = ra.loc[window_start:as_of_date], rb.loc[window_start:as_of_date]
        if len(slice_a) < 20 or len(slice_b) < 20: return None
        combined = pd.concat([slice_a, slice_b], axis=1).dropna()
        if len(combined) < 20: return None
        return float(combined.corr().iloc[0, 1])

    def _quarantine_session_handler(self, sector: str, current_date: date) -> None:
        # CRITICAL FIX: Safe session quarantine advancing mapped inside the trading sessions index
        try:
            idx = self.trading_dates.index(current_date)
            expiry_date = self.trading_dates[min(idx + SECTOR_QUARANTINE_SESSIONS, len(self.trading_dates) - 1)]
        except ValueError:
            expiry_date = current_date + timedelta(days=7)
        self.ledger.set_sector_quarantine(sector, expiry_date)

    def run(self, candidates: List[_Candidate]) -> List[_Candidate]:
        if not candidates: return []
        by_date = defaultdict(list)
        for c in candidates: by_date[c.entry_date].append(c)
        accepted = []

        for today in sorted(by_date.keys()):
            self.ledger.advance_day_backtest(today, self._quarantine_session_handler)
            acc_candidates = [
                AcceptanceCandidate(
                    symbol=c.symbol, sector=c.sector, fused_score=c.fused_score, entry_date=c.entry_date,
                    entry_price=c.entry_price, stop_loss=c.stop_loss, exit_date=c.exit_date, exit_reason=c.exit_reason,
                    adv_estimate=c.adv_estimate, ref=c
                ) for c in by_date[today]
            ]
            eval_result = self.engine.evaluate_batch(today, acc_candidates)
            for ac in eval_result.accepted:
                original: _Candidate = ac.candidate.ref
                new_shares = ac.position.final_shares
                if new_shares != original.shares:
                    original.shares = new_shares
                    original.gross_pnl = (original.exit_price - original.entry_price) * new_shares if original.is_long else (original.entry_price - original.exit_price) * new_shares
                    original.net_pnl = original.gross_pnl - (original.exit_price * new_shares * STT_SELL_RATE) - BROKERAGE_PER_TRADE * 2
                accepted.append(original)
        return accepted


def calculate_metrics(trades: List[TradeRecord], strategy_id: str, period_start: date, period_end: date, regime: Optional[str] = None, sector: Optional[str] = None) -> Optional[BacktestResult]:
    if not trades: return None
    returns = [t.return_pct for t in trades]
    net_ret = [t.net_pnl / (t.entry_price * t.shares) * 100 for t in trades]
    wins, losses = [t for t in trades if t.return_pct > 0], [t for t in trades if t.return_pct <= 0]

    win_rate = len(wins) / len(trades) * 100
    avg_return = float(np.mean(returns))
    avg_win = float(np.mean([t.return_pct for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.return_pct for t in losses])) if losses else 0.0
    profit_factor = sum(t.return_pct for t in wins) / abs(sum(t.return_pct for t in losses)) if losses else float("inf")

    days_in_period = (period_end - period_start).days or 1
    total_return = float(np.sum(returns))
    ann_return = total_return * (365 / days_in_period)

    sharpe = (np.mean([r - (6.5 / 252) for r in returns]) / (np.std([r - (6.5 / 252) for r in returns]) + 1e-9)) * math.sqrt(252) if len(returns) > 1 else 0.0
    pnl_curve = np.cumsum([t.net_pnl for t in sorted(trades, key=lambda t: t.exit_date)])
    max_dd = float(abs(np.min((settings.portfolio_value_inr + pnl_curve - np.maximum.accumulate(settings.portfolio_value_inr + pnl_curve)) / np.maximum.accumulate(settings.portfolio_value_inr + pnl_curve) * 100))) if len(pnl_curve) > 0 else 0.0

    net_sharpe = (np.mean(net_ret) / (np.std(net_ret) + 1e-9)) * math.sqrt(252) if len(net_ret) > 1 else 0.0
    meets = bool(sharpe >= settings.backtest_min_sharpe and max_dd <= settings.backtest_max_drawdown * 100 and win_rate >= settings.backtest_min_win_rate * 100)

    return BacktestResult(
        strategy_id, period_start, period_end, regime, sector, len(trades), len(wins), len(losses),
        round(win_rate, 2), round(avg_return, 4), round(total_return, 2), round(ann_return, 2),
        round(sharpe, 3), round(max_dd, 2), round(avg_win, 4), round(avg_loss, 4), round(profit_factor, 3),
        round(net_sharpe, 3), round(len([r for r in net_ret if r > 0]) / len(net_ret) * 100 if net_ret else 0, 2),
        round(float(np.sum(net_ret)) * (365 / days_in_period), 2), meets
    )


class BacktestEngine:
    def __init__(self, stocks: Optional[List[str]] = None, train_years: int = None, test_months: int = None, verbose: bool = True, save_to_db: bool = True):
        self.stocks = stocks or list(settings.watchlist)
        self.train_years = train_years or settings.backtest_train_years
        self.test_months = test_months or settings.backtest_walk_forward_months
        self.verbose = verbose; self.save_to_db = save_to_db; self.run_id = str(uuid.uuid4())

    def run(self) -> Dict:
        all_candidates: List[_Candidate] = []
        trading_dates_seen = set()
        returns_cache: Dict[str, pd.Series] = {}

        for symbol in self.stocks:
            df = self._load_data(symbol)
            if df is None or len(df) < 210: continue
            
            returns_cache[symbol] = df["Close"].pct_change()
            windows = generate_windows(df.index[0].date(), df.index[-1].date(), self.train_years, self.test_months)

            for window in windows:
                regime = self._detect_regime(df[df.index.date <= window.train_end])
                signals = simulate_signals_for_window(symbol, df, window, regime)
                for c in _build_candidates(symbol, df, signals, window, regime):
                    all_candidates.append(c)
                    trading_dates_seen.update([c.entry_date, c.exit_date])

        if not all_candidates: return {"all_trades": [], "aggregate": None, "passes_acceptance": False}

        simulator = BacktestPortfolioSimulator(list(trading_dates_seen), returns_cache)
        accepted_candidates = simulator.run(all_candidates)
        all_trades = [
            TradeRecord(
                c.window_id, c.symbol, c.sector, c.signal, c.regime, c.strategy_id, c.entry_date, c.exit_date,
                c.entry_price, c.exit_price, c.stop_loss, c.target_price, c.shares, c.gross_pnl, c.net_pnl,
                c.return_pct, c.hit_target, c.hit_stop, c.timed_out, c.exit_reason
            ) for c in accepted_candidates
        ]
        if not all_trades: return {"all_trades": [], "aggregate": None, "passes_acceptance": False}

        p_start, p_end = min(t.entry_date for t in all_trades), max(t.exit_date for t in all_trades)
        aggregate = calculate_metrics(all_trades, "combined", p_start, p_end)
        
        # ── RESTORE SEGMENTED BREAKDOWNS FOR REPORTING ────────────────
        results_by_strategy: Dict[str, BacktestResult] = {}
        for sid in set(t.strategy_id for t in all_trades):
            seg_trades = [t for t in all_trades if t.strategy_id == sid]
            r = calculate_metrics(seg_trades, sid, p_start, p_end)
            if r: results_by_strategy[sid] = r

        results_by_regime: Dict[str, BacktestResult] = {}
        for reg in set(t.regime for t in all_trades):
            seg_trades = [t for t in all_trades if t.regime == reg]
            r = calculate_metrics(seg_trades, "combined", p_start, p_end, regime=reg)
            if r: results_by_regime[reg] = r

        results_by_sector: Dict[str, BacktestResult] = {}
        for sec in set(t.sector for t in all_trades):
            seg_trades = [t for t in all_trades if t.sector == sec]
            r = calculate_metrics(seg_trades, "combined", p_start, p_end, sector=sec)
            if r: results_by_sector[sec] = r
        # ──────────────────────────────────────────────────────────────

        return {
            "run_id": self.run_id, 
            "all_candidates": all_candidates, 
            "all_trades": all_trades,
            "aggregate": aggregate, 
            "results_by_strategy": results_by_strategy,
            "results_by_regime": results_by_regime,
            "results_by_sector": results_by_sector,
            "passes_acceptance": aggregate.meets_acceptance_criteria if aggregate else False
        }

    def _load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            from app.data.validator import get_latest_ohlcv
            rows = get_latest_ohlcv(symbol, n=2000)
            return build_dataframe(rows) if len(rows) >= 210 else None
        except Exception: return None

    def _detect_regime(self, train_df: pd.DataFrame) -> str:
        try:
            import pandas_ta as ta
            if len(train_df) < 210: return "UNCERTAIN"
            close = train_df["Close"]
            high = train_df["High"] if "High" in train_df.columns else close
            low = train_df["Low"] if "Low" in train_df.columns else close
            regime, _ = classify_regime(
                float(close.iloc[-1]), float(ta.ema(close, length=20).iloc[-1]),
                float(ta.ema(close, length=50).iloc[-1]), float(ta.ema(close, length=200).iloc[-1]),
                float(ta.adx(high, low, close, length=14)["ADX_14"].iloc[-1])
            )
            return regime
        except Exception: 
            return "UNCERTAIN"