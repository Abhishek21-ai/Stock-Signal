"""
Strategy Correlation and Dependency Engine — Section 23

Two responsibilities:
  1. compute_correlation_matrix() — rolling 60-day Pearson correlation
     between the 5 strategy scores (Trend, Momentum, MeanReversion,
     Breakout, Volume) per stock. Stored in strategy_correlations table
     as a 5x5 JSONB matrix, refreshed daily.

  2. apply_correlation_penalty() — called by the Fusion Engine. Reads
     today's correlation matrix for a stock and downweights strategies
     that are highly correlated with each other (r > 0.7), since two
     strategies firing together on correlated signals shouldn't count
     as "two independent confirmations."

Formula (Section 23.1):
  CorrelationPenalty(i,j) = 0.2 x (r_ij - 0.7) / 0.3   for r_ij > 0.7
                           = 0                          otherwise
  AdjustedWeight_i = BaseWeight_i x (1 - max_j CorrelationPenalty(i,j))

Combined with the static co-firing penalty (Section 8: Trend+Breakout),
the total reduction for any single strategy is capped at 50%.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.db import get_sync_db
from app.logger import get_logger

logger = get_logger("correlation")

# ── Constants (Section 23.1) ──────────────────────────────────
ROLLING_WINDOW_DAYS   = 60
CORRELATION_THRESHOLD = 0.7
MAX_PENALTY_AT_R1     = 0.2     # penalty scales 0 -> 0.2 as r goes 0.7 -> 1.0
MAX_COMBINED_PENALTY  = 0.5     # hard cap: never reduce a weight by more than 50%
MIN_SAMPLES_REQUIRED  = 20      # need at least this many days of history

STRATEGIES = ["trend", "momentum", "reversion", "breakout", "volume"]

# Static co-firing penalty from Section 8 (Trend + Breakout fire together
# on the same breakout event more often than not — pre-existing rule)
STATIC_COFIRE_PAIRS: Dict[Tuple[str, str], float] = {
    ("trend", "breakout"): 0.20,   # 20% static reduction when both fire same direction
}


# ── Step 1: Compute rolling correlation matrix ──────────────────

def _fetch_strategy_score_history(
    stock: str,
    as_of_date: date,
    window_days: int = ROLLING_WINDOW_DAYS,
) -> Dict[str, List[float]]:
    """
    Fetch the last `window_days` of per-strategy scores for a stock
    from daily_signals. Returns {strategy: [scores...]} aligned by date.
    """
    cutoff = as_of_date - timedelta(days=window_days)
    scores: Dict[str, List[float]] = {s: [] for s in STRATEGIES}

    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT date, trend_score, momentum_score, reversion_score,
                       breakout_score, volume_score
                FROM daily_signals
                WHERE stock = %s
                  AND date BETWEEN %s AND %s
                ORDER BY date ASC
                """,
                (stock, cutoff, as_of_date),
            )
            rows = cursor.fetchall()
    except Exception as e:
        logger.warning(f"Could not fetch score history for {stock}: {e}")
        return scores

    for row in rows:
        scores["trend"].append(float(row["trend_score"] or 0))
        scores["momentum"].append(float(row["momentum_score"] or 0))
        scores["reversion"].append(float(row["reversion_score"] or 0))
        scores["breakout"].append(float(row["breakout_score"] or 0))
        scores["volume"].append(float(row["volume_score"] or 0))

    return scores


def compute_correlation_matrix(
    stock: str,
    as_of_date: Optional[date] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Computes the 5x5 Pearson correlation matrix between strategy scores
    over the rolling window. Returns None if insufficient history.

    Returns: {strategy_i: {strategy_j: correlation, ...}, ...}
    """
    as_of_date = as_of_date or date.today()
    scores = _fetch_strategy_score_history(stock, as_of_date)

    n_samples = len(scores["trend"])
    if n_samples < MIN_SAMPLES_REQUIRED:
        logger.debug(
            f"{stock}: only {n_samples} days of history "
            f"(need {MIN_SAMPLES_REQUIRED}) — skipping correlation calc"
        )
        return None

    # Build matrix as numpy array: rows = strategies, cols = days
    matrix_data = np.array([scores[s] for s in STRATEGIES])

    # Handle constant series (zero variance) which break correlation
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_matrix = np.corrcoef(matrix_data)

    # Replace NaN (from zero-variance strategies) with 0 correlation
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    result: Dict[str, Dict[str, float]] = {}
    for i, strat_i in enumerate(STRATEGIES):
        result[strat_i] = {}
        for j, strat_j in enumerate(STRATEGIES):
            result[strat_i][strat_j] = round(float(corr_matrix[i, j]), 4)

    return result


def save_correlation_matrix(
    stock: str,
    as_of_date: date,
    matrix: Dict[str, Dict[str, float]],
) -> None:
    """Upsert the correlation matrix into strategy_correlations table."""
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO strategy_correlations (stock, as_of_date, correlation_matrix_json)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (stock, as_of_date) DO UPDATE SET
                    correlation_matrix_json = EXCLUDED.correlation_matrix_json
                """,
                (stock, as_of_date, json.dumps(matrix)),
            )
    except Exception as e:
        logger.error(f"Failed to save correlation matrix for {stock}: {e}")


def get_correlation_matrix(
    stock: str,
    as_of_date: Optional[date] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Read the most recent stored correlation matrix for a stock."""
    as_of_date = as_of_date or date.today()
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT correlation_matrix_json FROM strategy_correlations
                WHERE stock = %s AND as_of_date <= %s
                ORDER BY as_of_date DESC
                LIMIT 1
                """,
                (stock, as_of_date),
            )
            row = cursor.fetchone()
            if row:
                return row["correlation_matrix_json"]
    except Exception as e:
        logger.warning(f"Could not fetch correlation matrix for {stock}: {e}")
    return None


# ── Step 2: Apply penalty in Signal Fusion ──────────────────────

def _dynamic_penalty(r: float) -> float:
    """
    Section 23.1 formula:
    CorrelationPenalty = 0.2 x (r - 0.7) / 0.3   for r > 0.7
                        = 0                       otherwise
    Scales linearly from 0 at r=0.7 to 0.2 at r=1.0.
    """
    if r <= CORRELATION_THRESHOLD:
        return 0.0
    r = min(r, 1.0)
    return MAX_PENALTY_AT_R1 * (r - CORRELATION_THRESHOLD) / (1.0 - CORRELATION_THRESHOLD)


def _static_cofire_penalty(strategy: str, active_strategies: set) -> float:
    """
    Section 8 static rule: Trend + Breakout firing together get a
    20% reduction each, since they often confirm the same breakout event.
    """
    penalty = 0.0
    for (a, b), p in STATIC_COFIRE_PAIRS.items():
        if strategy in (a, b):
            other = b if strategy == a else a
            if other in active_strategies:
                penalty = max(penalty, p)
    return penalty


def apply_correlation_penalty(
    stock: str,
    base_weights: Dict[str, float],
    active_strategies: Optional[set] = None,
    as_of_date: Optional[date] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Called by Signal Fusion Engine (Section 9) before final weighted sum.

    Args:
        stock:             symbol being scored
        base_weights:      regime.fusion_weights dict, e.g. {"trend": 0.30, ...}
                            (may also contain non-strategy keys like "risk" —
                            those pass through untouched)
        active_strategies: set of strategy_ids that fired (score != 0) today;
                            used for the static co-fire check. If None,
                            defaults to all keys present in base_weights.
        as_of_date:        date for matrix lookup

    Returns:
        (adjusted_weights, explanation_notes)
    """
    as_of_date = as_of_date or date.today()
    active_strategies = active_strategies or set(base_weights.keys())

    matrix = get_correlation_matrix(stock, as_of_date)
    adjusted = dict(base_weights)
    notes: List[str] = []

    for strat in base_weights:
        if strat not in STRATEGIES:
            continue   # e.g. "risk" weight key — not part of correlation matrix

        # ── Dynamic penalty from rolling correlation ────────────
        dynamic_pen = 0.0
        worst_pair  = None
        if matrix and strat in matrix:
            for other_strat, r in matrix[strat].items():
                if other_strat == strat or other_strat not in active_strategies:
                    continue
                pen = _dynamic_penalty(abs(r))
                if pen > dynamic_pen:
                    dynamic_pen = pen
                    worst_pair  = (other_strat, r)

        # ── Static co-firing penalty (Section 8) ────────────────
        static_pen = _static_cofire_penalty(strat, active_strategies)

        # ── Combine, capped at 50% total reduction ──────────────
        combined_pen = min(dynamic_pen + static_pen, MAX_COMBINED_PENALTY)

        if combined_pen > 0:
            original = adjusted[strat]
            adjusted[strat] = round(original * (1 - combined_pen), 4)
            note = f"{strat}: weight {original:.3f}→{adjusted[strat]:.3f} (-{combined_pen:.0%})"
            if worst_pair:
                note += f" [r={worst_pair[1]:.2f} vs {worst_pair[0]}]"
            if static_pen > 0:
                note += " [+static co-fire]"
            notes.append(note)

    # NOTE: no renormalization here. fuse() in app/fusion/engine.py already
    # normalizes the weighted sum by weight_used (sum of weights actually
    # applied across eligible strategies) — see the line
    # `if 0 < weight_used < 1.0: fused = fused / weight_used`.
    # Renormalizing here too would double-correct: it silently diluted a
    # computed 39% reduction down to ~24% effective, and inflated unrelated
    # weights (including "risk", which isn't a strategy weight at all but a
    # penalty subtractor in the final score — see Section 9) every time a
    # correlation penalty fired elsewhere. Returning the raw post-penalty
    # weights keeps the cap and the formula's effect intact and lets fuse()
    # own normalization in one place.
    return adjusted, notes


# ── Daily batch job ────────────────────────────────────────────

class CorrelationEngine:
    """
    Called by pipeline.py or a separate scheduled job (daily, post-close).
    Recomputes and stores the correlation matrix for all watchlist stocks.
    """

    def __init__(self, run_date: Optional[date] = None):
        self.run_date = run_date or date.today()

    def run(self, stocks: List[str]) -> Dict[str, bool]:
        """Returns {stock: success} for each stock processed."""
        results = {}
        for stock in stocks:
            matrix = compute_correlation_matrix(stock, self.run_date)
            if matrix:
                save_correlation_matrix(stock, self.run_date, matrix)
                results[stock] = True
                logger.debug(f"{stock}: correlation matrix updated")
            else:
                results[stock] = False

        computed = sum(1 for v in results.values() if v)
        logger.info(
            f"Correlation engine: {computed}/{len(stocks)} stocks "
            f"updated (others have insufficient history)"
        )
        return results