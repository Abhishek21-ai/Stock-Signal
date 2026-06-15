-- ============================================================
-- Stock Signal Platform — Initial Schema
-- Design Doc v3.1 | All tables from Sections 5.4, 22.1, 23.1
-- ============================================================

-- ── Extensions ───────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- text search on explanations

-- ── Enums ────────────────────────────────────────────────────

CREATE TYPE signal_type AS ENUM (
    'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL',
    'EXPIRED', 'UNAVAILABLE', 'CIRCUIT_HIT', 'STOP_INVALIDATED'
);

CREATE TYPE regime_type AS ENUM ('BULL', 'BEAR', 'SIDEWAYS', 'UNCERTAIN');

CREATE TYPE llm_override_type AS ENUM ('NONE', 'VETO', 'REDUCE_CONFIDENCE', 'TIMEOUT', 'FALLBACK');

CREATE TYPE trade_status AS ENUM ('PENDING', 'ACTIVE', 'CLOSED');

CREATE TYPE exit_reason AS ENUM (
    'TARGET', 'TARGET_PARTIAL', 'STOP', 'EXPIRY', 'MANUAL', 'CIRCUIT'
);

CREATE TYPE data_source_type AS ENUM ('KITE', 'YFINANCE', 'NSE_PYTHON', 'STALE_WARNING');

-- ── market_data ───────────────────────────────────────────────
-- Section 5.4: raw adjusted OHLCV (yfinance auto_adjust=True)
CREATE TABLE market_data (
    id              BIGSERIAL PRIMARY KEY,
    date            DATE        NOT NULL,
    stock           VARCHAR(20) NOT NULL,
    open            NUMERIC(12,4) NOT NULL,
    high            NUMERIC(12,4) NOT NULL,
    low             NUMERIC(12,4) NOT NULL,
    close           NUMERIC(12,4) NOT NULL,
    volume          BIGINT      NOT NULL,
    adjusted_close  NUMERIC(12,4) NOT NULL,
    data_source     data_source_type NOT NULL DEFAULT 'YFINANCE',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, stock)
);

CREATE INDEX idx_market_data_stock_date ON market_data (stock, date DESC);

-- ── regime_snapshots ─────────────────────────────────────────
-- Section 7: daily market regime + FII/DII context
CREATE TABLE regime_snapshots (
    id                  BIGSERIAL PRIMARY KEY,
    date                DATE NOT NULL UNIQUE,
    regime              regime_type NOT NULL,
    nifty_close         NUMERIC(10,2),
    nifty_ema20         NUMERIC(10,2),
    nifty_ema50         NUMERIC(10,2),
    nifty_ema200        NUMERIC(10,2),
    nifty_adx           NUMERIC(6,2),
    fii_net_crore       NUMERIC(12,2),     -- Section 19.4
    dii_net_crore       NUMERIC(12,2),
    rolling_5d_fii_sum  NUMERIC(14,2),
    regime_confidence   VARCHAR(20),        -- NORMAL | UNCERTAIN (FII-adjusted)
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── daily_signals ─────────────────────────────────────────────
-- Section 5.4 + 19.7 + 20.4: core signal output table
CREATE TABLE daily_signals (
    id                      BIGSERIAL PRIMARY KEY,
    signal_uuid             UUID NOT NULL DEFAULT uuid_generate_v4(),
    date                    DATE NOT NULL,
    stock                   VARCHAR(20) NOT NULL,
    sector                  VARCHAR(60),

    -- Signal output
    signal                  signal_type NOT NULL,
    quant_score             NUMERIC(5,2),        -- 0–100
    confidence_pct          NUMERIC(5,2),        -- raw (uncalibrated) 0–100
    calibrated_probability  NUMERIC(5,4),        -- post-calibration (Section 25)

    -- Strategy sub-scores
    trend_score             NUMERIC(5,2),
    momentum_score          NUMERIC(5,2),
    reversion_score         NUMERIC(5,2),
    breakout_score          NUMERIC(5,2),
    volume_score            NUMERIC(5,2),
    risk_penalty            NUMERIC(5,2),

    -- Prices — theoretical (Section 9) and realistic (Section 20)
    entry_price_theoretical     NUMERIC(10,2),
    entry_price_realistic       NUMERIC(10,2),
    exit_target_theoretical     NUMERIC(10,2),
    exit_target_realistic       NUMERIC(10,2),
    stop_loss_theoretical       NUMERIC(10,2),
    stop_loss_realistic         NUMERIC(10,2),
    slippage_factor_pct         NUMERIC(6,4),
    impact_cost_pct             NUMERIC(6,4),

    -- Position sizing (Section 21)
    position_size_shares    INTEGER,
    position_value_inr      NUMERIC(14,2),

    -- Regime + LLM context
    regime                  regime_type,
    llm_override            llm_override_type NOT NULL DEFAULT 'NONE',
    llm_status              VARCHAR(20),         -- OK | TIMEOUT | FALLBACK | ERROR
    llm_explanation         TEXT,

    -- Market microstructure flags (Section 19)
    circuit_hit             BOOLEAN NOT NULL DEFAULT FALSE,
    gap_open_type           VARCHAR(30),         -- NULL | CHASE_RISK | STOP_INVALIDATED
    macro_window_active     BOOLEAN NOT NULL DEFAULT FALSE,
    macro_event_type        VARCHAR(60),
    liquidity_warning       BOOLEAN NOT NULL DEFAULT FALSE,
    promoter_pledge_pct     NUMERIC(5,2),
    pledge_risk_flag        VARCHAR(20),         -- LOW | HIGH | CRITICAL

    -- Signal expiry (Section 19.7)
    valid_until             DATE NOT NULL,
    is_expired              BOOLEAN NOT NULL DEFAULT FALSE,

    -- Portfolio (Section 21.3)
    rejected_reason         VARCHAR(40),         -- NULL | PORTFOLIO_FULL | SECTOR_CAP

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (date, stock)
);

CREATE INDEX idx_daily_signals_date        ON daily_signals (date DESC);
CREATE INDEX idx_daily_signals_stock_date  ON daily_signals (stock, date DESC);
CREATE INDEX idx_daily_signals_signal      ON daily_signals (signal);
CREATE INDEX idx_daily_signals_expired     ON daily_signals (is_expired, date DESC);

-- ── signal_history ────────────────────────────────────────────
-- Section 5.4: outcome tracking — feeds calibration + meta-model
CREATE TABLE signal_history (
    id              BIGSERIAL PRIMARY KEY,
    signal_id       BIGINT REFERENCES daily_signals(id) ON DELETE CASCADE,
    stock           VARCHAR(20) NOT NULL,
    signal_date     DATE NOT NULL,
    signal          signal_type NOT NULL,
    regime_at_signal regime_type,
    sector          VARCHAR(60),

    -- Outcome
    outcome_direction   VARCHAR(10),    -- UP | DOWN | FLAT
    return_pct          NUMERIC(8,4),   -- actual return if held
    duration_days       INTEGER,
    was_correct         BOOLEAN,        -- signal direction matched outcome

    resolved_at         DATE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_signal_history_stock     ON signal_history (stock, signal_date DESC);
CREATE INDEX idx_signal_history_regime    ON signal_history (regime_at_signal);
CREATE INDEX idx_signal_history_correct   ON signal_history (was_correct);

-- ── backtest_results ──────────────────────────────────────────
-- Section 12 + 26: per-strategy, per-regime, per-segment results
CREATE TABLE backtest_results (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID NOT NULL DEFAULT uuid_generate_v4(),
    strategy_id     VARCHAR(40) NOT NULL,        -- trend | momentum | reversion | breakout | volume | combined
    period_start    DATE NOT NULL,
    period_end      DATE NOT NULL,
    regime          regime_type,                  -- NULL = aggregate
    sector          VARCHAR(60),                  -- Section 26.1 segmentation
    market_cap_tier VARCHAR(20),                  -- LARGE | MID | SMALL

    -- Core metrics (Section 12.3)
    sharpe_ratio        NUMERIC(6,3),
    max_drawdown_pct    NUMERIC(6,3),
    win_rate_pct        NUMERIC(6,3),
    annualized_return_pct NUMERIC(8,3),
    total_trades        INTEGER,
    avg_return_pct      NUMERIC(8,4),

    -- Execution-realistic versions (Section 20.3)
    sharpe_realistic    NUMERIC(6,3),
    win_rate_realistic  NUMERIC(6,3),
    annualized_return_realistic NUMERIC(8,3),

    -- Pass/fail (Section 12.3)
    meets_acceptance_criteria BOOLEAN,
    notes               TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_backtest_strategy ON backtest_results (strategy_id, regime);

-- ── trades ────────────────────────────────────────────────────
-- Section 22: active trade tracking with state machine
CREATE TABLE trades (
    id                          BIGSERIAL PRIMARY KEY,
    trade_uuid                  UUID NOT NULL DEFAULT uuid_generate_v4(),
    signal_id                   BIGINT REFERENCES daily_signals(id),
    stock                       VARCHAR(20) NOT NULL,

    -- Entry prices
    entry_price_theoretical     NUMERIC(10,2),
    entry_price_realistic       NUMERIC(10,2),
    entry_price_actual          NUMERIC(10,2),   -- what user actually paid

    -- Exit prices
    stop_loss_theoretical       NUMERIC(10,2),
    stop_loss_realistic         NUMERIC(10,2),
    target_price                NUMERIC(10,2),
    exit_price                  NUMERIC(10,2),
    exit_reason                 exit_reason,

    -- P&L
    pnl_pct                     NUMERIC(8,4),
    pnl_inr                     NUMERIC(14,2),

    -- Position
    position_size_shares        INTEGER,
    position_value_inr          NUMERIC(14,2),
    position_remaining_shares   INTEGER,         -- Section 22.2 partial exits (V2)

    -- Context
    regime_at_entry             regime_type,
    duration_days               INTEGER,
    status                      trade_status NOT NULL DEFAULT 'PENDING',

    signal_date                 DATE NOT NULL,
    entry_date                  DATE,
    exit_date                   DATE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trades_stock_status ON trades (stock, status);
CREATE INDEX idx_trades_status       ON trades (status);
CREATE INDEX idx_trades_signal_id    ON trades (signal_id);

-- ── strategy_correlations ────────────────────────────────────
-- Section 23: rolling 60-day correlation matrix per stock
CREATE TABLE strategy_correlations (
    id                      BIGSERIAL PRIMARY KEY,
    stock                   VARCHAR(20) NOT NULL,
    as_of_date              DATE NOT NULL,
    correlation_matrix_json JSONB NOT NULL,       -- 5x5 matrix as JSON
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (stock, as_of_date)
);

-- ── fii_dii_flows ─────────────────────────────────────────────
-- Section 19.4: scraped from NSE daily post-close
CREATE TABLE fii_dii_flows (
    id                  BIGSERIAL PRIMARY KEY,
    date                DATE NOT NULL UNIQUE,
    fii_net_crore       NUMERIC(12,2),
    dii_net_crore       NUMERIC(12,2),
    rolling_5d_fii_sum  NUMERIC(14,2),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── promoter_data ────────────────────────────────────────────
-- Section 19.5: quarterly pledging data from Screener.in
CREATE TABLE promoter_data (
    id                      BIGSERIAL PRIMARY KEY,
    stock                   VARCHAR(20) NOT NULL,
    quarter                 VARCHAR(10) NOT NULL,    -- e.g. "Q1FY26"
    promoter_holding_pct    NUMERIC(5,2),
    pledged_pct             NUMERIC(5,2),
    source                  VARCHAR(40) DEFAULT 'SCREENER',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (stock, quarter)
);

CREATE INDEX idx_promoter_stock ON promoter_data (stock, quarter DESC);

-- ── macro_events ─────────────────────────────────────────────
-- Section 19.6: structured calendar of high-impact events
CREATE TABLE macro_events (
    id              BIGSERIAL PRIMARY KEY,
    event_date      DATE NOT NULL,
    event_type      VARCHAR(60) NOT NULL,    -- RBI_MPC | BUDGET | FED_DECISION | FNO_EXPIRY | CPI | IIP
    event_name      VARCHAR(120),
    expected_impact VARCHAR(10) NOT NULL DEFAULT 'HIGH',   -- HIGH | MEDIUM | LOW
    is_special_session BOOLEAN NOT NULL DEFAULT FALSE,     -- Muhurat trading
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_macro_events_date ON macro_events (event_date);

-- ── nse_holiday_calendar ─────────────────────────────────────
-- Section 19.3: NSE trading holidays — checked before every run
CREATE TABLE nse_holiday_calendar (
    id              BIGSERIAL PRIMARY KEY,
    holiday_date    DATE NOT NULL UNIQUE,
    holiday_name    VARCHAR(120),
    year            INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── pipeline_runs ─────────────────────────────────────────────
-- Section 24: observability — one row per daily pipeline execution
CREATE TABLE pipeline_runs (
    id                  BIGSERIAL PRIMARY KEY,
    run_uuid            UUID NOT NULL DEFAULT uuid_generate_v4(),
    run_date            DATE NOT NULL UNIQUE,
    started_at          TIMESTAMPTZ NOT NULL,
    completed_at        TIMESTAMPTZ,

    -- Stage timings (ms)
    ingestion_ms        INTEGER,
    features_ms         INTEGER,
    regime_ms           INTEGER,
    strategies_ms       INTEGER,
    fusion_ms           INTEGER,
    llm_ms              INTEGER,
    notifications_ms    INTEGER,
    total_ms            INTEGER,

    -- Outcomes
    stocks_processed    INTEGER,
    stocks_skipped      INTEGER,
    signals_generated   INTEGER,
    llm_overrides       INTEGER,
    llm_timeouts        INTEGER,
    data_sla_breaches   TEXT[],       -- stocks that missed SLA

    status              VARCHAR(20),   -- SUCCESS | PARTIAL | FAILED | MARKET_CLOSED
    error_message       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── weekly_signals ────────────────────────────────────────────
-- Section 9.3 + 13.1: weekly rollup from daily signals
CREATE TABLE weekly_signals (
    id                  BIGSERIAL PRIMARY KEY,
    week_start          DATE NOT NULL,
    week_end            DATE NOT NULL,
    stock               VARCHAR(20) NOT NULL,
    signal              signal_type NOT NULL,
    buy_days            INTEGER,
    sell_days           INTEGER,
    hold_days           INTEGER,
    avg_confidence_pct  NUMERIC(5,2),
    score_std_dev       NUMERIC(5,2),     -- uncertainty indicator
    valid_until         DATE NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (week_start, stock)
);

-- ── monthly_signals ───────────────────────────────────────────
-- Section 13.2 + 13.3: monthly and long-term outlook
CREATE TABLE monthly_signals (
    id                  BIGSERIAL PRIMARY KEY,
    month               DATE NOT NULL,    -- first day of month
    stock               VARCHAR(20) NOT NULL,
    signal              signal_type NOT NULL,
    buy_weeks           INTEGER,
    sell_weeks          INTEGER,
    avg_confidence_pct  NUMERIC(5,2),
    long_term_outlook   VARCHAR(20),      -- ACCUMULATE | WATCHLIST | AVOID | NULL
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (month, stock)
);

-- ── Triggers ─────────────────────────────────────────────────

-- Auto-update trades.updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trades_updated_at
    BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Auto-mark daily_signals as expired
CREATE OR REPLACE FUNCTION mark_expired_signals()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.valid_until < CURRENT_DATE THEN
        NEW.is_expired = TRUE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER check_signal_expiry
    BEFORE INSERT OR UPDATE ON daily_signals
    FOR EACH ROW EXECUTE FUNCTION mark_expired_signals();

-- ── Seed: NSE holidays FY2026 ─────────────────────────────────
INSERT INTO nse_holiday_calendar (holiday_date, holiday_name, year) VALUES
    ('2026-01-26', 'Republic Day', 2026),
    ('2026-02-26', 'Mahashivratri', 2026),
    ('2026-03-02', 'Holi', 2026),
    ('2026-03-25', 'Holi (2nd Day)', 2026),
    ('2026-04-02', 'Ram Navami', 2026),
    ('2026-04-03', 'Good Friday', 2026),
    ('2026-04-14', 'Dr. Ambedkar Jayanti', 2026),
    ('2026-05-01', 'Maharashtra Day', 2026),
    ('2026-08-15', 'Independence Day', 2026),
    ('2026-08-27', 'Ganesh Chaturthi', 2026),
    ('2026-10-02', 'Gandhi Jayanti / Dussehra', 2026),
    ('2026-10-21', 'Diwali Laxmi Pujan', 2026),
    ('2026-10-22', 'Diwali Balipratipada', 2026),
    ('2026-11-05', 'Guru Nanak Jayanti', 2026),
    ('2026-12-25', 'Christmas', 2026)
ON CONFLICT (holiday_date) DO NOTHING;

-- ── Seed: macro events 2026 ───────────────────────────────────
-- F&O expiry = last Thursday of each month
INSERT INTO macro_events (event_date, event_type, event_name, expected_impact) VALUES
    ('2026-01-29', 'FNO_EXPIRY', 'Jan 2026 F&O Expiry', 'HIGH'),
    ('2026-02-26', 'FNO_EXPIRY', 'Feb 2026 F&O Expiry', 'HIGH'),
    ('2026-03-26', 'FNO_EXPIRY', 'Mar 2026 F&O Expiry', 'HIGH'),
    ('2026-04-30', 'FNO_EXPIRY', 'Apr 2026 F&O Expiry', 'HIGH'),
    ('2026-05-28', 'FNO_EXPIRY', 'May 2026 F&O Expiry', 'HIGH'),
    ('2026-06-25', 'FNO_EXPIRY', 'Jun 2026 F&O Expiry', 'HIGH'),
    ('2026-07-30', 'FNO_EXPIRY', 'Jul 2026 F&O Expiry', 'HIGH'),
    ('2026-08-27', 'FNO_EXPIRY', 'Aug 2026 F&O Expiry', 'HIGH'),
    ('2026-09-24', 'FNO_EXPIRY', 'Sep 2026 F&O Expiry', 'HIGH'),
    ('2026-10-29', 'FNO_EXPIRY', 'Oct 2026 F&O Expiry', 'HIGH'),
    ('2026-11-26', 'FNO_EXPIRY', 'Nov 2026 F&O Expiry', 'HIGH'),
    ('2026-12-31', 'FNO_EXPIRY', 'Dec 2026 F&O Expiry', 'HIGH'),
    -- RBI MPC meetings (approximate — update from RBI calendar)
    ('2026-02-07', 'RBI_MPC', 'RBI MPC Decision Feb 2026', 'HIGH'),
    ('2026-04-09', 'RBI_MPC', 'RBI MPC Decision Apr 2026', 'HIGH'),
    ('2026-06-06', 'RBI_MPC', 'RBI MPC Decision Jun 2026', 'HIGH'),
    ('2026-08-06', 'RBI_MPC', 'RBI MPC Decision Aug 2026', 'HIGH'),
    ('2026-10-08', 'RBI_MPC', 'RBI MPC Decision Oct 2026', 'HIGH'),
    ('2026-12-05', 'RBI_MPC', 'RBI MPC Decision Dec 2026', 'HIGH')
ON CONFLICT DO NOTHING;

COMMIT;
