-- ============================================================
-- Migration 002: health_alerts table — Section 24.1
-- Stores system health monitoring alerts for dashboard consumption.
-- ============================================================

CREATE TABLE IF NOT EXISTS health_alerts (
    id              BIGSERIAL PRIMARY KEY,
    run_date        DATE NOT NULL,
    metric          VARCHAR(50)  NOT NULL,
    severity        VARCHAR(20)  NOT NULL,   -- WARNING | CRITICAL
    message         TEXT         NOT NULL,
    value           NUMERIC(10,4),
    threshold       NUMERIC(10,4),
    action          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_health_alerts_date     ON health_alerts (run_date DESC);
CREATE INDEX IF NOT EXISTS idx_health_alerts_severity ON health_alerts (severity);
CREATE INDEX IF NOT EXISTS idx_health_alerts_metric   ON health_alerts (metric);
