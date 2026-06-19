-- ============================================================
-- Migration 002: Health Alerts table (Section 24 Monitoring)
-- Run this against your existing DB to add the missing table.
-- ============================================================

CREATE TABLE IF NOT EXISTS health_alerts (
    id              BIGSERIAL PRIMARY KEY,
    run_date        DATE        NOT NULL,
    metric          VARCHAR(60) NOT NULL,    -- e.g. data_ingestion_sla, llm_failure_rate
    severity        VARCHAR(20) NOT NULL,    -- WARNING | CRITICAL
    message         TEXT        NOT NULL,
    value           NUMERIC,                  -- the metric value that triggered the alert
    threshold       NUMERIC,                  -- the threshold that was breached
    action          TEXT,                     -- remediation action taken/suggested
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_health_alerts_date     ON health_alerts (run_date DESC);
CREATE INDEX IF NOT EXISTS idx_health_alerts_severity ON health_alerts (severity);
CREATE INDEX IF NOT EXISTS idx_health_alerts_metric   ON health_alerts (metric);

COMMIT;