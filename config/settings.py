"""
Central settings — loaded once at startup.
All values sourced from environment / .env file.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── PostgreSQL ───────────────────────────────────────────
    postgres_db: str = "stock_signals"
    postgres_user: str = "ssp_user"
    postgres_password: str = "change_me"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ────────────────────────────────────────────────
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = "change_me"

    @property
    def redis_url(self) -> str:
        return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"

    # ── Qdrant ───────────────────────────────────────────────
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "stock_knowledge"
    qdrant_vector_size: int = 384        # all-MiniLM-L6-v2 output dim

    # ── LLM ─────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout_seconds: int = 10
    openai_api_key: str = ""
    google_api_key: str = ""

    # ── Notifications ────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    resend_api_key: str = ""
    notification_email_to: str = ""

    # ── Data Sources ─────────────────────────────────────────
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_access_token: str = ""
    news_api_key: str = ""

    # ── Platform Config ──────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"
    pipeline_run_time: str = "15:45"
    signal_expiry_trading_days: int = 1
    max_llm_override_rate: float = 0.30

    # ── Watchlist ────────────────────────────────────────────
    watchlist: List[str] = Field(default=[
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    ])

    # ── Risk Config ──────────────────────────────────────────
    portfolio_value_inr: float = 1_000_000.0
    risk_per_trade_pct: float = 0.015
    max_single_stock_pct: float = 0.15
    max_open_positions: int = 8
    max_sector_exposure_pct: float = 0.30
    adv_order_cap_pct: float = 0.05

    # ── Backtesting ──────────────────────────────────────────
    backtest_train_years: int = 2
    backtest_walk_forward_months: int = 3
    backtest_min_history_years: int = 3
    backtest_min_sharpe: float = 1.0
    backtest_max_drawdown: float = 0.20
    backtest_min_win_rate: float = 0.45

    @field_validator("environment")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "production", "test"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def kite_enabled(self) -> bool:
        return bool(self.kite_api_key and self.kite_access_token)

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
