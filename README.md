# Stock Signal Platform

## Intelligent Multi-Timeframe Stock Signal Generation Platform

Production-ready decision-support system for Indian equities (NSE/BSE) that combines quantitative trading algorithms, market-regime detection, risk management, portfolio controls, and LLM-assisted explainability.

**Version:** 1.0.0
**Status:** Production-Ready Architecture  
**Market:** Indian Equity Markets (NSE/BSE)

---

## Overview

The Stock Signal Platform is an intelligent stock recommendation system designed to monitor Indian equities and generate actionable trade signals based on multiple quantitative trading strategies, market context, risk management rules, and AI-assisted reasoning.

The platform generates:

- Daily stock signals
- Weekly recommendations
- Monthly recommendations
- Long-term investment outlooks
- Portfolio-level insights
- Risk-adjusted position sizing suggestions

The goal is to assist investors and traders in making informed decisions while maintaining transparency and explainability.

---

## Key Features

### Multi-Strategy Signal Engine

The platform combines multiple trading strategies:

- Trend Following
- Momentum
- Mean Reversion
- Breakout Detection
- Volume Confirmation

Each strategy independently generates:

- Signal
- Confidence
- Score
- Explanation

---

### Regime Detection

Market conditions are classified into:

- Bull Market
- Bear Market
- Sideways Market

Strategy weights automatically adjust based on the detected regime.

---

### Signal Fusion Engine

Signals from multiple strategies are combined using dynamic weighting:

```text
Final Score =
Trend Weight × Trend Score
+ Momentum Weight × Momentum Score
+ Mean Reversion Weight × Mean Reversion Score
+ Breakout Weight × Breakout Score
+ Volume Weight × Volume Score
- Risk Penalty
```

Signal Mapping:

| Score Range | Signal |
|------------|---------|
| 80-100 | Strong Buy |
| 65-79 | Buy |
| 45-64 | Hold |
| 30-44 | Sell |
| <30 | Strong Sell |

---

### Risk Management

Built-in risk controls include:

- ATR-based volatility checks
- Portfolio drawdown controls
- Liquidity filtering
- Position sizing
- Sector concentration limits
- Maximum portfolio risk limits

---

### Portfolio Intelligence

Version 3.1 introduces:

- Position sizing
- Trade lifecycle management
- Sector exposure monitoring
- Correlation-aware signal selection
- Portfolio-level backtesting

---

### LLM-Assisted Explainability

The LLM does **not** directly influence signal scores.

Instead it provides:

- Context awareness
- Risk identification
- Event detection
- Signal conflict resolution
- Human-readable explanations

Supported providers:

- Groq (Primary)
- OpenAI (Fallback)

---

### Market Microstructure Awareness

The platform models real Indian market conditions:

- NSE holidays
- Muhurat trading sessions
- Circuit breakers
- Overnight gaps
- FII/DII flows
- Promoter pledge risks
- Macro events
- Signal expiry

---

### Execution Realism

Signals are adjusted for:

- Slippage
- Impact costs
- Liquidity constraints
- Realistic entry prices
- Realistic stop losses

This ensures backtesting and live recommendations remain practical.

---

## High-Level Architecture

```text
+----------------------+
| Data Ingestion Layer |
+----------------------+
           |
           v
+----------------------+
| Feature Engineering  |
+----------------------+
           |
           v
+----------------------+
| Strategy Engines     |
+----------------------+
           |
           v
+----------------------+
| Regime Detection     |
+----------------------+
           |
           v
+----------------------+
| Signal Fusion Engine |
+----------------------+
           |
           v
+----------------------+
| Risk Engine          |
+----------------------+
           |
           v
+----------------------+
| LLM Override Layer   |
+----------------------+
           |
           v
+-----------------------------+
| Market Microstructure Layer |
+-----------------------------+
           |
           v
+----------------------+
| Execution Realism    |
+----------------------+
           |
           v
+----------------------+
| Portfolio Management |
+----------------------+
           |
           v
+----------------------+
| Notifications        |
+----------------------+
```

---

## Technology Stack

| Component | Technology |
|------------|------------|
| Backend API | FastAPI |
| Scheduler | APScheduler |
| Database | PostgreSQL |
| Cache | Redis |
| Dashboard | Streamlit |
| Vector Database | Qdrant |
| LLM Provider | Groq |
| ML Models | LightGBM / XGBoost |
| Deployment | Docker Compose |
| Cloud | Azure / AWS |

---

## Data Sources

| Source | Purpose |
|----------|----------|
| NSE Python | OHLC and Market Data |
| yfinance | Historical Data |
| Screener.in | Fundamentals |
| Zerodha Kite | Real-Time Data |
| NewsAPI | News Headlines |
| RBI Data | Macro Indicators |
| MOSPI | Economic Indicators |

---

## Project Structure

```text
stock-signal-platform/
│
├── app/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── strategies/
│   ├── regime/
│   ├── fusion/
│   ├── risk/
│   ├── portfolio/
│   ├── llm/
│   ├── notifications/
│   └── backtesting/
│
├── streamlit_app/
│
├── sql/
│
├── tests/
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-org/stock-signal-platform.git

cd stock-signal-platform
```

### Create Virtual Environment

```bash
python -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/stocksignal

REDIS_URL=redis://localhost:6379/0

GROQ_API_KEY=your_key

OPENAI_API_KEY=your_key

NEWS_API_KEY=your_key

TELEGRAM_BOT_TOKEN=your_token

TELEGRAM_CHAT_ID=your_chat_id
```

---

## Running Infrastructure

Start supporting services:

```bash
docker compose up -d
```

---

## Run Database Migrations

```bash
psql $DATABASE_URL -f sql/schema.sql
```

---

## Start API

```bash
uvicorn app.main:app --reload
```

---

## Start Dashboard

```bash
streamlit run streamlit_app/app.py
```

---

## Run Daily Signal Pipeline

```bash
python -m app.jobs.daily_pipeline
```

Pipeline steps:

1. Fetch market data
2. Compute indicators
3. Detect market regime
4. Run strategy engines
5. Fuse strategy outputs
6. Apply risk checks
7. Execute LLM validation
8. Generate signals
9. Store results
10. Send notifications

---

## Example Signal Output

```text
STRONG BUY SIGNAL

Stock: RELIANCE.NS

Regime: Bull

Confidence: 83%

Theoretical Entry: ₹2,845
Realistic Entry: ₹2,848

Target Price: ₹3,050

Stop Loss: ₹2,757

Reason:
Strong trend alignment (EMA20 > EMA50 > EMA200),
ADX > 25,
Breakout above resistance,
Volume 2.3x average,
No earnings event within 10 days.
```

---

## Database Tables

### market_data

Stores:

- OHLC
- Volume
- Adjusted Close
- Source Information

### daily_signals

Stores:

- Signal
- Confidence
- Entry Price
- Exit Price
- Stop Loss
- Regime
- LLM Override
- Explanation

### signal_history

Stores:

- Historical Outcomes
- Returns
- Holding Period

### backtest_results

Stores:

- Sharpe Ratio
- Drawdown
- Win Rate
- Annual Returns

### trades

Stores:

- Trade Lifecycle
- Entry / Exit
- PnL
- Status

---

## Backtesting Framework

Features:

- Walk-forward validation
- Corporate-action adjusted data
- Regime segmentation
- Sector segmentation
- Market-cap segmentation
- Portfolio simulation
- Realistic execution costs

Acceptance Criteria:

| Metric | Requirement |
|----------|-------------|
| Sharpe Ratio | > 1.0 |
| Win Rate | > 45% |
| Max Drawdown | < 20% |
| Annual Return | Beat Nifty 50 |

---

## Monitoring & Observability

Tracked metrics:

- Data ingestion SLA
- LLM API health
- Pipeline latency
- Strategy win rate
- Portfolio Sharpe ratio
- Drawdown
- Sector exposure

Dashboard includes:

- Performance tracking
- Signal history
- Portfolio allocation
- Risk metrics

---

## Roadmap

### Version 1

- Daily signals
- Weekly recommendations
- Monthly recommendations
- Telegram alerts
- Streamlit dashboard

### Version 2

- React dashboard
- Kafka integration
- Intraday trading support
- ML Meta Model

### Version 3

- Portfolio optimization
- Kubernetes deployment
- Auto rebalancing

### Version 4

- Autonomous investment agent
- Multi-asset support
- Reinforcement learning

---

## Disclaimer

This project is intended for educational and research purposes only.

It does not constitute financial advice, investment advice, trading advice, or any other form of professional recommendation.

Always perform your own due diligence before making investment decisions.

---

## License

MIT License
