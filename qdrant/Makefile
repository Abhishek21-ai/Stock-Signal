# ── Stock Signal Platform — Developer Commands ─────────────────

.PHONY: help up down logs shell db-shell migrate seed test lint

help:
	@echo ""
	@echo "  make up          Start all services (postgres, redis, qdrant, app, streamlit)"
	@echo "  make down         Stop all services"
	@echo "  make logs         Tail app logs"
	@echo "  make shell        Shell into app container"
	@echo "  make db-shell     psql into postgres"
	@echo "  make migrate      Run SQL migrations manually"
	@echo "  make run-pipeline Trigger pipeline now (bypass scheduler)"
	@echo "  make test         Run test suite"
	@echo "  make lint         Run ruff linter"
	@echo ""

up:
	@cp -n .env.example .env 2>/dev/null || true
	docker compose up -d --build
	@echo "✅ Services up | Dashboard: http://localhost:8501 | DB: localhost:5432"

down:
	docker compose down

logs:
	docker compose logs -f app

shell:
	docker compose exec app bash

db-shell:
	docker compose exec postgres psql -U $$(grep POSTGRES_USER .env | cut -d= -f2) -d $$(grep POSTGRES_DB .env | cut -d= -f2)

migrate:
	docker compose exec postgres psql \
		-U $$(grep POSTGRES_USER .env | cut -d= -f2) \
		-d $$(grep POSTGRES_DB .env | cut -d= -f2) \
		-f /docker-entrypoint-initdb.d/001_initial_schema.sql

run-pipeline:
	docker compose exec app python -c "import asyncio; from app.pipeline import DailyPipeline; asyncio.run(DailyPipeline().run())"

test:
	docker compose exec app pytest tests/ -v

lint:
	docker compose exec app ruff check app/ config/
