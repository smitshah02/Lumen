SHELL := /bin/bash
PYTHON ?= python3.12
VENV := .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

.PHONY: help setup check-python install compose-check db-up schema models models-verify \
	demo-init test check-generated smoke-demo doctor validate

help:
	@echo "Lumen lifecycle targets"
	@echo "  setup          create the CPython 3.12 venv and install pinned dependencies"
	@echo "  compose-check  render research + demo Compose without starting services"
	@echo "  db-up           start the explicitly selected research Postgres volume"
	@echo "  schema          apply additive versioned schema upgrades"
	@echo "  models          fetch/verify pinned runtime retrieval weights"
	@echo "  models-verify   verify local runtime weights without network"
	@echo "  demo-init       initialize the synthetic Docker demo (explicit; downloads models)"
	@echo "  test            run the complete unit/regression suite"
	@echo "  smoke-demo      run demo safety + retrieval smoke checks against live services"
	@echo "  doctor          comprehensive read-only readiness report"
	@echo "  validate        static checks + tests; no services, downloads, or ingestion"

check-python:
	@$(PYTHON) -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

setup: check-python
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(VENV_PY) -c 'import sys; assert sys.version_info[:2] == (3, 12), "existing .venv is not CPython 3.12; recreate it deliberately"'
	$(VENV_PIP) install -r requirements.txt -r requirements-dev.txt
	$(VENV_PY) -m pip check

install: setup

compose-check:
	LUMEN_PG_PASSWORD=compose-check-only LUMEN_PG_VOLUME=lumen-compose-check-only docker compose config --quiet
	LUMEN_LLM_MAIN=compose-check-only LUMEN_LLM_FAST=compose-check-only docker compose -f docker-compose.demo.yml config --quiet

db-up: compose-check
	docker compose up -d --wait db

schema:
	$(VENV_PY) -m src.storage.schema --upgrade

models:
	$(VENV_PY) scripts/fetch_models.py --profile runtime

models-verify:
	$(VENV_PY) scripts/fetch_models.py --profile runtime --verify

demo-init:
	LUMEN_DATA_PLANE=demo scripts/init_demo_stack.sh

test:
	$(VENV_PY) -m pytest -q -p no:rerunfailures

check-generated:
	$(VENV_PY) -m src.demo_data.generate --check
	$(VENV_PY) -m src.demo_data.generate --validate

smoke-demo:
	scripts/lumen demo test

doctor:
	scripts/lumen research doctor

validate: compose-check check-generated test
	@echo "static validation complete; no service was started and no data was ingested"
