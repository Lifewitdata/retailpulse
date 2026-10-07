# RetailPulse - task runner. Every target wraps a verified entry point.
# Run `make help` for the list. All targets are safe to re-run (idempotent).

PY ?= python
PORT_API ?= 8010
PORT_DASH ?= 8501

.DEFAULT_GOAL := help

.PHONY: help install simulate validate analyze figures serve dashboard test lint docker-build docker-run clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	$(PY) -m pip install -r requirements.txt

simulate:  ## Generate raw users/events/assignments + simulation truth (seed 42)
	$(PY) -m src.simulate

validate:  ## Run the 39-check data-quality + assignment gate (non-blocking)
	$(PY) -m src.validate

analyze:  ## Full pipeline: simulate -> validate -> warehouse -> metrics -> experiments -> figures -> reports
	$(PY) -m src.analyze

figures:  ## Regenerate report figures only (assumes warehouse is built)
	$(PY) -m src.figures

serve:  ## Serve the FastAPI analytics readout on 127.0.0.1:$(PORT_API)
	$(PY) -m src.api --port $(PORT_API)

dashboard:  ## Serve the Streamlit dashboard on 127.0.0.1:$(PORT_DASH)
	$(PY) -m src.dashboard --port $(PORT_DASH)

test:  ## Run the pytest suite
	$(PY) -m pytest -q

lint:  ## Byte-compile all sources as a syntax gate
	$(PY) -m compileall -q src tests

docker-build:  ## Build the container image
	docker build -t retailpulse .

docker-run:  ## Run the container (builds the warehouse, then serves the API on $(PORT_API))
	docker run --rm -p $(PORT_API):8010 retailpulse

clean:  ## Remove generated artifacts + caches (keeps config, src, sql, tests)
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f data/retail.duckdb
	rm -rf data/raw/* data/interim/* data/processed/*
	rm -f reports/*.md reports/*.csv
	rm -rf reports/figures/*
	@echo "cleaned generated artifacts (data/, reports/figures, duckdb)"
