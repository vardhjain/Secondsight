# Makefile for person-reid-market1501.
#
# Uses `uv` (https://github.com/astral-sh/uv) as the package manager / runner.
# Override any variable on the command line, e.g.:
#   make train CONFIG=configs/default.yaml DATA_ROOT=/path/to/Market-1501-v15.09.15
#
# Common targets:
#   make install        Create the environment and install the package (+ dev deps).
#   make lint           Run ruff lint checks.
#   make format         Auto-format the codebase with ruff.
#   make test           Run the test suite.
#   make train          Train the strong baseline.
#   make eval           Evaluate a trained checkpoint.
#   make demo           Launch the Gradio demo.
#   make docker-build   Build the Docker image.

UV ?= uv
PYTHON ?= python
CONFIG ?= configs/market1501_strong_baseline.yaml
DATA_ROOT ?=
WEIGHTS ?= outputs/best.pth
OUTPUT_DIR ?= outputs
DEVICE ?= cuda
IMAGE ?= person-reid-market1501:latest

.DEFAULT_GOAL := help
.PHONY: help install install-dev sync lint format format-check typecheck test test-cov \
        precommit download train eval demo clean docker-build docker-run

help: ## Show this help message.
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create the environment and install the package with dev extras.
	$(UV) sync --extra dev

install-dev: install ## Alias for `install` (package + dev tooling).

sync: ## Re-sync the environment from pyproject (dev extras included).
	$(UV) sync --extra dev

lint: ## Run ruff lint checks.
	$(UV) run ruff check .

format: ## Auto-format and apply lint fixes with ruff.
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

format-check: ## Verify formatting without modifying files (CI-friendly).
	$(UV) run ruff format --check .
	$(UV) run ruff check .

typecheck: ## Run mypy static type checks.
	$(UV) run mypy

test: ## Run the test suite.
	$(UV) run pytest

test-cov: ## Run the test suite with coverage reporting.
	$(UV) run pytest --cov=reid --cov-report=term-missing

precommit: ## Run all pre-commit hooks across the repository.
	$(UV) run pre-commit run --all-files

download: ## Download the Market-1501 dataset via kagglehub.
	$(UV) run python scripts/download_data.py

train: ## Train the model (override CONFIG, DATA_ROOT, OUTPUT_DIR, DEVICE).
	$(UV) run python scripts/train.py --config $(CONFIG) \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),) \
		--output-dir $(OUTPUT_DIR) --device $(DEVICE)

eval: ## Evaluate a trained checkpoint (override WEIGHTS, CONFIG, DATA_ROOT).
	$(UV) run python scripts/evaluate.py --config $(CONFIG) --weights $(WEIGHTS) \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),) --device $(DEVICE)

demo: ## Launch the interactive Gradio demo (override WEIGHTS).
	$(UV) run python app/gradio_app.py --config $(CONFIG) --weights $(WEIGHTS)

clean: ## Remove caches and build artifacts (cross-platform).
	$(UV) run python -c "import shutil,pathlib; [shutil.rmtree(p,ignore_errors=True) for p in ['build','dist','.pytest_cache','.ruff_cache','.mypy_cache','htmlcov']]; [shutil.rmtree(d,ignore_errors=True) for d in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(d,ignore_errors=True) for d in pathlib.Path('.').glob('*.egg-info')]; pathlib.Path('.coverage').unlink(missing_ok=True)"

docker-build: ## Build the Docker image.
	docker build -t $(IMAGE) .

docker-run: ## Run the Gradio demo inside Docker (mounts ./outputs).
	docker run --rm -it -p 7860:7860 -v $(PWD)/outputs:/app/outputs $(IMAGE)
