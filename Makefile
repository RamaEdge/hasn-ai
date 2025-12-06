.DEFAULT_GOAL := help

# Configuration
PYTHON ?= python3
VENV_DIR ?= .venv
REQ_FILE ?= requirements.txt

IMAGE_NAME ?= hasn-ai
IMAGE_TAG ?= local
IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)

# Useful shortcuts
PY := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
RUFF := $(VENV_DIR)/bin/ruff
BLACK := $(VENV_DIR)/bin/black

# Directories/files to lint. Adjust as needed.
LINT_PATHS := src

.PHONY: help venv install deps lint docker-build trivy-image trivy-fs trivy-all clean format

help:
	@echo "Targets:"
	@echo "  venv         - Create Python virtualenv in $(VENV_DIR) and install deps + dev tools"
	@echo "  install      - Install only runtime deps from $(REQ_FILE) into $(VENV_DIR)"
	@echo "  deps         - Install/upgrade dev tools (ruff, black)"
	@echo "  lint         - Run ruff and black --check on $(LINT_PATHS)"
	@echo "  format       - Auto-format with black and ruff --fix"
	@echo "  docker-build - Build docker image $(IMAGE) locally"
	@echo "  trivy-image  - Trivy scan docker image $(IMAGE) (local or via dockerized trivy)"
	@echo "  trivy-fs     - Trivy filesystem scan of current directory (local or via dockerized trivy)"
	@echo "  trivy-all    - Run both image and filesystem scans"
	@echo "  clean        - Remove venv and caches"

# Create venv and install dependencies + dev tooling
venv: $(VENV_DIR)/bin/activate
	@echo "Installing runtime dependencies from $(REQ_FILE) (if present)"
	@if [ -f "$(REQ_FILE)" ]; then \
		$(PIP) install -U pip setuptools wheel && \
		$(PIP) install -r $(REQ_FILE); \
	else \
		echo "No $(REQ_FILE); skipping runtime deps"; \
	fi
	$(MAKE) deps

# Create venv only
$(VENV_DIR)/bin/activate:
	@echo "Creating venv in $(VENV_DIR)"
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Upgrading pip/setuptools/wheel"
	$(PIP) install -U pip setuptools wheel

# Install only runtime deps
install: $(VENV_DIR)/bin/activate
	@if [ -f "$(REQ_FILE)" ]; then \
		$(PIP) install -r $(REQ_FILE); \
	else \
		echo "No $(REQ_FILE); skipping runtime deps"; \
	fi

# Dev tools
deps: $(VENV_DIR)/bin/activate
	$(PIP) install -U ruff black

# Linting
lint: deps
	@echo "Running ruff (lint)"
	$(RUFF) check --exclude .venv --exclude .git $(LINT_PATHS)
	@echo "Running black --check"
	$(BLACK) --check --exclude '/(\.venv|\.git)/' .

# Optional formatting helper
format: deps
	$(BLACK) --exclude '/(\.venv|\.git)/' .
	$(RUFF) check --fix --exclude .venv --exclude .git $(LINT_PATHS)

# Docker
docker-build:
	docker build -t $(IMAGE) .

# Trivy scans (works if trivy installed locally; falls back to dockerized trivy otherwise)
trivy-image:
	@echo "Scanning image $(IMAGE) with Trivy (CRITICAL,HIGH)"
	@if command -v trivy >/dev/null 2>&1; then \
		trivy image --exit-code 1 --severity CRITICAL,HIGH $(IMAGE); \
	else \
		docker run --rm \
			-v /var/run/docker.sock:/var/run/docker.sock \
			-v $$HOME/.cache/trivy:/root/.cache/ \
			aquasec/trivy:latest \
			image --exit-code 1 --severity CRITICAL,HIGH $(IMAGE); \
	fi

trivy-fs:
	@echo "Scanning filesystem with Trivy (CRITICAL,HIGH)"
	@if command -v trivy >/dev/null 2>&1; then \
		trivy fs --exit-code 1 --severity CRITICAL,HIGH --scanners vuln,config --ignore-unfixed .; \
	else \
		docker run --rm \
			-v $$PWD:/project \
			-v $$HOME/.cache/trivy:/root/.cache/ \
			-w /project \
			aquasec/trivy:latest \
			fs --exit-code 1 --severity CRITICAL,HIGH --scanners vuln,config --ignore-unfixed .; \
	fi

trivy-all: trivy-image trivy-fs

clean:
	rm -rf $(VENV_DIR) .ruff_cache .mypy_cache .pytest_cache build dist __pycache__

