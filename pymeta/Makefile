# Makefile for pymeta development
.PHONY: help install install-dev test lint type format clean all docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest

lint:  ## Run linting
	ruff check .

type:  ## Run type checking
	mypy pymeta cli

format:  ## Format code
	ruff format .

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:  ## Build documentation
	@echo "Documentation build not implemented yet"

all: lint type test  ## Run all checks (lint, type, test)