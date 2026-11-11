# LineTwin developer tasks. Windows users without make: use .\scripts\dev.ps1 <target>.
# Recipes only call `uv` and `npm --prefix`, so they behave the same under sh and cmd.

.PHONY: help install dev minimal test lint fmt sim

help:
	@echo "Targets: install lint fmt test dev minimal sim"

install:
	uv sync --all-packages
	npm --prefix frontend ci

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	npm --prefix frontend run lint
	npm --prefix frontend run typecheck

fmt:
	uv run ruff format .
	uv run ruff check --fix .
	npm --prefix frontend run lint:fix

test:
	uv run pytest
	npm --prefix frontend test

dev:
	@echo "'dev' is not implemented yet; it becomes available in Phase 1." && exit 1

minimal:
	@echo "'minimal' is not implemented yet; it becomes available in Phase 1." && exit 1

sim:
	@echo "'sim' is not implemented yet; it becomes available in Phase 1." && exit 1
