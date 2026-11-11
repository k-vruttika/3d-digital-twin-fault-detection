# LineTwin developer tasks. Windows users without make: use .\scripts\dev.ps1 <target>.
# Recipes only call `uv` and `npm --prefix`, so they behave the same under sh and cmd.

.PHONY: help install dev minimal test lint fmt sim sim-quick sim-quick-single sim-plots

help:
	@echo "Targets: install lint fmt test dev minimal sim sim-quick sim-quick-single sim-plots"

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
	@echo "'dev' is not implemented yet; it needs the backend and frontend (Phase 2+)." && exit 1

minimal:
	@echo "'minimal' is not implemented yet; it needs the file adapter and frontend (Phase 2+)." && exit 1

sim:
	uv run python -m twin_sim generate --runs 200 --duration 3600 --seed 42 --out data/sim/

sim-quick:
	uv run python -m twin_sim generate --quick --out data/sim_quick/ --export csv,jsonl

sim-quick-single:
	uv run python -m twin_sim generate --quick --line config/line_single.yaml --scenario config/scenarios/quick_single.yaml --out data/sim_quick_single/ --export csv,jsonl

sim-plots:
	uv run python -m twin_sim plot-examples --out experiments/results/sim_examples/
