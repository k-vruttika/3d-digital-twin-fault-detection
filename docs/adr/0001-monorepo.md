# ADR 0001: Monorepo as a uv workspace

- **Status:** Accepted
- **Date:** 2026-11-11
- **Phase:** 0

## Context

LineTwin has four Python packages (`twin_common`, `twin_sim`, `twin_backend`, `twin_ml`) and a TypeScript frontend. The packages share contracts and config loaders from `twin_common`. Their dependency footprints differ a lot. `twin_ml` will pull in PyTorch, XGBoost, SHAP and so on. `twin_backend` must stay light enough to run in minimal mode. We need reproducible installs on Windows and in CI.

Options considered:

1. **One `pyproject.toml`** listing all four packages and one shared dependency list.
2. **A uv workspace.** A root `pyproject.toml` declares `[tool.uv.workspace]`. Each package has its own `pyproject.toml` with its own dependencies. A single `uv.lock` covers everything.
3. **Separate repositories.**

## Decision

Use a **uv workspace** (option 2).

- Workspace members: `common/`, `simulator/`, `backend/`, `ml/`. Each is a hatchling project with a flat package directory, matching CLAUDE.md section 5 (e.g. `common/twin_common/`).
- Members depend on each other through `twin-common = { workspace = true }`.
- The root project is virtual (`package = false`). It depends on all members, so a plain `uv sync` / `uv run` installs everything editable into one `.venv`. It also holds the dev tool group (ruff, mypy, pytest, pre-commit) and the shared tool configuration (ruff, mypy, pytest).
- There is one `uv.lock` at the root.

## Consequences

- Each package declares exactly what it needs. Heavy ML dependencies are added to `ml/pyproject.toml` only and never become a requirement of `twin_backend`.
- There is one lockfile and one virtualenv, so local, Windows CI and Linux CI resolve identical versions.
- Tests live next to each package (`common/tests/`, ...) and cross-package tests in root `tests/`. Test directories have **no `__init__.py`**, and **test file names must be unique across the repo** (e.g. `test_backend_import.py`). This lets mypy (with `mypy_path` set to the member roots) and pytest (`--import-mode=importlib`) treat all `tests` dirs as one namespace without collisions.
- Shared test fixtures that other packages will need (sample contracts) live in `twin_common.testing`, not in a test module.
- Separate repos (option 3) would make contract changes span several PRs. That was rejected for a single-team research project.
