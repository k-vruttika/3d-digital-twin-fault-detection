# ADR 0004: poethepoet as the only task runner

- **Status:** Accepted
- **Date:** 2026-11-12
- **Phase:** 1 (tooling)

## Context

Up to Phase 1, developer tasks (install, lint, fmt, test, dev, minimal, and the sim targets) were defined twice: in a `Makefile` for Linux CI, and in `scripts/dev.ps1` for Windows PowerShell 5.1. The two copies had to be kept in sync by hand, each new target was written twice, `make` was not installed on the development machine, and macOS was not covered. CI also needed separate jobs with shell-specific steps; the minimal-mode check, for example, set an environment variable differently in bash and in PowerShell.

Options considered:

1. **Keep Makefile + dev.ps1.**
2. **Plain Python scripts** (`python tools/tasks.py lint`). Cross-platform, but a hand-written mini task runner.
3. **poethepoet (`poe`)**, with tasks declared in `[tool.poe.tasks]` of the root `pyproject.toml` and run through `uv run poe <task>`.

## Decision

Use **poethepoet** (option 3) and remove the `Makefile` and `scripts/dev.ps1`.

- Tasks are `cmd`, `sequence` or `ref` tasks. Poe runs commands directly, without a shell, so a task behaves identically on Windows, Linux and macOS. Environment variables are set through poe's `env` option, not shell syntax.
- Commands use only cross-platform executables: tools from the uv environment (`ruff`, `mypy`, `pytest`, `python`), `uv` itself, and `npm --prefix frontend`. `uv run poe …` already runs inside the project environment, so the Python tools are called by name rather than through a nested `uv run`. That avoids a second environment sync per step, which in CI could also re-resolve without `--frozen`.
- Poe's **simple executor** is configured (`[tool.poe] executor = { type = "simple" }`). By default poe detects uv and wraps every command in `uv run`, but uv cannot launch `npm` on Windows, where it is `npm.cmd`. Found when first running the tasks locally. The simple executor runs commands directly, inside the environment that `uv run poe` already activated.
- CI is one job with an OS matrix (Ubuntu, Windows, macOS). Every step is `uv run --frozen poe <task>`, and the extra CI tasks (`minimal-data-check`, `build-frontend`) are poe tasks too.
- `uv run poe` with no task lists every task with its help text; this replaces the old `help` target.
- CLAUDE.md §7 records the rule that new developer tasks are added only as poe tasks.

## Consequences

- There is one definition per task, and adding a task means adding one TOML entry.
- `poethepoet` is a new dev dependency, pinned in `uv.lock`.
- A fresh clone needs one bootstrap command before poe exists: `uv sync --all-packages`, then `uv run poe install`.
- Tasks cannot use shell features (pipes, redirection, `&&`). Anything more complex goes into Python code (e.g. a CLI subcommand) that a task calls.
- The pre-commit hooks are unchanged; they already call `uv run --frozen …` and `npm --prefix frontend` directly.
