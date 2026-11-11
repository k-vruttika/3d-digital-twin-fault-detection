<#
.SYNOPSIS
    LineTwin developer tasks for Windows PowerShell (same targets as the Makefile).

.EXAMPLE
    .\scripts\dev.ps1 install
    .\scripts\dev.ps1 lint
    .\scripts\dev.ps1 test

.NOTES
    Works on Windows PowerShell 5.1 and PowerShell 7+. If script execution is blocked, run once:
        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
    or invoke with: powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 lint
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'install', 'dev', 'minimal', 'test', 'lint', 'fmt', 'sim', 'sim-quick', 'sim-quick-single', 'sim-plots')]
    [string]$Target = 'help'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Step {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][scriptblock]$Command
    )
    Write-Host "==> $Name" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Name' failed with exit code $LASTEXITCODE"
    }
}

function Show-NotYetAvailable {
    param([string]$What, [string]$Phase)
    Write-Host "'$What' is not implemented yet; it becomes available in $Phase." -ForegroundColor Yellow
    exit 1
}

function Show-Help {
    Write-Host @'
Usage: .\scripts\dev.ps1 <target>

  install  Install Python (uv) and frontend (npm) dependencies
  lint     ruff check, ruff format --check, mypy, eslint, tsc
  fmt      ruff format, ruff --fix, eslint --fix
  test     pytest, vitest
  dev        Full stack: backend + frontend                 (Phase 2+)
  minimal    3D twin from a JSON/CSV file, no DB/broker/ML  (Phase 2+)
  sim        Generate the full dataset: 200 runs x 3600 s -> data/sim/
  sim-quick  5 short fault-dense runs (+ CSV/JSONL) -> data/sim_quick/
  sim-quick-single  Same, single-station line (CNV-01 only) -> data/sim_quick_single/
  sim-plots  Example plots per fault type -> experiments/results/sim_examples/
'@
}

Push-Location $RepoRoot
try {
    switch ($Target) {
        'help' { Show-Help }
        'install' {
            Invoke-Step 'uv sync' { uv sync --all-packages }
            Invoke-Step 'npm ci' { npm --prefix frontend ci }
        }
        'lint' {
            Invoke-Step 'ruff check' { uv run ruff check . }
            Invoke-Step 'ruff format --check' { uv run ruff format --check . }
            Invoke-Step 'mypy' { uv run mypy }
            Invoke-Step 'eslint' { npm --prefix frontend run lint }
            Invoke-Step 'tsc' { npm --prefix frontend run typecheck }
        }
        'fmt' {
            Invoke-Step 'ruff format' { uv run ruff format . }
            Invoke-Step 'ruff check --fix' { uv run ruff check --fix . }
            Invoke-Step 'eslint --fix' { npm --prefix frontend run lint:fix }
        }
        'test' {
            Invoke-Step 'pytest' { uv run pytest }
            Invoke-Step 'vitest' { npm --prefix frontend test }
        }
        'dev' { Show-NotYetAvailable 'dev' 'Phase 2' }
        'minimal' { Show-NotYetAvailable 'minimal' 'Phase 2' }
        'sim' {
            Invoke-Step 'simulator: generate dataset' {
                uv run python -m twin_sim generate --runs 200 --duration 3600 --seed 42 --out data/sim/
            }
        }
        'sim-quick' {
            Invoke-Step 'simulator: quick dataset' {
                uv run python -m twin_sim generate --quick --out data/sim_quick/ --export csv,jsonl
            }
        }
        'sim-quick-single' {
            Invoke-Step 'simulator: quick dataset, single station' {
                uv run python -m twin_sim generate --quick --line config/line_single.yaml `
                    --scenario config/scenarios/quick_single.yaml --out data/sim_quick_single/ `
                    --export csv,jsonl
            }
        }
        'sim-plots' {
            Invoke-Step 'simulator: example plots' {
                uv run python -m twin_sim plot-examples --out experiments/results/sim_examples/
            }
        }
    }
    Write-Host "OK: $Target" -ForegroundColor Green
}
catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
