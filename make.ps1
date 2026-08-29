#Requires -Version 5.1
<#
.SYNOPSIS
    Task runner for the AI Agent Control Plane (Windows counterpart to Makefile).
.EXAMPLE
    .\make.ps1 setup
    .\make.ps1 test
    .\make.ps1 test-race
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'setup', 'up', 'down', 'logs', 'ps', 'testdb', 'migrate',
                 'revision', 'test', 'test-unit', 'test-db', 'test-race',
                 'lint', 'fmt', 'psql', 'api', 'clean')]
    [string]$Target = 'help',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$Py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$DevUrl = 'postgresql+psycopg://acp:acp@localhost:5434/acp'
$TestUrl = 'postgresql+psycopg://acp:acp@localhost:5434/acp_test'

function Assert-Venv {
    if (-not (Test-Path $Py)) {
        throw "No virtualenv found. Run: .\make.ps1 setup"
    }
}

function Invoke-Step($Label, [scriptblock]$Block) {
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
        throw "$Label failed (exit $LASTEXITCODE)"
    }
}

switch ($Target) {

    'help' {
        Write-Host ""
        Write-Host "  AI Agent Control Plane" -ForegroundColor Green
        Write-Host ""
        Write-Host "  setup       create venv and install dependencies (run once)"
        Write-Host "  up          start postgres (host port 5434)"
        Write-Host "  down        stop everything and delete volumes"
        Write-Host "  ps          show container status"
        Write-Host "  logs        follow api logs"
        Write-Host "  testdb      create the isolated acp_test database"
        Write-Host "  migrate     alembic upgrade head against the dev database"
        Write-Host "  revision    new migration:  .\make.ps1 revision 'message'"
        Write-Host ""
        Write-Host "  test        full suite"
        Write-Host "  test-unit   pure domain tests, no database"
        Write-Host "  test-db     integration + concurrency"
        Write-Host "  test-race   run the CAS race tests 20x to prove they are not flaky"
        Write-Host ""
        Write-Host "  lint / fmt  ruff"
        Write-Host "  psql        interactive shell on the dev database"
        Write-Host "  api         run the API locally (not in docker)"
        Write-Host ""
    }

    'setup' {
        Invoke-Step 'creating virtualenv' { python -m venv .venv }
        Invoke-Step 'upgrading pip' { & $Py -m pip install --quiet --upgrade pip }
        Invoke-Step 'installing acp[dev]' { & $Py -m pip install -e '.[dev]' }
        Write-Host "Done. Next: .\make.ps1 up" -ForegroundColor Green
    }

    'up' {
        Invoke-Step 'starting postgres' { docker compose up -d postgres }
        Write-Host "Waiting for postgres to accept connections..." -NoNewline
        for ($i = 0; $i -lt 30; $i++) {
            docker compose exec -T postgres pg_isready -U acp -d acp | Out-Null
            if ($LASTEXITCODE -eq 0) { Write-Host " ready" -ForegroundColor Green; break }
            Start-Sleep -Seconds 1
            Write-Host "." -NoNewline
        }
    }

    'down'   { docker compose down -v }
    'ps'     { docker compose ps }
    'logs'   { docker compose logs -f api }
    'psql'   { docker compose exec postgres psql -U acp -d acp }

    'testdb' {
        docker compose exec -T postgres psql -U acp -d postgres -c 'CREATE DATABASE acp_test'
        if ($LASTEXITCODE -ne 0) {
            Write-Host "acp_test already exists - continuing" -ForegroundColor DarkGray
        }
        $global:LASTEXITCODE = 0
    }

    'migrate' {
        Assert-Venv
        $env:ACP_DATABASE_URL = $DevUrl
        Invoke-Step 'alembic upgrade head' { & $Py -m alembic upgrade head }
    }

    'revision' {
        Assert-Venv
        if (-not $Rest) { throw "Usage: .\make.ps1 revision 'describe the change'" }
        $env:ACP_DATABASE_URL = $DevUrl
        & $Py -m alembic revision --autogenerate -m ($Rest -join ' ')
    }

    'test' {
        Assert-Venv
        & $PSCommandPath testdb
        $env:ACP_TEST_DATABASE_URL = $TestUrl
        & $Py -m pytest -q @Rest
    }

    'test-unit' {
        Assert-Venv
        & $Py -m pytest -q tests\unit @Rest
    }

    'test-db' {
        Assert-Venv
        & $PSCommandPath testdb
        $env:ACP_TEST_DATABASE_URL = $TestUrl
        & $Py -m pytest -q tests\integration tests\concurrency @Rest
    }

    'test-race' {
        # A race test that passes once has proven nothing. Twenty consecutive
        # runs is the bar for calling a concurrency guarantee demonstrated.
        Assert-Venv
        & $PSCommandPath testdb
        $env:ACP_TEST_DATABASE_URL = $TestUrl
        & $Py -m pytest -q tests\concurrency --count=20
    }

    'lint' { Assert-Venv; & $Py -m ruff check src tests }
    'fmt'  { Assert-Venv; & $Py -m ruff format src tests migrations; & $Py -m ruff check --fix src tests }

    'api' {
        Assert-Venv
        $env:ACP_DATABASE_URL = $DevUrl
        & $Py -m uvicorn acp.api.app:app --reload --port 8001
    }

    'clean' {
        Get-ChildItem -Path $PSScriptRoot -Include __pycache__, .pytest_cache, .ruff_cache `
            -Recurse -Directory | Remove-Item -Recurse -Force
        Write-Host "cleaned" -ForegroundColor Green
    }
}
