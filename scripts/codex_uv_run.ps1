param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$UvArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $repoRoot ".uv-python"
if (-not $env:PYTEST_ADDOPTS) {
    $pytestCache = (Join-Path $env:UV_CACHE_DIR "pytest-cache").Replace("\", "/")
    $env:PYTEST_ADDOPTS = "-o cache_dir=$pytestCache"
}

if ($UvArgs.Count -eq 0) {
    Write-Error "Usage: scripts\codex_uv_run.ps1 [uv run args], e.g. --with pytest python -m pytest -q"
}

$pythonInstall = Join-Path $env:UV_PYTHON_INSTALL_DIR "cpython-3.13-windows-x86_64-none\python.exe"
if (-not (Test-Path -LiteralPath $pythonInstall)) {
    Write-Error @"
Workspace-local uv Python is missing.
Install it once with:
  `$env:UV_CACHE_DIR='.uv-cache'; uv python install 3.13 --install-dir .uv-python --no-registry
"@
}

& uv run --no-project --isolated --managed-python --python 3.13 @UvArgs
exit $LASTEXITCODE
