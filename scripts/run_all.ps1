# Ejecuta todo el flujo del Ejercicio 06 (VAE, evaluación y Model Card).
#   .\scripts\run_all.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> 1/5. Entorno (uv, extra cpu)" -ForegroundColor Cyan
uv python install 3.12
uv sync --extra cpu
# Si tienes WSL2/Linux con NVIDIA ya configurado, usa en su lugar: uv sync --extra gpu
# Jupyter y reportlab no están en el grupo dev del proyecto base: se instalan aparte
uv pip install --python .venv/bin/python jupyter nbformat ipykernel reportlab

Write-Host "==> 2/5. Pruebas (esperado: 11 passed)" -ForegroundColor Cyan
uv run pytest -q

Write-Host "==> 3/5. Runtime y entrenamiento" -ForegroundColor Cyan
uv run python scripts/check_runtime.py
uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8

Write-Host "==> 4/5. Notebook del Ejercicio 06 (ejecutado de principio a fin)" -ForegroundColor Cyan
if ($env:SKIP_NOTEBOOK -ne "1") {
    uv run python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/01_ejercicio06_generativos.ipynb
} else {
    Write-Host "    OMITIDO por SKIP_NOTEBOOK=1" -ForegroundColor DarkGray
}

Write-Host "==> 5/5. PDF breve de entrega" -ForegroundColor Cyan
if (-not $env:REPO_URL) { $env:REPO_URL = "https://github.com/edvvdev/unidad04-06" }
uv run python scripts/build_ejercicio06_pdf.py

Write-Host "Listo. Revisa 'git status' y completa los campos PENDIENTE del PDF (nombre y matrícula)." -ForegroundColor Green
