#!/usr/bin/env bash
# Ejecuta todo el flujo del Ejercicio 06 (VAE, evaluación y Model Card).
#
#   ./scripts/run_all.sh            # todo: entorno → runtime → entrenamiento → notebook → PDF
#   SKIP_NOTEBOOK=1 ./scripts/run_all.sh   # omite el notebook (si solo quieres el script de entrenamiento)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> 1/5. Entorno (uv, extra cpu)"
uv python install 3.12
uv sync --extra cpu
# Si tienes WSL2/Linux con NVIDIA ya configurado, usa en su lugar: uv sync --extra gpu
# Jupyter y reportlab no están en el grupo dev del proyecto base: se instalan aparte
uv pip install --python .venv/bin/python jupyter nbformat ipykernel reportlab

echo "==> 2/5. Pruebas (esperado: 4 passed)"
uv run pytest -q

echo "==> 3/5. Runtime y entrenamiento"
uv run python scripts/check_runtime.py
uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8

echo "==> 4/5. Notebook del Ejercicio 06 (ejecutado de principio a fin)"
if [ "${SKIP_NOTEBOOK:-0}" != "1" ]; then
  uv run python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 notebooks/01_ejercicio06_generativos.ipynb
else
  echo "    OMITIDO por SKIP_NOTEBOOK=1"
fi

echo "==> 5/5. PDF breve de entrega"
REPO_URL="${REPO_URL:-https://github.com/edvvdev/unidad04-06}" \
  uv run python scripts/build_ejercicio06_pdf.py

echo "Listo. Revisa 'git status' y completa los campos PENDIENTE del PDF (nombre y matrícula)."
