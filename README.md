# INF-8239 · Ejercicio 06 · VAE, evaluación y Model Card

Autor académico: Edwin Ramón José Nolasco · Repositorio: https://github.com/edvvdev/unidad04-06

Consolida LAB10 (proyecto separado del de recomendadores, para no mezclar TensorFlow con el entorno
de PLN/recomendadores) en una entrega ejecutable: autoencoder y VAE sobre Fashion-MNIST, con
reconstrucción, muestreo, interpolación, métricas de fidelidad y diversidad, coste, Model Card y
límites. El dataset se descarga con `tf.keras.datasets` y **no se versiona en Git**.

## CPU
```bash
uv python install 3.12
uv sync --extra cpu
uv run python scripts/check_runtime.py
uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8
```

## GPU
Solo para WSL2/Linux con NVIDIA correctamente configurada:
```bash
uv sync --extra gpu
```

## Aplicación
```bash
uv run streamlit run app/streamlit_app.py
```

## Entrega del Ejercicio 06

| Elemento | Dónde está |
|---|---|
| Notebook ejecutado | `notebooks/01_ejercicio06_generativos.ipynb` (15 celdas, con salidas) |
| Código reutilizable | `src/inf8239_u03_gen/` (`data`, `models`) y `scripts/train_ae_vae.py` |
| Pruebas | `tests/` (`uv run pytest -q` → 4 passed) |
| Métricas y figuras | `reports/` |
| Requisitos | `requirements.txt` y `requirements-dev.txt` (exportados con `uv export`) |
| PDF breve de entrega | `docs/EJERCICIO_06_PDF_BRIEF.pdf` |
| Model Card | `MODEL_CARD.md` |
| Declaración de IA | `AI_USE_DECLARATION.md` |

Para ejecutar el notebook de principio a fin:

```bash
uv pip install --python .venv/bin/python jupyter nbformat ipykernel reportlab   # o: uv sync --group dev
uv run python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 notebooks/01_ejercicio06_generativos.ipynb
uv run python scripts/build_ejercicio06_pdf.py
```

## Resultados de referencia (2026-11-01, CPU sin GPU)

| Evidencia | Archivo | Resultado |
|---|---|---|
| Métricas | `reports/training_metrics.json` | AE test MSE **0,02048** · VAE test MSE **0,03375** · KL medio 6,3744 |
| Curvas | `reports/training_curves.png` | AE: loss 0,30604 vs val_loss 0,30710 (sin sobreajuste) · VAE: total 273,60, reconstructions 267,35, KL 6,26 |
| Panel no seleccionado | `reports/ae_vae_panel.png` | 16 ejemplos de prueba con semilla 42: originales, reconstrucción AE, reconstrucción VAE y muestras nuevas |
| Interpolación | `reports/interpolation.png` | 16 puntos entre dos extremos del espacio latente |
| Modelos | `models/autoencoder.keras`, `models/decoder.keras` | 331 KB y 416 KB · 25.888 y 101.520 parámetros |
| Tiempos | `reports/training_metrics.json` | AE 10,42 s (6 épocas) · VAE 17,23 s (8 épocas) |
| Consolidação E06 | `reports/ejercicio06_metricas.json` | AE MSE 0,02048 · VAE MSE 0,03375 · KL 6,3744 · diversidad media 4,8633 · distancia al vecino más cercano 3,2987 · AE 9,08 s · VAE 14,04 s · inferencia 0,46 ms por lote de 100 |

Entorno de la ejecución: Python 3.12.15 · Linux Debian 13 · TensorFlow 2.21.0 · `GPU: []`
(`scripts/check_runtime.py`).

## Documentación

| Documento | Contenido |
|---|---|
| `MODEL_CARD.md` | Modelos, uso previsto y fuera de alcance, dataset, arquitectura, métricas, hardware, riesgos |
| `docs/CIERRE_INTERPRETATIVO_LAB10.md` | Cierre interpretativo del LAB10 |
| `REQUISITOS_LAB10.md` | Requisitos y evidencia según el manual del profesor |
| `CAMBIOS_AL_BASE.md` | Cambios aplicados sobre el proyecto base y su justificación |
| `docs/EJERCICIO_06_PDF_BRIEF.pdf` | PDF breve de entrega (4 páginas) con la tabla principal y las figuras |
| `AI_USE_DECLARATION.md` | Uso honesto del asistente de IA y verificaciones pendientes del estudiante |

## Advertencia

Una imagen sintética no es automáticamente privada, original, verdadera ni libre de sesgo. No se
afirme privacidad por el solo hecho de generar datos sintéticos: falta el análisis de vecinos
cercanos en el conjunto de entrenamiento.
