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
| Código reutilizable | `src/inf8239_u03_gen/` (`data`, `models` con el VAE, `metrics`, `figures`) y `scripts/train_ae_vae.py` |
| Pruebas | `tests/` (`uv run pytest -q` → 15 passed) |
| Métricas y figuras | `reports/` |
| Requisitos | `requirements.txt` y `requirements-dev.txt` (exportados con `uv export`) |
| PDF breve de entrega | `docs/EJERCICIO_06_PDF_BRIEF.pdf` |
| Model Card | `MODEL_CARD.md` |
| Declaración de IA | `AI_USE_DECLARATION.md` |

Para ejecutar el notebook de principio a fin:

```bash
uv pip install --python .venv/bin/python jupyter nbformat ipykernel reportlab
uv run python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 notebooks/01_ejercicio06_generativos.ipynb
uv run python scripts/build_ejercicio06_pdf.py
```

## Ejecución completa (un solo comando)

```bash
./scripts/run_all.sh          # macOS / Linux
```
```powershell
.\scripts\run_all.ps1        # Windows
```

El script hace: entorno con uv → pruebas → descarga del dataset y auditoría → notebook ejecutado de
principio a fin → PDF de entrega. Instala además Jupyter y `reportlab`, que **no** vienen en el
grupo de desarrollo del proyecto base del profesor (`uv sync` solo instala pytest, pytest-cov y ruff).

## Resultados de referencia (corrida del 2026-11-05, CPU sin GPU)

Todas las cifras provienen de **una sola corrida**, la del 2026-11-05, y su fuente de verdad es
`reports/ejercicio06_metricas.json` (que registra `origen` y `fecha_ejecucion`). No hay un segundo
juego de números en el repositorio.

La corrida es **reproducible bit a bit**: el script y el notebook fijan TensorFlow a un solo hilo por
operación (`TF_NUM_INTRAOP_THREADS=1`, `TF_NUM_INTEROP_THREADS=1`) antes de importarlo. Sin esa
fijación, el orden de las sumas en las capas densas cambia entre corridas y el VAE entrenado varía en
el cuarto decimal, con lo que el párrafo canónico de la Model Card deja de cuadrar con el JSON. El
párrafo canónico se renderiza desde el JSON (`parrafo_canonico` en `src/inf8239_u03_gen/metrics.py`)
y `tests/test_canonical.py` verifica que la documentación está sincronizada.

| Evidencia | Archivo | Resultado |
|---|---|---|
| Métricas (fuente de verdad) | `reports/ejercicio06_metricas.json` | AE test MSE **0,02048** · VAE test MSE **0,03361** · KL medio 6,3072 |
| Diversidad y vecinos cercanos | `reports/ejercicio06_metricas.json` | diversidad media entre 64 muestras **4,8769** · distancia media al vecino más cercano **3,3011** (contra 1.000 imágenes de **prueba**, no de entrenamiento) · línea base de pares de prueba **11,3984** |
| Curvas | `reports/ejercicio06_curvas.png` | AE: loss 0,30604 vs val_loss 0,30710 (sin sobreajuste) · VAE: total 273,58, reconstrucción 267,33, KL 6,26 |
| Panel no seleccionado | `reports/ejercicio06_panel.png` | 16 ejemplos de prueba con semilla 42: originales, reconstrucción AE, reconstrucción VAE y muestras nuevas |
| Interpolación | `reports/ejercicio06_interpolacion.png` | 16 puntos entre dos extremos del espacio latente |
| Modelos | `models/autoencoder.keras`, `models/decoder.keras` | 331 KB y 416 KB · 25.888 y 101.520 parámetros · **regenerados por el script, no versionados en Git** |
| Tiempos | `reports/ejercicio06_metricas.json` | AE 6,63 s (6 épocas) · VAE 13,50 s (8 épocas) (de esa misma corrida) |
| Inferencia | `reports/ejercicio06_metricas.json` | 0,49 ms por lote de 100 imágenes |

Entorno de la ejecución: Python 3.12.15 · Linux Debian 13 · TensorFlow 2.21.0 · `GPU: []`
(`scripts/check_runtime.py`).

### Los modelos no se versionan

`models/autoencoder.keras` y `models/decoder.keras` están en `.gitignore` (`models/*`, salvo
`.gitkeep`): son binarios de 331 KB y 416 KB que **se regeneran** con
`uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8` o al ejecutar el notebook. El
README y la Model Card los citan como evidencia del modelo entrenado (parámetros y tamaño), no como
archivos contenidos en el árbol de Git.

## Documentación

| Documento | Contenido |
|---|---|
| `MODEL_CARD.md` | Modelos, uso previsto y fuera de alcance, dataset, arquitectura, métricas, hardware, riesgos |
| `docs/CIERRE_INTERPRETATIVO_LAB10.md` | Cierre interpretativo del LAB10 (cita el párrafo canónico de diversidad) |
| `REQUISITOS_LAB10.md` | Requisitos y evidencia según el manual del profesor |
| `CAMBIOS_AL_BASE.md` | Cambios aplicados sobre el proyecto base y su justificación |
| `docs/EJERCICIO_06_PDF_BRIEF.pdf` | PDF breve de entrega (4 páginas) con la tabla principal y las figuras |
| `AI_USE_DECLARATION.md` | Uso honesto del asistente de IA y verificaciones del estudiante con su salida real |

## Advertencia

Una imagen sintética no es automáticamente privada, original, verdadera ni libre de sesgo. El
análisis de vecinos cercanos de esta entrega se calculó contra el **conjunto de prueba**, así que no
permite descartar la memorización del conjunto de entrenamiento; antes de cualquier uso de datos
sintéticos hay que repetirlo contra el entrenamiento completo.
