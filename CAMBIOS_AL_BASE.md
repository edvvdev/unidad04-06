# Cambios sobre el proyecto base del profesor (LAB10)

El manual de la Unidad 03 pide evidencia que el proyecto base no generaba, y la auditoría de la
entrega publicada pidió además corregir contradicciones y duplicaciones. Todos los cambios están
documentados aquí y en el diff, y **no tocan `pyproject.toml` ni `uv.lock`**. El proyecto base
original se conserva en `~/Downloads/INF8239_U03_Generativos_Proyecto_Base_uv.zip`
(md5 idéntico a la copia re-descargada el 2026-11-01).

## `scripts/train_ae_vae.py` — evidencia que faltaba

El manual exige guardar «curvas, reconstrucciones, muestras, interpolaciones, métricas, el decodificador
y modelos entrenados», y evaluar «con paneles no seleccionados» (16 ejemplos de prueba con semilla
fija). El base solo guardaba panel (12 ejemplos, los primeros del test), interpolación, métricas de
tiempo/parámetros y los modelos.

**Cambios:**

1. **Curvas de entrenamiento** → `reports/ejercicio06_curvas.png`: panel AE (pérdida de entrenamiento vs
   validación) y panel VAE (pérdida total, reconstrucción y KL), con leyenda.
2. **Métricas de reconstrucción** añadidas a `reports/ejercicio06_metricas.json`: `mse_reconstruccion_ae`,
   `mse_reconstruccion_vae`, `kl_media_vae`, pérdidas finales por época, `seed`, `evaluation_examples`,
   `panel_examples`. Se calculan sobre **1.000 ejemplos de prueba elegidos con semilla 42**
   (`np.random.default_rng(SEED).choice`), sin selección manual de los mejores casos.
3. **Panel con 16 ejemplos** elegidos con esa misma semilla (el base usaba los 12 primeros del test)
   y rejilla de interpolación de 16 puntos, como pide el manual.

## Cambios de la auditoría de la entrega (2026-11-05)

| # | Defecto | Corrección | Archivos |
|---|---|---|---|
| D1 | La documentación afirmaba que no se había medido diversidad ni vecinos cercanos, pero el repositorio sí los medía | Párrafo canónico único en `MODEL_CARD.md`, citado literalmente en `docs/CIERRE_INTERPRETATIVO_LAB10.md` y en el PDF, con las tres cifras y su matiz (el vecino más cercano se calcula contra **prueba**, no contra entrenamiento) | `MODEL_CARD.md`, `docs/CIERRE_INTERPRETATIVO_LAB10.md`, `notebooks/…ipynb`, `scripts/build_ejercicio06_pdf.py` |
| D1 | Faltaba la línea base que hace interpretable el 3,2987 | Nueva métrica `distancia_media_entre_pares_de_prueba` (distancia media entre pares de las 1.000 imágenes de prueba) | `src/inf8239_u03_gen/metrics.py`, `scripts/train_ae_vae.py`, `notebooks/…ipynb` |
| D2 | La Tabla 1 del PDF repetía la diversidad en la columna del AE | El AE queda en «—» y se añade la nota al pie de que las métricas de muestreo aplican solo al VAE | `scripts/build_ejercicio06_pdf.py` |
| D3 | El PDF y los documentos citaban juegos de figuras distintos | Ruta única `reports/ejercicio06_{panel,interpolacion,curvas}.png`; se eliminan `ae_vae_panel.png`, `interpolation.png` y `training_curves.png` | `scripts/train_ae_vae.py`, `src/inf8239_u03_gen/figures.py`, `README.md`, `MODEL_CARD.md`, `docs/CIERRE_INTERPRETATIVO_LAB10.md` |
| D4 | Dos juegos de tiempos sin explicar y fechas desalineadas | Una sola fuente de verdad (`reports/ejercicio06_metricas.json`, con `origen` y `fecha_ejecucion`); se elimina `reports/training_metrics.json`; todas las fechas y todos los tiempos se refieren a la corrida del 2026-11-05 | `src/inf8239_u03_gen/metrics.py`, `scripts/train_ae_vae.py`, `notebooks/…ipynb`, `README.md`, `MODEL_CARD.md`, `docs/CIERRE_INTERPRETATIVO_LAB10.md` |
| D5 | Las casillas de verificación de la declaración de IA no estaban marcadas | Cada casilla incluye el comando y su salida real; se corrigen las afirmaciones contradichas por la evidencia | `AI_USE_DECLARATION.md` |
| D6 | La lógica del VAE estaba duplicada en el notebook y en el script | La clase `VAE` (muestreo, reparametrización, pérdidas y `train_step`) vive en `src/inf8239_u03_gen/models.py` (`build_vae`) y la importan ambos | `src/inf8239_u03_gen/models.py`, `scripts/train_ae_vae.py`, `notebooks/…ipynb` |
| D7 | `uv run ruff check .` fallaba con 5 hallazgos | Corregidos los cuatro `UP031` del notebook (formato con f-strings) y el `I001` del generador de PDF (imports ordenados) | `notebooks/…ipynb`, `scripts/build_ejercicio06_pdf.py` |
| D8 | `models/*.keras` se citaban como evidencia sin decir que no se versionan | Se aclara en el README y en la Model Card que se regeneran con el script y que no están en Git por tamaño | `README.md`, `MODEL_CARD.md` |

### Nota sobre `src/inf8239_u03_gen/models.py`

El archivo del profesor **no se reescribió**: `build_autoencoder` y `build_encoder_decoder` siguen
intactos, con la misma firma (reciben `tf` como argumento para que el paquete se pueda importar sin
TensorFlow). Lo que se **agrega** es `build_vae(tf, latent_dim=2)`, que devuelve el VAE con su
`train_step` propio: es la implementación que ahora usan tanto el script como el notebook, de modo
que la lógica de muestreo y reparametrización existe en un único sitio.

### Módulos nuevos en `src/inf8239_u03_gen/`

- `metrics.py`: `mean_pairwise_distance`, `mean_nearest_neighbor_distance`, `flatten` y
  `build_report` (que valida que el reporte tenga las claves canónicas y añade `origen` y
  `fecha_ejecucion`).
- `figures.py`: `guardar_panel`, `guardar_interpolacion` y `guardar_curvas`, usadas por el script y
  por el notebook para que las figuras tengan siempre el mismo formato.

### Pruebas

`tests/test_metrics.py` (nuevo) comprueba que las dos distancias calculadas coinciden con el
cálculo directo por fuerza bruta y que `build_report` rechaza reportes incompletos.
`tests/test_models.py` ahora cubre también `build_vae` (contrato de salidas y un `train_step` real).
Total: **11 passed**.

## `.gitignore` — la evidencia debe quedar versionada

El `.gitignore` del base ignoraba `reports/*` completo, de modo que las curvas, el panel, la
interpolación y las métricas exigidas por el manual no podían commitearse. Ahora solo se ignoran
los archivos que no son evidencia (`data/raw/*` y `models/*`), y se versionan `reports/*.json`,
`reports/*.csv` y `reports/*.png`. Los modelos entrenados (`models/*.keras`) siguen fuera de Git por
tamaño y se regeneran con el script; el README y la Model Card lo dicen de forma explícita.

## Comandos ejecutados para verificar (en esta máquina)

```bash
cd ~/source/uasd/cienciadedatos/tareas/Unidad_3/INF8239_U03_Generativos
uv sync --extra cpu && uv run pytest -q            # 11 passed
uv run ruff check .                                # All checks passed!
uv run python scripts/check_runtime.py             # TensorFlow 2.21.0, GPU: []
uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8
./scripts/run_all.sh                               # entorno → pruebas → runtime → notebook → PDF
```

## Cambio equivalente en el proyecto de recomendadores

El error de compatibilidad con NumPy 2 (`np.matrix`) y las extensiones de LAB08/LAB09 están
documentados en `../INF8239_U03_Recomendadores/CAMBIOS_AL_BASE.md`.
