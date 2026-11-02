# Cambios sobre el proyecto base del profesor (LAB10)

El manual de la Unidad 03 pide evidencia que el proyecto base no generaba. Estos cambios son
mínimos, están documentados aquí y en el diff, y **no tocan `pyproject.toml` ni `uv.lock`**. El
proyecto base original se conserva en `~/Downloads/INF8239_U03_Generativos_Proyecto_Base_uv.zip`
(md5 idéntico a la copia re-descargada el 2026-11-01).

## `scripts/train_ae_vae.py` — evidencia que faltaba

El manual exige guardar «curvas, reconstrucciones, muestras, interpolaciones, métricas, el decodificador
y modelos entrenados», y evaluar «con paneles no seleccionados» (16 ejemplos de prueba con semilla
fija). El base solo guardaba panel (12 ejemplos, los primeros del test), interpolación, métricas de
tiempo/parámetros y los modelos.

**Cambios:**

1. **Curvas de entrenamiento** → `reports/training_curves.png`: panel AE (pérdida de entrenamiento vs
   validación) y panel VAE (pérdida total, reconstrucción y KL), con leyenda.
2. **Métricas de reconstrucción** añadidas a `reports/training_metrics.json`: `ae_test_mse`,
   `vae_test_mse`, `vae_test_kl`, pérdidas finales por época y `seed`, `evaluation_examples`,
   `panel_examples`. Se calculan sobre **1.000 ejemplos de prueba elegidos con semilla 42**
   (`np.random.default_rng(SEED).choice`), sin selección manual de los mejores casos.
3. **Panel con 16 ejemplos** elegidos con esa misma semilla (el base usaba los 12 primeros del test)
   y rejilla de interpolación de 16 puntos, como pide el manual.

La lógica de los modelos (`src/inf8239_u03_gen/models.py`) y la normalización de datos
(`src/inf8239_u03_gen/data.py`) **no se tocaron**: son las del profesor.

## `.gitignore` — la evidencia debe quedar versionada

El `.gitignore` del base ignoraba `reports/*` completo, de modo que las curvas, el panel, la
interpolación y las métricas exigidas por el manual no podían commitearse. Ahora solo se ignoran
los archivos que no son evidencia (`data/raw/*` y `models/*`), y se versionan `reports/*.json`,
`reports/*.csv` y `reports/*.png`.

## Comandos ejecutados para verificar (en esta máquina)

```bash
cd ~/source/uasd/cienciadedatos/tareas/Unidad_3/INF8239_U03_Generativos
uv sync --extra cpu && uv run pytest -q            # 4 passed
uv run python scripts/check_runtime.py             # TensorFlow 2.21.0, GPU: []
uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8
```

## Cambio equivalente en el proyecto de recomendadores

El error de compatibilidad con NumPy 2 (`np.matrix`) y las extensiones de LAB08/LAB09 están
documentados en `../INF8239_U03_Recomendadores/CAMBIOS_AL_BASE.md`.
