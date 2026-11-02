# Declaración de uso de inteligencia artificial · LAB10

Proyecto: `INF8239_U03_Generativos` (Unidad 03) · Curso: INF-8239 Ciencia de Datos II
Fecha: 2026-11-01

## Herramientas utilizadas

- **Kilo** (asistente de IA en CLI, modelo MiniMax-M3) como apoyo de análisis, documentación y
  redacción de las tarjetas del proyecto.
- **Python 3.12 + TensorFlow 2.21 (extra `cpu`)** mediante `uv`, y las librerías del proyecto base
  (NumPy, pandas, scikit-learn, matplotlib, Streamlit) para el entrenamiento, las métricas y las
  figuras. No se usaron servicios en la nube, ni APIs externas, ni datasets generados por IA.

## Propósito de cada uso

| Uso | Detalle |
|---|---|
| Interpretación | Contrastar las curvas y las métricas del entrenamiento con lo que el manual pide interpretar (sobreajuste, equilibrio reconstructions/KL, diversidad) |
| Extensión del script base | `scripts/train_ae_vae.py` ahora guarda las curvas de entrenamiento, el MSE de reconstrucción en prueba y el panel con 16 ejemplos seleccionados por semilla (el base solo guardaba 12 ejemplos, sin curvas ni métricas de reconstrucción) |
| Documentación | `MODEL_CARD.md`, este documento, `REQUISITOS_LAB10.md` y el cierre interpretativo |
| Revisión de figuras | `reports/ae_vae_panel.png`, `reports/interpolation.png` y `reports/training_curves.png` revisadas para legibilidad |

## Prompts o decisiones relevantes

- Se siguió la estructura de cuatro niveles del manual: **observación → evidencia → interpretación →
  decisión**. Ninguna cifra del informe se escribe sin el archivo JSON o CSV que la respalda.
- Decisión técnica: evaluar el AE y el VAE sobre la **misma** selección de 1.000 ejemplos de prueba con
  `np.random.default_rng(42).choice`, para que el MSE de reconstrucción sea comparable; el panel usa
  los primeros 16 de esa misma selección.
- Decisión de alcance: no se/generated imágenes fuera del Fashion-MNIST ni se entrenó con datos
  sintéticos previos.

## Elementos verificados por el equipo

- [ ] `uv sync --extra cpu` y `uv run pytest -q` (aquí: **4 passed**, con TensorFlow 2.21.0).
- [ ] `uv run python scripts/check_runtime.py` para registrar el hardware y la versión de TensorFlow.
- [ ] `uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8` y contraste de
      `reports/training_metrics.json` con las cifras del informe.
- [ ] Revisión visual de las tres figuras del panel (`ae_vae_panel.png`, `interpolation.png`,
      `training_curves.png`): que no tengan textos encimados y que el panel no muestre solo los
      mejores casos.
- [ ] `uv run streamlit run app/streamlit_app.py`: comprobar que carga `models/decoder.keras` por ruta
      relativa y que no entrena al arrancar.
- [ ] Tomar las capturas con `cls` antes de cada una y nombrarlas con el prefijo del laboratorio.

## Cambios realizados sobre resultados generados

- Se añadió el cálculo de MSE de reconstrucción en 1.000 ejemplos de prueba y la divergencia KL media,
  porque el proyecto base solo reportaba tiempo de entrenamiento y número de parámetros.
- Se amplió el panel a 16 ejemplos y se añadió `reports/training_curves.png`.
- **No** se corrigió ni ajustó ninguna métrica para mejorar el desempeño: el VAE reproduce peor que el
  AE (MSE 0,03375 frente a 0,02048) y ese resultado **se publica tal cual**, con su explicación
  (la penalización KL compra espacio muestreable a cambio de nitidez).

## Responsabilidad asumida

El estudiante es responsable de: reproducir cada comando en su propio entorno, verificar que las
cifras del informe coinciden con los JSON/CSV, no atribuir al modelo propiedades que no se midieron
(diversidad, privacidad de las muestras) y declarar el uso del asistente de IA tal como aquí se
describe. Las cifras de este repositorio provienen de ejecuciones en esta máquina Linux y deben
reconfirmarse en el entorno de entrega del estudiante.


## Ejercicio 06 (consolidación) · prompts, verificaciones y correcciones

**Prompts relevantes que guiaron el trabajo**

1. «Consolida LAB10 en una entrega ejecutable: AE y VAE comparables, reconstrucción, muestreo,
   interpolación, métricas de fidelidad y diversidad, coste, memoria, Model Card y límites.»
2. «El panel debe construirse sobre una muestra con semilla fija, no sobre los mejores casos a mano.»
3. «Añade al script base lo que el manual exige y el base no guarda (curvas, MSE de reconstrucción) y
   documenta cada cambio.»
4. «Declara las herramientas de IA, los prompts relevantes, las partes verificadas y las correcciones
   realizadas; ninguna cifra puede quedar sin respaldo en un JSON o CSV del repositorio.»
5. «Una imagen sintética no es automáticamente privada, original ni libre de sesgo: esa limitación debe
   quedar escrita en la Model Card y en el informe.»

**Verificaciones que el estudiante debe repetir (casillas para marcar)**

- [ ] `uv sync --extra cpu` y `uv run pytest -q` → 4 passed.
- [ ] `uv run python scripts/check_runtime.py` → TensorFlow 2.21.0, `GPU: []`.
- [ ] `uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8` → AE MSE 0,02048 y VAE MSE
      0,03375 (idénticos a `reports/ejercicio06_metricas.json`).
- [ ] Ejecutar el notebook del Ejercicio 06 y revisar las tres figuras (panel, interpolación, curvas).
- [ ] Revisar que el panel usa la muestra con semilla 42 y no casos elegidos a mano.
- [ ] Regenerar `docs/EJERCICIO_06_PDF_BRIEF.pdf` y completar nombre, matrícula y enlace.

**Correcciones realizadas durante la ejecución**

| Corrección | Causa | Verificación |
|---|---|---|
| Curvas de entrenamiento guardadas en `reports/training_curves.png` | El manual exige las curvas y el base no las guardaba | Figura con leyenda y ejes rotulados |
| MSE de reconstrucción y divergencia KL añadidos a `reports/training_metrics.json` | Sin ellos no hay cifra para el informe | `ae_test_mse` 0,020484 y `vae_test_mse` 0,033752 |
| Panel ampliado a 16 ejemplos con semilla fija | El base usaba los 12 primeros del test | `reports/ae_vae_panel.png` con 16 columnas y cuatro filas |
| VAE del notebook con `train_step` propio | Una versión inline del notebook no calculaba pérdida y Keras fallaba con «No loss to compute» | Se adoptó la misma definición del script base del profesor |
| Métricas de diversidad y distancia al vecino más cercano | El manual pide diversidad; el base no la medía | `diversidad_media_entre_muestras` 4,8633 y `distancia_media_al_vecino_mas_cercano` 3,2987 |
| `requirements.txt` y `requirements-dev.txt` generados con `uv export` | El mandato los exige | Reproducibles desde `pyproject.toml` y `uv.lock` |
