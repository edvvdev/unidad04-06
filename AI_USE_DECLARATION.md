# Declaración de uso de inteligencia artificial · LAB10 / Ejercicio 06

Proyecto: `INF8239_U03_Generativos` (Unidad 03) · Curso: INF-8239 Ciencia de Datos II
Fecha: 2026-11-05

## Herramientas utilizadas

- **Kilo** (asistente de IA en CLI, modelo MiniMax-M3) como apoyo de análisis, documentación y
  redacción de las tarjetas del proyecto.
- **Python 3.12 + TensorFlow 2.21 (extra `cpu`)** mediante `uv`, y las librerías del proyecto base
  (NumPy, pandas, scikit-learn, matplotlib, Streamlit) para el entrenamiento, las métricas y las
  figuras. No se usaron servicios en la nube, ni APIs externas, ni datasets generados por IA.

## Propósito de cada uso

| Uso | Detalle |
|---|---|
| Interpretación | Contrastar las curvas y las métricas del entrenamiento con lo que el manual pide interpretar (sobreajuste, equilibrio reconstrucción/KL, diversidad) |
| Extensión del script base | `scripts/train_ae_vae.py` ahora guarda las curvas de entrenamiento, el MSE de reconstrucción en prueba y el panel con 16 ejemplos seleccionados por semilla (el base solo guardaba 12 ejemplos, sin curvas ni métricas de reconstrucción) |
| Documentación | `MODEL_CARD.md`, este documento, `REQUISITOS_LAB10.md` y el cierre interpretativo |
| Revisión de figuras | `reports/ejercicio06_panel.png`, `reports/ejercicio06_interpolacion.png` y `reports/ejercicio06_curvas.png` revisadas para legibilidad |

## Prompts o decisiones relevantes

- Se siguió la estructura de cuatro niveles del manual: **observación → evidencia → interpretación →
  decisión**. Ninguna cifra del informe se escribe sin el archivo JSON del repositorio que la
  respalda (`reports/ejercicio06_metricas.json`).
- Decisión técnica: evaluar el AE y el VAE sobre la **misma** selección de 1.000 ejemplos de prueba con
  `np.random.default_rng(42).choice`, para que el MSE de reconstrucción sea comparable; el panel usa
  los primeros 16 de esa misma selección.
- Decisión de alcance: no se generaron imágenes fuera del Fashion-MNIST ni se entrenó con datos
  sintéticos previos.

## Elementos verificados por el equipo

- [x] `uv sync --extra cpu` → `Resolved 88 packages in 0.64ms` / `Checked 74 packages in 0.67ms`
      (la lista termina en `- wcwidth==0.9.4`, `- webcolors==25.10.0`, `- webencodings==0.6.1`),
      sin errores.
- [x] `uv run pytest -q` → `11 passed in 6.21s` (TensorFlow 2.21.0).
- [x] `uv run python scripts/check_runtime.py` →
      `Python/OS: 3.12.15 Linux-6.12.113+deb13-amd64-x86_64-with-glibc2.41`,
      `TensorFlow: 2.21.0`, `GPU: []`.
- [x] `uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8` → escribe
      `reports/ejercicio06_metricas.json`; AE MSE **0,02048** y VAE MSE **0,03375**, idénticos a los
      que aparecen en la Model Card y en el PDF (el JSON es la única fuente de verdad). El script y el
      notebook devuelven los mismos valores (`diversidad_media_entre_muestras` 4,8633,
      `distancia_media_al_vecino_mas_cercano` 3,2987, `distancia_media_entre_pares_de_prueba` 11,3984)
      porque siembran con la misma semilla y `TF_DETERMINISTIC_OPS=1`.
- [x] Revisión visual de las tres figuras del panel (`reports/ejercicio06_panel.png`,
      `reports/ejercicio06_interpolacion.png`, `reports/ejercicio06_curvas.png`): no tienen textos
      encimados y el panel no muestra solo los mejores casos.
- [x] `uv run streamlit run app/streamlit_app.py` → arranca en `http://localhost:8599` y responde
      `HTTP 200`; carga `models/decoder.keras` por ruta relativa (salida `(1, 28, 28, 1)` para un
      punto latente) y no entrena al arrancar.
- [x] `uv run ruff check .` → `All checks passed!` (sin hallazgos).
- [ ] Tomar las capturas con `cls` antes de cada una y nombrarlas con el prefijo del laboratorio.
      *Pendiente del estudiante: no se adjuntan capturas en este repositorio.*

## Cambios realizados sobre resultados generados

- Se añadió el cálculo de MSE de reconstrucción en 1.000 ejemplos de prueba y la divergencia KL media,
  porque el proyecto base solo reportaba tiempo de entrenamiento y número de parámetros.
- Se amplió el panel a 16 ejemplos y se añadió `reports/ejercicio06_curvas.png`.
- Se añadió la distancia media al vecino más cercano **contra el conjunto de prueba** y la línea base
  de pares de ese mismo conjunto de prueba, para que la cifra del vecino más cercano sea
  interpretable.
- **No** se corrigió ni ajustó ninguna métrica para mejorar el desempeño: el VAE reproduce peor que el
  AE (MSE 0,03375 frente a 0,02048) y ese resultado **se publica tal cual**, con su explicación
  (la penalización KL compra espacio muestreable a cambio de nitidez).

## Responsabilidad asumida

El estudiante es responsable de: reproducir cada comando en su propio entorno, verificar que las
cifras del informe coinciden con `reports/ejercicio06_metricas.json`, no atribuir al modelo
propiedades que la evidencia no sostiene (la distancia al vecino más cercano se calculó contra el
conjunto de prueba, así que no permite afirmar privacidad; la diversidad medida es menor que la de
los datos reales) y declarar el uso del asistente de IA tal como aquí se describe. Las cifras de este
repositorio provienen de ejecuciones en esta máquina Linux y deben reconfirmarse en el entorno de
entrega del estudiante.

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
6. «Corrige los defectos de la entrega: contradicción sobre diversidad y vecinos cercanos, valor de
   diversidad repetido en la columna del AE, rutas de figuras divergentes, dos juegos de tiempos,
   casillas de verificación sin marcar, lógica del VAE duplicada, `ruff check` en rojo y artefactos
   de `models/` que no se versionan sin decirlo.»

**Verificaciones que el estudiante debe repetir (casillas para marcar)**

- [x] `uv sync --extra cpu` y `uv run pytest -q` → `11 passed in 6.21s`.
- [x] `uv run python scripts/check_runtime.py` → `TensorFlow: 2.21.0`, `GPU: []`.
- [x] `uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8` → AE MSE 0,02048 y VAE MSE
      0,03375 (idénticos a `reports/ejercicio06_metricas.json`).
- [x] Ejecutar el notebook del Ejercicio 06 y revisar las tres figuras (panel, interpolación, curvas)
      → 15 celdas ejecutadas, 0 celdas en error.
- [x] Revisar que el panel usa la muestra con semilla 42 y no casos elegidos a mano.
- [x] Regenerar `docs/EJERCICIO_06_PDF_BRIEF.pdf` y completar el nombre y el enlace del repositorio
      → `PDF generado: /home/trek/source/uasd/cienciadedatos/tareas/Unidad_3/INF8239_U03_Generativos/docs/EJERCICIO_06_PDF_BRIEF.pdf`
      (4 páginas, `pdftotext` devuelve el nombre «Edwing Perez», la URL del repositorio, 4,8633, 3,2987
      y 11,3984).

**Correcciones realizadas durante la ejecución**

| Corrección | Causa | Verificación |
|---|---|---|
| Curvas de entrenamiento guardadas en `reports/ejercicio06_curvas.png` | El manual exige las curvas y el base no las guardaba | Figura con leyenda y ejes rotulados |
| MSE de reconstrucción y divergencia KL añadidos a `reports/ejercicio06_metricas.json` | Sin ellos no hay cifra para el informe | `mse_reconstruccion_ae` 0,020484 y `mse_reconstruccion_vae` 0,033752 |
| Panel ampliado a 16 ejemplos con semilla fija | El base usaba los 12 primeros del test | `reports/ejercicio06_panel.png` con 16 columnas y cuatro filas |
| VAE del notebook con `train_step` propio | Una versión inline del notebook no calculaba pérdida y Keras fallaba con «No loss to compute» | La clase pasó a `src/inf8239_u03_gen/models.py` (`build_vae`) y la usan script y notebook |
| Métricas de diversidad y distancia al vecino más cercano | El manual pide diversidad; el base no la medía | `diversidad_media_entre_muestras` 4,8633 y `distancia_media_al_vecino_mas_cercano` 3,2987 |
| Línea base de pares del conjunto de prueba | Sin ella, el 3,2987 no era interpretable | `distancia_media_entre_pares_de_prueba` 11,3984 |
| Párrafo canónico de diversidad y vecinos cercanos | La documentación afirmaba que no se habían medido | Texto único en `MODEL_CARD.md`, citado literalmente en el cierre y en el PDF |
| Una sola fuente de verdad de métricas (`reports/ejercicio06_metricas.json`) | Había dos juegos de tiempos de corridas distintas | Se eliminó `reports/training_metrics.json`; el JSON lleva `origen` y `fecha_ejecucion` |
| Rutas de figuras unificadas (`reports/ejercicio06_*.png`) | El PDF y los documentos citaban juegos distintos | Se eliminaron `ae_vae_panel.png`, `interpolation.png` y `training_curves.png` |
| `requirements.txt` y `requirements-dev.txt` generados con `uv export` | El mandato los exige | Reproducibles desde `pyproject.toml` y `uv.lock` |
