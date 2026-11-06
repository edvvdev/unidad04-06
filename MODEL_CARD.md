# Model Card · Autoencoder y VAE (Fashion-MNIST)

Proyecto: `INF8239_U03_Generativos` · Unidad 03 · Fecha de la evaluación: **2026-11-05**

La evaluación corresponde a **una sola corrida** de la entrega. La fuente de verdad de todas las
cifras es `reports/ejercicio06_metricas.json`, que registra el origen de los datos
(`origen`), la marca de tiempo de la corrida (`fecha_ejecucion`) y el resto de métricas. Ningún
número de este documento procede de otra corrida.

## Modelos y versiones

| Modelo | Arquitectura | Parámetros | Artefacto |
|---|---|---|---|
| Autoencoder (AE) | Flatten(784) → Dense(16, relu) → Dense(784, sigmoid) → Reshape(28×28×1) | 25.888 | `models/autoencoder.keras` (331 KB) |
| VAE (encoder + decoder) | Encoder: Dense(128, relu) → `z_mean`(2) + `z_log_var`(2); Decoder: Dense(128, relu) → Dense(784, sigmoid) | Decoder: 101.520 | `models/decoder.keras` (416 KB) |

- Entrenamiento: AE 6 épocas y VAE 8 épocas, optimizador Adam, lote 256, `validation_split=0.1` con
  `EarlyStopping(patience=2, restore_best_weights=True)` en el AE, regularización KL con la
  reparametrización habitual (`z = media + exp(0.5·log_var)·ε`).
- La lógica del VAE (muestreo, reparametrización, pérdidas y `train_step`) está en
  `src/inf8239_u03_gen/models.py` (`build_vae`): la importan tanto `scripts/train_ae_vae.py` como el
  notebook, de modo que no hay dos copias que puedan divergir.
- Semilla global 42 (`SEED` en `scripts/train_ae_vae.py`, `TF_DETERMINISTIC_OPS=1`).
- Versiones del entorno: Python 3.12, TensorFlow 2.21.0 (extra `cpu`), NumPy 2.x, scikit-learn 1.5+.

### Los artefactos se regeneran; no se versionan

`models/autoencoder.keras` y `models/decoder.keras` **no están versionados en Git**: el
`.gitignore` del proyecto ignora `models/*` (salvo `.gitkeep`) porque son binarios de 331 KB y
416 KB. Se **regeneran** con `uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8` o
al ejecutar el notebook, que además escribe `reports/ejercicio06_metricas.json` y las tres figuras
de `reports/`. Los tamaños citados en esta tarjeta corresponden a los artefactos de la corrida
evaluada, no a archivos que árboles el repositorio.

## Uso previsto y usuarios

Uso **didáctico**: comparar reconstrucción (AE) con representación probabilística y muestreo (VAE)
sobre un benchmark público, y practicar la lectura crítica de un espacio latente de 2 dimensiones.
Usuario previsto: estudiante de INF-8239 y docente evaluador del laboratorio.

## Usos fuera de alcance

- **No** es generador de imágenes de producto, de moda realista ni de personas: las salidas son
  28×28 en escala de grises y de baja fidelidad.
- **No** es applicable a imágenes médicas, de vigilancia, de documentos o de OCR.
- **No** garantiza privacidad: una imagen sintética no es automáticamente privada ni original, y el
  análisis de vecinos cercanos de esta entrega se calculó contra el conjunto de prueba, no contra
  el de entrenamiento (ver «Diversidad y vecinos cercanos»).
- **No** se puede afirmar que el VAE «maximiza nitidez» ni que «garantiza diversidad»: la diversidad
  medida entre sus muestras queda por debajo de la línea base de los datos reales.

## Dataset, licencia y particiones

| Aspecto | Detalle |
|---|---|
| Dataset | Fashion-MNIST (Zalando), 60.000 entrenamiento + 10.000 prueba, 28×28, escala de grises, 10 clases |
| Licencia | MIT (declarada por los autores del dataset) |
| Preprocesamiento | Normalización a [0,1] en `src/inf8239_u03_gen/data.py`; `validate_images` comprueba forma y rango |
| Particiones | La prueba (10.000) se reserva para la evaluación final; el AE usa 10 % de entrenamiento como validación |
| Selección de ejemplos | 1.000 ejemplos de prueba elegidos con `np.random.default_rng(42).choice` (semilla fija, **sin** selección manual) |

## Arquitectura y espacio latente

El espacio latente del VAE tiene **2 dimensiones**, suficiente para visualizar pero insuficiente para
representar la variedad de las prendas reales: las proyecciones se agrupan y las transiciones entre
puntos interpolados no siempre son plausibles. Los ejes del espacio latente **no tienen significado
interpretable** (no son «tipo de prenda» ni «color»); el decodificador es el que define qué región
produce qué imagen.

## Métricas de reconstrucción y generación

`reports/ejercicio06_metricas.json` (corrida del 2026-11-05):

| Métrica | Valor | Lectura |
|---|---|---|
| MSE de reconstrucción (AE) en prueba | 0,02048 | El AE conserva información visual suficiente para ejemplos del propio dominio |
| MSE de reconstrucción (VAE) en prueba | 0,03375 | **Peor que el AE**: la penalización KL compra espacio muestreable a costa de nitidez |
| KL media en prueba (VAE) | 6,3744 | El latente se aleja de la referencia N(0,1); un KL mayor daría más regularidad y menos información |
| Pérdida de reconstrucción final (VAE) | 267,3488 | Pérdida suma por imagen, no MSE; no es comparable con el MSE |
| KL final por época (VAE) | 6,2594 | Estable: el equilibrio reconstrucción/KL no se dispara |
| Loss final de entrenamiento (AE) | 0,30604 | — |
| Loss final de validación (AE) | 0,30710 | Validez ≈ entrenamiento: **no hay sobreajuste visible** |

Sobre el AE y el VAE: el **error bajo no implica que el modelo genere muestras nuevas plausibles**
(error de reconstrucción y calidad de muestreo son tareas distintas, igual que en LAB09 RMSE y
HitRate@10 lo eran).

## Diversidad y vecinos cercanos

- Panel `reports/ejercicio06_panel.png`: 16 ejemplos de prueba con semilla fija, en cuatro filas
  (originales, reconstrucción AE, reconstrucción VAE, muestras nuevas del decoder). Al ser una
  selección aleatoria con semilla, **no** muestra solo los mejores casos.
- Interpolación `reports/ejercicio06_interpolacion.png`: 16 puntos entre dos extremos del espacio
  latente. La transición es gradual en el recorrido, lo que indica continuidad del decodificador,
  **no** que las imágenes intermedias sean plausibles.

### Párrafo canónico · diversidad y vecinos cercanos

La diversidad media entre 64 muestras del decoder es **4,8633** (distancia media euclídea en el espacio de píxeles aplanados). La distancia media al vecino más cercano es **3,2987**, calculada **contra 1.000 imágenes del conjunto de prueba** y **no** contra el conjunto de entrenamiento: por eso **no permite descartar la memorización** y no es concluyente como indicador. La línea base es la distancia media entre pares de ese mismo conjunto de prueba, **11,3984** (`distancia_media_entre_pares_de_prueba`): con ella se ve que las muestras del decoder se concentran en una región más estrecha del espacio de píxeles que los datos reales (4,8633 frente a 11,3984), coherente con la repetición de patrones visible en el panel y con un espacio latente de solo 2 dimensiones, y que el 3,2987 se explica por esa concentración y no por una copia de una imagen de prueba.

Este párrafo es el texto canónico del repositorio: se cita literalmente en
`docs/CIERRE_INTERPRETATIVO_LAB10.md` y en `docs/EJERCICIO_06_PDF_BRIEF.pdf`. Las tres cifras
están en `reports/ejercicio06_metricas.json` con las claves `diversidad_media_entre_muestras`,
`distancia_media_al_vecino_mas_cercano` y `distancia_media_entre_pares_de_prueba`.

## Hardware, tiempo y tamaño

| Elemento | Valor |
|---|---|
| Hardware | CPU de escritorio (sin GPU detectada); `scripts/check_runtime.py` reporta el entorno |
| Tiempo de entrenamiento AE | 7,63 s (6 épocas) |
| Tiempo de entrenamiento VAE | 13,21 s (8 épocas) |
| Tiempo de inferencia | 0,91 ms por lote de 100 imágenes |
| Tamaño del modelo AE | 331 KB (`models/autoencoder.keras`, regenerado, no versionado) |
| Tamaño del decoder | 416 KB (`models/decoder.keras`, regenerado, no versionado) |
| Curvas | `reports/ejercicio06_curvas.png` (pérdida AE con validación; VAE con total, reconstrucción y KL) |

Los tiempos son los de la corrida del **2026-11-05** (marcas de tiempo exactas en
`fecha_ejecucion` de `reports/ejercicio06_metricas.json`): son la única medición de costo de la
entrega y no se mezclan con ninguna otra corrida. En esa misma corrida el entrenamiento completo
sumó 20,85 s y la inferencia es la parte ligera.

## Limitaciones, riesgos y supervisión

| Riesgo / limitación | Descripción | Mitigación |
|---|---|---|
| Privacidad no garantiada | El análisis de vecinos cercanos (3,2987) se calculó contra el conjunto de prueba, no contra el de entrenamiento | Antes de cualquier uso de datos sintéticos, repetir el análisis contra el conjunto de entrenamiento completo |
| Diversidad por debajo de la de los datos | Las 64 muestras del decoder están a 4,8633 entre sí, frente a la línea base de 11,3984 de los datos de prueba | Declarar que las muestras son menos diversas que los datos y no presentarlas como sustituto del dataset |
| Sesgo heredado | El modelo reproduce la distribución y los sesgos de Fashion-MNIST | Declarar el dataset y su licencia en toda comunicación |
| Fidelidad baja | 28×28, grises, formas simples | No extrapolar a dominios photography/industrial |
| Espacio latente de 2 dimensiones | Interpolar no garantiza pluralistic | Usar las imágenes como ilustración, no como evidencia de calidad |
| Coste | Entrenamiento CPU de 20,85 s en total; la inferencia es la parte ligera | La aplicación no entrena, solo infiere |
| Uso responsable | Los datos generados pueden confundirse con imágenes reales | Marcar siempre las salidas como sintéticas |
