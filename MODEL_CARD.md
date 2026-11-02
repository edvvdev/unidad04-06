# Model Card · Autoencoder y VAE (Fashion-MNIST)

Proyecto: `INF8239_U03_Generativos` · Unidad 03 · Fecha de la evaluación: 2026-11-01

## Modelos y versiones

| Modelo | Arquitectura | Parámetros | Artefacto |
|---|---|---|---|
| Autoencoder (AE) | Flatten(784) → Dense(16, relu) → Dense(784, sigmoid) → Reshape(28×28×1) | 25.888 | `models/autoencoder.keras` (331 KB) |
| VAE (encoder + decoder) | Encoder: Dense(128, relu) → `z_mean`(2) + `z_log_var`(2); Decoder: Dense(128, relu) → Dense(784, sigmoid) | Decoder: 101.520 (encoder ~24.9 KB) | `models/decoder.keras` (416 KB) |

- Entrenamiento: AE 6 épocas y VAE 8 épocas, optimizador Adam, lote 256, `validation_split=0.1` con
  `EarlyStopping(patience=2, restore_best_weights=True)` en el AE, regularización KL con la
  reparametrización habitual (`z = mean + exp(0.5·log_var)·ε`).
- Semilla global 42 (`SEED` en `scripts/train_ae_vae.py`, `TF_DETERMINISTIC_OPS=1`).
- Versiones del entorno: Python 3.12, TensorFlow 2.21.0 (extra `cpu`), NumPy 2.x, scikit-learn 1.5+.

## Uso previsto y usuarios

Uso **didáctico**: comparar reconstrucción (AE) con representación probabilística y muestreo (VAE)
sobre un benchmark público, y practicar la lectura crítica de un espacio latente de 2 dimensiones.
Usuario previsto: estudiante de INF-8239 y docente evaluador del laboratorio.

## Usos fuera de alcance

- **No** es generador de imágenes de producto, de moda realista ni de personas: las salidas son
  28×28 en escala de grises y de baja fidelidad.
- **No** es applicable a imágenes médicas, de vigilancia, de documentos o de OCR.
- **No** garantiza privacidad: una imagen sintética no es automáticamente privada ni original.
- **No** se puede afirmar que el VAE "maximiza nitidez" ni que "garantiza diversidad" (ver el análisis
  de diversidad más abajo).

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
interpretable** (no son "tipo de prenda" ni "color"); el decodificador es el que define qué región
produce qué imagen.

## Métricas de reconstrucción y generación

`reports/training_metrics.json` (ejecución del 2026-11-01):

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

- Panel `reports/ae_vae_panel.png`: 16 ejemplos de prueba con semilla fija, en cuatro filas
  (originales, reconstrucción AE, reconstrucción VAE, muestras nuevas del decoder). Al ser una
  selección aleatoria con semilla, **no** muestra solo los mejores casos.
- Interpolación `reports/interpolation.png`: 16 puntos entre dos extremos del espacio latente. La
  transición es gradual en el recorrido, lo que indica continuidad del decodificador, **no** que las
  imágenes intermedias sean plausibles.
- No se ejecutó un análisis de vecinos cercanos (nearest neighbors en el conjunto de entrenamiento) ni
  una métrica de diversidad (por ejemplo, distancia media entre muestras). Esa es la principal
  evidencia que **falta** para sostener afirmaciones sobre diversity, y queda como siguiente
  experimento.

## Hardware, tiempo y tamaño

| Elemento | Valor |
|---|---|
| Hardware | CPU de escritorio (sin GPU detectada); `scripts/check_runtime.py` reporta el entorno |
| Tiempo de entrenamiento AE | 10,42 s (6 épocas) |
| Tiempo de entrenamiento VAE | 17,23 s (8 épocas) |
| Tamaño del modelo AE | 331 KB (`models/autoencoder.keras`) |
| Tamaño del decoder | 416 KB (`models/decoder.keras`) |
| Curvas | `reports/training_curves.png` (pérdida AE con validación; VAE con total, reconstrucción y KL) |

## Limitaciones, riesgos y supervisión

| Riesgo / limitación | Descripción | Mitigación |
|---|---|---|
| Privacidad no garantizada | Una muestra puede parecerse a un ejemplo de entrenamiento | Analizar vecinos cercanos antes de cualquier uso de datos sintéticos |
| Sesgo heredado | El modelo reproduce la distribución y los sesgos de Fashion-MNIST | Declarar el dataset y su licencia en toda comunicación |
| Fidelidad baja | 28×28, grises, formas simples | No extrapolar a dominios photography/industrial |
| Espacio latente de 2 dimensiones | Interpolar no garantiza pluralistic | Usar las imágenes como ilustración, no como evidencia de calidad |
| Coste | Entrenamiento CPU de ~28 s en total; la inferencia es la parte ligera | La aplicación no entrena, solo infiere |
| Uso responsable | Los datos generados pueden confundirse con imágenes reales | Marcar siempre las salidas como sintéticas |
