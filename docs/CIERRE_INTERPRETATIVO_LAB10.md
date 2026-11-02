# LAB10 · Cierre interpretativo

Proyecto: `INF8239_U03_Generativos` · Evidencia: `reports/`, `models/` · Fecha: 2026-11-01

## Resultado de reconstrucción

El autoencoder reconstruye con **MSE 0,02048** sobre 1.000 ejemplos de prueba y el VAE con
**MSE 0,03375** (`reports/training_metrics.json`). El AE es mejor reconstructor: su cuello de
botella de 16 unidades conserva suficiente información visual para el dominio de origen. El VAE
pierde ~65 % más error de reconstrucción porque paga ese precio a cambio de un espacio latente
muestreable (KL medio 6,3744). Las curvas (`reports/training_curves.png`) muestran la pérdida de
entrenamiento (0,30604) y de validación (0,30710) prácticamente iguales en el AE: **no hay
sobreajuste visible**; en el VAE la pérdida total se estabiliza (273,60) y la reconstructionsim KL
(267,35 y 6,26) se mantiene en un equilibrio estable, sin que la penalización KL se dispare.

## Resultado de generación

El VAE **sí** genera: el decodificador produce imágenes distintas para puntos distintos del espacio
latente (verificado: punto (0,0) con media 0,104; (1,5,−1,0) con media 0,053; (−2,0,1,5) con media
0,504), y la interpolación entre dos extremos produce una transición continua
(`reports/interpolation.png`). Pero la calidad es la de un benchmark didáctico: siluetas grises de
28×28, a veces reconocibles, a veces ambiguas. **Reconstruir bien y generar bien son tareas
distintas**: el error de reconstrucción del AE no dice nada sobre la calidad de las muestras del VAE.

## Evidencia de diversidad

La evidencia disponible es **cualitativa**: el panel `reports/ae_vae_panel.png` muestra 16 ejemplos
seleccionados con semilla fija (semilla 42, sin selección manual de los mejores casos), con las
cuatro filas originales / AE / VAE / muestras nuevas. Se observa variedad entre clases de prenda
(calzado, prendas de punto, bolsos) y también **repetición de patrones**, que es la señal visual de
posible colapso. Lo que **no** se midió: una métrica cuantitativa de diversidad (por ejemplo,
distancia media entre muestras) ni un análisis de vecinos cercanos en el conjunto de entrenamiento.
Por eso la afirmación defendible es «hay diversidad observable pero no cuantificada», y no «el VAE
no colapsa».

## Comportamiento del espacio latente

El espacio latente tiene **2 dimensiones**, suficiente para visualizar el decodificador pero
demasiado estrecho para la variedad real de las prendas. Las proyecciones se agrupan y las
interpolaciones son graduales pero no siempre plausibles: entre dos puntos las formas cambian de
prenda de forma abrupta. Los ejes **no tienen interpretación semántica** (no son «tipo de prenda» ni
«color»); lo que existe es la función que aprendió el decodificador.

## Costo computacional

| Elemento | Valor |
|---|---|
| Tiempo de entrenamiento AE (6 épocas) | 10,42 s |
| Tiempo de entrenamiento VAE (8 épocas) | 17,23 s |
| Parámetros AE / decoder | 25.888 / 101.520 |
| Tamaño `autoencoder.keras` / `decoder.keras` | 331 KB / 416 KB |
| Hardware | CPU de escritorio, sin GPU (`scripts/check_runtime.py`: TensorFlow 2.21.0, `GPU: []`) |
| Entrenamiento total | ~28 s; la aplicación solo hace inferencia |

## Riesgo o limitación

1. **Privacidad no garantizada:** una imagen sintética puede parecerse a un ejemplo de entrenamiento;
   sin análisis de vecinos cercanos no se puede descartar la memorización. No se afirma privacidad.
2. **Sesgo heredado:** el modelo reproduce la distribución de Fashion-MNIST (ropa y calzado de
   catálogo europeo); cualquier uso fuera de ese dominio es inválido.
3. **Fidelidad:** 28×28 en escala de grises; unsuitable para imágenes médicas, de vigilancia, de
   documentos o de OCR.
4. **Espacio latente de 2 dimensiones:** limita la diversidad y hace que la interpolación sea una
   demostración, no una herramienta de síntesis.
5. **Colapso no descartado:** la regularización KL (6,37) es moderada y el espacio de 2 dimensiones
   concentra muestras.

## Decisión sobre uso previsto

Se **acepta** el uso del AE/VAE como material didáctico del Ejercicio 06: ilustrar qué es
reconstruir, qué es muestrear y por qué un error bajo no garantiza generación plausible. Se
**rechaza** cualquier uso que requiera originalidad, privacidad, diversidad o fidelidad: si se
necesitara, el siguiente experimento es (1) aumentar el espacio latente a 16 dimensiones y repetir la
comparación de KL, (2) añadir métrica de diversidad y análisis de vecinos cercanos, y (3) marcar
explícitamente toda salida sintética como tal en la interfaz y en el informe.
