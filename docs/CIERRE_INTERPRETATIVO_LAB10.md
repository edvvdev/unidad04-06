# LAB10 · Cierre interpretativo

Proyecto: `INF8239_U03_Generativos` · Evidencia: `reports/ejercicio06_metricas.json` y `reports/*.png` · Fecha: 2026-11-05

Todas las cifras de este cierre salen de una única corrida, la del **2026-11-05**, registrada en
`reports/ejercicio06_metricas.json` (campos `origen` y `fecha_ejecucion`).

## Resultado de reconstrucción

El autoencoder reconstruye con **MSE 0,02048** sobre 1.000 ejemplos de prueba y el VAE con
**MSE 0,03375** (`reports/ejercicio06_metricas.json`). El AE es mejor reconstructor: su cuello de
botella de 16 unidades conserva suficiente información visual para el dominio de origen. El VAE
pierde ~65 % más error de reconstrucción porque paga ese precio a cambio de un espacio latente
muestreable (KL medio 6,3744). Las curvas (`reports/ejercicio06_curvas.png`) muestran la pérdida de
entrenamiento (0,30604) y de validación (0,30710) prácticamente iguales en el AE: **no hay
sobreajuste visible**; en el VAE la pérdida total se estabiliza (273,61) y la reconstrucción
(267,35) y la KL (6,26) se mantienen en un equilibrio estable, sin que la penalización KL se dispare.

## Resultado de generación

El VAE **sí** genera: el decodificador produce imágenes distintas para puntos distintos del espacio
latente (medias de píxel 0,104 para (0,0), 0,053 para (1,5,−1,0) y 0,504 para (−2,0,1,5), en
`reports/ejercicio06_metricas.json`), y la interpolación entre dos extremos produce una transición
continua (`reports/ejercicio06_interpolacion.png`). Pero la calidad es la de un benchmark didáctico:
siluetas grises de 28×28, a veces reconocibles, a veces ambiguas. **Reconstruir bien y generar bien
son tareas distintas**: el error de reconstrucción del AE no dice nada sobre la calidad de las
muestras del VAE.

## Evidencia de diversidad

La diversidad media entre 64 muestras del decoder es **4,8633** (distancia media euclídea en el espacio de píxeles aplanados). La distancia media al vecino más cercano es **3,2987**, calculada **contra 1.000 imágenes del conjunto de prueba** y **no** contra el conjunto de entrenamiento: por eso **no permite descartar la memorización** y no es concluyente como indicador. La línea base es la distancia media entre pares de ese mismo conjunto de prueba, **11,3984** (`distancia_media_entre_pares_de_prueba`): con ella se ve que las muestras del decoder se concentran en una región más estrecha del espacio de píxeles que los datos reales (4,8633 frente a 11,3984), coherente con la repetición de patrones visible en el panel y con un espacio latente de solo 2 dimensiones, y que el 3,2987 se explica por esa concentración y no por una copia de una imagen de prueba.

Esa es la evidencia cuantitativa de diversidad de la entrega: distancia media entre muestras
(4,8633), distancia media al vecino más cercano calculada contra el conjunto de prueba (3,2987) y
línea base de pares de ese mismo conjunto de prueba (11,3984). La evidencia cualitativa la aporta el
panel `reports/ejercicio06_panel.png`: 16 ejemplos con semilla fija (semilla 42, sin selección
manual de los mejores casos), con las cuatro filas originales / AE / VAE / muestras nuevas, donde se
observa variedad entre clases de prenda (calzado, prendas de punto, bolsos) y también repetición de
patrones.

Con estas cifras la afirmación defendible es «hay diversidad medible, pero menor que la de los datos
reales y sin verificación contra el conjunto de entrenamiento», y no «el VAE no colapsa».

## Comportamiento del espacio latente

El espacio latente tiene **2 dimensiones**, suficiente para visualizar el decodificador pero
demasiado estrecho para la variedad real de las prendas. Las proyecciones se agrupan y las
interpolaciones son graduales pero no siempre plausibles: entre dos puntos las formas cambian de
prenda de forma abrupta. Los ejes **no tienen interpretación semántica** (no son «tipo de prenda» ni
«color»); lo que existe es la función que aprendió el decodificador.

## Costo computacional

| Elemento | Valor |
|---|---|
| Tiempo de entrenamiento AE (6 épocas) | 7,63 s |
| Tiempo de entrenamiento VAE (8 épocas) | 13,21 s |
| Parámetros AE / decoder | 25.888 / 101.520 |
| Tamaño `autoencoder.keras` / `decoder.keras` | 331 KB / 416 KB (se regeneran; no se versionan) |
| Hardware | CPU de escritorio, sin GPU (`scripts/check_runtime.py`: TensorFlow 2.21.0, `GPU: []`) |
| Entrenamiento total | 20,85 s; la aplicación solo hace inferencia |

## Riesgo o limitación

1. **Privacidad no garantizada:** una imagen sintética puede parecerse a un ejemplo de entrenamiento;
   el análisis de vecinos cercanos de esta entrega se calculó contra el **conjunto de prueba**, de
   modo que **no permite descartar la memorización**. No se afirma privacidad.
2. **Sesgo heredado:** el modelo reproduce la distribución de Fashion-MNIST (ropa y calzado de
   catálogo europeo); cualquier uso fuera de ese dominio es inválido.
3. **Fidelidad:** 28×28 en escala de grises; unsuitable para imágenes médicas, de vigilancia, de
   documentos o de OCR.
4. **Espacio latente de 2 dimensiones:** limita la diversidad y hace que la interpolación sea una
   demostración, no una herramienta de síntesis.
5. **Diversidad por debajo de la de los datos:** las 64 muestras del decoder están a 4,8633 entre sí,
   frente a la línea base de 11,3984 de los datos de prueba. La concentración es compatible con un
   colapso parcial que la regularización KL (6,37) no evita.

## Decisión sobre uso previsto

Se **acepta** el uso del AE/VAE como material didáctico del Ejercicio 06: ilustrar qué es
reconstruir, qué es muestrear y por qué un error bajo no garantiza generación plausible. Se
**rechaza** cualquier uso que requiera originalidad, privacidad, diversidad o fidelidad: si se
necesitara, el siguiente experimento es (1) aumentar el espacio latente a 16 dimensiones y repetir
la comparación de KL, (2) repetir el análisis de vecinos cercanos **contra el conjunto de
entrenamiento** y ampliar la línea base a los 10.000 ejemplos de prueba, y (3) marcar explícitamente
toda salida sintética como tal en la interfaz y en el informe.
