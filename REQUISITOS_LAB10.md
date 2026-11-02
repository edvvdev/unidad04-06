# LAB10 · Requisitos y evidencia (Unidad 03 · Manual de laboratorios guiados)

Fuente: `~/Downloads/Unidad 03. Manual laboratorios guiados_.pdf` (sección `U03.LAB10`, págs. 12–14).
Proyecto base oficial: **INF8239_U03_Generativos_Proyecto_Base_uv.zip** (TensorFlow aislado del proyecto de recomendadores).

## Producto del laboratorio
Autoencoder y VAE comparados mediante reconstrucción, generación, espacio latente y costo; aplicación de muestreo, pruebas, Model Card y declaración de uso de IA.

## Pasos y requisitos
1. **Entorno CPU/GPU**: `uv sync --extra cpu` (vía CPU) o `uv sync --extra gpu` (WSL2/Linux con NVIDIA). Luego `uv run python scripts/check_runtime.py`. La GPU es opcional; en CPU usar menos épocas o `--limit`. No mezclar instalaciones globales de TensorFlow con el entorno del proyecto.
2. **Dataset y preprocesamiento**: Fashion-MNIST (60.000 / 10.000, 28×28, grises, licencia MIT); normalizar a [0,1] y reservar prueba para la evaluación final.
   `uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8`
   Resultado esperado: curvas, reconstrucciones, muestras, interpolaciones, métricas, el decodificador y los modelos entrenados (los valores exactos dependen de hardware, semilla y épocas).
3. **Interpretar el autoencoder**: reconstrucción nítida (el cuello de botella conserva información) · salida borrosa (la pérdida favorece promedios o el latente es limitado) · entrenamiento mejora y validación empeora (posible sobreajuste; conservar el mejor punto) · error bajo **no** implica que el modelo genere muestras nuevas plausibles.
4. **Interpretar el VAE**: la pérdida total combina reconstrucción y KL. Preguntas: ¿cuánta información visual se pierde? ¿el latente se aproxima a la distribución de referencia? ¿hay diversidad o salidas casi idénticas? ¿la interpolación es gradual y plausible? ¿la muestra parece copiar un ejemplo o producir una variación?
   Advertencia: una imagen sintética no es automáticamente privada, original, verdadera ni libre de sesgo; hay que analizar la similitud con el entrenamiento y declarar el origen de los datos.
5. **Comparar AE, VAE, GAN y transferencia de aprendizaje**, con su objetivo dominante y lo que **no** debe afirmarse de cada uno (AE: que toda decodificación sea plausible; VAE: que maximiza nitidez o garantiza diversidad; GAN: que evita colapso o es fácil de evaluar; Transfer: que elimina validar el nuevo dominio).
6. **Paneles no seleccionados**: semilla fija para elegir **16 ejemplos de prueba**; mostrar originales, reconstrucciones AE, reconstrucciones VAE y muestras nuevas; **no elegir manualmente solo los mejores**; describir diversidad, fallos y clases visualmente confusas; comparar con vecinos cercanos del entrenamiento.
7. **Aplicación local y nube**: `uv run streamlit run app/streamlit_app.py` (carga `models/decoder.keras` por ruta relativa) y `uv export --format requirements.txt --extra cpu --output-file requirements-cloud.txt`. En nube con memoria limitada: reducir dependencias y no entrenar en cada visita. Secretos en la plataforma, nunca en Git.
8. **Pruebas, Model Card y Git**: `uv run pytest -q`; commits `chore: initialize isolated generative project` y `feat: compare autoencoder and variational autoencoder` (incluye `MODEL_CARD.md`, `AI_USE_DECLARATION.md`, `reports/`, `models/`).

## Cierre interpretativo (redacción obligatoria)
Resultado de reconstrucción · Resultado de generación · Evidencia de diversidad · Comportamiento del espacio latente · Costo computacional · Riesgo o limitación · Decisión sobre uso previsto.

## Evidencia para entregar (Ejercicio 06)
Paneles no seleccionados, interpolación, métricas, costo, pruebas, aplicación local, Model Card, declaración de IA y conclusión crítica.
