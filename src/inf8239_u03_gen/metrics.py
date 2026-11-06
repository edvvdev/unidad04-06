"""Métricas de diversidad, vecinos cercanos y reporte canónico de la entrega."""

from __future__ import annotations

from datetime import datetime

import numpy as np

CLAVES_CANONICAS = (
    "origen",
    "fecha_ejecucion",
    "mse_reconstruccion_ae",
    "mse_reconstruccion_vae",
    "kl_media_vae",
    "medias_decodificacion_latentes",
    "diversidad_media_entre_muestras",
    "distancia_media_al_vecino_mas_cercano",
    "distancia_media_entre_pares_de_prueba",
    "segundos_entrenamiento_ae",
    "segundos_entrenamiento_vae",
    "ms_inferencia_por_lote_100",
    "kb_modelo_ae",
    "kb_decoder",
    "parametros_ae",
    "parametros_decoder",
    "ae_final_train_loss",
    "ae_final_val_loss",
    "vae_final_reconstruction",
    "vae_final_total",
    "vae_final_kl",
    "seed",
    "evaluation_examples",
    "panel_examples",
)

CLAVES_CANONICAS_DIVERSIDAD = (
    "diversidad_media_entre_muestras",
    "distancia_media_al_vecino_mas_cercano",
    "distancia_media_entre_pares_de_prueba",
)

ENCABEZADO_CANONICO = "### Párrafo canónico · diversidad y vecinos cercanos"


def numero_es(valor: float, decimales: int = 4) -> str:
    """Formato español: coma decimal y punto de millar (4,8769)."""
    return f"{valor:,.{decimales}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def parrafo_canonico(metricas: dict[str, object]) -> str:
    """Renderiza el párrafo canónico de diversidad con las cifras de `metricas`.

    Es la forma canónica de la entrega: se cita literalmente en `MODEL_CARD.md`,
    en `docs/CIERRE_INTERPRETATIVO_LAB10.md` y en el PDF breve.
    """
    diversidad = numero_es(float(metricas["diversidad_media_entre_muestras"]))
    vecino = numero_es(float(metricas["distancia_media_al_vecino_mas_cercano"]))
    base = numero_es(float(metricas["distancia_media_entre_pares_de_prueba"]))
    return (
        f"La diversidad media entre 64 muestras del decoder es **{diversidad}** (distancia media "
        "euclídea en el espacio de píxeles aplanados). La distancia media al vecino más cercano es "
        f"**{vecino}**, calculada **contra 1.000 imágenes del conjunto de prueba** y **no** contra el "
        "conjunto de entrenamiento: por eso **no permite descartar la memorización** y no es "
        "concluyente como indicador. La línea base es la distancia media entre pares de ese mismo "
        f"conjunto de prueba, **{base}** (`distancia_media_entre_pares_de_prueba`): con ella se ve que "
        "las muestras del decoder se concentran en una región más estrecha del espacio de píxeles que "
        f"los datos reales ({diversidad} frente a {base}), coherente con la repetición de patrones "
        "visible en el panel y con un espacio latente de solo 2 dimensiones, y que el "
        f"{vecino} se explica por esa concentración y no por una copia de una imagen de prueba."
    )


def extraer_parrafo_canonico(documento: str) -> str:
    """Extrae de un documento markdown el párrafo canónico de la Model Card."""
    partes = documento.split(ENCABEZADO_CANONICO, 1)
    if len(partes) != 2:
        raise ValueError("La Model Card no contiene el encabezado del párrafo canónico")
    return partes[1].strip().split("\n\n", 1)[0].strip()


def flatten(images: np.ndarray) -> np.ndarray:
    """Convierte imágenes (N, H, W, C) en vectores planos (N, H*W*C)."""
    return np.asarray(images).reshape(len(images), -1)


def mean_pairwise_distance(points: np.ndarray) -> float:
    """Distancia euclídea media entre todos los pares de puntos."""
    matrix = np.asarray(points, dtype=np.float64)
    norms = np.einsum("ij,ij->i", matrix, matrix)
    squared = norms[:, None] + norms[None, :] - 2.0 * (matrix @ matrix.T)
    np.maximum(squared, 0.0, out=squared)
    distances = np.sqrt(squared)
    upper = distances[np.triu_indices(len(matrix), k=1)]
    return float(upper.mean())


def mean_nearest_neighbor_distance(samples: np.ndarray, reference: np.ndarray) -> float:
    """Distancia media de cada muestra a su vecino más cercano dentro de `reference`."""
    samples = np.asarray(samples, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    norms_reference = np.einsum("ij,ij->i", reference, reference)
    norms_samples = np.einsum("ij,ij->i", samples, samples)
    squared = norms_reference[None, :] - 2.0 * (samples @ reference.T) + norms_samples[:, None]
    np.maximum(squared, 0.0, out=squared)
    return float(np.sqrt(squared.min(axis=1)).mean())


def build_report(values: dict[str, object], origen: str) -> dict[str, object]:
    """Arma el reporte canónico de métricas con las claves de `CLAVES_CANONICAS`."""
    reporte = dict(values)
    reporte.setdefault("origen", origen)
    reporte.setdefault("fecha_ejecucion", datetime.now().astimezone().isoformat(timespec="seconds"))
    faltantes = [clave for clave in CLAVES_CANONICAS if clave not in reporte]
    if faltantes:
        raise ValueError(f"Faltan claves en el reporte de métricas: {faltantes}")
    return {clave: reporte[clave] for clave in CLAVES_CANONICAS}
