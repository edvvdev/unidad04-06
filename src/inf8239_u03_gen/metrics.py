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
