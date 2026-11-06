from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

# Reproducibilidad bit a bit. La semilla global no basta: con varios hilos por
# operación, oneDNN cambia el orden de las sumas en las capas densas y el VAE
# entrenado varía entre corridas (mse 0,033678... frente a 0,033752... en dos
# corridas del mismo código). Con un solo hilo intra e inter, dos corridas dan
# exactamente el mismo modelo y las mismas métricas.
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np
import tensorflow as tf

from inf8239_u03_gen.data import normalize_images, validate_images
from inf8239_u03_gen.figures import (
    guardar_curvas,
    guardar_interpolacion,
    guardar_panel,
)
from inf8239_u03_gen.metrics import (
    build_report,
    flatten,
    mean_nearest_neighbor_distance,
    mean_pairwise_distance,
)
from inf8239_u03_gen.models import build_autoencoder, build_vae

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-ae", type=int, default=6)
    parser.add_argument("--epochs-vae", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    reports = ROOT / "reports"
    models = ROOT / "models"
    reports.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    if args.limit:
        x_train = x_train[: args.limit]
        x_test = x_test[: max(100, args.limit // 5)]
    validate_images(x_train)
    start = perf_counter()
    autoencoder = build_autoencoder(tf)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    ae_history = autoencoder.fit(x_train, x_train, validation_split=0.1, epochs=args.epochs_ae, batch_size=256,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)], verbose=2).history
    ae_seconds = perf_counter() - start
    vae = build_vae(tf, latent_dim=2)
    encoder, decoder = vae.encoder, vae.decoder
    vae.compile(optimizer="adam")
    start = perf_counter()
    vae_history = vae.fit(x_train, epochs=args.epochs_vae, batch_size=256, verbose=2).history
    vae_seconds = perf_counter() - start
    rng = np.random.default_rng(SEED)
    evaluation_index = rng.choice(len(x_test), 1000, replace=False)
    panel_index = evaluation_index[:16]
    originals = x_test[panel_index]
    ae_reconstructed = autoencoder.predict(originals, verbose=0)
    mean, _log_variance = encoder.predict(originals, verbose=0)
    vae_reconstructed = decoder.predict(mean, verbose=0)
    generated = decoder.predict(np.random.default_rng(SEED).normal(size=(16, 2)), verbose=0)
    guardar_panel(reports / "ejercicio06_panel.png", originals, ae_reconstructed, vae_reconstructed, generated)
    grid = np.array([(1 - value) * np.array([-2.0, -1.0]) + value * np.array([2.0, 1.0]) for value in np.linspace(0, 1, 16)])
    interpolation = decoder.predict(grid, verbose=0)
    guardar_interpolacion(reports / "ejercicio06_interpolacion.png", interpolation)
    guardar_curvas(reports / "ejercicio06_curvas.png", ae_history, vae_history)
    autoencoder.save(models / "autoencoder.keras")
    decoder.save(models / "decoder.keras")
    evaluation = x_test[evaluation_index]
    ae_predictions = autoencoder.predict(evaluation, verbose=0)
    vae_mean, vae_log_variance = encoder.predict(evaluation, verbose=0)
    vae_predictions = decoder.predict(vae_mean, verbose=0)
    kl_loss = float(-0.5 * np.mean(np.sum(1 + vae_log_variance - np.square(vae_mean) - np.exp(vae_log_variance), axis=1)))
    planas_prueba = flatten(evaluation)
    muestras = decoder.predict(np.random.default_rng(SEED).normal(size=(64, 2)), verbose=0)
    planas_muestras = flatten(muestras)
    diversidad = mean_pairwise_distance(planas_muestras)
    distancia_vecino = mean_nearest_neighbor_distance(planas_muestras[:32], planas_prueba)
    distancia_pares_prueba = mean_pairwise_distance(planas_prueba)
    puntos_latentes = [(0.0, 0.0), (1.5, -1.0), (-2.0, 1.5)]
    medias_latentes = [float(decoder.predict(np.array([punto]), verbose=0).mean()) for punto in puntos_latentes]
    inicio = perf_counter()
    decoder.predict(np.zeros((100, 2)), verbose=0)
    ms_inferencia = (perf_counter() - inicio) * 1000 / 100
    metrics = build_report({
        "mse_reconstruccion_ae": float(np.mean((evaluation - ae_predictions) ** 2)),
        "mse_reconstruccion_vae": float(np.mean((evaluation - vae_predictions) ** 2)),
        "kl_media_vae": kl_loss,
        "medias_decodificacion_latentes": medias_latentes,
        "diversidad_media_entre_muestras": diversidad,
        "distancia_media_al_vecino_mas_cercano": distancia_vecino,
        "distancia_media_entre_pares_de_prueba": distancia_pares_prueba,
        "segundos_entrenamiento_ae": ae_seconds,
        "segundos_entrenamiento_vae": vae_seconds,
        "ms_inferencia_por_lote_100": ms_inferencia,
        "kb_modelo_ae": (models / "autoencoder.keras").stat().st_size / 1024,
        "kb_decoder": (models / "decoder.keras").stat().st_size / 1024,
        "parametros_ae": autoencoder.count_params(),
        "parametros_decoder": decoder.count_params(),
        "ae_final_train_loss": float(ae_history["loss"][-1]),
        "ae_final_val_loss": float(ae_history["val_loss"][-1]),
        "vae_final_reconstruction": float(vae_history["reconstruction"][-1]),
        "vae_final_kl": float(vae_history["kl"][-1]),
        "vae_final_total": float(vae_history["loss"][-1]),
        "seed": SEED,
        "evaluation_examples": len(evaluation_index),
        "panel_examples": len(panel_index),
    }, origen="scripts/train_ae_vae.py")
    (reports / "ejercicio06_metricas.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
