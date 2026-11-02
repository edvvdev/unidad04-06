from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inf8239_u03_gen.data import normalize_images, validate_images
from inf8239_u03_gen.models import build_autoencoder, build_encoder_decoder

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = tf.keras.metrics.Mean(name="reconstruction")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_tracker, self.kl_tracker]

    def train_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            mean, log_variance = self.encoder(images, training=True)
            epsilon = tf.random.normal(tf.shape(mean))
            latent = mean + tf.exp(0.5 * log_variance) * epsilon
            reconstruction = self.decoder(latent, training=True)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(images, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_variance - tf.square(mean) - tf.exp(log_variance), axis=1))
            total_loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        self.reconstruction_tracker.update_state(reconstruction_loss)
        self.kl_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}


def image_panel(path: Path, originals, ae_reconstructed, vae_reconstructed, generated):
    columns = len(originals)
    figure, axes = plt.subplots(4, columns, figsize=(columns, 4))
    for column in range(columns):
        for row, collection in enumerate([originals, ae_reconstructed, vae_reconstructed, generated]):
            axes[row, column].imshow(collection[column].squeeze(), cmap="gray")
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=170)


def training_curves(path: Path, ae_history, vae_history):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ae_history["loss"], label="entrenamiento")
    axes[0].plot(ae_history["val_loss"], label="validación")
    axes[0].set_title("Autoencoder: pérdida binary crossentropy")
    axes[0].set_xlabel("época")
    axes[0].set_ylabel("pérdida")
    axes[0].legend()
    axes[1].plot(vae_history["loss"], label="total")
    axes[1].plot(vae_history["reconstruction"], label="reconstrucción")
    axes[1].plot(vae_history["kl"], label="KL")
    axes[1].set_title("VAE: pérdida total, reconstrucción y KL")
    axes[1].set_xlabel("época")
    axes[1].set_ylabel("pérdida")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")


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
    encoder, decoder = build_encoder_decoder(tf, latent_dim=2)
    vae = VAE(encoder, decoder)
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
    image_panel(reports / "ae_vae_panel.png", originals, ae_reconstructed, vae_reconstructed, generated)
    grid = np.array([(1 - value) * np.array([-2.0, -1.0]) + value * np.array([2.0, 1.0]) for value in np.linspace(0, 1, 16)])
    interpolation = decoder.predict(grid, verbose=0)
    figure, axes = plt.subplots(1, 16, figsize=(16, 1.3))
    for axis, image in zip(axes, interpolation):
        axis.imshow(image.squeeze(), cmap="gray")
        axis.axis("off")
    figure.savefig(reports / "interpolation.png", dpi=170, bbox_inches="tight")
    training_curves(reports / "training_curves.png", ae_history, vae_history)
    autoencoder.save(models / "autoencoder.keras")
    decoder.save(models / "decoder.keras")
    evaluation = x_test[evaluation_index]
    ae_predictions = autoencoder.predict(evaluation, verbose=0)
    vae_mean, vae_log_variance = encoder.predict(evaluation, verbose=0)
    vae_predictions = decoder.predict(vae_mean, verbose=0)
    kl_loss = float(-0.5 * np.mean(np.sum(1 + vae_log_variance - np.square(vae_mean) - np.exp(vae_log_variance), axis=1)))
    metrics = {"ae_train_seconds": ae_seconds, "vae_train_seconds": vae_seconds,
               "ae_parameters": autoencoder.count_params(), "decoder_parameters": decoder.count_params(),
               "seed": SEED, "evaluation_examples": len(evaluation_index), "panel_examples": len(panel_index),
               "ae_test_mse": float(np.mean((evaluation - ae_predictions) ** 2)),
               "vae_test_mse": float(np.mean((evaluation - vae_predictions) ** 2)),
               "vae_test_kl": kl_loss,
               "ae_final_train_loss": float(ae_history["loss"][-1]),
               "ae_final_val_loss": float(ae_history["val_loss"][-1]),
               "vae_final_reconstruction": float(vae_history["reconstruction"][-1]),
               "vae_final_kl": float(vae_history["kl"][-1])}
    (reports / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
