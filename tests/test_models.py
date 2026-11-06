import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from inf8239_u03_gen.models import build_autoencoder, build_encoder_decoder, build_vae


def test_autoencoder_output_contract():
    model = build_autoencoder(tf)
    result = model(np.zeros((2, 28, 28, 1), dtype="float32")).numpy()
    assert result.shape == (2, 28, 28, 1)


def test_decoder_output_contract():
    _, decoder = build_encoder_decoder(tf, latent_dim=2)
    result = decoder(np.zeros((2, 2), dtype="float32")).numpy()
    assert result.shape == (2, 28, 28, 1)


def test_vae_shares_the_encoder_and_the_decoder():
    vae = build_vae(tf, latent_dim=2)
    assert vae.encoder.output_shape == [(None, 2), (None, 2)]
    assert vae.decoder.output_shape == (None, 28, 28, 1)
    assert {metric.name for metric in vae.metrics} == {"loss", "reconstruction", "kl"}


def test_vae_train_step_updates_the_weights():
    vae = build_vae(tf, latent_dim=2)
    vae.compile(optimizer="adam")
    antes = len(vae.trainable_weights)
    historial = vae.fit(np.zeros((4, 28, 28, 1), dtype="float32"), epochs=1, batch_size=2, verbose=0)
    assert len(vae.trainable_weights) == antes > 0
    assert "loss" in historial.history
