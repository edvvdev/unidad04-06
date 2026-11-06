from __future__ import annotations


def build_autoencoder(tf, latent_dim: int = 16):
    inputs = tf.keras.Input((28, 28, 1))
    values = tf.keras.layers.Flatten()(inputs)
    latent = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent")(values)
    values = tf.keras.layers.Dense(784, activation="sigmoid")(latent)
    outputs = tf.keras.layers.Reshape((28, 28, 1))(values)
    return tf.keras.Model(inputs, outputs, name="autoencoder")


def build_encoder_decoder(tf, latent_dim: int = 2):
    inputs = tf.keras.Input((28, 28, 1))
    values = tf.keras.layers.Flatten()(inputs)
    values = tf.keras.layers.Dense(128, activation="relu")(values)
    mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(values)
    log_variance = tf.keras.layers.Dense(latent_dim, name="z_log_var")(values)
    encoder = tf.keras.Model(inputs, [mean, log_variance], name="encoder")
    latent_inputs = tf.keras.Input((latent_dim,))
    values = tf.keras.layers.Dense(128, activation="relu")(latent_inputs)
    values = tf.keras.layers.Dense(784, activation="sigmoid")(values)
    outputs = tf.keras.layers.Reshape((28, 28, 1))(values)
    decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")
    return encoder, decoder


def build_vae(tf, latent_dim: int = 2):
    """Construye el VAE (subclase de `tf.keras.Model`) con su `train_step` propio.

    La lógica de muestreo, reparametrización y pérdidas vive aquí para que el script de
    línea de comandos y el notebook compartan exactamente la misma implementación.
    """
    encoder, decoder = build_encoder_decoder(tf, latent_dim=latent_dim)

    class VAE(tf.keras.Model):
        """Autoencoder variacional: reconstrucción binaria + divergencia KL."""

        def __init__(self):
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
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.keras.losses.binary_crossentropy(images, reconstruction), axis=(1, 2))
                )
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + log_variance - tf.square(mean) - tf.exp(log_variance), axis=1)
                )
                total_loss = reconstruction_loss + kl_loss
            gradients = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            self.loss_tracker.update_state(total_loss)
            self.reconstruction_tracker.update_state(reconstruction_loss)
            self.kl_tracker.update_state(kl_loss)
            return {metric.name: metric.result() for metric in self.metrics}

    return VAE()
