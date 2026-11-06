"""Figuras del Ejercicio 06: panel de ejemplos, interpolación y curvas de entrenamiento."""

from __future__ import annotations

import matplotlib.pyplot as plt


def guardar_panel(ruta, originales, reconstruccion_ae, reconstruccion_vae, muestras):
    """Panel de ejemplos no seleccionados: originales, AE, VAE y muestras nuevas."""
    columnas = len(originales)
    figura, ejes = plt.subplots(4, columnas, figsize=(16, 4.6))
    for columna in range(columnas):
        for fila, coleccion in enumerate([originales, reconstruccion_ae, reconstruccion_vae, muestras]):
            ejes[fila, columna].imshow(coleccion[columna].squeeze(), cmap="gray")
            ejes[fila, columna].axis("off")
    for fila, titulo in enumerate(["Originales (prueba)", "Reconstrucción AE", "Reconstrucción VAE", "Muestras del VAE"]):
        ejes[fila, 0].set_ylabel(titulo, fontsize=7)
    figura.suptitle("Fashion-MNIST · 16 ejemplos de prueba elegidos con semilla 42 (panel no seleccionado)")
    figura.tight_layout()
    figura.savefig(ruta, dpi=170, bbox_inches="tight")
    plt.close(figura)


def guardar_interpolacion(ruta, interpolacion):
    """Rejilla con los puntos interpolados entre dos extremos del espacio latente."""
    figura, ejes = plt.subplots(1, len(interpolacion), figsize=(16, 1.5))
    for eje, imagen in zip(ejes, interpolacion):
        eje.imshow(imagen.squeeze(), cmap="gray")
        eje.axis("off")
    figura.suptitle("Interpolación en el espacio latente del VAE (16 puntos)")
    figura.tight_layout()
    figura.savefig(ruta, dpi=170, bbox_inches="tight")
    plt.close(figura)


def guardar_curvas(ruta, curvas_ae, curvas_vae):
    """Pérdida del AE (entrenamiento y validación) y del VAE (total, reconstrucción y KL)."""
    figura, (eje1, eje2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    eje1.plot(curvas_ae["loss"], label="entrenamiento")
    eje1.plot(curvas_ae["val_loss"], label="validación")
    eje1.set_title("Autoencoder · pérdida binary crossentropy")
    eje1.set_xlabel("época")
    eje1.set_ylabel("pérdida")
    eje1.legend(fontsize=8)
    eje1.grid(alpha=0.3)

    eje2.plot(curvas_vae["loss"], label="total")
    eje2.plot(curvas_vae["reconstruction"], label="reconstrucción")
    eje2.plot(curvas_vae["kl"], label="KL")
    eje2.set_title("VAE · pérdida total, reconstrucción y KL")
    eje2.set_xlabel("época")
    eje2.set_ylabel("pérdida")
    eje2.legend(fontsize=8)
    eje2.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(ruta, dpi=170, bbox_inches="tight")
    plt.close(figura)
