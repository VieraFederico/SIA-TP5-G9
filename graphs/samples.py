"""
graphs/samples.py — visualización ASCII de reconstrucciones en consola.

Imprime con visualize_font sólo los patrones cuya reconstrucción difiere del
original (tras binarizar a 0.5). Lo usan ae.py y vae.py por igual: vivir acá
corta el acople en que vae.py importaba este helper desde ae.py.
"""
import numpy as np

from evaluation import binarize
from font import visualize_font


def visualize_samples(clean, x_input, reconstructed, with_noise: bool, labels) -> None:
    """Muestra (ASCII) sólo los patrones mal reconstruidos. with_noise agrega la
    entrada ruidosa además del original limpio."""
    clean_bin = binarize(clean)
    recon_bin = binarize(reconstructed)
    mismatched = [
        i for i in range(len(clean_bin))
        if not np.array_equal(clean_bin[i], recon_bin[i])
    ]

    if not mismatched:
        print("All reconstructions match the originals (after thresholding at 0.5).")
        return

    for i in mismatched:
        name = labels[i]
        visualize_font(x_input[i],
                       f"Original '{name}' Noise" if with_noise else f"Original '{name}'")
        if with_noise:
            visualize_font(clean[i], f"Original '{name}'")
        visualize_font(reconstructed[i], f"Reconstructed '{name}'")
