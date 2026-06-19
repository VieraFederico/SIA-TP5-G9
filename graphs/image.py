"""
graphs/image.py — visualización de imágenes (imshow), en dark mode.

plot_sample fue MOVIDO tal cual desde data_management/digit_dataset_loader.py
(mismo nombre, mismo comportamiento). Es legacy: dibuja un dígito de un dataset
de otro TP y no lo usa la corrida del AE/VAE.
"""
import matplotlib.pyplot as plt

from graphs.style import BLACK, FG, IMSHOW_CMAP
from data_management.digit_dataset_loader import get_image


def plot_sample(row) -> None:
    """Muestra un dígito como imagen sobre fondo negro (imshow + plt.show())."""
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor(BLACK)
    ax.set_facecolor(BLACK)
    ax.imshow(get_image(row), cmap=IMSHOW_CMAP, vmin=0, vmax=1)
    ax.set_title(f"Label: {int(row['label'])}", fontsize=13, color=FG)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
