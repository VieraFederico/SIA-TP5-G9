"""
graphs/latent.py — gráficos del espacio latente 2D (AE y VAE), en dark mode.

Funciones PURAS: reciben arrays (ya extraídos del modelo) y devuelven la ruta
del .png generado. No conocen network/* ni el tipo de modelo. Los callers
(ae.py / vae.py) sacan los datos del modelo y eligen qué función llamar.

El tema oscuro (fondo negro puro + paleta) vive en graphs/style.py; el loop de
etiquetas (_annotate) se comparte acá entre AE y VAE.
"""
import numpy as np
from matplotlib.patches import Ellipse

from activation.activation import Array
from graphs.style import (
    BLUE, FG, ORANGE, RED,
    add_subtitle, dark_figure, dark_grid, dark_legend, save_dark,
)


def _annotate(ax, labels: list[str] | None, points: Array) -> None:
    """Etiqueta de texto (clara, para que se lea sobre negro) al lado de cada punto."""
    if labels is None:
        return
    for label, (x, y) in zip(labels, points):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), color=FG)


def plot_latent_points(
    positions: Array,
    labels: list[str] | None = None,
    output_path: str = "latent_space.png",
    title: str = "2D latent space",
    subtitle: str | None = None,
) -> str:
    """Espacio latente del AE: un punto por patrón en el plano 2D."""
    if positions.shape[1] != 2:
        raise ValueError(f"Expected 2D latent space, got shape {positions.shape}")

    fig, ax = dark_figure(figsize=(8, 6))
    ax.scatter(positions[:, 0], positions[:, 1], color=BLUE)
    _annotate(ax, labels, positions)

    ax.set_title(title)
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    dark_grid(ax)
    add_subtitle(fig, subtitle)
    return save_dark(fig, output_path)


def plot_latent_distributions(
    means: Array,
    standard_deviations: Array,
    labels: list[str] | None = None,
    output_path: str = "latent_space.png",
    title: str = "2D latent space",
    subtitle: str | None = None,
    samples_per_pattern: int = 20,
    ellipse_std: float = 1.0,
) -> str:
    """Espacio latente del VAE: por patrón, nube de samples + elipse 1σ + media μ."""
    if means.shape[1] != 2:
        raise ValueError(f"Expected 2D latent space, got shape {means.shape}")

    fig, ax = dark_figure(figsize=(9, 7))

    for mean, standard_deviation in zip(means, standard_deviations):
        if samples_per_pattern > 0:
            samples = np.random.normal(
                loc=mean,
                scale=standard_deviation,
                size=(samples_per_pattern, 2),
            )
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                color=BLUE,
                alpha=0.18,
                s=12,
                linewidths=0,
            )

        ellipse = Ellipse(
            xy=mean,
            width=2 * ellipse_std * standard_deviation[0],
            height=2 * ellipse_std * standard_deviation[1],
            angle=0,
            edgecolor=ORANGE,
            facecolor="none",
            alpha=0.7,
            linewidth=1.2,
        )
        ax.add_patch(ellipse)

    ax.scatter(
        means[:, 0],
        means[:, 1],
        color=RED,
        s=35,
        label="Latent mean μ",
        zorder=3,
    )
    _annotate(ax, labels, means)

    ax.set_title(title)
    ax.set_xlabel("Latent mean dimension 1")
    ax.set_ylabel("Latent mean dimension 2")
    dark_grid(ax)
    dark_legend(ax, ["Samples z", f"{ellipse_std:g}σ region", "Latent mean μ"])
    add_subtitle(fig, subtitle)
    return save_dark(fig, output_path)
