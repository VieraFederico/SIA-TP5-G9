"""
graphs/studies.py — figuras de los estudios de grid (arquitectura e hiperparámetros).

Los scripts grid_*.py calculan los números; el dibujo vive acá para no repetir
matplotlib en cada estudio y mantener un solo estilo dark.
"""
import numpy as np

from graphs.style import (
    BLACK, BLUE, FG, FG_DIM, ORANGE, RED,
    add_subtitle, dark_figure, dark_grid, save_dark,
)

# Colores para curvas superpuestas (legibles sobre negro).
PALETTE = [BLUE, ORANGE, RED, "#7ee787", "#c78bff", "#ffd866"]


def _legend(ax):
    leg = ax.legend(facecolor=BLACK, edgecolor=FG_DIM, fontsize=8)
    for t in leg.get_texts():
        t.set_color(FG)


def bar_study(labels, means, stds, ylabel, title, path,
              subtitle=None, color=BLUE, target=None, target_label=None, rotate=0):
    """Barras con barra de error (el desvío entre seeds) y una línea de objetivo opcional."""
    x = np.arange(len(labels))
    fig, ax = dark_figure(figsize=(10, 6) if rotate else (8, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color=color)
    if target is not None:
        ax.axhline(target, color=RED, linestyle="--", linewidth=1, label=target_label)
        if target_label:
            _legend(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    dark_grid(ax)
    add_subtitle(fig, subtitle)
    return save_dark(fig, path)


def overlaid_curves(series, xlabel, ylabel, title, path, subtitle=None, logy=False):
    """Varias curvas en un mismo eje. series = [(etiqueta, valores_y), ...]."""
    fig, ax = dark_figure(figsize=(10, 6))
    draw = ax.semilogy if logy else ax.plot
    for i, (label, ys) in enumerate(series):
        draw(ys, label=label, color=PALETTE[i % len(PALETTE)], linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _legend(ax)
    dark_grid(ax)
    add_subtitle(fig, subtitle)
    return save_dark(fig, path)
