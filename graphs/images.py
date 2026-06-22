"""
graphs/images.py — patrones 7x5 dibujados como imágenes (imshow), en dark mode.

Las figuras de presentación (Fase 7) que muestran patrones como imagen y no como
ASCII: reconstrucciones X vs X', tríptico del denoising, muestras generadas y la
curva de loss del entrenamiento. No recalculan nada: reciben lo que el pipeline ya
produjo (run_experiment / generate*.py) y sólo lo grafican.
"""
import numpy as np

import matplotlib.pyplot as plt

from graphs.style import (
    BLACK, BLUE, FG, FG_DIM, ORANGE,
    add_subtitle, dark_figure, dark_grid, save_dark,
)

PATTERN_SHAPE = (7, 5)   # font.h y emojis: 7 filas x 5 columnas = 35 píxeles
IMG_KW = dict(cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")


def _grid_figure(nrows, ncols, figsize):
    """subplots con fondo negro; devuelve (fig, axes aplanados)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor(BLACK)
    for ax in axes.ravel():
        ax.set_facecolor(BLACK)
    return fig, axes


def _show(ax, pattern, title=None):
    """Pinta un patrón de 35 valores como una grilla 7x5 sin ticks."""
    grid = np.asarray(pattern, dtype=float).reshape(PATTERN_SHAPE)
    ax.imshow(grid, **IMG_KW)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(FG_DIM)
    if title is not None:
        ax.set_title(title, color=FG, fontsize=8)


def plot_loss_curve(history, path, title="Curva de loss", subtitle=None):
    """Error de entrenamiento por época (y validación si la hay)."""
    train = history.get("train_error") or []
    if not train:
        return None

    fig, ax = dark_figure(figsize=(9, 6))
    ax.plot(range(1, len(train) + 1), train, color=BLUE, linewidth=1.4, label="train")
    val = history.get("val_error") or []
    if any(v is not None for v in val):
        ax.plot(range(1, len(val) + 1), val, color=ORANGE, linewidth=1.4, label="val")
        leg = ax.legend(facecolor=BLACK, edgecolor=FG_DIM)
        for t in leg.get_texts():
            t.set_color(FG)

    ax.set_xlabel("Época")
    ax.set_ylabel("Error de reconstrucción (BCE)")
    ax.set_title(title)
    dark_grid(ax)
    add_subtitle(fig, subtitle)
    return save_dark(fig, path)


def plot_reconstructions(originals, reconstructions, labels, path,
                         title="Entrada vs reconstrucción", subtitle=None, cols=8):
    """Los patrones originales y su reconstrucción, en pares (original arriba, X' abajo)."""
    n = len(originals)
    blocks = int(np.ceil(n / cols))
    fig, axes = _grid_figure(blocks * 2, cols, figsize=(cols * 1.1, blocks * 2 * 1.1))
    for ax in axes.ravel():
        ax.set_visible(False)

    for i in range(n):
        block, col = divmod(i, cols)
        name = str(labels[i]) if labels is not None else None
        ax_o, ax_r = axes[block * 2, col], axes[block * 2 + 1, col]
        ax_o.set_visible(True)
        ax_r.set_visible(True)
        _show(ax_o, originals[i], title=name)
        _show(ax_r, reconstructions[i])

    fig.suptitle(title, color=FG)
    add_subtitle(fig, subtitle)
    return save_dark(fig, path)


def plot_triptych(clean, noisy, recon, labels, path,
                  title="Denoising", subtitle=None, indices=None, max_items=8):
    """Filas limpio / ruidoso / reconstruido para unos pocos patrones (DAE)."""
    if indices is None:
        indices = list(range(min(max_items, len(clean))))
    rows = [("limpio", clean), ("ruidoso", noisy), ("reconstruido", recon)]

    fig, axes = _grid_figure(3, len(indices), figsize=(len(indices) * 1.1, 3 * 1.2))
    for col, idx in enumerate(indices):
        for r, (row_name, data) in enumerate(rows):
            ax = axes[r, col]
            name = str(labels[idx]) if (r == 0 and labels is not None) else None
            _show(ax, data[idx], title=name)
            if col == 0:
                ax.set_ylabel(row_name, color=FG, fontsize=8)

    fig.suptitle(title, color=FG)
    add_subtitle(fig, subtitle)
    return save_dark(fig, path)


def plot_generated(patterns, latents, path, title="Patrones generados", subtitle=None, cols=None):
    """Muestras generadas (decode de z), cada una anotada con su z latente."""
    n = len(patterns)
    cols = cols or min(n, 8)
    rows = int(np.ceil(n / cols))
    fig, axes = _grid_figure(rows, cols, figsize=(cols * 1.3, rows * 1.5))
    flat = axes.ravel()
    for ax in flat:
        ax.set_visible(False)

    for i in range(n):
        ax = flat[i]
        ax.set_visible(True)
        z = None if latents is None else np.asarray(latents[i]).ravel()
        label = f"z=({z[0]:.2f}, {z[1]:.2f})" if (z is not None and z.size >= 2) else None
        _show(ax, patterns[i], title=label)

    fig.suptitle(title, color=FG)
    add_subtitle(fig, subtitle)
    return save_dark(fig, path)
