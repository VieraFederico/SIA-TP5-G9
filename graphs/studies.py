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

GREEN = "#3fb950"          # seleccionado por el criterio
TOP_TINT = "#10202b"       # fondo tenue del TOP-N en las tablas
SEL_TINT = "#16331a"       # fondo del seleccionado en las tablas
HEADER_BG = "#11151c"


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


def grouped_bar_study(labels, series, ylabel, title, path,
                      subtitle=None, target=None, target_label=None, rotate=0, ylim=None):
    """Barras AGRUPADAS: varias series por categoría, lado a lado, con barra de error.

    series = [(nombre, means, stds, color), ...]. Pensado para comparar las dos
    series de ruido del DAE (salt 0.1 / 0.2) sobre el mismo eje y escala. ylim fija
    la escala vertical para que las dos series se lean comparables.
    """
    x = np.arange(len(labels))
    n = max(len(series), 1)
    width = 0.8 / n
    fig, ax = dark_figure(figsize=(10, 6) if rotate else (9, 5.5))
    for i, (name, means, stds, color) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, color=color, label=name)
    if target is not None:
        ax.axhline(target, color=RED, linestyle="--", linewidth=1, label=target_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    _legend(ax)
    dark_grid(ax)
    add_subtitle(fig, subtitle)
    return save_dark(fig, path)


def ranked_bar_study(labels, means, stds, ylabel, title, path,
                     subtitle=None, selected_idx=None, top_n=10, rotate=0, ylim=None):
    """Barras de error medio (ya ordenadas mejor→peor por el caller) con banda de σ.

    Resalta el TOP-N (azul fuerte) y marca el seleccionado en verde. selected_idx es
    el índice (0-based) en la lista ya ordenada."""
    x = np.arange(len(labels))
    fig, ax = dark_figure(figsize=(11, 5.5))
    colors = [BLUE if i < top_n else FG_DIM for i in range(len(labels))]
    if selected_idx is not None and 0 <= selected_idx < len(colors):
        colors[selected_idx] = GREEN
    ax.bar(x, means, yerr=stds, capsize=3, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    dark_grid(ax)
    add_subtitle(fig, subtitle)
    return save_dark(fig, path)


def loss_band_curve(curves, title, path, subtitle=None, logy=True, ylim=None,
                    label=None, color=None):
    """Curva de loss media ± σ entre seeds para UNA combinación.

    curves = lista de curvas por seed (longitudes posiblemente distintas: se trunca a
    la más corta). Banda sombreada = ± σ entre seeds."""
    curves = [list(c) for c in curves if c is not None and len(c) > 0]
    fig, ax = dark_figure(figsize=(8, 5))
    if curves:
        length = min(len(c) for c in curves)
        mat = np.array([c[:length] for c in curves], dtype=float)
        mean, std = mat.mean(axis=0), mat.std(axis=0)
        x = np.arange(length)
        col = color or BLUE
        (ax.semilogy if logy else ax.plot)(x, mean, color=col, linewidth=1.4, label=label)
        ax.fill_between(x, np.clip(mean - std, 1e-12, None), mean + std,
                        color=col, alpha=0.2, linewidth=0)
    ax.set_xlabel("Época")
    ax.set_ylabel("Train loss (BCE)")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    if label:
        _legend(ax)
    dark_grid(ax)
    add_subtitle(fig, subtitle)
    return save_dark(fig, path)


def table_figure(headers, rows, title, path, subtitle=None,
                 selected_ids=None, top_n=10, id_col=0):
    """Tabla de presentación (PNG), filas ya ordenadas por el criterio (rank asc).

    Resalta el TOP-N y marca en verde la(s) fila(s) cuyo id (columna id_col) está en
    selected_ids. rows = lista de listas de strings."""
    selected_ids = {str(s) for s in (selected_ids or [])}
    n = len(rows)
    fig_h = max(2.2, 0.4 * (n + 1) + 1.0)
    fig, ax = dark_figure(figsize=(11, fig_h))
    ax.axis("off")
    ax.set_title(title, color=FG, pad=14)

    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(FG_DIM)
        text = cell.get_text()
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            text.set_color(FG)
            text.set_weight("bold")
            continue
        # r>=1 son filas de datos (1-based porque la 0 es el header).
        rid = rows[r - 1][id_col]
        if rid in selected_ids:
            cell.set_facecolor(SEL_TINT)
        elif r <= top_n:
            cell.set_facecolor(TOP_TINT)
        else:
            cell.set_facecolor(BLACK)
        text.set_color(FG)

    add_subtitle(fig, subtitle)
    return save_dark(fig, path)


def overlaid_curves(series, xlabel, ylabel, title, path, subtitle=None, logy=False, ylim=None):
    """Varias curvas en un mismo eje. series = [(etiqueta, valores_y), ...].

    ylim opcional fija la escala vertical (útil para comparar figuras hermanas, p.ej.
    convergencia a salt=0.1 vs salt=0.2 con la misma escala)."""
    fig, ax = dark_figure(figsize=(10, 6))
    draw = ax.semilogy if logy else ax.plot
    for i, (label, ys) in enumerate(series):
        draw(ys, label=label, color=PALETTE[i % len(PALETTE)], linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    _legend(ax)
    dark_grid(ax)
    add_subtitle(fig, subtitle)
    return save_dark(fig, path)
