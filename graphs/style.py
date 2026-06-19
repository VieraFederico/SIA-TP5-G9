from pathlib import Path

import matplotlib.pyplot as plt

# Fondo y foreground
BLACK = "#000000"      # fondo PURO
FG = "#e6e9ef"         # texto, ticks, labels, título
FG_DIM = "#3a4150"     # spines / bordes tenues
GRID = "#262b36"       # grilla
MUTED = "#8b94a6"      # subtítulo de hiperparámetros (gris legible sobre negro)

# Paleta de datos (dark mode)
BLUE = "#4aa3ff"       # puntos latentes / samples
ORANGE = "#ffb454"     # elipses 1σ
RED = "#ff6b6b"        # medias μ
IMSHOW_CMAP = "magma"  # mapa de color para imágenes (queda bien sobre negro)


def dark_figure(figsize):
    """Crea (fig, ax) con fondo negro puro y ejes/ticks/labels en dark mode."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BLACK)
    ax.set_facecolor(BLACK)
    for spine in ax.spines.values():
        spine.set_color(FG_DIM)
    ax.tick_params(colors=FG, which="both")
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    return fig, ax


def dark_grid(ax) -> None:
    """Grilla tenue acorde al fondo negro."""
    ax.grid(True, color=GRID, alpha=0.6)


def dark_legend(ax, labels):
    """Leyenda con caja negra y texto claro."""
    legend = ax.legend(labels, facecolor=BLACK, edgecolor=FG_DIM)
    for text in legend.get_texts():
        text.set_color(FG)
    return legend


def add_subtitle(fig, text: str | None) -> None:
    """Caption con los hiperparámetros, centrado al pie de la figura."""
    if not text:
        return
    fig.text(
        0.5, 0.012, text,
        ha="center", va="bottom",
        color=MUTED, fontsize=8, family="monospace",
    )


def save_dark(fig, output_path: str) -> str:
    """Guarda con el fondo negro (que savefig NO lo pinte de blanco) y cierra."""
    path = Path(output_path)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BLACK, edgecolor="none")
    plt.close(fig)
    return str(path)
