"""
graphs/ — librería central de gráficos del TP5.

Toda la visualización vive acá. Las clases de modelo (network/*) ya no dibujan:
exponen datos (get_latent_positions / get_latent_distributions) y los callers
(ae.py / vae.py) eligen qué función de graphs llamar e inyectan el resultado en
experiment.run_experiment.

El backend "Agg" (sin ventana, sirve en WSL/headless) se setea UNA sola vez acá,
antes de importar pyplot en los submódulos.

API pública:
    plot_latent_points(positions, labels, path, title)             # espacio latente AE
    plot_latent_distributions(means, stds, labels, path, ...)      # espacio latente VAE
    plot_reconstructions / plot_triptych / plot_generated / plot_loss_curve  # imágenes (Fase 7)
    visualize_font(pattern, char_name)                             # ASCII a consola (def. en font.py)
"""
import matplotlib
matplotlib.use("Agg")

from graphs.latent import (
    plot_latent_clouds_generated,
    plot_latent_distributions,
    plot_latent_distributions_with_generated,
    plot_latent_points,
    plot_latent_with_generated,
)
from graphs.images import (
    plot_generated, plot_loss_curve, plot_reconstructions, plot_triptych,
)
from graphs.studies import bar_study, overlaid_curves
from font import visualize_font  # se mantiene definido en font.py; acá sólo se re-exporta

__all__ = [
    "plot_latent_points",
    "plot_latent_distributions",
    "plot_latent_distributions_with_generated",
    "plot_latent_with_generated",
    "plot_latent_clouds_generated",
    "plot_reconstructions",
    "plot_triptych",
    "plot_generated",
    "plot_loss_curve",
    "bar_study",
    "overlaid_curves",
    "visualize_font",
]
