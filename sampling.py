"""
sampling.py — helpers de muestreo del espacio latente, compartidos por los
generadores (generate.py / generate_vae.py) y los plots del latente.

Centraliza tres operaciones que estaban repetidas: fijar la semilla, muestrear
del prior N(0, I) y obtener el rango (bounding box) ocupado por unas posiciones.
"""
import numpy as np


def set_seed(seed: int | None) -> None:
    """Fija la semilla global de numpy si se pasa una (None = no tocar el estado)."""
    if seed is not None:
        np.random.seed(seed)


def sample_prior(n: int, dim: int = 2, seed: int | None = None) -> np.ndarray:
    """n muestras del prior z ~ N(0, I) de dimensión dim. Si seed no es None, la fija antes."""
    set_seed(seed)
    return np.random.standard_normal((n, dim))


def latent_bounds(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mínimo, máximo) por dimensión del rango ocupado por unas posiciones latentes."""
    positions = np.asarray(positions)
    return positions.min(axis=0), positions.max(axis=0)
