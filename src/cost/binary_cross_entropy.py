from src.activation.activation import Array
from src.cost.cost import CostFunction
import numpy as np


class BinaryCrossEntropyCost(CostFunction):
    """E(ζ, O) = -1/N Σ [ζᵢ log(Oᵢ) + (1 - ζᵢ) log(1 - Oᵢ)]"""

    def compute(self, zeta: Array, O: Array) -> float:
        """E = -1/N Σ [ζ log(O) + (1 - ζ) log(1 - O)]"""
        # Añadimos un pequeño valor epsilon para evitar log(0)
        epsilon = 1e-12
        O_clipped = np.clip(O, epsilon, 1.0 - epsilon)

        # N es el número de ejemplos (batch size)
        N = zeta.shape[0] if len(zeta.shape) > 0 else 1

        # Calculamos la sumatoria y dividimos por N
        cost = -np.sum(zeta * np.log(O_clipped) + (1 - zeta) * np.log(1 - O_clipped)) / N
        return float(cost)

    def gradient(self, zeta: Array, O: Array) -> Array:
        """∂E/∂O = -(ζ/O - (1-ζ)/(1-O)) / N"""
        # Añadimos epsilon para evitar la división por 0
        epsilon = 1e-12
        O_clipped = np.clip(O, epsilon, 1.0 - epsilon)

        N = zeta.shape[0] if len(zeta.shape) > 0 else 1

        # Calculamos el gradiente
        grad = -(zeta / O_clipped - (1 - zeta) / (1 - O_clipped)) / N
        return grad
