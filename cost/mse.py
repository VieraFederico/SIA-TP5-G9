import numpy as np

from activation.activation import Array
from cost.cost import CostFunction


class MSECost(CostFunction):
    """E(ζ, O) = (1/2N) Σ (ζᵢ - Oᵢ)²"""

    def compute(self, zeta: Array, O: Array) -> float:
        """E = (1/2N) Σ (ζ - O)²"""
        N = len(zeta) if hasattr(zeta, '__len__') else 1
        return (1 / (2 * N)) * np.sum((zeta - O) ** 2)

    def gradient(self, zeta: Array, O: Array) -> Array:
        """∂E/∂O = -(ζ - O) / N"""
        N = len(zeta) if hasattr(zeta, '__len__') else 1
        return -(zeta - O) / N