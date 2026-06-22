import numpy as np
from src.activation.activation import ActivationFunction, Array


class StepActivation(ActivationFunction):
    """θ(h) = 1 if h >= 0 else 0. No diferenciable."""

    def compute(self, h: Array) -> Array:
        """θ(h) = 1 if h >= 0 else 0"""
        return np.where(h >= 0, 1.0, 0.0)

    def derivative(self, h: Array) -> Array:
        """No definida para step. Retorna ceros por convención."""
        return np.zeros_like(h)

    def is_differentiable(self) -> bool:
        return False
