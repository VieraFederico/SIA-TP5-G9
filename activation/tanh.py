import numpy as np
from src.activation.activation import ActivationFunction, Array


class TanhActivation(ActivationFunction):
    """θ(h) = tanh(β·h),  θ'(h) = β(1 - θ²(h))"""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def compute(self, h: Array) -> Array:
        """θ(h) = tanh(β·h)"""
        return np.tanh(self.beta * h)

    def derivative(self, h: Array) -> Array:
        """θ'(h) = β(1 - θ²(h))"""
        return self.beta * (1 - np.tanh(self.beta * h) ** 2)