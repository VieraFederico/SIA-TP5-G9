import numpy as np

from activation.activation import ActivationFunction, Array


class LogisticActivation(ActivationFunction):
    """θ(h) = 1 / (1 + e^(-β·h)),  θ'(h) = β·θ(h)·(1 - θ(h))"""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def compute(self, h: Array) -> Array:
        """θ(h) = 1 / (1 + e^(-β·h))"""
        h_clipped = np.clip(-self.beta * h, -250, 250)
        return 1.0 / (1.0 + np.exp(h_clipped))

    def derivative(self, h: Array) -> Array:
        """θ'(h) = β·θ(h)·(1 - θ(h))"""
        o = self.compute(h)
        return self.beta * o * (1.0 - o)