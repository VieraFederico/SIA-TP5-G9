import numpy as np
from src.activation.activation import ActivationFunction, Array


class Neuron:
    """Una neurona. Calcula h = Σ xᵢwᵢ + w₀, luego O = θ(h)."""

    def __init__(self, n_inputs: int, activation: ActivationFunction, seed: int = None) -> None:
        self.n_inputs = n_inputs
        self.activation = activation
        rng = np.random.default_rng(seed)
        self.weights: Array = rng.uniform(-0.5, 0.5, n_inputs)
        self.bias: float = 0.0
        self._h: float
        self.last_input: Array

    def forward(self, x: Array) -> float:
        """Calcula h y O. Guarda h para regla de la cadena en MLP para backpropagation (epa)."""
        self.last_input = x
        self.h = np.dot(x, self.weights) + self.bias  # guarda h
        self.O = self.activation.compute(self.h)  # O = θ(h)
        return self.O

    def backward(self, grad_output):
        """Calculamos Δw, no separamos del Learning rate ya que no ace falta simerpe aca"""
        if self.activation.is_differentiable(): #Literalmente para este if Existe is differentiable
            delta = grad_output * self.activation.derivative(self.h)  # (ζ-O) · θ'(h)
        else:
            # Regla de Rosenblatt: step no tiene derivada, se trata θ'(h) = 1
            delta = grad_output
        self.grad_weights = delta * self.last_input  # Δw sin η
        self.grad_bias = delta
        return delta  # para backpropagation para seguir la regla de la cadena
