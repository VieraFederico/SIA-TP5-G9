from abc import ABC, abstractmethod

from src.activation.activation import Array


class CostFunction(ABC):
    """Función de costo E(O). Es lo que el entrenamiento minimiza."""

    @abstractmethod
    def compute(self, zeta: Array, O: Array) -> float:
        """Calcula E(ζ, O). Devuelve el error escalar."""
        ...

    @abstractmethod
    def gradient(self, zeta: Array, O: Array) -> Array:
        """Calcula ∂E/∂O. Arranca la retropropagación."""
        ...
