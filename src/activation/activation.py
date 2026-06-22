from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class ActivationFunction(ABC):
    """Función de activación θ(h). Transforma la excitación h en la salida O = θ(h)."""

    @abstractmethod
    def compute(self, h: Array) -> Array:
        """Aplica θ sobre la excitación h. Devuelve O = θ(h)."""
        ...

    @abstractmethod
    def derivative(self, h: Array) -> Array:
        """Devuelve θ'(h). Necesario para calcular δ en backpropagation."""
        ...

    def is_differentiable(self) -> bool:
        """Retorna True si la función es diferenciable. Step retorna False."""
        return True
