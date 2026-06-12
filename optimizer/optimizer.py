from abc import ABC, abstractmethod

from activation.activation import Array


class Optimizer(ABC):
    """Regla de actualización de pesos. Implementa Δw = f(∂E/∂w).

    Recibe y devuelve listas de pares (weights, bias) — una tupla por capa.
    Así el Trainer no necesita aplanar ni reconstruir la estructura del modelo.
    """

    @abstractmethod
    def update(
        self,
        params: list[tuple[Array, Array]],
        grads:  list[tuple[Array, Array]],
    ) -> list[tuple[Array, Array]]:
        """Aplica la regla Δw y devuelve los pares (weights, bias) actualizados."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Limpia el estado interno (velocidades en Momentum, momentos en Adam)."""
        ...
