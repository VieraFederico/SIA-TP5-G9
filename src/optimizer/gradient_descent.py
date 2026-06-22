from src.activation.activation import Array
from src.optimizer.optimizer import Optimizer


class GradientDescent(Optimizer):
    """Descenso de gradiente estándar. Δw = -η · ∂E/∂w"""

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def update(self, params: list[tuple[Array, Array]], grads:  list[tuple[Array, Array]],) -> list[tuple[Array, Array]]:
        updated = []
        for (w, b), (gw, gb) in zip(params, grads):
            w_nuevo = w - self.learning_rate * gw
            b_nuevo = b - self.learning_rate * gb
            updated.append((w_nuevo, b_nuevo))
        return updated

    def reset(self) -> None:
        pass
