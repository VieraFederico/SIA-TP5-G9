from abc import ABC, abstractmethod



class Metric(ABC):
    """Métrica de evaluación del modelo."""

    @abstractmethod
    def compute(self, false_pos: float, false_neg: float, true_pos: float, true_neg:float) -> float:
        """Calcula la métrica comparando salida esperada ζ contra obtenida O."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Nombre de la métrica para logging."""
        ...
