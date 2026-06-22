from src.metric.metric import Metric
from src.activation.activation import Array


class PrecisionMetric(Metric):
    """Precisión = TP / (TP + FP). Para clasificación binaria o macro-averaged."""

    def compute(self, false_pos: float, false_neg: float, true_pos: float, true_neg:float) -> float:
        return true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0

    def name(self) -> str:
        return "precision"
