from src.metric.metric import Metric
from src.activation.activation import Array


class RecallMetric(Metric):
    """Recall = TP / (TP + FN). Para clasificación binaria o macro-averaged."""

    def compute(self, false_pos: float, false_neg: float, true_pos: float, true_neg:float) -> float:
        return true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0

    def name(self) -> str:
        return "recall"
