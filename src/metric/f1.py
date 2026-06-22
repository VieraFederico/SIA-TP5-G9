from src.metric.metric import Metric
from src.metric.precision import PrecisionMetric
from src.metric.recall import RecallMetric

class F1Metric(Metric):
    """F1 = 2 · precision · recall / (precision + recall)"""

    def compute(self, false_pos: float, false_neg: float, true_pos: float, true_neg:float) -> float:
        """F1 = 2 · P · R / (P + R)"""
        precision = PrecisionMetric().compute(false_pos, false_neg, true_pos, true_neg)
        recall = RecallMetric().compute(false_pos, false_neg, true_pos, true_neg)
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def name(self) -> str:
        return "f1"
