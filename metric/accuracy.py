from src.metric.metric import Metric
from src.activation.activation import Array


class AccuracyMetric(Metric):

    def compute(self, false_pos: float, false_neg: float, true_pos: float, true_neg:float) -> float:
        return (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg) if (true_pos + true_neg + false_pos + false_neg) > 0 else 0.0

    def name(self) -> str:
        return "accuracy"
