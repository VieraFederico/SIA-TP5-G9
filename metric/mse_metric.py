import numpy as np

from src.metric.metric import Metric
from src.activation.activation import Array


class MSEMetric(Metric):
    """MSE como métrica de evaluación: (1/N) Σ (ζᵢ - Oᵢ)²"""

    def compute(self, false_pos: float, false_neg: float, true_pos: float, true_neg:float) -> float:
        raise NotImplementedError("TODO")

    # def compute(self, zeta: Array, O: Array) -> float:
        # """mse = (1/N) Σ (ζ - O)²"""
        # mse = np.sum(np.square(zeta - O)) / len(zeta)
        # return mse
    def name(self) -> str:
        return "mse"
