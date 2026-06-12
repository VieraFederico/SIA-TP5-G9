from src.cost.cost import CostFunction
from src.activation.activation import Array


class BinaryCrossEntropyCost(CostFunction):
    """E(ζ, O) = -1/N Σ [ζᵢ log(Oᵢ) + (1 - ζᵢ) log(1 - Oᵢ)]"""

    def compute(self, zeta: Array, O: Array) -> float:
        """E = -1/N Σ [ζ log(O) + (1 - ζ) log(1 - O)]"""
        raise NotImplementedError("TODO")

    def gradient(self, zeta: Array, O: Array) -> Array:
        """∂E/∂O = -(ζ/O - (1-ζ)/(1-O)) / N"""
        raise NotImplementedError("TODO")
