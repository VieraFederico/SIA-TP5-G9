from abc import ABC, abstractmethod

from activation.activation import Array


class Noise(ABC):


    @abstractmethod
    def add_noise(self, zeta: Array) -> Array:
        """Devuelve un dataset con ruido"""
        ...