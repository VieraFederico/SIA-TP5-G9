import numpy as np

from activation.activation import Array
from noise.noise import Noise


class SaltNPepperNoise(Noise):
    def __init__(self, p=0.1):
        """
        p: probability of flipping each pixel (default 0.1 = 10%)
           - flips 0 -> 1 or 1 -> 0 with equal probability
        """
        self.p = p

    def add_noise(self, x: Array) -> Array:
        """Apply salt-and-pepper noise to input array"""
        noisy = x.copy()

        # Generate random mask for which pixels to flip
        mask = np.random.random(x.shape) < self.p

        # Flip those pixels (0->1, 1->0)
        noisy[mask] = 1.0 - noisy[mask]

        return noisy