import random

import numpy as np

from src.activation.activation import Array
from src.noise.noise import Noise


class GaussianNoise(Noise):
    def __init__(self, sigma=1.0):
        self.sigma = sigma


    def add_noise(self, x: Array) -> Array:
        noisy = x + np.random.normal(0, self.sigma, size=x.shape)
        return np.clip(noisy, 0.0, 1.0)