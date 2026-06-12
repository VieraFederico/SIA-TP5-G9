import random

import numpy as np

from activation.activation import Array
from noise.noise import Noise


class GaussianNoise(Noise):
    def __init__(self, sigma=1.0):
        self.sigma = sigma


    def add_noise(self, x: Array) -> Array:
        return x + np.random.normal(0, self.sigma, size=x.shape)