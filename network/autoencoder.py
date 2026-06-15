from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from activation.activation import Array
from network.model import Model
from network.multilayer_perceptron import MultilayerPerceptron


class Autoencoder(Model):
    """Autoencoder split into encoder, latent-space mapper and decoder.

    The split is conceptual and practical: training still treats the whole
    chain as a single differentiable model.
    """

    def __init__(
        self,
        encoder: MultilayerPerceptron,
        latent_space: MultilayerPerceptron,
        decoder: MultilayerPerceptron,
    ) -> None:
        self.encoder = encoder
        self.latent_space = latent_space
        self.decoder = decoder

    def encode(self, x: Array) -> Array:
        return self.latent_space.forward(self.encoder.forward(x))

    def get_latent_positions(self, X: Array) -> Array:
        return np.array([self.encode(x) for x in X])

    def plot_latent_space(
        self,
        X: Array,
        labels: list[str] | None = None,
        output_path: str = "latent_space.png",
        title: str = "2D latent space",
    ) -> str:
        positions = self.get_latent_positions(X)
        if positions.shape[1] != 2:
            raise ValueError(f"Expected 2D latent space, got shape {positions.shape}")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(positions[:, 0], positions[:, 1], color="tab:blue")

        if labels is not None:
            for label, (x, y) in zip(labels, positions):
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))

        ax.set_title(title)
        ax.set_xlabel("Latent dimension 1")
        ax.set_ylabel("Latent dimension 2")
        ax.grid(True, alpha=0.3)

        path = Path(output_path)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def decode(self, z: Array) -> Array:
        return self.decoder.forward(z)

    def forward(self, x: Array) -> Array:
        return self.decode(self.encode(x))

    def backward(self, grad_output: Array) -> None:
        delta = self.decoder.backward(grad_output)
        delta = self.latent_space.backward(delta)
        self.encoder.backward(delta)

    def get_weights(self) -> list[tuple[Array, Array]]:
        return (
            self.encoder.get_weights()
            + self.latent_space.get_weights()
            + self.decoder.get_weights()
        )

    def get_grads(self) -> list[tuple[Array, Array]]:
        return (
            self.encoder.get_grads()
            + self.latent_space.get_grads()
            + self.decoder.get_grads()
        )

    def zero_grads(self) -> None:
        self.encoder.zero_grads()
        self.latent_space.zero_grads()
        self.decoder.zero_grads()

    def set_weights(self, weights: list[tuple[Array, Array]]) -> None:
        encoder_end = len(self.encoder.layers)
        latent_end = encoder_end + len(self.latent_space.layers)

        self.encoder.set_weights(weights[:encoder_end])
        self.latent_space.set_weights(weights[encoder_end:latent_end])
        self.decoder.set_weights(weights[latent_end:])
