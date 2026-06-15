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
