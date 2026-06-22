import numpy as np

from src.activation.activation import Array
from src.network.model import Model
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer


class VariationalAutoencoder(Model):
    """Variational autoencoder with encoder, mean/log-variance heads and decoder."""

    def __init__(
        self,
        encoder: MultilayerPerceptron,
        mean_layer: NeuronLayer,
        log_variance_layer: NeuronLayer,
        decoder: MultilayerPerceptron,
        kl_weight: float = 0.01,
    ) -> None:
        self.encoder = encoder
        self.mean_layer = mean_layer
        self.log_variance_layer = log_variance_layer
        self.decoder = decoder
        self.kl_weight = kl_weight
        self._mean: Array | None = None
        self._log_variance: Array | None = None
        self._epsilon: Array | None = None

    def _safe_log_variance(self, log_variance: Array) -> Array:
        return np.clip(log_variance, -10.0, 10.0)

    def encode_distribution(self, x: Array) -> tuple[Array, Array]:
        hidden = self.encoder.forward(x)
        mean = self.mean_layer.forward(hidden)
        log_variance = self.log_variance_layer.forward(hidden)
        self._mean = mean
        self._log_variance = log_variance
        return mean, log_variance

    def sample(self, mean: Array, log_variance: Array) -> Array:
        self._epsilon = np.random.standard_normal(mean.shape)
        standard_deviation = np.exp(0.5 * self._safe_log_variance(log_variance))
        return mean + standard_deviation * self._epsilon

    def encode(self, x: Array) -> Array:
        mean, _ = self.encode_distribution(x)
        return mean

    def decode(self, z: Array) -> Array:
        return self.decoder.forward(z)

    def forward(self, x: Array) -> Array:
        mean, log_variance = self.encode_distribution(x)
        z = self.sample(mean, log_variance)
        return self.decode(z)

    def backward(self, grad_output: Array) -> None:
        if self._mean is None or self._log_variance is None or self._epsilon is None:
            raise RuntimeError("forward must be called before backward")

        grad_z = self.decoder.backward(grad_output)

        safe_log_variance = self._safe_log_variance(self._log_variance)
        standard_deviation = np.exp(0.5 * safe_log_variance)
        grad_mean = grad_z + self.kl_weight * self._mean
        grad_log_variance = (
            grad_z * self._epsilon * 0.5 * standard_deviation
            + self.kl_weight * 0.5 * (np.exp(safe_log_variance) - 1)
        )

        grad_hidden_from_mean = self.mean_layer.backward(grad_mean)
        grad_hidden_from_log_variance = self.log_variance_layer.backward(grad_log_variance)
        self.encoder.backward(grad_hidden_from_mean + grad_hidden_from_log_variance)

    def kl_divergence(self, X: Array) -> float:
        total_kl = 0.0
        for x in X:
            mean, log_variance = self.encode_distribution(x)
            safe_log_variance = self._safe_log_variance(log_variance)
            total_kl += -0.5 * np.sum(1 + safe_log_variance - mean ** 2 - np.exp(safe_log_variance))
        return total_kl / len(X)

    def reconstruct(self, x: Array) -> Array:
        return self.decode(self.encode(x))

    def get_latent_positions(self, X: Array) -> Array:
        return np.array([self.encode(x) for x in X])

    def get_latent_distributions(self, X: Array) -> tuple[Array, Array]:
        means = []
        standard_deviations = []
        for x in X:
            mean, log_variance = self.encode_distribution(x)
            standard_deviation = np.exp(0.5 * self._safe_log_variance(log_variance))
            means.append(mean)
            standard_deviations.append(standard_deviation)
        return np.array(means), np.array(standard_deviations)

    def get_weights(self) -> list[tuple[Array, Array]]:
        return (
            self.encoder.get_weights()
            + [self.mean_layer.get_weights()]
            + [self.log_variance_layer.get_weights()]
            + self.decoder.get_weights()
        )

    def get_grads(self) -> list[tuple[Array, Array]]:
        return (
            self.encoder.get_grads()
            + [(self.mean_layer.grad_weights, self.mean_layer.grad_bias)]
            + [(self.log_variance_layer.grad_weights, self.log_variance_layer.grad_bias)]
            + self.decoder.get_grads()
        )

    def zero_grads(self) -> None:
        self.encoder.zero_grads()
        self.mean_layer.zero_grads()
        self.log_variance_layer.zero_grads()
        self.decoder.zero_grads()

    def set_weights(self, weights: list[tuple[Array, Array]]) -> None:
        encoder_end = len(self.encoder.layers)
        mean_index = encoder_end
        log_variance_index = encoder_end + 1
        decoder_start = encoder_end + 2

        self.encoder.set_weights(weights[:encoder_end])
        self.mean_layer.set_weights(*weights[mean_index])
        self.log_variance_layer.set_weights(*weights[log_variance_index])
        self.decoder.set_weights(weights[decoder_start:])
