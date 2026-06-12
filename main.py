import numpy as np

from activation.logistic import LogisticActivation
from activation.tanh import TanhActivation
from cost.binary_cross_entropy import BinaryCrossEntropyCost
from cost.mse import MSECost
from font import load_fonts, visualize_font
from network.multilayer_perceptron import MultilayerPerceptron
from network.neuron_layer import NeuronLayer
from activation.relu import ReLUActivation  # or tanh, logistic, etc.
from activation.identity import IdentityActivation
from noise.gaussean_noise import GaussianNoise
from optimizer.adam import AdamOptimizer

from optimizer.gradient_descent import GradientDescent
from trainer import Trainer
from config import ExperimentConfig

def main():

    with_noise = True




    # 1. Load your font data
    X_train = load_fonts()  # Shape: (32, 35) - 32 fonts, 35 features each
    Zeta = X_train
    orig_X = X_train
    if with_noise:
        gauss = GaussianNoise(0.5)
        X_train = gauss.add_noise(X_train)

    # 2. Create activation functions
    relu = ReLUActivation()
    identity = IdentityActivation()
    tanh = TanhActivation(beta=0.4)


    # 3. Build the multilayered network
    # Architecture: 35 (input) → 16 (hidden) → 8 (hidden) → 35 (output/reconstruction)
    # Create activation functions

    logistic = LogisticActivation(beta=1.0)
    # ...
    layers = [
        NeuronLayer(n_inputs=35, n_neurons=30, activation=relu),
        NeuronLayer(n_inputs=30, n_neurons=25, activation=relu),
        NeuronLayer(n_inputs=25, n_neurons=20, activation=relu),
        NeuronLayer(n_inputs=20, n_neurons=16, activation=relu),
        NeuronLayer(n_inputs=16, n_neurons=8, activation=relu),
        NeuronLayer(n_inputs=8, n_neurons=4, activation=relu),

        NeuronLayer(n_inputs=4, n_neurons=2, activation=relu),  # bottleneck

        NeuronLayer(n_inputs=2, n_neurons=4, activation=relu),
        NeuronLayer(n_inputs=4, n_neurons=8, activation=relu),
        NeuronLayer(n_inputs=8, n_neurons=16, activation=relu),
        NeuronLayer(n_inputs=16, n_neurons=20, activation=relu),
        NeuronLayer(n_inputs=20, n_neurons=25, activation=relu),
        NeuronLayer(n_inputs=25, n_neurons=30, activation=relu),
        NeuronLayer(n_inputs=30, n_neurons=35, activation=logistic),  # <--- Changed
    ]

    model = MultilayerPerceptron(layers=layers)

    # 4. Create training components
    bce = BinaryCrossEntropyCost()
    mse = MSECost()
    adam = AdamOptimizer(learning_rate=0.001)
    grad = GradientDescent(learning_rate=0.001)
    trainer = Trainer(
        cost_fn=mse,
        optimizer=adam,
        metrics=[],
        cfg=ExperimentConfig(
            name="autoencoder_fonts",
            seed=42,
            data_path="",
            target_column="",
            preprocessing="normalize",
            split_train=1.0,
            split_val=0.0,
            split_test=0.0,
            activation="relu",
            beta=1.0,
            architecture=[35, 16, 8,2,8,16, 35],
            cost_function="mse",
            optimizer="adam",
            eta=0.01,
            training_mode="batch",
            batch_size=5,
            epochs=1000,
            epsilon=1e-3,
        )
    )

    # 5. Train the autoencoder (reconstruct fonts from themselves)
    history = trainer.fit(
        model=model,
        X_train=X_train,
        zeta_train=X_train,  # For autoencoder: output = input
        X_val=None,
        zeta_val=None
    )

    # 6. Test the reconstruction
    reconstructed = np.array([model.forward(font) for font in X_train])

    # 7. Visualize

    visualize_font(X_train[1], "Original 'a' Noise" if with_noise else "Original 'a'")
    if with_noise:
        visualize_font(orig_X[1], "Original 'a'")

    visualize_font(reconstructed[1], "Reconstructed 'a'")

    visualize_font(X_train[2], "Original 'b' Noise" if with_noise else "Original 'b'")
    if with_noise:
        visualize_font(orig_X[2], "Original 'b'")
    visualize_font(reconstructed[2], "Reconstructed 'b'")

    visualize_font(X_train[3], "Original 'c' Noise" if with_noise else "Original 'c'")
    if with_noise:
        visualize_font(orig_X[3], "Original 'c'")
    visualize_font(reconstructed[3], "Reconstructed 'c'")

    visualize_font(X_train[4], "Original 'd' Noise" if with_noise else "Original 'd'")
    if with_noise:
        visualize_font(orig_X[4], "Original 'd'")
    visualize_font(reconstructed[4], "Reconstructed 'd'")

    visualize_font(X_train[5], "Original 'e' Noise" if with_noise else "Original 'e'")
    if with_noise:
        visualize_font(orig_X[5], "Original 'e'")
    visualize_font(reconstructed[5], "Reconstructed 'e'")






if __name__ == '__main__':
    main()

