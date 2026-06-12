import numpy as np

from activation.logistic import LogisticActivation
from cost.mse import MSECost
from font import load_fonts, visualize_font
from network.multilayer_perceptron import MultilayerPerceptron
from network.neuron_layer import NeuronLayer
from activation.relu import ReLUActivation  # or tanh, logistic, etc.
from activation.identity import IdentityActivation
from optimizer.adam import AdamOptimizer

from optimizer.gradient_descent import GradientDescent
from trainer import Trainer
from config import ExperimentConfig

def main():


    # 1. Load your font data
    X_train = load_fonts()  # Shape: (32, 35) - 32 fonts, 35 features each

    # 2. Create activation functions
    relu = ReLUActivation()
    identity = IdentityActivation()

    # 3. Build the multilayered network
    # Architecture: 35 (input) → 16 (hidden) → 8 (hidden) → 35 (output/reconstruction)
    # Create activation functions

    logistic = LogisticActivation(beta=1.0)
    # ...
    layers = [
        NeuronLayer(n_inputs=35, n_neurons=16, activation=relu),
        NeuronLayer(n_inputs=16, n_neurons=8, activation=relu),
        NeuronLayer(n_inputs=8, n_neurons=2, activation=relu),  # bottleneck
        NeuronLayer(n_inputs=2, n_neurons=8, activation=relu),
        NeuronLayer(n_inputs=8, n_neurons=16, activation=relu),
        NeuronLayer(n_inputs=16, n_neurons=35, activation=logistic),  # <--- Changed
    ]

    model = MultilayerPerceptron(layers=layers)

    # 4. Create training components
    cost_fn = MSECost()
    optimizer = AdamOptimizer(learning_rate=0.001)
    trainer = Trainer(
        cost_fn=cost_fn,
        optimizer=optimizer,
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
            epochs=5000,
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

    visualize_font(X_train[1], "Original 'a'")
    visualize_font(reconstructed[1], "Reconstructed 'a'")

    # visualize_font(X_train[2], "Original 'b'")
    # visualize_font(reconstructed[2], "Reconstructed 'b'")
    #
    # visualize_font(X_train[3], "Original 'c'")
    # visualize_font(reconstructed[3], "Reconstructed 'c'")






if __name__ == '__main__':
    main()

