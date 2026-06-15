import numpy as np

from activation.logistic import LogisticActivation
from activation.tanh import TanhActivation
from cost.binary_cross_entropy import BinaryCrossEntropyCost
from cost.mse import MSECost
from font import load_fonts, visualize_font
from network.multilayer_perceptron import MultilayerPerceptron
from network.neuron_layer import NeuronLayer
from network.variational_autoencoder import VariationalAutoencoder
from activation.relu import ReLUActivation  # or tanh, logistic, etc.
from activation.identity import IdentityActivation
from noise.gaussean_noise import GaussianNoise
from noise.salt_n_pepper import SaltNPepperNoise
from optimizer.adam import AdamOptimizer

from optimizer.gradient_descent import GradientDescent
from trainer import Trainer
from config import ExperimentConfig

def main():

    font_datatype = "emoji"
    with_noise = True

    # 1. Load your font data
    X_train = load_fonts(font_datatype)  # Shape: (32, 35) - 32 fonts, 35 features each



    orig_X = X_train.copy()
    if with_noise:
        gauss = GaussianNoise(0.5)
        salt = SaltNPepperNoise(0.01)
        X_train = salt.add_noise(X_train)
    zeta_train = orig_X if with_noise else X_train

    # 2. Create activation functions
    relu = ReLUActivation()
    identity = IdentityActivation()
    tanh = TanhActivation(beta=0.4)


    # 3. Build the multilayered network
    # Architecture: 35 (input) → 16 (hidden) → 8 (hidden) → 35 (output/reconstruction)
    # Create activation functions

    logistic = LogisticActivation(beta=1.0)
    encoder = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=35, n_neurons=30, activation=relu),
        NeuronLayer(n_inputs=30, n_neurons=25, activation=relu),
        NeuronLayer(n_inputs=25, n_neurons=20, activation=relu),
        NeuronLayer(n_inputs=20, n_neurons=16, activation=relu),
        NeuronLayer(n_inputs=16, n_neurons=8, activation=relu),
    ])

    mean_layer = NeuronLayer(n_inputs=8, n_neurons=2, activation=identity)
    log_variance_layer = NeuronLayer(n_inputs=8, n_neurons=2, activation=identity)

    decoder = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=2, n_neurons=8, activation=relu),
        NeuronLayer(n_inputs=8, n_neurons=16, activation=relu),
        NeuronLayer(n_inputs=16, n_neurons=20, activation=relu),
        NeuronLayer(n_inputs=20, n_neurons=25, activation=relu),
        NeuronLayer(n_inputs=25, n_neurons=30, activation=relu),
        NeuronLayer(n_inputs=30, n_neurons=35, activation=logistic),  # <--- Changed
    ])

    model = VariationalAutoencoder(
        encoder=encoder,
        mean_layer=mean_layer,
        log_variance_layer=log_variance_layer,
        decoder=decoder,
        kl_weight=0.01,
    )

    # 4. Create training components
    bce = BinaryCrossEntropyCost()
    mse = MSECost()
    adam = AdamOptimizer(learning_rate=0.001)
    grad = GradientDescent(learning_rate=0.001)
    trainer = Trainer(
        cost_fn=bce,
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
            architecture=[35, 30, 25, 20, 16, 8, 2, 8, 16, 20, 25, 30, 35],
            cost_function="binary_cross_entropy + kl_divergence",
            optimizer="adam",
            eta=0.01,
            training_mode="batch",
            batch_size=5,
            epochs=7500,
            epsilon=1e-3,
        )
    )

    # 5. Train the autoencoder (reconstruct fonts from themselves)
    history = trainer.fit(
        model=model,
        X_train=X_train,
        zeta_train=zeta_train,  # regular AE: input -> input; denoising AE: noisy -> clean
        X_val=None,
        zeta_val=None
    )

    # 6. Test the reconstruction
    reconstructed = np.array([model.reconstruct(font) for font in X_train])
    reconstruction_error = bce.compute(zeta_train, reconstructed)
    kl_error = model.kl_divergence(X_train)
    vae_error = reconstruction_error + model.kl_weight * kl_error
    print(f"Training epochs: {history['epochs']}")
    print(f"Reconstruction BCE: {reconstruction_error:.6f}")
    print(f"KL divergence: {kl_error:.6f}")
    print(f"VAE loss: {vae_error:.6f}")

    if font_datatype == "emoji":
        glyph_labels = [
            "happy", "sad", "wink", "surprise", "heart", "star", "check", "cross",
            "up", "down", "left", "right", "house", "music", "sun", "moon",
            "cloud", "umbrella", "lightning", "flower", "tree", "cat", "dog", "fish",
            "ghost", "skull", "robot", "alien", "rocket", "car", "coffee", "cake",
        ]
    else:
        glyph_labels = [chr(code) if code < 0x7f else "DEL" for code in range(0x60, 0x80)]

    latent_plot_path = model.plot_latent_space(
        X=orig_X,
        labels=glyph_labels,
        output_path=f"latent_space_{font_datatype}_vae.png",
        title=f"{font_datatype.capitalize()} VAE latent distributions",
    )
    print(f"Latent space plot saved to: {latent_plot_path}")

    # 7. Visualize



if __name__ == '__main__':
    main()
