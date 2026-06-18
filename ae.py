"""
ae.py — Autoencoder / Denoising Autoencoder sobre font.h.

Sólo arma la topología del AE; el entrenamiento, reporte y gráfico los maneja
experiment.run_experiment. Con with_noise=True la entrada se corrompe y el
objetivo es el patrón limpio -> denoising AE.
"""
from font import visualize_font
from network.autoencoder import Autoencoder
from network.multilayer_perceptron import MultilayerPerceptron
from network.neuron_layer import NeuronLayer

from experiment import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SALT_P,
    load_dataset,
    make_activations,
    make_trainer,
    resolve_labels,
    run_experiment,
)

# Tamaños por capa (sólo metadata para el log de ExperimentConfig).
AE_ARCHITECTURE = [35, 30, 25, 20, 16, 8, 4, 2, 4, 8, 16, 20, 25, 30, 35]


def build_ae_model(act: dict) -> Autoencoder:
    """encoder (35->4) + cuello de botella (4->2 tanh) + decoder (2->35)."""
    encoder = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=35, n_neurons=30, activation=act["relu"]),
        NeuronLayer(n_inputs=30, n_neurons=25, activation=act["relu"]),
        NeuronLayer(n_inputs=25, n_neurons=20, activation=act["relu"]),
        NeuronLayer(n_inputs=20, n_neurons=16, activation=act["relu"]),
        NeuronLayer(n_inputs=16, n_neurons=8, activation=act["relu"]),
        NeuronLayer(n_inputs=8, n_neurons=4, activation=act["relu"]),
    ])
    latent_space = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=4, n_neurons=2, activation=act["tanh"]),  # bottleneck
    ])
    decoder = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=2, n_neurons=4, activation=act["relu"]),
        NeuronLayer(n_inputs=4, n_neurons=8, activation=act["relu"]),
        NeuronLayer(n_inputs=8, n_neurons=16, activation=act["relu"]),
        NeuronLayer(n_inputs=16, n_neurons=20, activation=act["relu"]),
        NeuronLayer(n_inputs=20, n_neurons=25, activation=act["relu"]),
        NeuronLayer(n_inputs=25, n_neurons=30, activation=act["relu"]),
        NeuronLayer(n_inputs=30, n_neurons=35, activation=act["logistic"]),
    ])
    return Autoencoder(encoder=encoder, latent_space=latent_space, decoder=decoder)


def _visualize_samples(clean, x_input, reconstructed, with_noise: bool) -> None:
    """Muestra como ASCII los primeros 5 caracteres (a-e): entrada y reconstrucción."""
    names = ["a", "b", "c", "d", "e"]
    for i, name in enumerate(names, start=1):
        visualize_font(x_input[i], f"Original '{name}' Noise" if with_noise else f"Original '{name}'")
        if with_noise:
            visualize_font(clean[i], f"Original '{name}'")
        visualize_font(reconstructed[i], f"Reconstructed '{name}'")


def run_ae(
    *,
    datatype: str = "letters",
    with_noise: bool = True,
    load_path: str | None = None,
    save: bool = False,
    show_viz: bool = True,
):
    act = make_activations()
    clean, x_input, target = load_dataset(datatype, with_noise)
    model = build_ae_model(act)
    trainer, bce = make_trainer(AE_ARCHITECTURE, "binary_cross_entropy")

    hp = {
        "data": datatype,
        "noise": with_noise,
        "salt": SALT_P if with_noise else None,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "batch": BATCH_SIZE,
    }

    result = run_experiment(
        model,
        trainer,
        bce,
        clean=clean,
        x_input=x_input,
        target=target,
        labels=resolve_labels(datatype),
        model_type="ae",
        hp=hp,
        plot_title=f"{datatype.capitalize()} patterns in 2D latent space",
        reconstruct=model.forward,
        load_path=load_path,
        save=save,
    )

    if show_viz:
        _visualize_samples(clean, x_input, result["reconstructed"], with_noise)
    return result
