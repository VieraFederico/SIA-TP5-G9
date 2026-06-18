"""
vae.py — Variational Autoencoder sobre los emojis (font_emoji.h).

Sólo arma la topología del VAE (encoder + cabezas μ y logσ² + decoder); el
entrenamiento, reporte y gráfico los maneja experiment.run_experiment.
El término KL se suma en el backward del modelo, no en la función de costo.
"""
from network.multilayer_perceptron import MultilayerPerceptron
from network.neuron_layer import NeuronLayer
from network.variational_autoencoder import VariationalAutoencoder

from experiment import (
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
VAE_ARCHITECTURE = [35, 30, 25, 20, 16, 8, 2, 8, 16, 20, 25, 30, 35]
KL_WEIGHT = 0.01


def build_vae_model(act: dict) -> VariationalAutoencoder:
    """encoder (35->8) + cabezas μ y logσ² (8->2 identity) + decoder (2->35)."""
    encoder = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=35, n_neurons=30, activation=act["relu"]),
        NeuronLayer(n_inputs=30, n_neurons=25, activation=act["relu"]),
        NeuronLayer(n_inputs=25, n_neurons=20, activation=act["relu"]),
        NeuronLayer(n_inputs=20, n_neurons=16, activation=act["relu"]),
        NeuronLayer(n_inputs=16, n_neurons=8, activation=act["relu"]),
    ])
    mean_layer = NeuronLayer(n_inputs=8, n_neurons=2, activation=act["identity"])
    log_variance_layer = NeuronLayer(n_inputs=8, n_neurons=2, activation=act["identity"])
    decoder = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=2, n_neurons=8, activation=act["relu"]),
        NeuronLayer(n_inputs=8, n_neurons=16, activation=act["relu"]),
        NeuronLayer(n_inputs=16, n_neurons=20, activation=act["relu"]),
        NeuronLayer(n_inputs=20, n_neurons=25, activation=act["relu"]),
        NeuronLayer(n_inputs=25, n_neurons=30, activation=act["relu"]),
        NeuronLayer(n_inputs=30, n_neurons=35, activation=act["logistic"]),
    ])
    return VariationalAutoencoder(
        encoder=encoder,
        mean_layer=mean_layer,
        log_variance_layer=log_variance_layer,
        decoder=decoder,
        kl_weight=KL_WEIGHT,
    )


def _vae_report(model, x_input, recon_bce) -> None:
    """Métricas extra del VAE: KL y loss total (BCE + kl_weight·KL)."""
    kl = model.kl_divergence(x_input)
    print(f"KL divergence: {kl:.6f}")
    print(f"VAE loss: {recon_bce + model.kl_weight * kl:.6f}")


def run_vae(
    *,
    datatype: str = "emoji",
    with_noise: bool = True,
    load_path: str | None = None,
    save: bool = False,
):
    act = make_activations()
    clean, x_input, target = load_dataset(datatype, with_noise)
    model = build_vae_model(act)
    trainer, bce = make_trainer(VAE_ARCHITECTURE, "binary_cross_entropy + kl_divergence")

    hp = {
        "data": datatype,
        "noise": with_noise,
        "salt": SALT_P if with_noise else None,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "kl": KL_WEIGHT,
    }

    return run_experiment(
        model,
        trainer,
        bce,
        clean=clean,
        x_input=x_input,
        target=target,
        labels=resolve_labels(datatype),
        model_type="vae",
        hp=hp,
        plot_title=f"{datatype.capitalize()} VAE latent distributions",
        reconstruct=model.reconstruct,  # determinista: usa μ, no samplea
        load_path=load_path,
        save=save,
        extra_report=_vae_report,
    )
