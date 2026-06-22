"""
vae.py — Variational Autoencoder sobre los emojis (font_emoji.h).

Sólo arma la topología del VAE (encoder + cabezas μ y logσ² + decoder); el
entrenamiento, reporte y gráfico los maneja experiment.run_experiment.
El término KL se suma en el backward del modelo, no en la función de costo.
"""
import numpy as np

from ae import _visualize_samples, _write_reconstruction_csv
from font import EMOJI_LABELS
from graphs import plot_latent_distributions
from network.multilayer_perceptron import MultilayerPerceptron
from network.neuron_layer import NeuronLayer
from network.variational_autoencoder import VariationalAutoencoder

from experiment import (
    EPOCHS,
    KL_WEIGHT,
    LEARNING_RATE,
    SALT_P,
    SEED,
    load_dataset,
    make_activations,
    make_trainer,
    resolve_labels,
    run_experiment,
)

# Tamaños por capa (sólo metadata para el log de ExperimentConfig).
# KL_WEIGHT (β del VAE) ahora sale de config.json vía experiment.
VAE_ARCHITECTURE = [35, 30, 25, 20, 16, 8, 2, 8, 16, 20, 25, 30, 35]


def build_vae_model(act: dict, seed : int | None = None) -> VariationalAutoencoder:
    """encoder (35->8) + cabezas μ y logσ² (8->2 identity) + decoder (2->35)."""
    encoder = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=35, n_neurons=30, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=30, n_neurons=25, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=25, n_neurons=20, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=20, n_neurons=16, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=16, n_neurons=8, activation=act["relu"],rand_seed=seed),
    ])
    mean_layer = NeuronLayer(n_inputs=8, n_neurons=2, activation=act["identity"], rand_seed=seed)
    log_variance_layer = NeuronLayer(n_inputs=8, n_neurons=2, activation=act["identity"], rand_seed=seed)
    decoder = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=2, n_neurons=8, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=8, n_neurons=16, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=16, n_neurons=20, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=20, n_neurons=25, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=25, n_neurons=30, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=30, n_neurons=35, activation=act["logistic"],rand_seed=seed),
    ])
    return VariationalAutoencoder(
        encoder=encoder,
        mean_layer=mean_layer,
        log_variance_layer=log_variance_layer,
        decoder=decoder,
        kl_weight=KL_WEIGHT,
    )


def _plot_latent(model, clean, labels, output_path, title, subtitle) -> str:
    """Saca (μ, σ) por patrón del VAE y delega el gráfico en graphs/."""
    means, standard_deviations = model.get_latent_distributions(clean)
    return plot_latent_distributions(means, standard_deviations, labels, output_path, title, subtitle)


def _vae_report(model, x_input, recon_bce) -> None:
    """Métricas extra del VAE: KL y loss total (BCE + kl_weight·KL)."""
    kl = model.kl_divergence(x_input)
    print(f"KL divergence: {kl:.6f}")
    print(f"VAE loss: {recon_bce + model.kl_weight * kl:.6f}")


def run_vae(
    *,
    datatype: str = "emoji",
    with_noise: bool = True,
    kl_weight: float | None = None,
    load_path: str | None = None,
    save: bool = False,
    seed: int | None = None,
):
    if seed is None:               # --seed overridea; si no, usa el default de config.json
        seed = SEED
    if seed is not None:
        np.random.seed(seed)
    if kl_weight is None:
        kl_weight = KL_WEIGHT
    act = make_activations()
    clean, x_input, target = load_dataset(datatype, with_noise)
    model = build_vae_model(act,seed)
    model.kl_weight = kl_weight
    trainer, bce = make_trainer(VAE_ARCHITECTURE, "binary_cross_entropy + kl_divergence")

    hp = {
        "data": datatype,
        "noise": with_noise,
        "salt": SALT_P if with_noise else None,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "kl": kl_weight,
        "Seed" : seed
    }

    result =  run_experiment(
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
        plot=_plot_latent,
        reconstruct=model.reconstruct,  # determinista: usa μ, no samplea
        load_path=load_path,
        save=save,
        extra_report=_vae_report,
    )



    _visualize_samples(clean, x_input, result["reconstructed"], with_noise,EMOJI_LABELS)
    _write_reconstruction_csv(clean, result["reconstructed"], output_dir="output")
    return result
