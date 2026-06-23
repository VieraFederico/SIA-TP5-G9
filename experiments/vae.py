"""
vae.py — Variational Autoencoder sobre los emojis (font_emoji.h).

Sólo arma la topología del VAE (encoder + cabezas μ y logσ² + decoder); el
entrenamiento, reporte y gráfico los maneja experiment.run_experiment.
El término KL se suma en el backward del modelo, no en la función de costo.
"""
import numpy as np

from src.utils.evaluation import write_reconstruction_csv
from src.data.font import EMOJI_LABELS
from graphs import plot_latent_distributions, visualize_samples
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.network.variational_autoencoder import VariationalAutoencoder
from src.utils.sampling import set_seed

from src.utils.config import EPOCHS, KL_WEIGHT, LEARNING_RATE, OUTPUT_ROOT, SALT_P, SEED
from experiments.experiment import (
    LATENT_DIM,
    load_dataset,
    make_activations,
    make_trainer,
    resolve_labels,
    run_experiment,
)

# Encoder del VAE = ganador del estudio de arquitectura AE ("deep 35-30-20-10-2").
# El decoder lo espeja. KL_WEIGHT (β del VAE) sale de config.json vía experiment.
VAE_ENCODER_WIDTHS = [35, 30, 20, 10]
VAE_ARCHITECTURE = (
    VAE_ENCODER_WIDTHS + [LATENT_DIM] + VAE_ENCODER_WIDTHS[::-1]
)  # 35-30-20-10-2-10-20-30-35


def build_vae_model(act: dict, seed: int | None = None) -> VariationalAutoencoder:
    """encoder (35->10) + cabezas μ y logσ² (10->2 identity) + decoder (2->35).

    Topología = ganador del estudio de arquitectura AE; el decoder espeja el encoder."""
    widths = VAE_ENCODER_WIDTHS
    encoder = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=widths[i], n_neurons=widths[i + 1], activation=act["relu"], rand_seed=seed)
        for i in range(len(widths) - 1)
    ])
    last = widths[-1]
    mean_layer = NeuronLayer(n_inputs=last, n_neurons=LATENT_DIM, activation=act["identity"], rand_seed=seed)
    log_variance_layer = NeuronLayer(n_inputs=last, n_neurons=LATENT_DIM, activation=act["identity"], rand_seed=seed)

    dec_widths = [LATENT_DIM] + widths[::-1]  # 2 -> 10 -> 20 -> 30 -> 35
    decoder = MultilayerPerceptron(layers=[
        NeuronLayer(
            n_inputs=dec_widths[i], n_neurons=dec_widths[i + 1],
            activation=act["logistic"] if i == len(dec_widths) - 2 else act["relu"],
            rand_seed=seed,
        )
        for i in range(len(dec_widths) - 1)
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
    set_seed(seed)
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



    visualize_samples(clean, x_input, result["reconstructed"], with_noise, EMOJI_LABELS)
    write_reconstruction_csv(clean, result["reconstructed"], output_dir=str(OUTPUT_ROOT / "vae"))
    return result
