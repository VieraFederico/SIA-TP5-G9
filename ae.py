"""
ae.py — Autoencoder / Denoising Autoencoder sobre font.h.

Sólo arma la topología del AE; el entrenamiento, reporte y gráfico los maneja
experiment.run_experiment. Con with_noise=True la entrada se corrompe y el
objetivo es el patrón limpio -> denoising AE.
"""
import csv
from pathlib import Path

import numpy as np

from font import FONT_LABELS
from graphs import plot_latent_points, visualize_font
from network.autoencoder import Autoencoder
from network.multilayer_perceptron import MultilayerPerceptron
from network.neuron_layer import NeuronLayer
from noise.salt_n_pepper import SaltNPepperNoise

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


def build_ae_model(act: dict, seed : int | None = None) -> Autoencoder:
    """encoder (35->4) + cuello de botella (4->2 tanh) + decoder (2->35)."""
    encoder = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=35, n_neurons=30, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=30, n_neurons=25, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=25, n_neurons=20, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=20, n_neurons=16, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=16, n_neurons=8, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=8, n_neurons=4, activation=act["relu"],rand_seed=seed),
    ])
    latent_space = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=4, n_neurons=2, activation=act["tanh"],rand_seed=seed),  # bottleneck
    ])
    decoder = MultilayerPerceptron(layers=[
        NeuronLayer(n_inputs=2, n_neurons=4, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=4, n_neurons=8, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=8, n_neurons=16, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=16, n_neurons=20, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=20, n_neurons=25, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=25, n_neurons=30, activation=act["relu"],rand_seed=seed),
        NeuronLayer(n_inputs=30, n_neurons=35, activation=act["logistic"],rand_seed=seed),
    ])
    return Autoencoder(encoder=encoder, latent_space=latent_space, decoder=decoder)


def _plot_latent(model, clean, labels, output_path, title, subtitle) -> str:
    """Saca las posiciones latentes del AE y delega el gráfico en graphs/."""
    positions = model.get_latent_positions(clean)
    return plot_latent_points(positions, labels, output_path, title, subtitle)


def _visualize_samples(clean, x_input, reconstructed, with_noise: bool, labels) -> None:
    """
    Visualize only samples where reconstruction differs from the clean original.
    Comparison is done after thresholding both arrays to 0/1.
    """

    clean_bin = (np.array(clean, dtype=float) >= 0.5).astype(int)
    recon_bin = (np.array(reconstructed, dtype=float) >= 0.5).astype(int)

    # mismatched_indices = [
    #     i for i in range(len(clean_bin))
    #     if not np.array_equal(clean_bin[i], recon_bin[i])
    # ]

    mismatched_indices = [
        i for i in range(9)
    ]

    if not mismatched_indices:
        print("All reconstructions match the originals (after thresholding at 0.5).")
        return


    # Label letters as a, b, c... when possible; fallback to index otherwise.
    for i in mismatched_indices:
        name = labels[i]

        visualize_font(
            x_input[i],
            f"Original '{name}' Noise" if with_noise else f"Original '{name}'"
        )
        if with_noise:
            visualize_font(clean[i], f"Original '{name}'")
        visualize_font(reconstructed[i], f"Reconstructed '{name}'")

def _row_to_font_string(row) -> str:
    """
    Convert a 35-length vector into a compact 7x5 string using 0/1 chars.
    Rows are separated by '|', e.g. '01110|10001|...'
    """
    arr = np.array(row, dtype=float).reshape(7, 5)
    bits = (arr >= 0.5).astype(int)
    return "|".join("".join(str(v) for v in r) for r in bits)


def _similarity_percentage(original_row, reconstructed_row) -> float:
    """
    Percentage of equal pixels after thresholding both vectors to 0/1.
    """
    o = (np.array(original_row, dtype=float) >= 0.5).astype(int).reshape(-1)
    r = (np.array(reconstructed_row, dtype=float) >= 0.5).astype(int).reshape(-1)
    return float((o == r).mean() * 100.0)


def _write_reconstruction_csv(clean, reconstructed, output_dir: str = "output") -> str:
    """
    Writes CSV with columns:
    1) original representation
    2) reconstructed representation
    3) similarity_percentage
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "ae_reconstruction_similarity.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_letter", "reconstructed_letter", "similarity_percentage"])

        for i in range(len(clean)):
            original_repr = _row_to_font_string(clean[i])
            reconstructed_repr = _row_to_font_string(reconstructed[i])
            similarity = _similarity_percentage(clean[i], reconstructed[i])
            writer.writerow([original_repr, reconstructed_repr, f"{similarity:.2f}"])

    print(f"Reconstruction CSV written to: {csv_path}")
    return str(csv_path)

def run_ae(
    *,
    datatype: str = "letters",
    with_noise: bool = True,
    salt_p: float = SALT_P,
    resample_noise: bool = True,
    load_path: str | None = None,
    save: bool = False,
    show_viz: bool = True,
    seed: int | None = None,
):
    if seed is not None:
        np.random.seed(seed)

    if salt_p is None:
        salt_p = SALT_P

    act = make_activations()
    clean, x_input, target = load_dataset(datatype, with_noise, salt_p)
    model = build_ae_model(act, seed)
    trainer, bce = make_trainer(AE_ARCHITECTURE, "binary_cross_entropy")

    # Denoising AE: re-sampleamos el ruido en cada época (corrupción distinta de
    # cada patrón) para que aprenda a limpiar y no memorice un ruido fijo
    noise_fn = None
    if with_noise and resample_noise:
        noise_fn = lambda: SaltNPepperNoise(salt_p).add_noise(clean.copy())

    hp = {
        "data": datatype,
        "noise": with_noise,
        "salt": salt_p if with_noise else None,
        "resample": resample_noise if with_noise else None,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "batch": BATCH_SIZE,
        "Seed" : seed
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
        plot=_plot_latent,
        reconstruct=model.forward,
        load_path=load_path,
        save=save,
        noise_fn=noise_fn,
    )

    if show_viz:

        _visualize_samples(clean, x_input, result["reconstructed"], with_noise, FONT_LABELS)

    _write_reconstruction_csv(clean, result["reconstructed"], output_dir="output")
    return result
