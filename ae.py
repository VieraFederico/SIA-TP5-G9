"""
ae.py — Autoencoder / Denoising Autoencoder sobre font.h.

Sólo arma la topología del AE; el entrenamiento, reporte y gráfico los maneja
experiment.run_experiment. Con with_noise=True la entrada se corrompe y el
objetivo es el patrón limpio -> denoising AE.
"""
from evaluation import write_reconstruction_csv
from font import FONT_LABELS
from graphs import plot_latent_points, visualize_samples
from network.autoencoder import Autoencoder
from network.multilayer_perceptron import MultilayerPerceptron
from network.neuron_layer import NeuronLayer
from noise.salt_n_pepper import SaltNPepperNoise
from sampling import set_seed

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, OUTPUT_ROOT, SALT_P, SEED
from experiment import (
    load_dataset,
    make_activations,
    make_trainer,
    resolve_labels,
    run_experiment,
)

# Anchos del encoder, de la entrada (35) a la capa que alimenta el bottleneck.
# El decoder los espeja; AE_ARCHITECTURE es la lista completa (para el log).
AE_ENCODER_WIDTHS = [35, 30, 25, 20, 16, 8, 4]
AE_ARCHITECTURE = AE_ENCODER_WIDTHS + [2] + AE_ENCODER_WIDTHS[::-1]


def build_ae_model(act: dict, seed: int | None = None,
                   encoder_widths: list[int] = AE_ENCODER_WIDTHS,
                   hidden_act: str = "relu", init_scheme: str = "he") -> Autoencoder:
    """Autoencoder con cuello de botella 2D.

    encoder_widths va de la entrada (35) a la capa que alimenta el bottleneck; el
    decoder espeja esos anchos y cierra con una sigmoide. Por defecto arma la
    topología del TP (35-30-25-20-16-8-4-2 y vuelta). El estudio de arquitectura
    reusa esta misma función con otros anchos; el de hiperparámetros, con otro
    init_scheme (he/xavier/normal).
    """
    hidden = act[hidden_act]

    encoder = MultilayerPerceptron(layers=[
        NeuronLayer(encoder_widths[i], encoder_widths[i + 1], hidden,
                    init_scheme=init_scheme, rand_seed=seed)
        for i in range(len(encoder_widths) - 1)
    ])

    latent_space = MultilayerPerceptron(layers=[
        NeuronLayer(encoder_widths[-1], 2, act["tanh"],
                    init_scheme=init_scheme, rand_seed=seed),  # bottleneck 2D
    ])

    widths = [2] + encoder_widths[::-1]  # 2 -> ... -> 35
    decoder = MultilayerPerceptron(layers=[
        NeuronLayer(
            widths[i], widths[i + 1],
            act["logistic"] if i == len(widths) - 2 else hidden,  # sigmoide sólo en la salida
            init_scheme=init_scheme, rand_seed=seed,
        )
        for i in range(len(widths) - 1)
    ])

    return Autoencoder(encoder=encoder, latent_space=latent_space, decoder=decoder)


def _plot_latent(model, clean, labels, output_path, title, subtitle) -> str:
    """Saca las posiciones latentes del AE y delega el gráfico en graphs/."""
    positions = model.get_latent_positions(clean)
    return plot_latent_points(positions, labels, output_path, title, subtitle)


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
    if seed is None:               # --seed overridea; si no, usa el default de config.json
        seed = SEED
    set_seed(seed)

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

    # AE sin ruido (1.a) y denoising AE (1.b) caen en carpetas de primer nivel distintas.
    model_type = "dae" if with_noise else "ae"

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
        model_type=model_type,
        hp=hp,
        plot_title=f"{datatype.capitalize()} patterns in 2D latent space",
        plot=_plot_latent,
        reconstruct=model.forward,
        load_path=load_path,
        save=save,
        noise_fn=noise_fn,
    )

    if show_viz:

        visualize_samples(clean, x_input, result["reconstructed"], with_noise, FONT_LABELS)

    write_reconstruction_csv(clean, result["reconstructed"], output_dir=str(OUTPUT_ROOT / model_type))
    return result
