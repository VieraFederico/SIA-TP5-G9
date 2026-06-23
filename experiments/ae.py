"""
ae.py — Autoencoder / Denoising Autoencoder sobre font.h.

Sólo arma la topología del AE; el entrenamiento, reporte y gráfico los maneja
experiment.run_experiment. Con with_noise=True la entrada se corrompe y el
objetivo es el patrón limpio -> denoising AE.
"""
from src.utils.evaluation import write_reconstruction_csv
from src.data.font import FONT_LABELS
from graphs import plot_latent_points, visualize_samples
from src.network.autoencoder import Autoencoder
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.noise.salt_n_pepper import SaltNPepperNoise
from src.utils.sampling import set_seed

from src.optimizer.adam import AdamOptimizer
from src.utils.config import (
    ADAM_BETA1, ADAM_BETA2, BATCH_SIZE, DAE_LEARNING_RATE, EPOCHS, LEARNING_RATE,
    OUTPUT_ROOT, SALT_P, SEED, TRAINING_MODE,
)
from experiments.experiment import (
    LATENT_DIM,
    load_dataset,
    make_activations,
    make_trainer,
    resolve_labels,
    run_experiment,
)

# Anchos del encoder, de la entrada (35) a la capa que alimenta el bottleneck.
# El decoder los espeja; *_ARCHITECTURE es la lista completa (para el log).
# Arquitecturas canónicas elegidas para los experimentos.
AE_ENCODER_WIDTHS = [35, 30, 20, 10]
DAE_ENCODER_WIDTHS = [35, 30, 20, 10]
AE_ARCHITECTURE = AE_ENCODER_WIDTHS + [LATENT_DIM] + AE_ENCODER_WIDTHS[::-1]
DAE_ARCHITECTURE = DAE_ENCODER_WIDTHS + [LATENT_DIM] + DAE_ENCODER_WIDTHS[::-1]
AE_ARCH_LABEL = "-".join(map(str, AE_ENCODER_WIDTHS + [LATENT_DIM]))
DAE_ARCH_LABEL = "-".join(map(str, DAE_ENCODER_WIDTHS + [LATENT_DIM]))


def build_ae_model(act: dict, seed: int | None = None,
                   encoder_widths: list[int] = AE_ENCODER_WIDTHS,
                   hidden_act: str = "relu", init_scheme: str = "he") -> Autoencoder:
    """Autoencoder con cuello de botella 2D.

    encoder_widths va de la entrada (35) a la capa que alimenta el bottleneck; el
    decoder espeja esos anchos y cierra con una sigmoide. Por defecto arma la
    topología AE canónica (35-30-20-10-2 y vuelta). El estudio de arquitectura
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
        NeuronLayer(encoder_widths[-1], LATENT_DIM, act["tanh"],
                    init_scheme=init_scheme, rand_seed=seed),  # bottleneck 2D
    ])

    widths = [LATENT_DIM] + encoder_widths[::-1]  # 2 -> ... -> 35
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
    epochs: int | None = None,
):
    if seed is None:               # --seed overridea; si no, usa el default de config.json
        seed = SEED
    set_seed(seed)
    n_epochs = EPOCHS if epochs is None else epochs

    if salt_p is None:
        salt_p = SALT_P

    act = make_activations()
    clean, x_input, target = load_dataset(datatype, with_noise, salt_p)
    # Config canónica: AE puro y DAE pueden tener arquitecturas y learning rates distintos.
    encoder_widths = DAE_ENCODER_WIDTHS if with_noise else AE_ENCODER_WIDTHS
    architecture = DAE_ARCHITECTURE if with_noise else AE_ARCHITECTURE
    lr = DAE_LEARNING_RATE if with_noise else LEARNING_RATE
    model = build_ae_model(act, seed, encoder_widths=encoder_widths)
    optimizer = AdamOptimizer(learning_rate=lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2)
    trainer, bce = make_trainer(architecture, "binary_cross_entropy",
                                optimizer=optimizer, epochs=n_epochs)

    print(f"[config efectiva] {'DAE' if with_noise else 'AE'}: arch={architecture} · "
          f"opt=adam · lr={lr} · init=he · mode={TRAINING_MODE} · act=relu · "
          f"{'salt=' + str(salt_p) + ' resample=' + ('on' if resample_noise else 'off') if with_noise else 'sin ruido'}")

    # Denoising AE: re-sampleamos el ruido en cada época (corrupción distinta de
    # cada patrón) para que aprenda a limpiar y no memorice un ruido fijo
    noise_fn = None
    if with_noise and resample_noise:
        noise_fn = lambda: SaltNPepperNoise(salt_p).add_noise(clean.copy())

    # AE sin ruido (1.a) y denoising AE (1.b) caen en carpetas de primer nivel distintas.
    model_type = "dae" if with_noise else "ae"

    hp = {
        "data": datatype,
        "arch": DAE_ARCH_LABEL if with_noise else AE_ARCH_LABEL,
        "noise": with_noise,
        "salt": salt_p if with_noise else None,
        "resample": resample_noise if with_noise else None,
        "epochs": n_epochs,
        "lr": lr,
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
