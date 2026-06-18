"""
experiment.py — pasos compartidos entre el autoencoder (AE/DAE) y el VAE.

Acá vive todo lo que NO depende del tipo de modelo: carga de datos + ruido,
creación de activaciones e hiperparámetros de entrenamiento, el loop de
entrenar/cargar pesos, el reporte de error y el gráfico del espacio latente.

Los módulos ae.py y vae.py sólo arman su topología y delegan en run_experiment.
"""
from pathlib import Path

import numpy as np

from activation.identity import IdentityActivation
from activation.logistic import LogisticActivation
from activation.relu import ReLUActivation
from activation.tanh import TanhActivation
from config import ExperimentConfig
from cost.binary_cross_entropy import BinaryCrossEntropyCost
from evaluation import pixel_error_report
from font import load_fonts
from noise.salt_n_pepper import SaltNPepperNoise
from optimizer.adam import AdamOptimizer
from trainer import Trainer
from weights_io import load_weights, save_weights

# Nombres de los 32 emojis (mismo orden que EMOJI_FONTS_DATA en font.py).
EMOJI_LABELS = [
    "happy", "sad", "wink", "surprise", "heart", "star", "check", "cross",
    "up", "down", "left", "right", "house", "music", "sun", "moon",
    "cloud", "umbrella", "lightning", "flower", "tree", "cat", "dog", "fish",
    "ghost", "skull", "robot", "alien", "rocket", "car", "coffee", "cake",
]

# Hiperparámetros de entrenamiento (únicos, compartidos AE/VAE). Una sola
# fuente: make_trainer los usa y los slugs de carpeta los reportan.
LEARNING_RATE = 0.001
EPOCHS = 7500
BATCH_SIZE = 5
EPSILON = 1e-3
TRAINING_MODE = "batch"
SALT_P = 0.01

# Todas las salidas van a:  output/<modelo>/<tipo>/<hiperparams>/<archivo>
OUTPUT_ROOT = Path("output")


def hyperparams_slug(hp: dict) -> str:
    """Convierte {clave: valor} en un nombre de carpeta estable y seguro."""
    parts = []
    for key, value in hp.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "on" if value else "off"
        parts.append(f"{key}-{value}")
    return "_".join(parts)


def output_path(model_type: str, kind: str, hp: dict, filename: str) -> str:
    """Arma output/<model_type>/<kind>/<slug>/<filename> y crea las carpetas."""
    directory = OUTPUT_ROOT / model_type / kind / hyperparams_slug(hp)
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / filename)


def make_activations() -> dict:
    """Activaciones compartidas. Los betas (tanh=0.4, logistic=1.0) viven acá."""
    return {
        "relu": ReLUActivation(),
        "identity": IdentityActivation(),
        "tanh": TanhActivation(beta=0.4),
        "logistic": LogisticActivation(beta=1.0),
    }


def load_dataset(datatype: str, with_noise: bool, salt_p: float = SALT_P):
    """
    Devuelve (clean, x_input, target):
        clean   : patrones limpios (32, 35) = lo que se grafica y el objetivo del DAE.
        x_input : entrada al modelo (ruidosa si with_noise, si no = clean).
        target  : objetivo de reconstrucción (clean si with_noise, si no = x_input).
    """
    clean = load_fonts(datatype)
    x_input = clean.copy()
    if with_noise:
        x_input = SaltNPepperNoise(salt_p).add_noise(x_input)
    target = clean if with_noise else x_input
    return clean, x_input, target


def resolve_labels(datatype: str) -> list[str]:
    """Etiqueta por patrón: emojis por nombre, letras por carácter ASCII."""
    if datatype == "emoji":
        return list(EMOJI_LABELS)
    return [chr(code) if code < 0x7f else "DEL" for code in range(0x60, 0x80)]


def make_trainer(architecture: list[int], cost_name: str):
    """Trainer + función de costo, con los hiperparámetros fijos del TP."""
    bce = BinaryCrossEntropyCost()
    adam = AdamOptimizer(learning_rate=LEARNING_RATE)
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
            architecture=architecture,
            cost_function=cost_name,
            optimizer="adam",
            eta=0.01,
            training_mode=TRAINING_MODE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            epsilon=EPSILON,
        ),
    )
    return trainer, bce


def run_experiment(
    model,
    trainer,
    bce,
    *,
    clean,
    x_input,
    target,
    labels,
    model_type: str,
    hp: dict,
    plot_title: str,
    reconstruct,
    load_path: str | None = None,
    save: bool = False,
    extra_report=None,
):
    """
    Entrena (o carga pesos), imprime el error, grafica el latente y corre el
    reporte de píxeles. Devuelve dict con history, reconstructed y recon_bce.

    Las salidas (pesos, gráfico) van a output/<model_type>/<tipo>/<slug-hp>/.

    Args:
        model_type:   "ae" o "vae"; primer nivel de la carpeta de salida.
        hp:           hiperparámetros del run -> nombre de la subcarpeta.
        reconstruct:  función x -> x' determinista (model.forward para AE,
                      model.reconstruct para VAE).
        save:         si True, guarda los pesos en output/.../weights/.../weights.npz.
        extra_report: callable(model, x_input, recon_bce) para métricas extra
                      del modelo (ej. KL + loss total del VAE). Opcional.
    """
    if load_path:
        load_weights(model, load_path)
        history = {"epochs": 0}
    else:
        history = trainer.fit(
            model=model,
            X_train=x_input,
            zeta_train=target,
            X_val=None,
            zeta_val=None,
        )
        if save:
            save_weights(model, output_path(model_type, "weights", hp, "weights.npz"))

    reconstructed = np.array([reconstruct(pattern) for pattern in x_input])
    recon_bce = bce.compute(target, reconstructed)
    print(f"Training epochs: {history['epochs']}")
    print(f"Reconstruction BCE: {recon_bce:.6f}")
    if extra_report is not None:
        extra_report(model, x_input, recon_bce)

    plot = model.plot_latent_space(
        X=clean,
        labels=labels,
        output_path=output_path(model_type, "latent_space", hp, "latent_space.png"),
        title=plot_title,
    )
    print(f"Latent space plot saved to: {plot}")

    pixel_error_report(
        model,
        X_target=clean,
        X_input=x_input,
        labels=labels,
        reconstruct=reconstruct,
    )

    return {"history": history, "reconstructed": reconstructed, "recon_bce": recon_bce}
