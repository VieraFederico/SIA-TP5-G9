"""
experiment.py — pasos compartidos entre el autoencoder (AE/DAE) y el VAE.

Acá vive todo lo que NO depende del tipo de modelo: carga de datos + ruido,
creación de activaciones e hiperparámetros de entrenamiento, el loop de
entrenar/cargar pesos, el reporte de error y el gráfico del espacio latente.

Los módulos ae.py y vae.py sólo arman su topología y delegan en run_experiment.
"""
import dataclasses

import numpy as np

from activation.identity import IdentityActivation
from activation.logistic import LogisticActivation
from activation.relu import ReLUActivation
from activation.tanh import TanhActivation
from config import (
    CFG, ADAM_BETA1, ADAM_BETA2, BATCH_SIZE, EPOCHS, EPSILON,
    LEARNING_RATE, OUTPUT_ROOT, SALT_P, TRAINING_MODE,
)
from cost.binary_cross_entropy import BinaryCrossEntropyCost
from evaluation import pixel_error_report
from font import load_fonts, EMOJI_LABEL_NAMES
from graphs import plot_loss_curve, plot_reconstructions, plot_triptych
from noise.salt_n_pepper import SaltNPepperNoise
from optimizer.adam import AdamOptimizer
from trainer import Trainer
from weights_io import load_weights, save_weights

# Los hiperparámetros del TP son una única fuente de verdad en config.py (config.json).
# Acá sólo se importan; experiment ya no los deriva ni hace de hub: cada módulo los lee
# de config directamente.


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


def hyperparams_subtitle(hp: dict) -> str:
    """Todos los hiperparámetros del run, en una línea, para el pie del gráfico.

    Suma a los del run (data, ruido, epochs, lr, batch/kl...) los fijos del TP
    que no viven en hp: optimizador Adam (β1, β2), modo de entrenamiento y ε.
    """
    full = {
        **hp,
        "opt": "adam",
        "b1": ADAM_BETA1,
        "b2": ADAM_BETA2,
        "mode": TRAINING_MODE,
        "adam_eps": 1e-8,   # ε interno de Adam
        "tol": EPSILON,     # criterio de corte por convergencia (no es el ε de Adam)
    }
    parts = []
    for key, value in full.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "on" if value else "off"
        parts.append(f"{key}={value}")
    return "  ·  ".join(parts)


def study_subtitle(base: dict, hp: dict, varied: str | None = None) -> str:
    """Pie de figura de un estudio con TODOS los hiperparámetros, no sólo el que se barre.

    base va primero (data, seeds, épocas); hp es el resto. Si varied coincide con una clave
    de hp, esa se muestra como '(barrido)'. Así dos figuras del mismo estudio se comparan
    sabiendo qué quedó fijo y qué se movió.
    """
    parts = [f"{k}={v}" for k, v in base.items()]
    for key, value in hp.items():
        parts.append(f"{key}=" + ("(barrido)" if key == varied else str(value)))
    return "  ·  ".join(parts)


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
        return list(EMOJI_LABEL_NAMES)
    return [chr(code) if code < 0x7f else "DEL" for code in range(0x60, 0x80)]


def make_trainer(architecture: list[int], cost_name: str, seed: int | None = None, *,
                 epochs: int | None = None, training_mode: str | None = None,
                 optimizer=None):
    """Trainer + función de costo, con los hiperparámetros del TP.

    Los estudios pasan overrides (epochs/training_mode/optimizer) acá, en la
    construcción, en vez de mutar trainer.cfg después de crearlo. None = usa el
    default de config.json.
    """
    bce = BinaryCrossEntropyCost()
    if optimizer is None:
        optimizer = AdamOptimizer(learning_rate=LEARNING_RATE, beta1=ADAM_BETA1, beta2=ADAM_BETA2)
    # Partimos del CFG global y sólo pisamos lo que el estudio quiera overridear.
    cfg = dataclasses.replace(
        CFG,
        seed=seed,
        epochs=CFG.epochs if epochs is None else epochs,
        training_mode=CFG.training_mode if training_mode is None else training_mode,
    )
    trainer = Trainer(cost_fn=bce, optimizer=optimizer, metrics=[], cfg=cfg)
    return trainer, bce


def train_once(model, X_train, zeta_train, architecture, cost_name="binary_cross_entropy", *,
               seed=None, epochs=None, training_mode=None, optimizer=None,
               X_val=None, zeta_val=None, noise_fn=None):
    """Construye el trainer (con los overrides en la construcción, sin mutar cfg después),
    entrena el modelo y devuelve el history. Lo usan los estudios para no tocar internals."""
    trainer, _ = make_trainer(architecture, cost_name, seed=seed, epochs=epochs,
                              training_mode=training_mode, optimizer=optimizer)
    return trainer.fit(model=model, X_train=X_train, zeta_train=zeta_train,
                       X_val=X_val, zeta_val=zeta_val, noise_fn=noise_fn)


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
    plot,
    reconstruct,
    load_path: str | None = None,
    save: bool = False,
    extra_report=None,
    noise_fn=None,
):
    """
    Entrena (o carga pesos), imprime el error, grafica el latente y corre el
    reporte de píxeles. Devuelve dict con history, reconstructed y recon_bce.

    Las salidas (pesos, gráfico) van a output/<model_type>/<tipo>/<slug-hp>/.

    Args:
        model_type:   "ae" o "vae"; primer nivel de la carpeta de salida.
        hp:           hiperparámetros del run -> nombre de la subcarpeta.
        plot:         callable(model, clean, labels, output_path, title, subtitle)
                      -> ruta. Lo inyectan ae.py / vae.py: sacan los datos del
                      modelo y llaman a la función de graphs/ que corresponda. El
                      subtitle es la línea de hiperparámetros (hyperparams_subtitle).
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
            noise_fn=noise_fn,
        )
        if save:
            save_weights(model, output_path(model_type, "weights", hp, "weights.npz"))

    reconstructed = np.array([reconstruct(pattern) for pattern in x_input])
    recon_bce = bce.compute(target, reconstructed)
    print(f"Training epochs: {history['epochs']}")
    print(f"Reconstruction BCE: {recon_bce:.6f}")
    if extra_report is not None:
        extra_report(model, x_input, recon_bce)

    plot_file = plot(
        model,
        clean,
        labels,
        output_path(model_type, "latent_space", hp, "latent_space.png"),
        plot_title,
        hyperparams_subtitle(hp),
    )
    print(f"Latent space plot saved to: {plot_file}")

    pixel_error_report(
        model,
        X_target=clean,
        X_input=x_input,
        labels=labels,
        reconstruct=reconstruct,
    )

    # Figuras de presentación: imágenes (no ASCII) sobre lo que ya se calculó.
    subtitle = hyperparams_subtitle(hp)
    present = lambda name: output_path(model_type, "presentation", hp, name)

    recon_file = plot_reconstructions(
        clean, reconstructed, labels, present("reconstructions.png"),
        title=f"{plot_title}: entrada vs reconstrucción", subtitle=subtitle,
    )
    print(f"Reconstructions plot saved to: {recon_file}")

    if history.get("train_error"):
        loss_file = plot_loss_curve(
            history, present("loss.png"),
            title=f"{plot_title}: curva de loss", subtitle=subtitle,
        )
        print(f"Loss curve saved to: {loss_file}")

    # Tríptico sólo cuando hay denoising (la entrada difiere del patrón limpio).
    if not np.array_equal(np.asarray(clean), np.asarray(x_input)):
        tript_file = plot_triptych(
            clean, x_input, reconstructed, labels, present("denoising_triptych.png"),
            title=f"{plot_title}: limpio / ruidoso / reconstruido", subtitle=subtitle,
        )
        print(f"Denoising triptych saved to: {tript_file}")

    return {"history": history, "reconstructed": reconstructed, "recon_bce": recon_bce}
