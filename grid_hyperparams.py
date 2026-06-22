"""
Estudio de hiperparámetros del autoencoder, con la arquitectura fija (build_ae_model).

Barremos un eje por vez (los otros quedan en su default de config.json):

    learning_rate     paso del optimizador
    training_mode     online / batch / minibatch
    init              inicialización de pesos: He / Xavier / normal chico
    optimizer         adam / gd (gradient descent)
    activation        relu / tanh en las capas ocultas
    epochs            cuántas épocas entrena (fracciones de --epochs)

Por celda medimos cuántos de los 32 se reconstruyen con <=1 px. Cada celda corre con
varias semillas y la figura muestra media ± desvío. El eje kl (β del VAE) no está acá:
lo cubre sweep_kl.py.

Cada figura se guarda en una carpeta nombrada por la COMBINACIÓN de hiperparámetros fijos
(los que no se barren), así corridas con distintos fijos no se pisan y se pueden comparar.

    python3 grid_hyperparams.py --epochs 300 --seeds 2   # rápido para probar
    python3 grid_hyperparams.py --axis opt               # un solo eje
"""
import argparse
from pathlib import Path

import numpy as np

from ae import AE_ARCHITECTURE, build_ae_model
from evaluation import pixel_error_counts
from sampling import set_seed
from config import ADAM_BETA1, ADAM_BETA2, EPSILON, LEARNING_RATE, TRAINING_MODE
from experiment import hyperparams_slug, make_activations, study_subtitle, train_once
from font import load_fonts
from graphs.studies import bar_study
from optimizer.adam import AdamOptimizer
from optimizer.gradient_descent import GradientDescent

# Niveles por eje. epochs se calcula a partir de --epochs (ver epoch_levels).
LR_LEVELS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
MODE_LEVELS = ["online", "batch", "minibatch"]
INIT_LEVELS = ["he", "xavier", "normal"]
OPT_LEVELS = ["adam", "gd"]
ACT_LEVELS = ["relu", "tanh"]

# Nombre lindo para el título + lista de niveles. epochs se llena en runtime.
AXES = {
    "lr":     ("learning_rate", LR_LEVELS),
    "mode":   ("training_mode", MODE_LEVELS),
    "init":   ("init", INIT_LEVELS),
    "opt":    ("optimizer", OPT_LEVELS),
    "act":    ("activation", ACT_LEVELS),
    "epochs": ("epochs", None),
}


def epoch_levels(max_epochs):
    """Niveles del eje epochs: fracciones de --epochs (así --epochs también lo abarata)."""
    return sorted({max(1, int(max_epochs * f)) for f in (0.1, 0.25, 0.5, 1.0)})


def make_optimizer(name, lr):
    """adam (con los β de config) o gradient descent plano."""
    if name == "adam":
        return AdamOptimizer(lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2)
    return GradientDescent(lr)


def run_cell(seed, clean, *, lr, mode, init, opt, act, epochs):
    """El AE fijo entrenado con un set concreto de hiperparámetros y una seed.
    Devuelve (pasan, peor_px)."""
    set_seed(seed)

    model = build_ae_model(make_activations(), seed=seed, hidden_act=act, init_scheme=init)

    train_once(model, clean, clean, AE_ARCHITECTURE,
               epochs=epochs, training_mode=mode, optimizer=make_optimizer(opt, lr))

    passed, worst, _ = pixel_error_counts(model, clean)
    return passed, worst


def sweep_axis(levels, cell, base_seed, n_seeds):
    """Para cada valor del eje corre n_seeds celdas y resume media ± desvío de los que pasan.
    cell(value, seed) -> (pasan, peor_px)."""
    pass_mean, pass_std = [], []
    for v in levels:
        runs = [cell(v, base_seed + k) for k in range(n_seeds)]
        passed = np.array([p for p, _ in runs], dtype=float)
        worst = np.array([w for _, w in runs], dtype=float)

        pass_mean.append(passed.mean())
        pass_std.append(passed.std())
        print(f"    {str(v):>10}: pasan {passed.mean():4.1f}±{passed.std():3.1f}/32  "
              f"peor {worst.mean():4.1f}±{worst.std():3.1f} px")

    return pass_mean, pass_std


def defaults_for(args):
    """Valores fijos por defecto de cada eje (los que no se barren)."""
    return {"lr": LEARNING_RATE, "mode": TRAINING_MODE, "init": "he",
            "opt": "adam", "act": "relu", "epochs": args.epochs}


def combo_dir(outdir, axis, defaults, args):
    """Carpeta nombrada por la combinación de HP FIJOS (todos menos el eje barrido)."""
    fixed = {k: v for k, v in defaults.items() if k != axis}
    folder = outdir / hyperparams_slug({"data": args.data, "seeds": args.seeds, **fixed})
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def study_axis(axis, args, clean, outdir):
    """Barre un eje: corre las celdas, imprime y guarda la figura en su carpeta de combinación."""
    title, levels = AXES[axis]
    if axis == "epochs":
        levels = epoch_levels(args.epochs)

    print(f"Eje {title}:")
    defaults = defaults_for(args)
    cell = lambda value, seed: run_cell(seed, clean, **{**defaults, axis: value})
    pass_mean, pass_std = sweep_axis(levels, cell, args.seed, args.seeds)

    folder = combo_dir(outdir, axis, defaults, args)
    subtitle = study_subtitle(
        {"data": args.data, "seeds": args.seeds},
        {"arch": "default", **defaults, "bottleneck": 2, "ε": EPSILON},
        varied=axis,
    )
    bar_study(
        [str(v) for v in levels], pass_mean, pass_std,
        "Patrones con <= 1 px (de 32)", f"Hiperparámetro: {title}",
        str(folder / f"hp_{axis}.png"), subtitle=subtitle,
        target=32, target_label="objetivo: 32/32",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estudio de hiperparámetros (arquitectura fija).")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=3, help="semillas por celda")
    parser.add_argument("--seed", type=int, default=42, help="seed base (las demás son base+1, ...)")
    parser.add_argument("--axis", choices=["all", *AXES], default="all")
    parser.add_argument("--data", choices=["letters", "emoji"], default="letters")
    parser.add_argument("--output-dir", default="output/study/hyperparams")
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    clean = load_fonts(args.data)

    axes = list(AXES) if args.axis == "all" else [args.axis]
    for axis in axes:
        study_axis(axis, args, clean, outdir)

    print(f"\nTodo guardado en: {outdir}/")


if __name__ == "__main__":
    main()
