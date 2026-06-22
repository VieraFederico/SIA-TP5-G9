"""
Estudio de arquitectura del autoencoder, con el cuello de botella fijo en 2.

Probamos varias formas de encoder (el decoder siempre las espeja) y comparamos, sobre
los 32 patrones: cuántos se reconstruyen con <=1 px, el peor caso, y en qué época corta
el entrenamiento. Cada arquitectura se corre con varias semillas; las figuras muestran
media ± desvío. Es una comparación descriptiva, no un intento de bajar a 1px moviendo capas.

    python3 grid_architecture.py                          # corrida completa
    python3 grid_architecture.py --epochs 300 --seeds 2   # rápido para probar
"""
import argparse
from pathlib import Path

import numpy as np

from ae import build_ae_model
from evaluation import pixel_error_counts
from experiment import (
    ADAM_BETA1, ADAM_BETA2, EPSILON, LEARNING_RATE, TRAINING_MODE,
    make_activations, make_trainer, study_subtitle,
)
from font import load_fonts
from graphs.studies import bar_study, overlaid_curves
from graphs.style import BLUE, ORANGE

# Cada arquitectura = anchos del encoder, de la entrada (35) a la capa previa al bottleneck.
ARCHITECTURES = {
    "directo 35-2":       [35],
    "shallow 35-16-2":    [35, 16],
    "medium 35-20-10-2":  [35, 20, 10],
    "deep 35-30-20-10-2": [35, 30, 20, 10],
    "default 35..4-2":    [35, 30, 25, 20, 16, 8, 4],
}


def run_seed(widths, hidden_act, seed, epochs, clean):
    """Entrena una arquitectura con una seed. Devuelve sus métricas y la curva de loss."""
    np.random.seed(seed)

    model = build_ae_model(make_activations(), seed=seed,
                           encoder_widths=widths, hidden_act=hidden_act)

    trainer, _ = make_trainer(widths, "binary_cross_entropy")
    trainer.cfg.epochs = epochs
    history = trainer.fit(model=model, X_train=clean, zeta_train=clean, X_val=None, zeta_val=None)

    passed, worst, _ = pixel_error_counts(model, clean)
    return {
        "pass": passed,
        "worst": worst,
        "epochs": history["epochs"],
        "loss_curve": history["train_error"],
    }


def run_architecture(widths, hidden_act, base_seed, n_seeds, epochs, clean):
    """Corre la arquitectura con n_seeds semillas y resume media ± desvío."""
    runs = [run_seed(widths, hidden_act, base_seed + k, epochs, clean) for k in range(n_seeds)]
    col = lambda key: np.array([r[key] for r in runs], dtype=float)

    return {
        "pass_mean": col("pass").mean(),
        "pass_std": col("pass").std(),
        "worst_mean": col("worst").mean(),
        "worst_std": col("worst").std(),
        "epochs_mean": col("epochs").mean(),
        "reaches": col("worst").mean() <= 1.0,
        "loss_curve": runs[0]["loss_curve"],
    }


def print_table(names, results):
    print(f"\n{'arquitectura':22s} | {'pasan/32':>12} | {'peor px':>12} | {'épocas':>8} | <=1px?")
    print("-" * 76)
    for name in names:
        r = results[name]
        print(f"{name:22s} | {r['pass_mean']:5.1f} ± {r['pass_std']:4.1f} | "
              f"{r['worst_mean']:5.1f} ± {r['worst_std']:4.1f} | {r['epochs_mean']:8.0f} | "
              f"{'SÍ' if r['reaches'] else 'no'}")


def save_figures(names, results, outdir, subtitle):
    bar_study(
        names,
        [results[n]["pass_mean"] for n in names],
        [results[n]["pass_std"] for n in names],
        "Patrones con <= 1 px (de 32)", "Arquitectura: patrones dentro del objetivo",
        str(outdir / "arch_pass_count.png"),
        subtitle=subtitle, color=BLUE, target=32, target_label="objetivo: 32/32", rotate=20,
    )
    bar_study(
        names,
        [results[n]["worst_mean"] for n in names],
        [results[n]["worst_std"] for n in names],
        "Peor caso (px mal, de 35)", "Arquitectura: peor patrón vs objetivo",
        str(outdir / "arch_worst_error.png"),
        subtitle=subtitle, color=ORANGE, target=1, target_label="objetivo <= 1 px", rotate=20,
    )
    overlaid_curves(
        [(n, results[n]["loss_curve"]) for n in names],
        "Época", "Train loss (BCE)", "Arquitectura: convergencia",
        str(outdir / "arch_convergence.png"),
        subtitle=subtitle, logy=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estudio de arquitectura (bottleneck=2).")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=3, help="semillas por arquitectura")
    parser.add_argument("--seed", type=int, default=42, help="seed base (las demás son base+1, ...)")
    parser.add_argument("--hidden-act", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--data", choices=["letters", "emoji"], default="letters")
    parser.add_argument("--output-dir", default="output/study/architecture")
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    clean = load_fonts(args.data)

    names = list(ARCHITECTURES.keys())
    results = {}
    for name in names:
        print(f"Arquitectura {name:22s} — {args.seeds} seeds x {args.epochs} épocas (act={args.hidden_act})...")
        results[name] = run_architecture(
            ARCHITECTURES[name], args.hidden_act, args.seed, args.seeds, args.epochs, clean)

    print_table(names, results)

    # El eje x es la arquitectura; el resto de los HP son fijos e iguales para todas.
    subtitle = study_subtitle(
        {"data": args.data, "seeds": args.seeds, "épocas": args.epochs},
        {"lr": LEARNING_RATE, "mode": TRAINING_MODE, "init": "he", "act": args.hidden_act,
         "bottleneck": 2, "opt": f"adam({ADAM_BETA1},{ADAM_BETA2})", "ε": EPSILON},
    )
    save_figures(names, results, outdir, subtitle)
    print(f"\nTodo guardado en: {outdir}/")


if __name__ == "__main__":
    main()
