"""
Estudio de arquitectura del DENOISING autoencoder, con el cuello de botella fijo en 2.

Mismo conjunto de encoders que el estudio del AE (grid_architecture.ARCHITECTURES),
pero entrenados en modo denoising: entrada ruidosa, objetivo limpio, ruido re-sampleado
por época (resample=on). Se corren DOS niveles de ruido —salt 0.1 y 0.2— como DOS series
en cada figura, para ver si el ranking de arquitecturas se mantiene cuando sube el ruido.

NO duplica el loop de grid: reusa run_architecture de grid_architecture.py (la misma
función que usa el estudio del AE), pasándole with_noise=True y el salt. La métrica es el
error de reconstrucción contra el patrón LIMPIO (pixel_errors_per_pattern vs clean), nunca
contra el ruidoso; NO se usa el criterio "objetivo <=1px" del AE puro.

    python3 main.py study architecture-dae                      # corrida completa
    python3 main.py study architecture-dae --epochs 30 --seeds 2   # rápido para probar
"""
import argparse
from pathlib import Path

from experiments.grid_architecture import ARCHITECTURES, run_architecture
from experiments.experiment import study_subtitle
from src.utils.config import ADAM_BETA1, ADAM_BETA2, EPSILON, LEARNING_RATE, TRAINING_MODE
from src.data.font import load_fonts
from graphs.studies import grouped_bar_study, overlaid_curves
from graphs.style import BLUE, ORANGE

# Dos niveles de ruido = dos series en cada figura (no dos corridas sueltas).
SALT_LEVELS = [0.1, 0.2]
SALT_COLORS = {0.1: BLUE, 0.2: ORANGE}


def print_table(names, results):
    """results[salt][name] -> dict de run_architecture. Tabla por nivel de salt."""
    for salt in SALT_LEVELS:
        print(f"\nsalt={salt}")
        print(f"{'arquitectura':22s} | {'err medio px':>14} | {'peor px':>12} | {'épocas':>8} | (<=1px)")
        print("-" * 80)
        for name in names:
            r = results[salt][name]
            print(f"{name:22s} | {r['mean_mean']:6.2f} ± {r['mean_std']:4.2f} | "
                  f"{r['worst_mean']:5.1f} ± {r['worst_std']:4.1f} | {r['epochs_mean']:8.0f} | "
                  f"{r['pass_mean']:4.1f}/32")


def save_figures(names, results, outdir, subtitle):
    # Barras agrupadas: error MEDIO de píxel vs limpio, una serie por nivel de salt.
    # Escala vertical compartida para que las dos series se comparen sin engaño.
    mean_series = [(f"salt={salt}",
                    [results[salt][n]["mean_mean"] for n in names],
                    [results[salt][n]["mean_std"] for n in names],
                    SALT_COLORS[salt]) for salt in SALT_LEVELS]
    top = max(m + s for _, ms, ss, _ in mean_series for m, s in zip(ms, ss))
    grouped_bar_study(
        names, mean_series,
        "Error medio de píxel (vs limpio, de 35)",
        "Arquitectura-DAE: error medio de reconstrucción",
        str(outdir / "arch_dae_mean_error.png"),
        subtitle=subtitle, rotate=20, ylim=(0, top * 1.15),
    )
    # Peor caso (descriptivo, sin línea de objetivo: el eje del DAE es el error medio).
    worst_series = [(f"salt={salt}",
                     [results[salt][n]["worst_mean"] for n in names],
                     [results[salt][n]["worst_std"] for n in names],
                     SALT_COLORS[salt]) for salt in SALT_LEVELS]
    grouped_bar_study(
        names, worst_series,
        "Peor caso (px mal, de 35)",
        "Arquitectura-DAE: peor caso de reconstrucción",
        str(outdir / "arch_dae_worst_error.png"),
        subtitle=subtitle, rotate=20,
    )
    # Convergencia: una figura por nivel de salt, con la misma escala vertical.
    curve_top = max(max(results[salt][n]["loss_curve"]) for salt in SALT_LEVELS for n in names)
    curve_bot = min(min(results[salt][n]["loss_curve"]) for salt in SALT_LEVELS for n in names)
    for salt in SALT_LEVELS:
        overlaid_curves(
            [(n, results[salt][n]["loss_curve"]) for n in names],
            "Época", "Train loss (BCE)", f"Arquitectura-DAE: convergencia (salt={salt})",
            str(outdir / f"arch_dae_convergence_salt{salt}.png"),
            subtitle=subtitle, logy=True, ylim=(curve_bot * 0.9, curve_top * 1.1),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estudio de arquitectura del DAE (bottleneck=2, salt 0.1/0.2).")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=3, help="semillas por arquitectura")
    parser.add_argument("--seed", type=int, default=42, help="seed base (las demás son base+1, ...)")
    parser.add_argument("--hidden-act", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--data", choices=["letters", "emoji"], default="letters")
    parser.add_argument("--output-dir", default="output/study/architecture-dae")
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    clean = load_fonts(args.data)

    names = list(ARCHITECTURES.keys())
    results = {salt: {} for salt in SALT_LEVELS}
    for salt in SALT_LEVELS:
        for name in names:
            print(f"[salt={salt}] Arquitectura {name:22s} — {args.seeds} seeds x {args.epochs} épocas "
                  f"(act={args.hidden_act})...")
            results[salt][name] = run_architecture(
                ARCHITECTURES[name], args.hidden_act, args.seed, args.seeds, args.epochs, clean,
                with_noise=True, salt=salt)

    print_table(names, results)

    # El eje x es la arquitectura; las series son los niveles de salt. El resto fijo.
    subtitle = study_subtitle(
        {"data": args.data, "seeds": args.seeds, "épocas": args.epochs},
        {"lr": LEARNING_RATE, "mode": TRAINING_MODE, "init": "he", "act": args.hidden_act,
         "bottleneck": 2, "opt": f"adam({ADAM_BETA1},{ADAM_BETA2})", "ε": EPSILON,
         "resample": "on", "target": "clean", "salt(series)": "0.1 / 0.2", "banda": "media ± σ"},
    )
    save_figures(names, results, outdir, subtitle)
    print(f"\nTodo guardado en: {outdir}/")


if __name__ == "__main__":
    main()
