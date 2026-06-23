"""
Estudio de hiperparámetros del DENOISING autoencoder, con la arquitectura fija.

Mismos ejes que el estudio del AE (grid_hyperparams.AXES: learning_rate / modo / init /
optimizer / activation / epochs), pero entrenando en modo denoising: entrada ruidosa,
objetivo limpio, ruido re-sampleado por época (resample=on). Se barren DOS niveles de
ruido —salt 0.1 y 0.2— como DOS series en cada figura, para ver si el ranking de
hiperparámetros se mantiene cuando sube el ruido.

NO duplica el loop de grid: reusa sweep_axis / run_cell de grid_hyperparams.py (la misma
maquinaria que usa el estudio del AE), pasándole with_noise=True y el salt. La métrica es
el error de reconstrucción contra el patrón LIMPIO (pixel_errors_per_pattern vs clean),
nunca contra el ruidoso; NO se usa el criterio "objetivo <=1px" del AE puro.

    python3 main.py study hyperparams-dae --epochs 30 --seeds 2 --axis all
    python3 main.py study hyperparams-dae --axis lr
"""
import argparse
from pathlib import Path

from experiments.grid_hyperparams import (
    AXES, combo_dir, defaults_for, epoch_levels, run_cell, sweep_axis,
)
from experiments.experiment import study_subtitle
from src.utils.config import EPSILON
from src.data.font import load_fonts
from graphs.studies import grouped_bar_study
from graphs.style import BLUE, ORANGE

# Dos niveles de ruido = dos series en cada figura (no dos corridas sueltas).
SALT_LEVELS = [0.1, 0.2]
SALT_COLORS = {0.1: BLUE, 0.2: ORANGE}


def study_axis_dae(axis, args, clean, outdir):
    """Barre un eje en modo denoising para los dos salt y guarda una figura con dos series.
    Reusa sweep_axis/run_cell de grid_hyperparams (sin clonar el loop)."""
    title, levels = AXES[axis]
    if axis == "epochs":
        levels = epoch_levels(args.epochs)

    print(f"Eje {title}:")
    defaults = defaults_for(args)
    series = []
    for salt in SALT_LEVELS:
        print(f"  salt={salt}")
        # salt fijado por kwarg-default para que el lambda capture el valor del bucle.
        cell = lambda value, seed, salt=salt: run_cell(
            seed, clean, **{**defaults, axis: value}, with_noise=True, salt=salt)
        stats = sweep_axis(levels, cell, args.seed, args.seeds)
        series.append((f"salt={salt}", stats["mean_mean"], stats["mean_std"], SALT_COLORS[salt]))

    folder = combo_dir(outdir, axis, defaults, args)
    subtitle = study_subtitle(
        {"data": args.data, "seeds": args.seeds},
        {"arch": "default", **defaults, "bottleneck": 2, "ε": EPSILON,
         "resample": "on", "target": "clean", "salt(series)": "0.1 / 0.2", "banda": "media ± σ"},
        varied=axis,
    )
    # Escala vertical compartida entre las dos series para una comparación honesta.
    top = max(m + s for _, ms, ss, _ in series for m, s in zip(ms, ss))
    grouped_bar_study(
        [str(v) for v in levels], series,
        "Error medio de píxel (vs limpio, de 35)", f"Hiperparámetro-DAE: {title}",
        str(folder / f"hp_dae_{axis}.png"), subtitle=subtitle, ylim=(0, top * 1.15),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estudio de hiperparámetros del DAE (arquitectura fija, salt 0.1/0.2).")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=3, help="semillas por celda")
    parser.add_argument("--seed", type=int, default=42, help="seed base (las demás son base+1, ...)")
    parser.add_argument("--axis", choices=["all", *AXES], default="all")
    parser.add_argument("--data", choices=["letters", "emoji"], default="letters")
    parser.add_argument("--output-dir", default="output/study/hyperparams-dae")
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    clean = load_fonts(args.data)

    axes = list(AXES) if args.axis == "all" else [args.axis]
    for axis in axes:
        study_axis_dae(axis, args, clean, outdir)

    print(f"\nTodo guardado en: {outdir}/")


if __name__ == "__main__":
    main()
