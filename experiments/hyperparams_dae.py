"""
Estudio de hiperparámetros del DENOISING autoencoder — GRID CRUZADO real × salt.

Mismo grid que el AE (opt × lr × init, study_engine), pero en modo denoising: entrada
ruidosa, objetivo limpio, ruido re-sampleado por época (resample=on). Se corre el grid
COMPLETO a DOS niveles de ruido —salt 0.1 y 0.2— → 30 combos × 2 salt = 60 celdas. La
métrica es el error de reconstrucción contra el patrón LIMPIO; NO se usa "objetivo <=1px".

Reusa la MISMA maquinaria que el AE: build_configs/add_grid_args/resolve_grid de
grid_hyperparams y el motor único study_engine.run_study. No duplica nada.

    python3 main.py study hyperparams-dae                              # grid completo × salt 0.1/0.2
    python3 main.py study hyperparams-dae --smoke
    python3 main.py study hyperparams-dae --epochs 20 --seeds 2 --salts 0.1,0.2 --lrs 0.001,0.01 --inits he
"""
import argparse
from pathlib import Path

from experiments.grid_hyperparams import add_grid_args, build_configs, resolve_grid, FIXED_ACT, FIXED_MODE
from experiments.experiment import hyperparams_slug
from experiments.study_engine import run_study
from experiments.study_selection import standard_tiebreaker
from src.data.font import load_fonts

SALT_LEVELS = [0.1, 0.2]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estudio de hiperparámetros DAE (grid opt×lr×init × salt 0.1/0.2).")
    add_grid_args(parser, with_salts=True)
    parser.add_argument("--output-dir", default="output/study/hyperparams-dae")
    args = parser.parse_args(argv)

    epochs, seeds, opts, lrs, inits = resolve_grid(args)
    salts = args.salts if args.salts else SALT_LEVELS
    clean = load_fonts(args.data)
    configs = build_configs(opts, lrs, inits)

    slug = hyperparams_slug({"data": args.data, "mode": FIXED_MODE, "act": FIXED_ACT,
                             "epochs": epochs, "seeds": seeds})
    outdir = Path(args.output_dir) / slug

    print(f"Grid hiperparámetros DAE: {len(configs)} combinaciones × {len(salts)} salt "
          f"× {seeds} seeds × {epochs} épocas = {len(configs) * len(salts) * seeds} celdas")
    run_study(
        study="hyperparams-dae", kind="hyperparams", configs=configs, clean=clean,
        seeds=seeds, base_seed=args.seed, with_noise=True, salts=salts, epochs=epochs,
        outdir=outdir, data=args.data, tiebreaker=standard_tiebreaker,
    )


if __name__ == "__main__":
    main()
