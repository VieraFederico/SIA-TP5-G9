"""
Estudio de arquitectura del DENOISING autoencoder, con el cuello de botella fijo en 2.

Mismo conjunto de arquitecturas que el estudio del AE (grid_architecture.ARCHITECTURES) y
los mismos HP fijos, pero en modo denoising: entrada ruidosa, objetivo limpio, ruido
re-sampleado por época (resample=on). Por default se corre con salt 0.1, que es el nivel
canónico del DAE; --salts permite probar otros niveles manualmente. La métrica es el error
contra el patrón LIMPIO; NO se usa "objetivo <=1px".

Reusa la MISMA maquinaria que el AE: build_configs/add_arch_args/resolve_arch de
grid_architecture y el motor único study_engine.run_study. No duplica nada.

    python3 main.py study architecture-dae                       # completo × salt 0.1
    python3 main.py study architecture-dae --epochs 20 --seeds 2   # smoke
"""
import argparse
from pathlib import Path

from experiments.grid_architecture import add_arch_args, build_configs, resolve_arch
from experiments.experiment import hyperparams_slug
from experiments.study_engine import run_study
from experiments.study_selection import simplest_tiebreaker
from src.data.font import load_fonts

SALT_LEVELS = [0.1]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estudio de arquitectura DAE (bottleneck=2, salt 0.1).")
    add_arch_args(parser, with_salts=True)
    parser.add_argument("--output-dir", default="output/study/architecture-dae")
    args = parser.parse_args(argv)

    epochs, seeds = resolve_arch(args)
    salts = args.salts if args.salts else SALT_LEVELS
    clean = load_fonts(args.data)
    configs = build_configs(args.hidden_act)

    slug = hyperparams_slug({"data": args.data, "salt": "-".join(str(s) for s in salts),
                             "act": args.hidden_act,
                             "epochs": epochs, "seeds": seeds})
    outdir = Path(args.output_dir) / slug

    print(f"Estudio de arquitectura DAE: {len(configs)} arquitecturas × {len(salts)} salt "
          f"× {seeds} seeds × {epochs} épocas")
    run_study(
        study="architecture-dae", kind="architecture", configs=configs, clean=clean,
        seeds=seeds, base_seed=args.seed, with_noise=True, salts=salts, epochs=epochs,
        outdir=outdir, data=args.data, tiebreaker=simplest_tiebreaker,
    )


if __name__ == "__main__":
    main()
