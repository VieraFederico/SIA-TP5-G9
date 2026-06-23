"""
Estudio de hiperparámetros del autoencoder — GRID CRUZADO real (no barridos 1D).

Producto cartesiano opt × lr × init (itertools.product → 30 combinaciones por defecto),
con la arquitectura fija (build_ae_model default) y los ejes mode=batch, act=relu FIJOS
y declarados (no entran al grid; epochs tampoco es un eje: es el presupuesto del grid).

Sólo arma la lista de combinaciones y delega TODO en study_engine.run_study (motor único
compartido con los otros tres estudios): entrenar → CSV crudo+resumen → selección única
(study_selection) → tablas/barras/curvas de loss. with_noise=off (AE puro).

    python3 main.py study hyperparams                                  # grid completo (2000 ep)
    python3 main.py study hyperparams --smoke                          # smoke rápido
    python3 main.py study hyperparams --epochs 20 --seeds 2 --opts adam,gd --lrs 0.001,0.01 --inits he,xavier
"""
import argparse
import itertools
from pathlib import Path

from experiments.ae import AE_ENCODER_WIDTHS
from experiments.experiment import hyperparams_slug
from experiments.study_engine import run_study
from experiments.study_selection import standard_tiebreaker
from src.data.font import load_fonts

# Ejes del grid cruzado (DEFINITIVO). Son flags para poder achicar en smoke.
OPTS = ["adam", "gd"]
LRS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
INITS = ["he", "xavier", "normal"]
# Fijos FUERA del grid (declarados en subtítulo y CSV).
FIXED_MODE = "batch"
FIXED_ACT = "relu"


def build_configs(opts, lrs, inits, *, widths=AE_ENCODER_WIDTHS, mode=FIXED_MODE, act=FIXED_ACT):
    """Producto cartesiano opt×lr×init. combo_id estable por orden de generación."""
    configs = []
    for i, (opt, lr, init) in enumerate(itertools.product(opts, lrs, inits), 1):
        configs.append({
            "combo_id": i, "label": f"{opt}|lr{lr:g}|{init}", "widths": list(widths),
            "opt": opt, "lr": lr, "init": init, "mode": mode, "act": act,
            "architecture": "default",
        })
    return configs


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _strs(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def add_grid_args(parser, *, with_salts=False):
    """Flags comunes a los dos estudios de hiperparámetros (AE y DAE)."""
    parser.add_argument("--epochs", type=int, default=None, help="presupuesto del grid (default 2000; smoke 20)")
    parser.add_argument("--seeds", type=int, default=None, help="seeds por celda (default 3; smoke 2)")
    parser.add_argument("--seed", type=int, default=42, help="seed base (las demás son base+1, ...)")
    parser.add_argument("--data", choices=["letters", "emoji"], default="letters")
    parser.add_argument("--opts", type=_strs, default=None, help="optimizers, coma-separados (default adam,gd)")
    parser.add_argument("--lrs", type=_floats, default=None, help="learning rates, coma-separados")
    parser.add_argument("--inits", type=_strs, default=None, help="inits, coma-separados (he,xavier,normal)")
    parser.add_argument("--smoke", action="store_true", help="defaults chicos para validar el pipeline")
    if with_salts:
        parser.add_argument("--salts", type=_floats, default=None, help="niveles de salt (default 0.1,0.2)")


def resolve_grid(args):
    """Aplica defaults full/smoke a epochs/seeds/ejes (smoke recorta lo no provisto)."""
    epochs = args.epochs if args.epochs is not None else (20 if args.smoke else 2000)
    seeds = args.seeds if args.seeds is not None else (2 if args.smoke else 3)
    opts = args.opts if args.opts else (["adam", "gd"] if not args.smoke else ["adam", "gd"])
    lrs = args.lrs if args.lrs else (LRS if not args.smoke else [1e-3, 1e-2])
    inits = args.inits if args.inits else (INITS if not args.smoke else ["he"])
    return epochs, seeds, opts, lrs, inits


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estudio de hiperparámetros AE (grid opt×lr×init).")
    add_grid_args(parser)
    parser.add_argument("--output-dir", default="output/study/hyperparams")
    args = parser.parse_args(argv)

    epochs, seeds, opts, lrs, inits = resolve_grid(args)
    clean = load_fonts(args.data)
    configs = build_configs(opts, lrs, inits)

    slug = hyperparams_slug({"data": args.data, "mode": FIXED_MODE, "act": FIXED_ACT,
                             "epochs": epochs, "seeds": seeds})
    outdir = Path(args.output_dir) / slug

    print(f"Grid hiperparámetros AE: {len(opts)}×{len(lrs)}×{len(inits)} = {len(configs)} "
          f"combinaciones × {seeds} seeds × {epochs} épocas")
    run_study(
        study="hyperparams", kind="hyperparams", configs=configs, clean=clean,
        seeds=seeds, base_seed=args.seed, with_noise=False, salts=[None], epochs=epochs,
        outdir=outdir, data=args.data, tiebreaker=standard_tiebreaker,
    )


if __name__ == "__main__":
    main()
