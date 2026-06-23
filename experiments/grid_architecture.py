"""
Estudio de arquitectura del autoencoder, con el cuello de botella fijo en 2.

Barre la lista de arquitecturas de encoder (el decoder siempre las espeja) con los
hiperparámetros FIJOS en su default y DECLARADOS (lr=1e-3, mode=batch, init=he, act=relu,
opt=adam). NO se cruza con hiperparámetros: es un estudio aparte.

Sólo arma la lista de combinaciones (una por arquitectura) y delega TODO en el motor único
study_engine.run_study (mismo que usan los otros tres estudios): entrenar → CSV
crudo+resumen → selección única (study_selection; desempate por arquitectura más simple) →
tabla/barras/curvas de loss + convergencia. with_noise=off (AE puro).

    python3 main.py study architecture                       # corrida completa (2000 ep)
    python3 main.py study architecture --epochs 20 --seeds 2   # smoke
"""
import argparse
from pathlib import Path

from experiments.experiment import hyperparams_slug
from experiments.study_engine import run_study
from experiments.study_selection import simplest_tiebreaker
from src.utils.config import LEARNING_RATE
from src.data.font import load_fonts

# Cada arquitectura = anchos del encoder, de la entrada (35) a la capa previa al bottleneck.
ARCHITECTURES = {
    "directo 35-2":       [35],
    "shallow 35-16-2":    [35, 16],
    "medium 35-20-10-2":  [35, 20, 10],
    "deep 35-30-20-10-2": [35, 30, 20, 10],
    "default 35..4-2":    [35, 30, 25, 20, 16, 8, 4],
}
# Hiperparámetros fijos (declarados) con los que se barre la arquitectura.
FIXED = {"lr": LEARNING_RATE, "mode": "batch", "init": "he", "act": "relu", "opt": "adam"}


def build_configs(hidden_act=FIXED["act"]):
    """Una combinación por arquitectura. combo_id estable por orden de la lista."""
    configs = []
    for i, (name, widths) in enumerate(ARCHITECTURES.items(), 1):
        configs.append({
            "combo_id": i, "label": name, "widths": list(widths), "architecture": name,
            "opt": FIXED["opt"], "lr": FIXED["lr"], "init": FIXED["init"],
            "mode": FIXED["mode"], "act": hidden_act,
        })
    return configs


def add_arch_args(parser, *, with_salts=False):
    parser.add_argument("--epochs", type=int, default=None, help="presupuesto del grid (default 2000; smoke 20)")
    parser.add_argument("--seeds", type=int, default=None, help="seeds por arquitectura (default 3; smoke 2)")
    parser.add_argument("--seed", type=int, default=42, help="seed base (las demás son base+1, ...)")
    parser.add_argument("--hidden-act", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--data", choices=["letters", "emoji"], default="letters")
    parser.add_argument("--smoke", action="store_true", help="defaults chicos para validar el pipeline")
    if with_salts:
        from experiments.grid_hyperparams import _floats
        parser.add_argument("--salts", type=_floats, default=None, help="niveles de salt (default 0.1,0.2)")


def resolve_arch(args):
    epochs = args.epochs if args.epochs is not None else (20 if args.smoke else 2000)
    seeds = args.seeds if args.seeds is not None else (2 if args.smoke else 3)
    return epochs, seeds


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estudio de arquitectura AE (bottleneck=2).")
    add_arch_args(parser)
    parser.add_argument("--output-dir", default="output/study/architecture")
    args = parser.parse_args(argv)

    epochs, seeds = resolve_arch(args)
    clean = load_fonts(args.data)
    configs = build_configs(args.hidden_act)

    slug = hyperparams_slug({"data": args.data, "act": args.hidden_act,
                             "epochs": epochs, "seeds": seeds})
    outdir = Path(args.output_dir) / slug

    print(f"Estudio de arquitectura AE: {len(configs)} arquitecturas × {seeds} seeds × {epochs} épocas")
    run_study(
        study="architecture", kind="architecture", configs=configs, clean=clean,
        seeds=seeds, base_seed=args.seed, with_noise=False, salts=[None], epochs=epochs,
        outdir=outdir, data=args.data, tiebreaker=simplest_tiebreaker,
    )


if __name__ == "__main__":
    main()
