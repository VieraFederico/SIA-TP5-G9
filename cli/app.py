"""
cli/app.py — Interfaz de línea de comandos del TP5.

Un único entry point con todos los subcomandos. Regla: TODO entra por main.py;
los scripts sueltos conservan su main(argv) sólo como detalle interno, no como
interfaz pública.

    python main.py ae       [--data letters|emoji] [--noise/--no-noise] [--load P] [--save] [--no-viz]
    python main.py vae      [--data emoji|letters]  [--noise/--no-noise] [--load P] [--save] [--kl W]
    python main.py generate {ae|vae}  --weights P [flags del generador]
    python main.py plot     latent    --weights P [flags del plot]
    python main.py study    {architecture|hyperparams|denoising|kl|architecture-dae|hyperparams-dae} [flags del estudio]

Ejemplos:
    python main.py ae                      # autoencoder sobre letras, con ruido (DAE)
    python main.py ae --data letters --no-noise --save
    python main.py ae --load weights_letters.npz   # sin reentrenar
    python main.py vae --data emoji --save
    python main.py generate ae  --weights output/ae/.../weights.npz --plot
    python main.py generate vae --weights output/vae/.../weights.npz
    python main.py plot latent  --weights output/vae/.../weights.npz
    python main.py study architecture --epochs 300 --seeds 2
    python main.py study kl --seeds 3
"""
import argparse
import sys

from experiments.ae import run_ae
from experiments.vae import run_vae


def _add_common_args(parser: argparse.ArgumentParser, default_data: str) -> None:
    parser.add_argument(
        "--data", choices=["letters", "emoji"], default=default_data,
        help="dataset a usar (default: %(default)s)",
    )
    parser.add_argument(
        "--noise", action=argparse.BooleanOptionalAction, default=True,
        help="corromper la entrada con Salt & Pepper (--no-noise para desactivar)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed opcional para reproducibilidad",
    )
    parser.add_argument(
        "--load", dest="load_path", default=None, metavar="PATH",
        help="cargar pesos .npz y saltear el entrenamiento",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="guardar los pesos en output/<modelo>/weights/<hiperparams>/weights.npz",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="TP5 — Autoencoders (AE/DAE) y Variational Autoencoder (VAE).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ae = subparsers.add_parser("ae", help="autoencoder / denoising AE sobre font.h")
    _add_common_args(ae, default_data="letters")
    ae.add_argument(
        "--salt", type=float, default=None, metavar="P",
        help="probabilidad de ruido slt-n-pepper (default: experiment.SALT_P)",
    )
    ae.add_argument(
        "--resample", action=argparse.BooleanOptionalAction, default=True,
        help="re-samplear el ruido en cada época (denoising real, default). "
             "--no-resample = ruido FIJO: la red memoriza una corrupción puntual, no es denoising real",
    )
    ae.add_argument(
        "--no-viz", dest="show_viz", action="store_false", default=True,
        help="no mostrar las reconstrucciones en ASCII",
    )
    ae.add_argument(
        "--epochs", type=int, default=None, metavar="N",
        help="épocas de entrenamiento (default: config.epochs = 7500)",
    )

    vae = subparsers.add_parser("vae", help="variational autoencoder sobre los emojis")
    _add_common_args(vae, default_data="emoji")
    vae.add_argument(
        "--kl", type=float, default=None, metavar="W",
        help="peso del término KL (β-VAE); default: vae.KL_WEIGHT. kl=0 → solo reconstrucción",
    )
    vae.add_argument(
        "--epochs", type=int, default=None, metavar="N",
        help="épocas de entrenamiento (default: config.epochs = 7500)",
    )

    # Generación de muestras nuevas. El CLI sólo elige el modelo; los flags propios
    # (--weights, --sampling, ...) se pasan tal cual al script generador.
    generate = subparsers.add_parser("generate", help="generar muestras nuevas (AE a.4 / VAE 2.c)")
    generate.add_argument("kind", choices=["ae", "vae"], help="modelo del que generar")
    generate.add_argument(
        "rest", nargs=argparse.REMAINDER,
        help="flags del generador (ej: --weights P --sampling ...). Probá 'generate ae --help'.",
    )

    # Visualizaciones de un modelo entrenado (plot del latente con generados).
    plot = subparsers.add_parser("plot", help="visualizar un modelo entrenado")
    plot.add_argument("kind", choices=["latent"], help="qué graficar")
    plot.add_argument(
        "rest", nargs=argparse.REMAINDER,
        help="flags del plot (ej: --weights P --sampling ...). Probá 'plot latent --help'.",
    )

    # Estudios comparativos. El CLI sólo elige cuál correr; los flags propios de cada
    # estudio (--epochs, --seeds, --axis, ...) se pasan tal cual al script.
    study = subparsers.add_parser("study", help="estudios comparativos / grid search")
    study.add_argument(
        "kind",
        choices=["architecture", "hyperparams", "denoising", "kl",
                 "architecture-dae", "hyperparams-dae"],
        help="qué estudio correr",
    )
    study.add_argument(
        "rest", nargs=argparse.REMAINDER,
        help="flags del estudio (ej: --epochs 300 --seeds 2). Probá 'study architecture --help'.",
    )

    return parser


# Cada subcomando que delega es un script con su propio main(argv); el CLI sólo
# elige el módulo y le pasa los flags crudos. Nada de lógica acá.
_STUDIES = {
    "architecture": "experiments.grid_architecture",
    "hyperparams": "experiments.grid_hyperparams",
    "denoising": "experiments.sweep_denoising",
    "kl": "experiments.sweep_kl",
    "architecture-dae": "experiments.architecture_dae",
    "hyperparams-dae": "experiments.hyperparams_dae",
}
_GENERATORS = {"ae": "experiments.generate", "vae": "experiments.generate_vae"}
_PLOTS = {"latent": "experiments.plot_latent_combined"}


def _run_module(module_name: str, rest: list[str]) -> None:
    import importlib
    importlib.import_module(module_name).main(rest)


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # Sin argumentos → menú interactivo (la TUI arma los flags y vuelve a entrar acá).
    if not argv:
        from cli.tui import run_tui
        run_tui()
        return

    args = build_parser().parse_args(argv)

    if args.command == "ae":
        run_ae(
            datatype=args.data,
            with_noise=args.noise,
            salt_p=args.salt,
            resample_noise=args.resample,
            load_path=args.load_path,
            save=args.save,
            show_viz=args.show_viz,
            seed=args.seed,
            epochs=args.epochs,
        )
    elif args.command == "vae":
        run_vae(
            datatype=args.data,
            with_noise=args.noise,
            kl_weight=args.kl,
            load_path=args.load_path,
            save=args.save,
            seed=args.seed,
            epochs=args.epochs,
        )
    elif args.command == "generate":
        _run_module(_GENERATORS[args.kind], args.rest)
    elif args.command == "plot":
        _run_module(_PLOTS[args.kind], args.rest)
    elif args.command == "study":
        _run_module(_STUDIES[args.kind], args.rest)
