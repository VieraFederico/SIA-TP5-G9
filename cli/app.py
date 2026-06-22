"""
cli/app.py — Interfaz de línea de comandos del TP5.

Un único entry point con dos subcomandos:

    python main.py ae  [--data letters|emoji] [--noise/--no-noise] [--load P] [--save] [--no-viz]
    python main.py vae [--data emoji|letters]  [--noise/--no-noise] [--load P] [--save]

Ejemplos:
    python main.py ae                      # autoencoder sobre letras, con ruido (DAE)
    python main.py ae --data letters --no-noise --save
    python main.py ae --load weights_letters.npz   # sin reentrenar
    python main.py vae --data emoji --save
"""
import argparse

from ae import run_ae
from vae import run_vae


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
        help="re-samplear el ruido en cada época (--no-resample para ruido fijo)",
    )
    ae.add_argument(
        "--no-viz", dest="show_viz", action="store_false", default=True,
        help="no mostrar las reconstrucciones en ASCII",
    )

    vae = subparsers.add_parser("vae", help="variational autoencoder sobre los emojis")
    _add_common_args(vae, default_data="emoji")
    vae.add_argument(
        "--kl", type=float, default=None, metavar="W",
        help="peso del término KL (β-VAE); default: vae.KL_WEIGHT. kl=0 → solo reconstrucción",
    )

    return parser


def main(argv=None) -> None:
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
        )
    elif args.command == "vae":
        run_vae(
            datatype=args.data,
            with_noise=args.noise,
            kl_weight=args.kl,
            load_path=args.load_path,
            save=args.save,
            seed=args.seed,
        )
