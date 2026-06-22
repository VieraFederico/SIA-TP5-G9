"""
Estudio de hiperparámetros del autoencoder, con la arquitectura fija (build_ae_model).

Barremos un eje por vez (los otros dos quedan en su default de config.json):

    learning_rate     paso del Adam
    training_mode     online / batch / minibatch
    init              inicialización de pesos: He / Xavier / normal chico

Por celda medimos cuántos de los 32 se reconstruyen con <=1 px. Cada celda corre con
varias semillas y la figura muestra media ± desvío. El eje kl (β del VAE) no está acá:
lo cubre sweep_kl.py.

    python3 grid_hyperparams.py --epochs 300 --seeds 2   # rápido para probar
    python3 grid_hyperparams.py --axis lr                # un solo eje
"""
import argparse
from pathlib import Path

import numpy as np

from ae import AE_ARCHITECTURE, build_ae_model
from evaluation import pixel_error_counts
from experiment import (
    ADAM_BETA1, ADAM_BETA2, EPSILON, LEARNING_RATE, TRAINING_MODE,
    make_activations, make_trainer, study_subtitle,
)
from font import load_fonts
from graphs.studies import bar_study
from optimizer.adam import AdamOptimizer

LR_LEVELS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
MODE_LEVELS = ["online", "batch", "minibatch"]
INIT_LEVELS = ["he", "xavier", "normal"]


def apply_init(model, scheme, seed):
    """Reinicia los pesos con el esquema elegido (lo hacemos acá porque NeuronLayer no
    deja elegir el init por capa):

        he      ~ N(0,1) * sqrt(2/n_in)   (el del repo, va bien con ReLU)
        xavier  ~ N(0,1) * sqrt(1/n_in)
        normal  ~ N(0,1) * 0.1
    """
    rng = np.random.default_rng(seed)
    for mlp in (model.encoder, model.latent_space, model.decoder):
        for layer in mlp.layers:
            n_in, n_out = layer.n_inputs, layer.n_neurons
            if scheme == "he":
                scale = np.sqrt(2.0 / n_in)
            elif scheme == "xavier":
                scale = np.sqrt(1.0 / n_in)
            else:  # normal
                scale = 0.1
            layer.weights = rng.standard_normal((n_in, n_out)) * scale
            layer.bias = np.zeros(n_out)


def run_seed(lr, mode, init, seed, epochs, clean):
    """El AE fijo con (lr, mode, init) y una seed. Devuelve (pasan, peor_px)."""
    np.random.seed(seed)

    model = build_ae_model(make_activations(), seed=seed)
    apply_init(model, init, seed)

    trainer, _ = make_trainer(AE_ARCHITECTURE, "binary_cross_entropy")
    trainer.cfg.epochs = epochs
    trainer.cfg.training_mode = mode
    trainer.optimizer = AdamOptimizer(lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2)
    trainer.fit(model=model, X_train=clean, zeta_train=clean, X_val=None, zeta_val=None)

    passed, worst, _ = pixel_error_counts(model, clean)
    return passed, worst


def sweep_axis(values, cell, base_seed, n_seeds):
    """Para cada valor del eje corre n_seeds celdas y resume media ± desvío de los que pasan.
    cell(value, seed) -> (pasan, peor_px)."""
    pass_mean, pass_std = [], []
    for v in values:
        runs = [cell(v, base_seed + k) for k in range(n_seeds)]
        passed = np.array([p for p, _ in runs], dtype=float)
        worst = np.array([w for _, w in runs], dtype=float)

        pass_mean.append(passed.mean())
        pass_std.append(passed.std())
        print(f"    {str(v):>10}: pasan {passed.mean():4.1f}±{passed.std():3.1f}/32  "
              f"peor {worst.mean():4.1f}±{worst.std():3.1f} px")

    return pass_mean, pass_std


def axis_subtitle(varied, args):
    """Pie con TODOS los hiperparámetros; el eje que se barre queda marcado como (barrido)
    y los otros dos muestran su valor fijo, para poder comparar las tres figuras entre sí."""
    return study_subtitle(
        {"data": args.data, "seeds": args.seeds, "épocas": args.epochs},
        {"arch": "default", "lr": LEARNING_RATE, "mode": TRAINING_MODE, "init": "he", "act": "relu",
         "bottleneck": 2, "opt": f"adam({ADAM_BETA1},{ADAM_BETA2})", "ε": EPSILON},
        varied=varied,
    )


def study_axis(title, varied, values, cell, outdir, filename, args):
    """Barre un eje, lo imprime y guarda la figura de patrones-que-pasan."""
    print(f"Eje {title}:")
    pass_mean, pass_std = sweep_axis(values, cell, args.seed, args.seeds)
    bar_study(
        [str(v) for v in values], pass_mean, pass_std,
        "Patrones con <= 1 px (de 32)", f"Hiperparámetro: {title}",
        str(outdir / filename), subtitle=axis_subtitle(varied, args),
        target=32, target_label="objetivo: 32/32",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estudio de hiperparámetros (arquitectura fija).")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=3, help="semillas por celda")
    parser.add_argument("--seed", type=int, default=42, help="seed base (las demás son base+1, ...)")
    parser.add_argument("--axis", choices=["lr", "mode", "init", "all"], default="all")
    parser.add_argument("--data", choices=["letters", "emoji"], default="letters")
    parser.add_argument("--output-dir", default="output/ae/grid_hyperparams")
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    clean = load_fonts(args.data)

    if args.axis in ("lr", "all"):
        study_axis("learning_rate", "lr", LR_LEVELS,
                   lambda v, s: run_seed(v, TRAINING_MODE, "he", s, args.epochs, clean),
                   outdir, "hp_learning_rate.png", args)

    if args.axis in ("mode", "all"):
        study_axis("training_mode", "mode", MODE_LEVELS,
                   lambda v, s: run_seed(LEARNING_RATE, v, "he", s, args.epochs, clean),
                   outdir, "hp_training_mode.png", args)

    if args.axis in ("init", "all"):
        study_axis("init", "init", INIT_LEVELS,
                   lambda v, s: run_seed(LEARNING_RATE, TRAINING_MODE, v, s, args.epochs, clean),
                   outdir, "hp_init.png", args)

    print(f"\nTodo guardado en: {outdir}/")


if __name__ == "__main__":
    main()
