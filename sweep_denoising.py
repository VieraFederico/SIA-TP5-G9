"""
Barrido de ruido del (denoising) autoencoder.

Reentrena, con SEMILLA FIJA, varios modelos:
    - AE básico (sin ruido) -> baseline, no aprende a limpiar
    - DAE entrenado a salt=0.05 / 0.10 / 0.20 (ruido RE_SAMPLEADO por época)

y evalúa cada uno sobre ruido NUEVO a intensidad creciente. Genera UNA figura:
    eje X = nivel de ruido en la prueba
    eje Y = error promedio de reconstrucción (pixeles_errados / 35)
    una curva por modelo entrenado.
"""
import argparse
from pathlib import Path

import numpy as np

from ae import build_ae_model, AE_ARCHITECTURE
from experiment import (
    ADAM_BETA1, ADAM_BETA2, EPOCHS, EPSILON, LEARNING_RATE, TRAINING_MODE,
    make_activations, make_trainer, study_subtitle,
)
from font import load_fonts
from graphs.style import (
    BLACK, BLUE, FG, FG_DIM, ORANGE, RED,
    add_subtitle, dark_figure, dark_grid, save_dark,
)
from noise.salt_n_pepper import SaltNPepperNoise

# Niveles de entrenamiento: None = AE básico (sin ruido); el resto = DAE.
TRAIN_LEVELS = [None, 0.05, 0.10, 0.20]
TEST_GRID = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
COLORS = {None: FG, 0.05: BLUE, 0.10: ORANGE, 0.20: RED}


def train_model(salt, seed, epochs, clean):
    """Entrena un AE (salt=None) o DAE (salt>0, ruido re-sampleado por época)."""
    np.random.seed(seed)
    act = make_activations()
    model = build_ae_model(act, seed=seed)
    trainer, _ = make_trainer(AE_ARCHITECTURE, "binary_cross_entropy")
    trainer.cfg.epochs = epochs

    if salt is None:
        x_input, noise_fn = clean.copy(), None
    else:
        x_input = SaltNPepperNoise(salt).add_noise(clean.copy())
        noise_fn = lambda: SaltNPepperNoise(salt).add_noise(clean.copy())

    trainer.fit(model=model, X_train=x_input, zeta_train=clean,
                X_val=None, zeta_val=None, noise_fn=noise_fn)
    return model


def eval_curve(model, clean, seed, n_real=5):
    """Error promedio en píxeles sobre ruido NUEVO, para cada nivel de TEST_GRID."""
    np.random.seed(seed + 1000)  # stream de evaluación, igual para todos los modelos
    cb = (clean >= 0.5).astype(int)
    curve = []
    for p in TEST_GRID:
        errs = []
        for _ in range(n_real):
            noisy = clean.copy() if p == 0.0 else SaltNPepperNoise(p).add_noise(clean.copy())
            recon = np.array([model.forward(x) for x in noisy])
            rb = (recon >= 0.5).astype(int)
            errs.append((cb != rb).sum(axis=1).mean())
        curve.append(float(np.mean(errs)))
    return curve


def curves_over_seeds(salt, base_seed, n_seeds, epochs, clean, realizations):
    """Entrena n_seeds modelos para un nivel y devuelve la matriz (n_seeds, len(TEST_GRID))
    de curvas error-vs-ruido, para después graficar media ± desvío entre seeds."""
    rows = []
    for k in range(n_seeds):
        seed = base_seed + k
        model = train_model(salt, seed, epochs, clean)
        rows.append(eval_curve(model, clean, seed, realizations))
    return np.array(rows)   # (n_seeds, len(TEST_GRID))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Barrido de ruido del DAE.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"épocas (default {EPOCHS})")
    parser.add_argument("--seed", type=int, default=42, help="seed base; las demás son base+1, ...")
    parser.add_argument("--seeds", type=int, default=3,
                        help="cantidad de seeds por nivel (banda de varianza media ± σ)")
    parser.add_argument("--data", choices=["letters", "emoji"], default="letters")
    parser.add_argument("--realizations", type=int, default=5,
                        help="realizaciones de ruido promediadas por punto")
    parser.add_argument("--output", default="output/study/denoising/denoising_sweep.png")
    args = parser.parse_args(argv)

    clean = load_fonts(args.data)
    means, stds = {}, {}

    for salt in TRAIN_LEVELS:
        name = "AE básico" if salt is None else f"DAE salt={salt}"
        print(f"Entrenando {name} ({args.seeds} seeds x {args.epochs} épocas, seed base={args.seed})...")
        mat = curves_over_seeds(salt, args.seed, args.seeds, args.epochs, clean, args.realizations)
        means[salt] = mat.mean(axis=0)   # media entre seeds
        stds[salt] = mat.std(axis=0)     # desvío entre seeds (ancho de la banda)

    # Tabla (media entre seeds)
    header = "train \\ test | " + " | ".join(f"{p:>5.2f}" for p in TEST_GRID)
    print("\n" + header)
    print("-" * len(header))
    for salt in TRAIN_LEVELS:
        name = "AE básico" if salt is None else f"DAE {salt:>4}"
        row = " | ".join(f"{v:>5.2f}" for v in means[salt])
        print(f"{name:>12} | {row}")

    # Figura: línea = media entre seeds; banda sombreada = media ± desvío entre seeds
    fig, ax = dark_figure(figsize=(9, 6))
    for salt in TRAIN_LEVELS:
        label = "AE básico (sin ruido)" if salt is None else f"DAE entrenado a salt={salt}"
        style = "--" if salt is None else "-"
        ax.plot(TEST_GRID, means[salt], marker="o", linestyle=style,
                color=COLORS[salt], label=label)
        ax.fill_between(TEST_GRID, means[salt] - stds[salt], means[salt] + stds[salt],
                        color=COLORS[salt], alpha=0.18, linewidth=0)   # banda de varianza

    ax.set_title("Denoising: error de reconstrucción vs ruido de prueba")
    ax.set_xlabel("Nivel de ruido en la prueba (salt-and-pepper)")
    ax.set_ylabel("Error promedio (pixeles_malos / 35)")
    dark_grid(ax)
    leg = ax.legend(facecolor=BLACK, edgecolor=FG_DIM)
    for t in leg.get_texts():
        t.set_color(FG)
    # las curvas (legend) son los niveles de ruido de entrenamiento; el resto de los HP
    # es igual para todos los modelos y va en el pie para poder comparar.
    add_subtitle(fig, study_subtitle(
        {"data": args.data, "seeds": args.seeds, "épocas": args.epochs},
        {"arch": "default", "lr": LEARNING_RATE, "mode": TRAINING_MODE, "init": "he",
         "bottleneck": 2, "opt": f"adam({ADAM_BETA1},{ADAM_BETA2})", "ε": EPSILON,
         "realizaciones/punto": args.realizations, "ruido": "re-sampleado/época",
         "banda": "media ± σ"},
    ))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nFigura guardada en: {save_dark(fig, args.output)}")


if __name__ == "__main__":
    main()
