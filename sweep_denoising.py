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
from experiment import EPOCHS, make_activations, make_trainer
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Barrido de ruido del DAE.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"épocas (default {EPOCHS})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", choices=["letters", "emoji"], default="letters")
    parser.add_argument("--realizations", type=int, default=5,
                        help="realizaciones de ruido promediadas por punto")
    parser.add_argument("--output", default="output/dae/denoising_sweep.png")
    args = parser.parse_args(argv)

    clean = load_fonts(args.data)
    curves = {}

    for salt in TRAIN_LEVELS:
        name = "AE básico" if salt is None else f"DAE salt={salt}"
        print(f"Entrenando {name} ({args.epochs} épocas, seed={args.seed})...")
        model = train_model(salt, args.seed, args.epochs, clean)
        curves[salt] = eval_curve(model, clean, args.seed, args.realizations)

    # Tabla
    header = "train \\ test | " + " | ".join(f"{p:>5.2f}" for p in TEST_GRID)
    print("\n" + header)
    print("-" * len(header))
    for salt in TRAIN_LEVELS:
        name = "AE básico" if salt is None else f"DAE {salt:>4}"
        row = " | ".join(f"{v:>5.2f}" for v in curves[salt])
        print(f"{name:>12} | {row}")

    # Figura
    fig, ax = dark_figure(figsize=(9, 6))
    for salt in TRAIN_LEVELS:
        label = "AE básico (sin ruido)" if salt is None else f"DAE entrenado a salt={salt}"
        style = "--" if salt is None else "-"
        ax.plot(TEST_GRID, curves[salt], marker="o", linestyle=style,
                color=COLORS[salt], label=label)

    ax.set_title("Denoising: error de reconstrucción vs ruido de prueba")
    ax.set_xlabel("Nivel de ruido en la prueba (salt-and-pepper)")
    ax.set_ylabel("Error promedio (píxeles mal, sobre 35)")
    dark_grid(ax)
    leg = ax.legend(facecolor=BLACK, edgecolor=FG_DIM)
    for t in leg.get_texts():
        t.set_color(FG)
    add_subtitle(fig, f"data={args.data}  ·  seed={args.seed}  ·  epochs={args.epochs}  "
                      f"·  {args.realizations} realizaciones/punto  ·  ruido re-sampleado por época")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nFigura guardada en: {save_dark(fig, args.output)}")


if __name__ == "__main__":
    main()
