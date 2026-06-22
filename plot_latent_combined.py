"""
Espacio latente del VAE con los generados superpuestos.

Dibuja tres capas sobre un VAE ya entrenado:
    nubes   (z ~ q(z|x), tenue)  = territorio explorado en el entrenamiento
    medias  (μ, un punto por patrón, etiquetado)
    generados (z ~ N(0,1) o latent_bounds, estrellas)

Entrada pública por main.py (este main(argv) es sólo detalle interno):
    python main.py plot latent --weights <ruta.npz>
    python main.py plot latent --weights <ruta.npz> --output output/vae/combined_kl0.png
"""
import argparse
from pathlib import Path

import numpy as np

from experiment import make_activations, resolve_labels
from font import load_fonts
from graphs.style import (
    BLACK, BLUE, FG, FG_DIM, ORANGE, RED, add_subtitle, dark_figure, dark_grid, save_dark,
)
from vae import build_vae_model
from weights_io import load_weights


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot del espacio latente del VAE con generados.")
    parser.add_argument("--weights", required=True, help="ruta al .npz del VAE entrenado")
    parser.add_argument("--data", default="emoji", choices=["emoji", "letters"])
    parser.add_argument("--n-generated", type=int, default=8)
    parser.add_argument("--cloud-samples", type=int, default=40, help="z por patrón (nubes)")
    parser.add_argument("--sampling", default="normal", choices=["normal", "bounds"],
                        help="normal: z~N(0,1)  ·  bounds: uniforme dentro del rango latente ocupado")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="ruta del .png; por defecto se deriva del nombre de los pesos")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    act = make_activations()
    clean = load_fonts(args.data)
    labels = resolve_labels(args.data)
    model = build_vae_model(act)
    load_weights(model, args.weights)

    means = model.get_latent_positions(clean)          # μ por patrón
    _, stds = model.get_latent_distributions(clean)    # σ por patrón

    # ---- generación ----
    if args.sampling == "normal":
        gen = np.random.standard_normal((args.n_generated, 2))
    else:  # bounds: uniforme dentro del rango ocupado por las medias
        lo, hi = means.min(axis=0), means.max(axis=0)
        gen = np.random.uniform(lo, hi, size=(args.n_generated, 2))

    # ---- distancia de generación (px al patrón real más cercano) ----
    cb = (clean >= 0.5).astype(int)
    dists = [int((cb != (model.decode(z) >= 0.5).astype(int)).sum(axis=1).min()) for z in gen]
    print(f"Distancia de generación (px al patrón real más cercano): "
          f"promedio {np.mean(dists):.2f}, por muestra {dists}")

    # ---- figura ----
    fig, ax = dark_figure(figsize=(8, 7))
    eps = np.random.standard_normal((len(clean), args.cloud_samples, 2))
    clouds = (means[:, None, :] + stds[:, None, :] * eps).reshape(-1, 2)
    ax.scatter(clouds[:, 0], clouds[:, 1], s=6, color=BLUE, alpha=0.15, linewidths=0,
               label="z ~ q(z|x) (explorado)")
    ax.scatter(means[:, 0], means[:, 1], s=40, color=RED, zorder=3, label="medias μ")
    ax.scatter(gen[:, 0], gen[:, 1], s=180, marker="*", color=ORANGE,
               edgecolors="black", linewidths=0.6, zorder=4, label=f"generados ({args.sampling})")
    for label, (x, y) in zip(labels, means):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 4), color=FG, fontsize=8)

    ax.set_title("VAE: espacio latente con generados")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    dark_grid(ax)
    leg = ax.legend(facecolor=BLACK, edgecolor=FG_DIM, loc="upper right", fontsize=8)
    for t in leg.get_texts():
        t.set_color(FG)
    add_subtitle(fig, f"data={args.data}  ·  seed={args.seed}  ·  sampling={args.sampling}  "
                      f"·  dist gen media={np.mean(dists):.2f}px")

    output = args.output or f"output/vae/combined_{Path(args.weights).parent.name}.png"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    print(f"Figura guardada en: {save_dark(fig, output)}")


if __name__ == "__main__":
    main()
