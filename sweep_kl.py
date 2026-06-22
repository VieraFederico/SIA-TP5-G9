"""
Barrido del peso del KL en el VAE (β-VAE) sobre emojis.

Entrena, con SEMILLA FIJA, el VAE a varios kl_weight y mide trade-off:

    - RECONSTRUCCIÓN  (costo)     : error en px al reconstruir los 32 emojis.
    - GENERACIÓN      (beneficio) : z ~ N(0,1) -> decode

Guarda TODO bajo output/vae/kl_sweep/ en figuras SEPARADAS:
    latent_kl-<kl>.png      -> un espacio latente por nivel (nubes+medias+generados)
    recon_error.png         -> reconstrucción vs kl
    kl_divergence.png       -> KL vs kl

Uso:
    python sweep_kl.py
"""
import argparse
from pathlib import Path

import numpy as np

from experiment import EPOCHS, make_activations, make_trainer
from font import load_fonts
from graphs.style import (
    BLACK, BLUE, FG, FG_DIM, ORANGE, RED, add_subtitle, dark_figure, dark_grid, save_dark,
)
from vae import VAE_ARCHITECTURE, build_vae_model

KL_LEVELS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
PANEL_LIMITS = (-3.5, 3.5)


def train_vae(kl_weight, seed, epochs, x):
    np.random.seed(seed)
    act = make_activations()
    model = build_vae_model(act, seed=seed)
    model.kl_weight = kl_weight
    trainer, _ = make_trainer(VAE_ARCHITECTURE, "binary_cross_entropy + kl_divergence")
    trainer.cfg.epochs = epochs
    trainer.fit(model=model, X_train=x, zeta_train=x, X_val=None, zeta_val=None)
    return model


def recon_pixel_error(model, clean):
    recon = np.array([model.reconstruct(x) for x in clean])
    cb = (clean >= 0.5).astype(int)
    rb = (recon >= 0.5).astype(int)
    return float((cb != rb).sum(axis=1).mean())


def generation_distance(model, clean, gen_z):
    cb = (clean >= 0.5).astype(int)
    dists = []
    for z in gen_z:
        out = (model.decode(z) >= 0.5).astype(int)
        dists.append((cb != out).sum(axis=1).min())
    return float(np.mean(dists))


def save_latent_figure(model, clean, gen_z, cloud_rng, kl, outdir, cloud_samples):
    """Un espacio latente por nivel: nubes + medias + generados (estrellas)."""
    means, stds = model.get_latent_distributions(clean)
    eps = cloud_rng.standard_normal((len(clean), cloud_samples, 2))
    clouds = (means[:, None, :] + stds[:, None, :] * eps).reshape(-1, 2)

    fig, ax = dark_figure(figsize=(7, 6))
    ax.scatter(clouds[:, 0], clouds[:, 1], s=6, color=BLUE, alpha=0.15, linewidths=0,
               label="z ~ q(z|x) (explorado)")
    ax.scatter(means[:, 0], means[:, 1], s=30, color=RED, zorder=3, label="medias μ")
    ax.scatter(gen_z[:12, 0], gen_z[:12, 1], s=150, marker="*", color=ORANGE,
               edgecolors="black", linewidths=0.5, zorder=4, label="generados z~N(0,1)")
    ax.set_title(f"Espacio latente VAE — kl = {kl}")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_xlim(*PANEL_LIMITS)
    ax.set_ylim(*PANEL_LIMITS)
    dark_grid(ax)
    leg = ax.legend(facecolor=BLACK, edgecolor=FG_DIM, loc="upper right", fontsize=8)
    for t in leg.get_texts():
        t.set_color(FG)
    save_dark(fig, str(outdir / f"latent_kl-{kl}.png"))


def save_metric_figure(kls, values, ylabel, title, color, filename, outdir, subtitle):
    fig, ax = dark_figure(figsize=(8, 5))
    ax.plot(kls, values, marker="o", color=color)
    ax.set_title(title)
    ax.set_xlabel("kl_weight")
    ax.set_ylabel(ylabel)
    dark_grid(ax)
    add_subtitle(fig, subtitle)
    save_dark(fig, str(outdir / filename))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Barrido amplio de kl_weight del VAE.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-samples", type=int, default=300)
    parser.add_argument("--cloud-samples", type=int, default=40)
    parser.add_argument("--output-dir", default="output/vae/kl_sweep")
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    clean = load_fonts("emoji")
    gen_z = np.random.default_rng(args.seed + 2000).standard_normal((args.gen_samples, 2))
    cloud_rng = np.random.default_rng(args.seed + 3000)

    recs, gens, kls_div, sigmas = [], [], [], []
    for kl in KL_LEVELS:
        print(f"Entrenando VAE kl={kl} ({args.epochs} épocas, seed={args.seed})...")
        model = train_vae(kl, args.seed, args.epochs, clean)
        recs.append(recon_pixel_error(model, clean))
        gens.append(generation_distance(model, clean, gen_z))
        kls_div.append(model.kl_divergence(clean))
        _, stds = model.get_latent_distributions(clean)
        sigmas.append(float(stds.mean()))
        save_latent_figure(model, clean, gen_z, cloud_rng, kl, outdir, args.cloud_samples)

    # Tabla
    print(f"\n{'kl':>7} | {'recon px':>9} | {'gen dist':>9} | {'KL':>7} | {'σ medio':>7}")
    print("-" * 50)
    for kl, r, g, k, s in zip(KL_LEVELS, recs, gens, kls_div, sigmas):
        print(f"{kl:>7} | {r:>9.2f} | {g:>9.2f} | {k:>7.2f} | {s:>7.3f}")

    sub = f"VAE emojis  ·  seed={args.seed}  ·  epochs={args.epochs}"
    save_metric_figure(KL_LEVELS, recs, "Error px reconstrucción (sobre 35)",
                       "Costo: reconstrucción vs kl", RED, "recon_error.png", outdir, sub)
    save_metric_figure(KL_LEVELS, kls_div, "KL divergence",
                       "KL vs kl (→0 = posterior collapse)", ORANGE, "kl_divergence.png", outdir, sub)

    print(f"\nTodo guardado en: {outdir}/")


if __name__ == "__main__":
    main()
