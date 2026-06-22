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

from evaluation import nearest_pattern_distance, pixel_errors_per_pattern
from sampling import set_seed
from experiment import (
    ADAM_BETA1, ADAM_BETA2, EPOCHS, EPSILON, LEARNING_RATE, TRAINING_MODE,
    make_activations, study_subtitle, train_once,
)
from font import load_fonts
from graphs import plot_latent_clouds_generated
from graphs.style import ORANGE, RED, add_subtitle, dark_figure, dark_grid, save_dark
from vae import VAE_ARCHITECTURE, build_vae_model

KL_LEVELS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
PANEL_LIMITS = (-3.5, 3.5)


def hp_line(args, kl=None):
    """Pie con todos los HP del barrido. kl=None lo marca '(barrido)' (figuras kl-vs-métrica);
    con un valor concreto queda fijo (figura de latente de ese nivel)."""
    return study_subtitle(
        {"data": "emoji", "seeds": args.seeds, "épocas": args.epochs},
        {"arch": "VAE", "kl": "(barrido)" if kl is None else kl, "lr": LEARNING_RATE,
         "mode": TRAINING_MODE, "init": "he", "bottleneck": 2,
         "opt": f"adam({ADAM_BETA1},{ADAM_BETA2})", "ε": EPSILON},
    )


def train_vae(kl_weight, seed, epochs, x):
    set_seed(seed)
    act = make_activations()
    model = build_vae_model(act, seed=seed)
    model.kl_weight = kl_weight
    train_once(model, x, x, VAE_ARCHITECTURE, "binary_cross_entropy + kl_divergence", epochs=epochs)
    return model


def recon_pixel_error(model, clean):
    recon = np.array([model.reconstruct(x) for x in clean])
    return float(pixel_errors_per_pattern(clean, recon).mean())


def generation_distance(model, clean, gen_z):
    generated = np.array([model.decode(z) for z in gen_z])
    return float(nearest_pattern_distance(generated, clean).mean())


def save_latent_figure(model, clean, gen_z, cloud_rng, kl, outdir, cloud_samples, subtitle=None):
    """Un espacio latente por nivel: nubes + medias + generados (estrellas)."""
    means, stds = model.get_latent_distributions(clean)
    eps = cloud_rng.standard_normal((len(clean), cloud_samples, 2))
    clouds = (means[:, None, :] + stds[:, None, :] * eps).reshape(-1, 2)

    plot_latent_clouds_generated(
        clouds, means, gen_z[:12],
        output_path=str(outdir / f"latent_kl-{kl}.png"),
        title=f"Espacio latente VAE — kl = {kl}",
        subtitle=subtitle, gen_label="generados z~N(0,1)",
        figsize=(7, 6), mean_size=30, gen_size=150, gen_edge_lw=0.5,
        panel_limits=PANEL_LIMITS,
    )


def save_metric_figure(kls, values, ylabel, title, color, filename, outdir, subtitle, stds=None):
    fig, ax = dark_figure(figsize=(8, 5))
    ax.plot(kls, values, marker="o", color=color)
    if stds is not None:
        # banda: media ± desvío entre seeds
        v, s = np.array(values), np.array(stds)
        ax.fill_between(kls, v - s, v + s, color=color, alpha=0.18, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("kl_weight")
    ax.set_ylabel(ylabel)
    dark_grid(ax)
    add_subtitle(fig, subtitle)
    save_dark(fig, str(outdir / filename))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Barrido amplio de kl_weight del VAE.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=42, help="seed base; las demás son base+1, ...")
    parser.add_argument("--seeds", type=int, default=3,
                        help="seeds por nivel de kl (banda de varianza media ± σ)")
    parser.add_argument("--gen-samples", type=int, default=300)
    parser.add_argument("--cloud-samples", type=int, default=40)
    parser.add_argument("--output-dir", default="output/study/kl")
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    clean = load_fonts("emoji")
    gen_z = np.random.default_rng(args.seed + 2000).standard_normal((args.gen_samples, 2))
    cloud_rng = np.random.default_rng(args.seed + 3000)

    # por cada nivel entrenamos varias seeds y guardamos media ± desvío de cada métrica
    rec_m, rec_s, gen_m, kl_m, kl_s, sig_m = [], [], [], [], [], []
    for kl in KL_LEVELS:
        print(f"Entrenando VAE kl={kl} ({args.seeds} seeds x {args.epochs} épocas, seed base={args.seed})...")
        r_runs, g_runs, k_runs, s_runs, first_model = [], [], [], [], None
        for j in range(args.seeds):
            model = train_vae(kl, args.seed + j, args.epochs, clean)
            first_model = first_model or model
            r_runs.append(recon_pixel_error(model, clean))
            g_runs.append(generation_distance(model, clean, gen_z))
            k_runs.append(model.kl_divergence(clean))
            _, stds = model.get_latent_distributions(clean)
            s_runs.append(float(stds.mean()))
        rec_m.append(np.mean(r_runs)); rec_s.append(np.std(r_runs))
        gen_m.append(np.mean(g_runs))
        kl_m.append(np.mean(k_runs)); kl_s.append(np.std(k_runs))
        sig_m.append(np.mean(s_runs))
        # latente de la primera seed (una figura por nivel, no una por seed)
        save_latent_figure(first_model, clean, gen_z, cloud_rng, kl, outdir, args.cloud_samples,
                           subtitle=hp_line(args, kl=kl))

    # Tabla (media ± σ entre seeds)
    print(f"\n{'kl':>7} | {'recon px':>13} | {'gen dist':>9} | {'KL':>13} | {'σ medio':>7}")
    print("-" * 62)
    for kl, r, rs, g, k, ks, s in zip(KL_LEVELS, rec_m, rec_s, gen_m, kl_m, kl_s, sig_m):
        print(f"{kl:>7} | {r:5.2f} ± {rs:4.2f} | {g:>9.2f} | {k:6.2f} ± {ks:4.2f} | {s:>7.3f}")

    sub = hp_line(args)  # kl en el eje x -> marcado (barrido); el resto de los HP, fijo
    save_metric_figure(KL_LEVELS, rec_m, "Error px reconstrucción (sobre 35)",
                       "Costo: reconstrucción vs kl", RED, "recon_error.png", outdir, sub, stds=rec_s)
    save_metric_figure(KL_LEVELS, kl_m, "KL divergence",
                       "KL vs kl (→0 = posterior collapse)", ORANGE, "kl_divergence.png", outdir, sub, stds=kl_s)

    print(f"\nTodo guardado en: {outdir}/")


if __name__ == "__main__":
    main()
