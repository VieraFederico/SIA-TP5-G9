#!/usr/bin/env python3
"""
generate_vae.py — Generación de patrones nuevos desde un VAE entrenado (2.c).

Modos de muestreo (--sampling):
    prior      z ~ N(0, I) y decode  → generación canónica "desde cero" (DEFAULT, §4.5/§4-F).
    posterior  z ~ N(μ_i, σ_i·scale) alrededor de los patrones aprendidos (comportamiento previo).

Opcional --grid: si el latente es 2D, barre una malla (z1,z2) sobre el rango ocupado y
decodifica cada celda. El mosaico se RENDERIZA en la Fase 6; acá sólo se genera y guarda.

Ejemplos:
    python3 generate_vae.py --weights output/vae/.../weights.npz                 # prior (default)
    python3 generate_vae.py --weights ... --sampling posterior --scale 1.0       # posterior
    python3 generate_vae.py --weights ... --grid --grid-n 12                      # malla 2D
"""
import argparse
from pathlib import Path
import numpy as np

from graphs.latent import plot_latent_distributions_with_generated
from vae import build_vae_model
from font import load_fonts
from experiment import (
    make_activations,
    resolve_labels,
    output_path,
    SEED,
)
from graphs import visualize_font
from weights_io import load_weights


def generate_prior_samples(model, num_samples: int, latent_dim: int = 2, seed: int | None = None):
    """z ~ N(0, I) y decode. Generación canónica del VAE (2.c).

    Mismo patrón que generate.py (AE) y sweep_kl/plot_latent_combined: muestrear el
    PRIOR y decodificar — no depende de los patrones de entrenamiento.
    """
    if seed is not None:
        np.random.seed(seed)
    latent_samples = np.random.standard_normal((num_samples, latent_dim))
    generated = np.array([model.decode(z) for z in latent_samples])
    return latent_samples, generated


def generate_samples_around_means(
    model,
    means: np.ndarray,
    stds: np.ndarray,
    num_samples: int = 5,
    samples_per_mean: int = 1,
    seed: int | None = None,
    scale: float = 1.0,
):
    """z ~ N(μ_i, σ_i·scale) alrededor de patrones de entrenamiento (modo posterior).

    Devuelve (latent_samples, generated).
    """
    if seed is not None:
        np.random.seed(seed)

    n_patterns, latent_dim = means.shape
    latent_samples = []

    # Patrones alrededor de los que se muestrea (con repetición permitida).
    chosen_indices = np.random.choice(n_patterns, size=num_samples, replace=True)
    for idx in chosen_indices:
        mean = means[idx]
        std = stds[idx] * scale
        for _ in range(samples_per_mean):
            z = np.random.normal(loc=mean, scale=std, size=(latent_dim,))
            latent_samples.append(z)

    latent_samples = np.array(latent_samples)
    generated = np.array([model.decode(z) for z in latent_samples])
    return latent_samples, generated


def generate_grid_samples(model, means: np.ndarray, grid_n: int = 12, margin: float = 0.1):
    """Malla (z1,z2) sobre el rango latente ocupado (get_latent_distributions);
    decodifica cada celda. Orden row-major (fila 0 = z2 alto), listo para el
    mosaico imshow de la Fase 6. Devuelve (latent, generated)."""
    lo = means.min(axis=0)
    hi = means.max(axis=0)
    pad = (hi - lo) * margin
    lo, hi = lo - pad, hi + pad
    z1 = np.linspace(lo[0], hi[0], grid_n)
    z2 = np.linspace(hi[1], lo[1], grid_n)   # de arriba (z2 alto) hacia abajo
    latent = np.array([[a, b] for b in z2 for a in z1])  # (grid_n*grid_n, 2)
    generated = np.array([model.decode(z) for z in latent])
    return latent, generated


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="generate_vae.py",
        description="Genera patrones nuevos desde un VAE entrenado (2.c).",
    )
    parser.add_argument("--weights", required=True, help="ruta a los pesos .npz del VAE entrenado")
    parser.add_argument("--sampling", choices=["prior", "posterior"], default="prior",
                        help="prior: z~N(0,I) (canónico, DEFAULT); posterior: z~N(μ_i,σ_i·scale)")
    parser.add_argument("-n", "--num-samples", type=int, default=8,
                        help="cantidad de muestras (no aplica a --grid; default 8)")
    parser.add_argument("--samples-per-mean", type=int, default=1,
                        help="(posterior) muestras por media elegida (default 1)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="(posterior) factor sobre los σ aprendidos (default 1.0)")
    parser.add_argument("--grid", action="store_true",
                        help="latente 2D: barre una malla (z1,z2) y decodifica (mosaico → Fase 6)")
    parser.add_argument("--grid-n", type=int, default=12,
                        help="resolución NxN de la malla para --grid (default 12)")
    parser.add_argument("--datatype", choices=["letters", "emoji"], default="emoji",
                        help="dataset (default emoji)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed; si no se pasa, usa el de config.json (experiment.SEED)")
    parser.add_argument("--output", type=str, default="output/vae/generated",
                        help="directorio de salida")
    parser.add_argument("--plot", action="store_true",
                        help="grafica el latente con los generados superpuestos")
    args = parser.parse_args(argv)

    if not Path(args.weights).exists():
        print(f"Error: weights file not found: {args.weights}")
        return

    seed = args.seed if args.seed is not None else SEED   # --seed overridea config.json

    print(f"Loading pre-trained VAE from: {args.weights}")
    act = make_activations()
    model = build_vae_model(act, seed=seed)
    model = load_weights(model, args.weights)

    print("Loading training data...")
    training_data = load_fonts(args.datatype)
    means, stds = model.get_latent_distributions(training_data)
    if means.shape[1] != 2:
        raise ValueError("Expected latent dim 2")

    if args.grid:
        latent_samples, generated = generate_grid_samples(model, means, grid_n=args.grid_n)
        mode_desc = f"grid {args.grid_n}x{args.grid_n}"
        tag = "grid"
    elif args.sampling == "prior":
        latent_samples, generated = generate_prior_samples(model, args.num_samples, means.shape[1], seed)
        mode_desc = "prior z~N(0,I)"
        tag = "prior"
    else:
        latent_samples, generated = generate_samples_around_means(
            model, means, stds,
            num_samples=args.num_samples,
            samples_per_mean=args.samples_per_mean,
            seed=seed,
            scale=args.scale,
        )
        mode_desc = f"posterior scale={args.scale}"
        tag = "posterior"

    total = len(generated)
    print(f"Generated {total} patterns [{mode_desc}].")

    if args.grid:
        print(f"Grid {args.grid_n}x{args.grid_n}: se renderiza como mosaico en la Fase 6.")
    else:
        for i, pattern in enumerate(generated, start=1):
            visualize_font(pattern, f"Generated VAE Sample {i} [{mode_desc}]")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"vae_generated_{tag}.npz"
    np.savez(output_file, generated=generated, latent_samples=latent_samples, sampling=tag)
    print(f"Generated samples saved to: {output_file}")

    if args.plot:
        try:
            hp = {"sampling": tag, "generated": total, "datatype": args.datatype, "seed": seed}
            plot_file = output_path("vae", "latent_space_generated", hp, "latent_with_generated.png")
            plot_latent_distributions_with_generated(
                means,
                stds,
                latent_samples,
                labels=resolve_labels(args.datatype),
                output_path=plot_file,
                title=f"{args.datatype.capitalize()} VAE latente + {total} generados [{mode_desc}]",
                subtitle=f"distribución de entrenamiento + generados ({mode_desc})",
            )
            print(f"Latent space plot saved to: {plot_file}")
        except Exception as e:
            print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()
