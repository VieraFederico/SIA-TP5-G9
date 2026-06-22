#!/usr/bin/env python3
"""
generate_vae.py — Generate synthetic font patterns from a trained VAE.

Sampling is done around the learned latent distributions: z ~ N(mean_i, std_i * scale)
for randomly chosen training patterns (or all patterns if desired).

Example usage:
    python generate_vae.py --weights output/vae/.../weights.npz -n 8 --datatype emoji --samples-per-mean 2 --scale 1.0 --plot
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
)
from graphs import visualize_font, plot_latent_with_generated
from weights_io import load_weights


def generate_samples_around_means(
    model,
    means: np.ndarray,
    stds: np.ndarray,
    num_samples: int = 5,
    samples_per_mean: int = 1,
    seed: int | None = None,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate latent_samples and decoded patterns sampling around the provided means.
    - means: array shape (n_patterns, latent_dim)
    - stds: array shape (n_patterns, latent_dim)
    Returns (latent_samples, generated_patterns)
    """
    if seed is not None:
        np.random.seed(seed)

    n_patterns, latent_dim = means.shape
    latent_samples = []

    # Choose which training patterns we'll sample around. We allow repeats.
    chosen_indices = np.random.choice(n_patterns, size=num_samples, replace=True)

    for idx in chosen_indices:
        mean = means[idx]
        std = stds[idx] * scale
        # If samples_per_mean > 1, sample that many around this chosen mean
        for _ in range(samples_per_mean):
            z = np.random.normal(loc=mean, scale=std, size=(latent_dim,))
            latent_samples.append(z)

    latent_samples = np.array(latent_samples)  # shape (num_samples * samples_per_mean, latent_dim)
    generated = np.array([model.decode(z) for z in latent_samples])
    return latent_samples, generated


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="generate_vae.py",
        description="Generate synthetic font patterns from a trained VAE.",
    )
    parser.add_argument("--weights", required=True, help="Path to pre-trained VAE weights (.npz)")
    parser.add_argument("-n", "--num-samples", type=int, default=5, help="Number of (center) samples to draw (default: 5)")
    parser.add_argument("--samples-per-mean", type=int, default=1, help="How many samples to draw per chosen mean (default: 1)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor applied to learned std devs (default: 1.0)")
    parser.add_argument("--datatype", choices=["letters", "emoji"], default="emoji", help="Dataset type (default: emoji)")

    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility")
    parser.add_argument("--output", type=str, default="output/vae/generated", help="Output directory for generated samples")
    parser.add_argument("--plot", action="store_true", help="Generate a latent space plot overlaying generated samples on training data")
    args = parser.parse_args(argv)

    if not Path(args.weights).exists():
        print(f"Error: weights file not found: {args.weights}")
        return

    print(f"Loading pre-trained VAE from: {args.weights}")
    act = make_activations()
    model = build_vae_model(act, seed=args.seed)
    model = load_weights(model, args.weights)

    print("Loading training data...")
    training_data = load_fonts(args.datatype)

    # Get per-pattern latent distributions (means, stds)
    means, stds = model.get_latent_distributions(training_data)

    if means.shape[1] != 2:
        raise ValueError(f"Expected latent dim 2")

    print(f"Sampling {args.num_samples} means, {args.samples_per_mean} samples per mean (scale={args.scale})...")
    latent_samples, generated = generate_samples_around_means(
        model,
        means,
        stds,
        num_samples=args.num_samples,
        samples_per_mean=args.samples_per_mean,
        seed=args.seed,
        scale=args.scale,
    )

    total_generated = len(generated)
    print(f"Generated {total_generated} patterns. Visualizing...")

    for i, pattern in enumerate(generated, start=1):
        visualize_font(pattern, f"Generated VAE Sample {i}")

    # Save generated patterns + latent samples
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "vae_generated_samples.npz"
    np.savez(output_file, generated=generated, latent_samples=latent_samples)
    print(f"Generated samples saved to: {output_file}")

    # Optionally create a latent plot with generated samples over training means
    if args.plot:
        try:
            hp = {
                "generated": total_generated,
                "datatype": args.datatype,
                "seed": args.seed,
                "scale": args.scale,
            }
            plot_file = output_path(
                "vae",
                "latent_space_generated",
                hp,
                "latent_with_generated.png"
            )
            # For plotting, use training means as "positions" and overlay latent_samples
            plot_latent_distributions_with_generated(
                means,
                stds,
                latent_samples,
                labels=resolve_labels(args.datatype),
                output_path=plot_file,
                title=f"{args.datatype.capitalize()} VAE latent space with {total_generated} generated samples",
                subtitle=f"Original training distribution + generated samples, scale={args.scale}",
            )
            print(f"Latent space plot saved to: {plot_file}")
        except Exception as e:
            print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()