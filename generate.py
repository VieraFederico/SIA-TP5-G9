"""
generate_from_latent.py — Generate synthetic font patterns from random latent space points.

Loads a pre-trained autoencoder and samples random points from the latent space,
then decodes them to generate new (synthetic) font patterns.

Usage:
    python generate.py --weights output/ae/letters/...../weights.npz --num-samples 5
    python generate.py --weights path/to/weights.npz -n 8 --datatype letters
"""
import argparse
from pathlib import Path

import numpy as np

from ae import build_ae_model
from experiment import make_activations, resolve_labels, output_path, hyperparams_slug, hyperparams_subtitle
from graphs import visualize_font, plot_latent_with_generated
from weights_io import load_weights


def generate_samples(
    model,
    num_samples: int = 5,
    latent_dim: int = 2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample random points from latent space and decode them.

    Args:
        model: Autoencoder instance
        num_samples: how many patterns to generate
        latent_dim: dimensionality of latent space (e.g., 2 for tanh bottleneck)
        seed: optional seed for reproducibility

    Returns:
        Tuple of (latent_samples, generated_patterns)
            - latent_samples: Array of shape (num_samples, 2)
            - generated_patterns: Array of shape (num_samples, 35)
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample from standard normal in latent space
    # Using tanh activation, the latent space is roughly [-1, 1]
    # Standard normal samples work well for exploration
    latent_samples = np.random.standard_normal((num_samples, latent_dim))

    # Decode each latent point
    generated = np.array([model.decode(z) for z in latent_samples])

    return latent_samples, generated


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate synthetic font patterns from a trained autoencoder.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to pre-trained weights (.npz file)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to generate (default: 5)",
    )
    parser.add_argument(
        "--datatype",
        choices=["letters", "emoji"],
        default="letters",
        help="Dataset type (for architecture consistency, default: letters)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        help="Latent space dimensionality (default: 2, must match trained model)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/generated",
        help="Output directory for generated samples (default: output/generated)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a latent space plot overlaying generated samples on training data",
    )

    args = parser.parse_args(argv)

    # Verify weights file exists
    if not Path(args.weights).exists():
        print(f"Error: weights file not found: {args.weights}")
        return

    print(f"Loading pre-trained model from: {args.weights}")
    act = make_activations()
    model = build_ae_model(act, seed=args.seed)
    model = load_weights(model, args.weights)

    print(f"\nGenerating {args.num_samples} samples from latent space...")
    latent_samples, generated = generate_samples(
        model,
        num_samples=args.num_samples,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )

    print(f"\nVisualizing {args.num_samples} generated patterns:\n")
    for i, pattern in enumerate(generated, start=1):
        visualize_font(pattern, f"Generated Sample {i}")

    # Optionally save generated patterns
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "generated_samples.npz"
    np.savez(output_file, generated=generated, latent_samples=latent_samples)
    print(f"\nGenerated samples saved to: {output_file}")

    # Generate plot if requested
    if args.plot:
        print("\nGenerating latent space plot with training data and generated samples...")
        training_positions = model.get_latent_positions(
            np.zeros((1, 35))  # Placeholder; we'll need actual training data
        )
        # For this we'd need to reload training data; better approach:
        # pass --plot-path to save the plot directly
        try:
            from font import load_fonts
            training_data = load_fonts(args.datatype)
            training_positions = model.get_latent_positions(training_data)

            hp = {
                "generated": args.num_samples,
                "datatype": args.datatype,
                "seed": args.seed,
            }

            plot_file = output_path(
                "ae",
                "latent_space_generated",
                hp,
                "latent_with_generated.png"
            )
            plot_latent_with_generated(
                training_positions,
                latent_samples,
                labels=resolve_labels(args.datatype),
                output_path=plot_file,
                title=f"{args.datatype.capitalize()} latent space with {args.num_samples} generated samples",
                subtitle=f"Generated samples (orange stars) overlaid on training data (blue dots)",
            )
            print(f"Latent space plot saved to: {plot_file}")
        except Exception as e:
            print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()