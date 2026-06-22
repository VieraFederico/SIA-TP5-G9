"""
generate_from_latent.py — Generate synthetic font patterns from random latent space points.

Loads a pre-trained autoencoder and samples random points from the latent space,
then decodes them to generate new (synthetic) font patterns.

Entrada pública por main.py (este main(argv) es sólo detalle interno):
    python main.py generate ae --weights output/ae/.../weights.npz -n 5
    python main.py generate ae --weights .../weights.npz -n 8 --data letters --sampling normal
"""
import argparse
from pathlib import Path
from font import load_fonts
import numpy as np

from ae import build_ae_model
from experiment import make_activations, resolve_labels, output_path, hyperparams_slug, hyperparams_subtitle
from graphs import visualize_font, plot_latent_with_generated, plot_generated
from sampling import latent_bounds, sample_prior, set_seed
from weights_io import load_weights


def generate_samples(
    model,
    num_samples: int = 5,
    latent_dim: int = 2,
    seed: int | None = None,
    sampling_method: str = "latent_bounds",
    latent_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)

    if sampling_method == "latent_bounds":
        if latent_bounds is None:
            raise ValueError("latent_bounds is required when sampling_method='latent_bounds'")

        latent_min, latent_max = latent_bounds
        latent_samples = np.random.uniform(
            low=latent_min,
            high=latent_max,
            size=(num_samples, latent_dim),
        )

    elif sampling_method == "normal":
        latent_samples = sample_prior(num_samples, latent_dim)   # seed ya fijada arriba

    else:
        raise ValueError(f"Unknown sampling_method: {sampling_method!r}")

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
        "--sampling",
        choices=["normal", "bounds"],
        default="bounds",
        help="normal: z~N(0,1)  ·  bounds: uniforme dentro del rango latente ocupado",
    )

    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to generate (default: 5)",
    )
    parser.add_argument(
        "--data", "--datatype", dest="data",
        choices=["letters", "emoji"],
        default="letters",
        help="dataset (debe coincidir con el modelo entrenado, default: letters)",
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
        default="output/ae/generated",
        help="Output directory for generated samples (default: output/ae/generated)",
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
    training_data = load_fonts(args.data)
    training_positions = model.get_latent_positions(training_data)
    latent_min, latent_max = latent_bounds(training_positions)

    # "bounds" es el nombre público; internamente la estrategia se llama "latent_bounds".
    method = "normal" if args.sampling == "normal" else "latent_bounds"
    latent_samples, generated = generate_samples(
        model,
        num_samples=args.num_samples,
        latent_dim=args.latent_dim,
        seed=args.seed,
        sampling_method=method,
        latent_bounds=(latent_min, latent_max),
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

    image_file = plot_generated(
        generated, latent_samples, str(output_dir / "generated_samples.png"),
        title=f"{args.data.capitalize()}: muestras generadas (AE)",
        subtitle=f"z muestreado del latente ({args.sampling}) · decode → patrón",
    )
    print(f"Generated images saved to: {image_file}")

    # Generate plot if requested
    if args.plot:
        print("\nGenerating latent space plot with training data and generated samples...")

        # For this we'd need to reload training data; better approach:
        # pass --plot-path to save the plot directly
        try:


            hp = {
                "generated": args.num_samples,
                "datatype": args.data,
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
                labels=resolve_labels(args.data),
                output_path=plot_file,
                title=f"{args.data.capitalize()} latent space with {args.num_samples} generated samples",
                subtitle=f"Generated samples (orange stars) overlaid on training data (blue dots)",
            )
            print(f"Latent space plot saved to: {plot_file}")
        except Exception as e:
            print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()