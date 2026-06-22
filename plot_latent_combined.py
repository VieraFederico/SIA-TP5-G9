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

from evaluation import nearest_pattern_distance
from experiment import make_activations, resolve_labels
from font import load_fonts
from graphs import plot_latent_clouds_generated
from sampling import latent_bounds, sample_prior, set_seed
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

    set_seed(args.seed)

    act = make_activations()
    clean = load_fonts(args.data)
    labels = resolve_labels(args.data)
    model = build_vae_model(act)
    load_weights(model, args.weights)

    means = model.get_latent_positions(clean)          # μ por patrón
    _, stds = model.get_latent_distributions(clean)    # σ por patrón

    # ---- generación ----
    if args.sampling == "normal":
        gen = sample_prior(args.n_generated, 2)   # seed ya fijada arriba
    else:  # bounds: uniforme dentro del rango ocupado por las medias
        lo, hi = latent_bounds(means)
        gen = np.random.uniform(lo, hi, size=(args.n_generated, 2))

    # ---- distancia de generación (px al patrón real más cercano) ----
    generated = np.array([model.decode(z) for z in gen])
    dists = [int(d) for d in nearest_pattern_distance(generated, clean)]
    print(f"Distancia de generación (px al patrón real más cercano): "
          f"promedio {np.mean(dists):.2f}, por muestra {dists}")

    # ---- figura (nubes muestreadas acá; el dibujo vive en graphs/) ----
    eps = np.random.standard_normal((len(clean), args.cloud_samples, 2))
    clouds = (means[:, None, :] + stds[:, None, :] * eps).reshape(-1, 2)

    output = args.output or f"output/vae/combined_{Path(args.weights).parent.name}.png"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    saved = plot_latent_clouds_generated(
        clouds, means, gen,
        output_path=output,
        title="VAE: espacio latente con generados",
        subtitle=f"data={args.data}  ·  seed={args.seed}  ·  sampling={args.sampling}  "
                 f"·  dist gen media={np.mean(dists):.2f}px",
        labels=labels, gen_label=f"generados ({args.sampling})",
    )
    print(f"Figura guardada en: {saved}")


if __name__ == "__main__":
    main()
