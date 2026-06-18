"""
main.py — Punto de entrada del TP5.

Lanzador delgado: toda la lógica de CLI está en el paquete cli/.
La lógica de cada modelo está en ae.py (autoencoder / denoising) y vae.py
(variational), y los pasos compartidos en experiment.py.

Uso:
    python main.py ae            # autoencoder / denoising AE sobre font.h
    python main.py vae           # variational autoencoder sobre emojis
    python main.py --help        # ver todas las opciones
"""
from cli.app import main

if __name__ == "__main__":
    main()
