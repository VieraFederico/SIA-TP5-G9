"""
weights_io.py — Guardar y cargar los pesos del modelo entre corridas.

Tarea 2: guardar los pesos del modelo final y poder inicializar un modelo
con esos pesos en una corrida futura (sin reentrenar).

Se apoya en model.get_weights() / model.set_weights(), que ya existen en la
interfaz Model y devuelven/aceptan la lista de (W, b) por capa en el orden
correcto. Solo depende de numpy.

Se usa .npz (no .csv) a propósito: los pesos son matrices 2-D y .npz preserva
forma y tipo. CSV las aplanaría y habría que reconstruir la forma a mano.
"""
import numpy as np


def save_weights(model, path: str = "weights.npz") -> None:
    """Guarda los pesos del modelo en un archivo .npz."""
    flat = {}
    for i, (w, b) in enumerate(model.get_weights()):
        flat[f"layer_{i}_w"] = np.asarray(w)
        flat[f"layer_{i}_b"] = np.asarray(b)
    np.savez(path, **flat)
    print(f"Pesos guardados en {path}  ({len(flat) // 2} capas)")


def load_weights(model, path: str = "weights.npz"):
    """
    Carga pesos guardados por save_weights() en un modelo YA construido con la
    MISMA arquitectura (mismas capas, mismos tamaños). Devuelve el modelo.
    """
    data = np.load(path)
    weights = []
    i = 0
    while f"layer_{i}_w" in data:
        weights.append((data[f"layer_{i}_w"], data[f"layer_{i}_b"]))
        i += 1
    if not weights:
        raise ValueError(f"No se encontraron pesos en {path}")
    model.set_weights(weights)
    print(f"Pesos cargados desde {path}  ({len(weights)} capas)")
    return model
