"""
evaluation.py — Reporte de error de reconstrucción por carácter.

Tarea 1: al final de una corrida, decir cuánto error hay entre cada letra
original y su reconstrucción. El error principal es en PÍXELES (de 35),
porque el enunciado pide "error máximo de 1 píxel incorrecto".

Solo depende de numpy, así que corre sin importar el lío de imports src./no-src.
"""
import numpy as np


def binarize(x, threshold: float = 0.5) -> np.ndarray:
    """Pasa una salida continua a 0/1 con un umbral (0.5 por defecto)."""
    return (np.asarray(x, dtype=float) >= threshold).astype(int)


def pixel_errors_per_pattern(target, recon, threshold: float = 0.5) -> np.ndarray:
    """Píxeles distintos por patrón entre target y reconstrucción (ambos binarizados).

    Acepta (n, d) -> (n,) o un patrón suelto (d,) -> escalar.
    """
    return (binarize(target, threshold) != binarize(recon, threshold)).sum(axis=-1)


def nearest_pattern_distance(generated, clean, threshold: float = 0.5) -> np.ndarray:
    """Para cada patrón generado, distancia en píxeles al patrón limpio MÁS cercano.

    Mide cuán "real" es lo generado: 0 = idéntico a una letra del set.
    """
    clean_bin = binarize(clean, threshold)
    return np.array([
        int((clean_bin != binarize(g, threshold)).sum(axis=1).min())
        for g in np.atleast_2d(generated)
    ])


def pixel_error_counts(model, X_target, X_input=None, reconstruct=None,
                       threshold: float = 0.5, max_errors_ok: int = 1):
    """Versión silenciosa de pixel_error_report para los barridos (corremos decenas de
    modelos y no queremos el detalle por carácter en consola). Mismo criterio: binariza
    a 0.5 y cuenta píxeles distintos por patrón.

    Devuelve (cuántos pasan <= max_errors_ok, peor caso, promedio) en píxeles.
    """
    if X_input is None:
        X_input = X_target
    if reconstruct is None:
        reconstruct = model.forward

    errors = np.array([
        int(pixel_errors_per_pattern(X_target[i], reconstruct(X_input[i]), threshold))
        for i in range(len(X_target))
    ])
    return int((errors <= max_errors_ok).sum()), int(errors.max()), float(errors.mean())


def pixel_error_report(
    model,
    X_target,
    X_input=None,
    labels=None,
    reconstruct=None,
    threshold: float = 0.5,
    max_errors_ok: int = 1,
):
    """
    Reconstruye cada patrón, lo binariza y cuenta píxeles mal contra el original.

    Args:
        model:       modelo entrenado (Autoencoder o VariationalAutoencoder).
        X_target:    (n, 35) patrones ORIGINALES limpios = lo que se espera a la salida.
        X_input:     (n, 35) lo que se le da de entrada al modelo.
                     Si es None, usa X_target (caso AE básico: entrada = objetivo).
                     Para el DAE pasá acá la versión RUIDOSA y dejá X_target limpio.
        labels:      lista de nombres por patrón, ej ['a','b',...]. Opcional.
        reconstruct: función x -> x'. Por defecto model.forward.
                     IMPORTANTE: para el VAE pasá model.reconstruct (es determinista;
                     model.forward samplea y daría resultados distintos cada vez).
        threshold:   umbral para pasar la salida continua a 0/1 (0.5 por defecto).
        max_errors_ok: umbral del enunciado (1 píxel).

    Returns:
        dict con el detalle por carácter y el resumen.
    """
    if X_input is None:
        X_input = X_target
    if reconstruct is None:
        reconstruct = model.forward

    rows = []
    for i in range(len(X_target)):
        x_hat = np.asarray(reconstruct(X_input[i])).reshape(-1)
        pixel_errors = int(pixel_errors_per_pattern(X_target[i], x_hat, threshold))
        # error continuo, por si se quiere comparar (no es el criterio del enunciado)
        mse = float(np.mean((np.asarray(X_target[i]).reshape(-1) - x_hat) ** 2))

        label = labels[i] if labels is not None else str(i)
        rows.append({"label": label, "pixel_errors": pixel_errors, "mse": mse})

    pass_count = sum(1 for r in rows if r["pixel_errors"] <= max_errors_ok)
    max_err = max(r["pixel_errors"] for r in rows)
    mean_err = sum(r["pixel_errors"] for r in rows) / len(rows)
    total = len(rows)

    print("\n" + "=" * 46)
    print("  RECONSTRUCCIÓN — error por carácter")
    print("=" * 46)
    print(f"  {'char':>6}  {'px err':>7}  {'mse':>8}  {'<=' + str(max_errors_ok) + '?':>5}")
    print("-" * 46)
    for r in rows:
        ok = "OK" if r["pixel_errors"] <= max_errors_ok else "X"
        print(f"  {r['label']:>6}  {r['pixel_errors']:>7}  {r['mse']:>8.4f}  {ok:>5}")
    print("-" * 46)
    print(f"  Caracteres con <= {max_errors_ok} px : {pass_count}/{total}")
    print(f"  Peor caso           : {max_err} px")
    print(f"  Promedio            : {mean_err:.2f} px")
    objetivo = "ALCANZADO" if max_err <= max_errors_ok else "NO alcanzado"
    print(f"  Objetivo enunciado  : {objetivo}")
    print("=" * 46 + "\n")

    return {
        "per_char": rows,
        "pass_count": pass_count,
        "max_error": max_err,
        "mean_error": mean_err,
        "total": total,
    }
