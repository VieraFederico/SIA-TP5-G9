"""
study_engine.py — MOTOR de estudio compartido por los cuatro estudios (arquitectura-AE,
hiperparámetros-AE, arquitectura-DAE, hiperparámetros-DAE).

UNA sola función `run_study`: por cada (config × [salt] × seed) entrena con train_once,
evalúa con la evaluación canónica (en DAE, contra el patrón LIMPIO), escribe la celda al
CSV crudo APENAS termina (reanudable), agrega por combinación, corre la selección única
(study_selection.rank_and_select) y plotea (tablas, barras, curvas de loss, overlay top-10).

Los cuatro estudios sólo arman su lista `configs` (lista de arquitecturas, o el producto
cartesiano opt×lr×init) y llaman acá. No hay lógica de entrenamiento/selección duplicada.

Reanudable: cada celda se escribe al CSV apenas termina y su curva de loss a un sidecar
JSON. Al rearrancar se saltean las celdas ya presentes (clave label+salt+seed), así una
interrupción no cuesta la corrida entera.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.ae import build_ae_model
from experiments.experiment import make_activations, study_subtitle, train_once
from src.utils.evaluation import pixel_error_counts
from src.utils.sampling import set_seed
from src.optimizer.adam import AdamOptimizer
from src.optimizer.gradient_descent import GradientDescent
from src.utils.config import ADAM_BETA1, ADAM_BETA2, EPSILON
from src.noise.salt_n_pepper import SaltNPepperNoise
from experiments.study_selection import rank_and_select
from graphs.studies import (
    grouped_bar_study, loss_band_curve, overlaid_curves, ranked_bar_study, table_figure,
)
from graphs.style import BLUE, ORANGE

LATENT_DIM = 2

RAW_COLUMNS = [
    "study", "combo_id", "label", "architecture", "opt", "lr", "init", "mode", "act",
    "salt", "with_noise", "seed", "mean_px_err", "worst_px_err", "patterns_le_1px",
    "final_loss", "epochs_ran",
]
SUMMARY_COLUMNS = [
    "study", "combo_id", "rank", "label", "architecture", "opt", "lr", "init", "mode",
    "act", "salt", "with_noise", "seeds", "mean_px_err_mean", "mean_px_err_std",
    "worst_px_err_mean", "worst_px_err_std", "patterns_le_1px_mean", "n_params", "selected",
]

SALT_COLORS = {0.1: BLUE, 0.2: ORANGE}


def make_optimizer(name, lr):
    """adam (con los β de config) o gradient descent plano. Único punto de mapeo."""
    if name == "adam":
        return AdamOptimizer(lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2)
    return GradientDescent(lr)


def param_count(widths):
    """Nº de parámetros (pesos + bias) de la arquitectura AE espejada con bottleneck 2D.
    widths = anchos del encoder de la entrada a la capa previa al bottleneck."""
    full = list(widths) + [LATENT_DIM] + list(widths)[::-1]
    return int(sum(full[i] * full[i + 1] + full[i + 1] for i in range(len(full) - 1)))


# --- entrenamiento + evaluación de una celda -----------------------------------------

def _train_eval_cell(cfg, seed, clean, with_noise, salt, epochs):
    """Entrena UNA combinación con UNA seed y evalúa SIEMPRE contra clean.

    DAE (with_noise): entrada ruidosa, ruido re-sampleado por época (resample=on),
    objetivo limpio; evalúa sobre ruido NUEVO (stream seed+1000). AE: entrada=objetivo."""
    set_seed(seed)
    model = build_ae_model(make_activations(), seed=seed,
                           encoder_widths=cfg["widths"], hidden_act=cfg["act"],
                           init_scheme=cfg["init"])

    if with_noise:
        x_input = SaltNPepperNoise(salt).add_noise(clean.copy())
        noise_fn = lambda: SaltNPepperNoise(salt).add_noise(clean.copy())
    else:
        x_input, noise_fn = clean, None

    history = train_once(model, x_input, clean, cfg["widths"], epochs=epochs,
                         training_mode=cfg["mode"],
                         optimizer=make_optimizer(cfg["opt"], cfg["lr"]),
                         noise_fn=noise_fn)

    if with_noise:
        set_seed(seed + 1000)  # ruido NUEVO para evaluar, distinto del de entrenamiento
        x_eval = SaltNPepperNoise(salt).add_noise(clean.copy())
    else:
        x_eval = clean
    passed, worst, mean_px = pixel_error_counts(model, clean, X_input=x_eval)

    curve = list(history.get("train_error", []) or [])
    final_loss = float(curve[-1]) if curve else float("nan")
    return {
        "mean": float(mean_px), "worst": int(worst), "passed": int(passed),
        "final_loss": final_loss, "epochs_ran": int(history["epochs"]), "curve": curve,
    }


# --- CSV crudo reanudable -------------------------------------------------------------

def _cell_key(label, salt_str, seed):
    return (str(label), str(salt_str), str(seed))


def _load_done(raw_path):
    """Celdas ya presentes en el CSV crudo, por clave label+salt+seed."""
    if not raw_path.exists():
        return set()
    with raw_path.open(newline="") as f:
        return {_cell_key(r["label"], r["salt"], r["seed"]) for r in csv.DictReader(f)}


def _salt_str(salt):
    return "" if salt is None else str(salt)


# --- agregación + selección -----------------------------------------------------------

def _aggregate(raw_rows, configs_by_id):
    """Agrupa filas crudas por (combo_id, salt) y resume media ± σ de cada métrica.
    Devuelve {salt_str: [summary_por_combo, ...]}."""
    groups = defaultdict(list)
    for r in raw_rows:
        groups[(int(r["combo_id"]), r["salt"])].append(r)

    per_salt = defaultdict(list)
    for (combo_id, salt_str), rows in groups.items():
        mean = np.array([float(r["mean_px_err"]) for r in rows])
        worst = np.array([float(r["worst_px_err"]) for r in rows])
        le1 = np.array([float(r["patterns_le_1px"]) for r in rows])
        cfg = configs_by_id[combo_id]
        per_salt[salt_str].append({
            "combo_id": combo_id, "label": cfg["label"], "architecture": cfg["architecture"],
            "opt": cfg["opt"], "lr": cfg["lr"], "init": cfg["init"], "mode": cfg["mode"],
            "act": cfg["act"], "n_params": param_count(cfg["widths"]),
            "mean_mean": float(mean.mean()), "mean_std": float(mean.std()),
            "worst_mean": float(worst.mean()), "worst_std": float(worst.std()),
            "le1_mean": float(le1.mean()), "n_seeds": len(rows),
        })
    return per_salt


# --- plotting -------------------------------------------------------------------------

def _curve_for(cache, label, salt_str, seeds, base_seed):
    """Curvas de loss por seed para una combinación (las que estén en el sidecar)."""
    out = []
    for k in range(seeds):
        key = f"{label}|{salt_str}|{base_seed + k}"
        if key in cache:
            out.append(cache[key])
    return out


def _fmt_lr(lr):
    return f"{float(lr):.0e}"


def _plot_tables(kind, ranked, best_id, outdir, subtitle, salt_suffix):
    """Tabla(s) de presentación, ordenadas por rank. Arquitectura: una con todas.
    Hiperparámetros: parte 1 (rank 1-15) y parte 2 (16-30)."""
    if kind == "architecture":
        headers = ["#", "arquitectura", "err medio ± σ", "peor px", "≤1px/32", "rank"]
        rows = [[str(r["combo_id"]), r["architecture"],
                 f"{r['mean_mean']:.2f} ± {r['mean_std']:.2f}", f"{r['worst_mean']:.1f}",
                 f"{r['le1_mean']:.1f}", str(r["rank"])] for r in ranked]
        table_figure(headers, rows, "Arquitectura — ranking por error medio",
                     str(outdir / f"table{salt_suffix}.png"), subtitle=subtitle,
                     selected_ids=[best_id], top_n=10)
        return

    headers = ["#", "opt", "lr", "init", "err medio ± σ", "peor px", "≤1px/32", "rank"]
    rows = [[str(r["combo_id"]), r["opt"], _fmt_lr(r["lr"]), r["init"],
             f"{r['mean_mean']:.2f} ± {r['mean_std']:.2f}", f"{r['worst_mean']:.1f}",
             f"{r['le1_mean']:.1f}", str(r["rank"])] for r in ranked]
    half = (len(rows) + 1) // 2
    for part, chunk in ((1, rows[:half]), (2, rows[half:])):
        if not chunk:
            continue
        lo, hi = chunk[0][-1], chunk[-1][-1]
        table_figure(headers, chunk,
                     f"Hiperparámetros — ranking por error medio (ranks {lo}-{hi})",
                     str(outdir / f"table_part{part}{salt_suffix}.png"), subtitle=subtitle,
                     selected_ids=[best_id], top_n=10)


def _plot_loss(ranked, cache, seeds, base_seed, loss_dir, subtitle, salt_str, salt_suffix):
    """Una curva de loss por combinación (media ± σ entre seeds), numerada por combo_id,
    + un overlay con las curvas medias del TOP-10."""
    by_id = {r["combo_id"]: r for r in ranked}
    for combo_id in sorted(by_id):
        curves = _curve_for(cache, by_id[combo_id]["label"], salt_str, seeds, base_seed)
        loss_band_curve(
            curves, f"Loss combo #{combo_id} — {by_id[combo_id]['label']}",
            str(loss_dir / f"combo_{combo_id:02d}{salt_suffix}.png"),
            subtitle=subtitle, logy=True,
        )

    top = sorted(ranked, key=lambda r: r["rank"])[:10]
    series = []
    for r in top:
        curves = _curve_for(cache, r["label"], salt_str, seeds, base_seed)
        curves = [c for c in curves if c]
        if not curves:
            continue
        length = min(len(c) for c in curves)
        mean = np.array([c[:length] for c in curves], dtype=float).mean(axis=0)
        series.append((f"#{r['combo_id']} (rank {r['rank']})", mean))
    if series:
        overlaid_curves(series, "Época", "Train loss (BCE)",
                        "TOP-10 por criterio — curvas de loss",
                        str(loss_dir.parent / f"top10_overlay{salt_suffix}.png"),
                        subtitle=subtitle, logy=True)


def _plot_bars(kind, per_salt, selected, salts, outdir, subtitle):
    """Barras de ranking. Arquitectura: por arquitectura (orden de generación); DAE con
    dos series de salt. Hiperparámetros: ordenadas mejor→peor, seleccionado en verde."""
    if kind == "architecture":
        any_salt = next(iter(per_salt))
        # eje x = arquitecturas en orden de generación (combo_id)
        base = sorted(per_salt[any_salt], key=lambda r: r["combo_id"])
        labels = [r["architecture"] for r in base]
        if len(salts) > 1:
            series = []
            for salt in salts:
                ss = _salt_str(salt)
                rows = sorted(per_salt[ss], key=lambda r: r["combo_id"])
                series.append((f"salt={salt}", [r["mean_mean"] for r in rows],
                               [r["mean_std"] for r in rows], SALT_COLORS.get(salt, BLUE)))
            top = max(m + s for _, ms, ss_, _ in series for m, s in zip(ms, ss_))
            grouped_bar_study(labels, series, "Error medio de píxel (vs limpio, de 35)",
                              "Arquitectura-DAE: error medio (series por salt)",
                              str(outdir / "arch_mean_error.png"), subtitle=subtitle,
                              rotate=20, ylim=(0, top * 1.15))
        else:
            ss = _salt_str(salts[0])
            means = [r["mean_mean"] for r in base]
            stds = [r["mean_std"] for r in base]
            sel_idx = next((i for i, r in enumerate(base)
                            if r["combo_id"] == selected[ss]), None)
            ranked_bar_study(labels, means, stds, "Error medio de píxel (de 35)",
                             "Arquitectura: error medio por arquitectura",
                             str(outdir / "arch_mean_error.png"), subtitle=subtitle,
                             selected_idx=sel_idx, top_n=len(labels), rotate=20)
        return

    # hiperparámetros: una figura de barras ordenadas por salt
    for salt in salts:
        ss = _salt_str(salt)
        ranked = sorted(per_salt[ss], key=lambda r: r["rank"])
        labels = [f"#{r['combo_id']}" for r in ranked]
        means = [r["mean_mean"] for r in ranked]
        stds = [r["mean_std"] for r in ranked]
        sel_idx = next((i for i, r in enumerate(ranked)
                        if r["combo_id"] == selected[ss]), None)
        suffix = "" if salt is None else f"_salt{salt}"
        title = "Hiperparámetros: error medio (mejor→peor)" + (
            "" if salt is None else f" — salt={salt}")
        ranked_bar_study(labels, means, stds, "Error medio de píxel (de 35)", title,
                         str(outdir / f"hp_ranked{suffix}.png"), subtitle=subtitle,
                         selected_idx=sel_idx, top_n=10)


def _plot_convergence(per_salt, cache, seeds, base_seed, salts, outdir, subtitle):
    """Arquitectura: curvas de convergencia superpuestas (media entre seeds) por salt."""
    for salt in salts:
        ss = _salt_str(salt)
        rows = sorted(per_salt[ss], key=lambda r: r["combo_id"])
        series = []
        for r in rows:
            curves = [c for c in _curve_for(cache, r["label"], ss, seeds, base_seed) if c]
            if not curves:
                continue
            length = min(len(c) for c in curves)
            mean = np.array([c[:length] for c in curves], dtype=float).mean(axis=0)
            series.append((r["architecture"], mean))
        if series:
            suffix = "" if salt is None else f"_salt{salt}"
            title = "Arquitectura: convergencia" + ("" if salt is None else f" (salt={salt})")
            overlaid_curves(series, "Época", "Train loss (BCE)", title,
                            str(outdir / f"arch_convergence{suffix}.png"),
                            subtitle=subtitle, logy=True)


# --- motor ----------------------------------------------------------------------------

def run_study(*, study, kind, configs, clean, seeds, base_seed, with_noise, salts,
              epochs, outdir, data, tiebreaker):
    """Corre un estudio completo: entrena (reanudable) → CSV → selección → figuras.

    study   : nombre para el CSV ('hyperparams', 'architecture-dae', ...).
    kind    : 'architecture' | 'hyperparams' (layout de figuras/tablas).
    configs : lista de dicts con combo_id, label, widths, opt, lr, init, mode, act, architecture.
    salts   : [None] para AE; [0.1, 0.2] para DAE (dos series / dos niveles).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    loss_dir = outdir / "loss"
    loss_dir.mkdir(exist_ok=True)
    raw_path = outdir / "raw.csv"
    summary_path = outdir / "summary.csv"
    curves_path = outdir / "loss_curves.json"
    configs_by_id = {c["combo_id"]: c for c in configs}

    done = _load_done(raw_path)
    cache = json.loads(curves_path.read_text()) if curves_path.exists() else {}

    new_file = not raw_path.exists()
    raw_f = raw_path.open("a", newline="")
    writer = csv.DictWriter(raw_f, fieldnames=RAW_COLUMNS)
    if new_file:
        writer.writeheader()
        raw_f.flush()

    n_trained = n_skipped = 0
    total = len(salts) * len(configs) * seeds
    for salt in salts:
        ss = _salt_str(salt)
        for cfg in configs:
            for k in range(seeds):
                seed = base_seed + k
                key = _cell_key(cfg["label"], ss, seed)
                if key in done:
                    n_skipped += 1
                    continue
                m = _train_eval_cell(cfg, seed, clean, with_noise, salt, epochs)
                writer.writerow({
                    "study": study, "combo_id": cfg["combo_id"], "label": cfg["label"],
                    "architecture": cfg["architecture"], "opt": cfg["opt"], "lr": cfg["lr"],
                    "init": cfg["init"], "mode": cfg["mode"], "act": cfg["act"], "salt": ss,
                    "with_noise": "on" if with_noise else "off", "seed": seed,
                    "mean_px_err": f"{m['mean']:.4f}", "worst_px_err": m["worst"],
                    "patterns_le_1px": m["passed"], "final_loss": f"{m['final_loss']:.6f}",
                    "epochs_ran": m["epochs_ran"],
                })
                raw_f.flush()
                cache[f"{cfg['label']}|{ss}|{seed}"] = m["curve"]
                curves_path.write_text(json.dumps(cache))
                done.add(key)
                n_trained += 1
                print(f"  [{n_trained + n_skipped}/{total}] {study} #{cfg['combo_id']} "
                      f"{cfg['label']} salt={ss or '-'} seed={seed}: "
                      f"medio {m['mean']:.2f}px peor {m['worst']}px ep={m['epochs_ran']}")
    raw_f.close()
    print(f"Celdas entrenadas: {n_trained}  ·  reusadas del CSV (reanudación): {n_skipped}")

    # Agregación + selección por nivel de salt.
    raw_rows = list(csv.DictReader(raw_path.open(newline="")))
    per_salt = _aggregate(raw_rows, configs_by_id)

    ranked_by_salt, selected, tie_by_salt = {}, {}, {}
    for ss, summaries in per_salt.items():
        ranked, best_id, tie_ids = rank_and_select(summaries, tiebreaker=tiebreaker)
        ranked_by_salt[ss] = ranked
        selected[ss] = best_id
        tie_by_salt[ss] = tie_ids

    # CSV resumen.
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for salt in salts:
            ss = _salt_str(salt)
            for r in ranked_by_salt[ss]:
                mark = ("best" if r["combo_id"] == selected[ss]
                        else "tie" if r["combo_id"] in tie_by_salt[ss] else "")
                w.writerow({
                    "study": study, "combo_id": r["combo_id"], "rank": r["rank"],
                    "label": r["label"], "architecture": r["architecture"], "opt": r["opt"],
                    "lr": r["lr"], "init": r["init"], "mode": r["mode"], "act": r["act"],
                    "salt": ss, "with_noise": "on" if with_noise else "off",
                    "seeds": r["n_seeds"], "mean_px_err_mean": f"{r['mean_mean']:.4f}",
                    "mean_px_err_std": f"{r['mean_std']:.4f}",
                    "worst_px_err_mean": f"{r['worst_mean']:.2f}",
                    "worst_px_err_std": f"{r['worst_std']:.2f}",
                    "patterns_le_1px_mean": f"{r['le1_mean']:.2f}",
                    "n_params": r["n_params"], "selected": mark,
                })

    # Subtítulo común: ejes fijos declarados + presupuesto de épocas del grid.
    sample = configs[0]
    fixed = {"mode": sample["mode"], "act": sample["act"], "épocas(grid)": epochs,
             "seeds": seeds, "target": "clean" if with_noise else "self"}
    if with_noise:
        fixed["salt"] = "/".join(str(s) for s in salts)
        fixed["resample"] = "on"
    if kind == "architecture":
        fixed["lr"] = sample["lr"]
        fixed["init"] = sample["init"]
        fixed["opt"] = sample["opt"]
    subtitle = study_subtitle({"data": data}, fixed)

    # Figuras.
    for salt in salts:
        ss = _salt_str(salt)
        salt_suffix = "" if salt is None else f"_salt{salt}"
        _plot_tables(kind, ranked_by_salt[ss], selected[ss], outdir, subtitle, salt_suffix)
        _plot_loss(ranked_by_salt[ss], cache, seeds, base_seed, loss_dir, subtitle,
                   ss, salt_suffix)
    _plot_bars(kind, per_salt, selected, salts, outdir, subtitle)
    if kind == "architecture":
        _plot_convergence(per_salt, cache, seeds, base_seed, salts, outdir, subtitle)

    # Reporte a consola del criterio.
    print(f"\nSelección por criterio (menor error medio; empate dentro de ±1σ):")
    for salt in salts:
        ss = _salt_str(salt)
        best = next(r for r in ranked_by_salt[ss] if r["combo_id"] == selected[ss])
        tag = "" if salt is None else f"[salt={salt}] "
        tie = tie_by_salt[ss]
        print(f"  {tag}elegido: combo #{selected[ss]} ({best['label']}) "
              f"= {best['mean_mean']:.2f} ± {best['mean_std']:.2f} px  (rank {best['rank']})")
        if len(tie) > 1:
            print(f"  {tag}indistinguibles (±1σ): combos {sorted(tie)} "
                  f"→ no hay ganador único que la varianza respalde")
    print(f"\nSalidas en: {outdir}/  (raw.csv, summary.csv, tablas, barras, loss/, overlay)")
    return {"ranked": ranked_by_salt, "selected": selected, "ties": tie_by_salt,
            "n_trained": n_trained, "n_skipped": n_skipped}
