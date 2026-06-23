"""
study_selection.py — criterio de selección ÚNICO, compartido por los cuatro estudios.

Métrica primaria: error MEDIO de píxel sobre los 32 patrones (en DAE, contra el patrón
limpio), promediado sobre seeds. Menor = mejor. Peor caso y nº de patrones <=1px se
reportan como descriptivos, NO como criterio.

Desempate por banda: si la banda ±1σ (σ entre seeds) del mejor se solapa con la de otra
combinación, son estadísticamente INDISTINGUIBLES. La regla secundaria (tiebreaker)
elige DENTRO de ese conjunto; nunca se canta un ganador que la varianza no respalda.
"""


def rank_and_select(summaries, *, tiebreaker=None):
    """summaries: lista de dicts con al menos 'combo_id', 'mean_mean', 'mean_std'.

    Devuelve (ranked, best_id, tie_ids):
        ranked   : misma lista ordenada por mean_mean asc, con 'rank' (1..N) agregado.
        tie_ids  : combos cuya banda ±1σ se solapa con la del mejor (indistinguibles).
        best_id  : el seleccionado. rank 1 salvo que tiebreaker elija otro del tie set.
    """
    ranked = sorted(summaries, key=lambda r: r["mean_mean"])
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    if not ranked:
        return ranked, None, []

    best = ranked[0]
    hi_best = best["mean_mean"] + best["mean_std"]
    # r (mean >= best.mean) se solapa con el mejor si su borde inferior cae bajo el
    # borde superior del mejor.
    tie = [r for r in ranked if (r["mean_mean"] - r["mean_std"]) <= hi_best]
    tie_ids = [r["combo_id"] for r in tie]

    chosen = best
    if tiebreaker is not None and len(tie) > 1:
        chosen = tiebreaker(tie) or best
    return ranked, chosen["combo_id"], tie_ids


def simplest_tiebreaker(tie):
    """Arquitectura: ante empate, la más simple (menos parámetros)."""
    return min(tie, key=lambda r: (r.get("n_params", float("inf")), r["rank"]))


def standard_tiebreaker(tie):
    """Hiperparámetros: ante empate, preferir la config estándar (adam, lr=1e-3, he);
    si no está en el empate, el de mejor rank."""
    standard = [r for r in tie
                if r.get("opt") == "adam" and abs(float(r.get("lr", 0)) - 1e-3) < 1e-12
                and r.get("init") == "he"]
    return standard[0] if standard else min(tie, key=lambda r: r["rank"])
