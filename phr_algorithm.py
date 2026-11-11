"""Zusammenstellung der Verfahren: benannte Varianten und die Kaskade. Dünn: nur Aufrufe in der richtigen Reihenfolge, keine eigene Regel.

Namen der Varianten (wie in der Landkarte): R_lock, R_near (θ = 0,5), R_prop (θ = 0,5), D_frac (20 Rückschritte), D_fracP (mit Propagation), D_coefP, FP_rs (Pump mit
Neustarts), FP_objrs (Pump mit Zielgewicht und Neustarts) und K1 (Kaskade R_prop → D_fracP → FP_rs)."""

from __future__ import annotations

import numpy as np

from phr_diving import dive
from phr_lp import frac_indices
from phr_pump import pump
from phr_result import Params, Result, Step
from phr_rounding import begin, round_lock, round_near, round_prop, solve_root

METHODS = ("R_lock", "R_near", "R_prop", "D_frac", "D_fracP", "D_coefP", "FP_rs", "FP_objrs", "K1")
CASCADE_ORDER = ("R_prop", "D_fracP", "FP_rs")

METHOD_LABELS = {
    "R_lock": "Runden nach Sperren",
    "R_near": "Runden mit Schwelle θ = 0,5 (nächste ganze Zahl)",
    "R_prop": "Runden mit Propagation",
    "D_frac": "Tauchen (Fractional, 20 Rückschritte)",
    "D_fracP": "Tauchen mit Propagation (20 Rückschritte)",
    "D_coefP": "Tauchen nach Sperren (Coefficient) mit Propagation",
    "FP_rs": "Pump mit Neustarts",
    "FP_objrs": "Pump mit Zielgewicht und Neustarts",
    "K1": "Kaskade Runden → Tauchen → Pump",
}


def run_named(m, root, name, seed=0, trace=False):
    """Eine benannte Variante mit den Standardeinstellungen der Vormessung ausführen (ohne Kaskade)."""
    if name == "R_lock":
        return round_lock(m, root, trace=trace)
    if name == "R_near":
        return round_near(m, root, 0.5, trace=trace)
    if name == "R_prop":
        return round_prop(m, root, 0.5, trace=trace)
    if name == "D_frac":
        return dive(m, root, Params(rule="frac", propagate=False), trace=trace)
    if name == "D_fracP":
        return dive(m, root, Params(rule="frac", propagate=True), trace=trace)
    if name == "D_coefP":
        return dive(m, root, Params(rule="coef", propagate=True), trace=trace)
    if name == "FP_rs":
        return pump(m, root, Params(objective_weight=0.0), seed=seed, trace=trace)
    if name == "FP_objrs":
        return pump(m, root, Params(objective_weight=1.0), seed=seed, trace=trace)
    raise ValueError(f"unbekannte Variante: {name}")


def combine_cascade(results, order=CASCADE_ORDER):
    """Kaskade aus bereits gelaufenen Einzelverfahren zusammensetzen: gelöste LPs = Wurzel einmal + (LPs − 1) je gelaufener Stufe.
    Rückgabe (gefunden, Zielwert, LPs, Stufe); Stufe 0 bis len(order) − 1, bei Misserfolg len(order)."""
    lps = 1
    for stage, name in enumerate(order):
        r = results[name]
        lps += r.lps - 1
        if r.found:
            return True, r.obj, lps, stage
    return False, None, lps, len(order)


def cascade(m, root, seed=0, params=Params(), trace=False):
    """Kaskade: Runden mit Propagation; nur wenn das scheitert Tauchen mit Propagation; nur wenn das scheitert der Pump mit Neustarts.
    Die Wurzel-LP wird einmal gezählt. `info["stage"]` ist die Stufe, die gelöst hat (0, 1, 2), bei Misserfolg 3."""
    stages = [
        ("R_prop", "Stufe 0: Runden mit Propagation (kein weiteres LP)", lambda: round_prop(m, root, params.theta, trace=trace)),
        ("D_fracP", "Stufe 1: Tauchen mit Propagation und Rückschritten", lambda: dive(m, root, Params(rule="frac", propagate=True, backtracks=params.backtracks,
                                                                                                         max_lps=params.max_lps), trace=trace)),
        ("FP_rs", "Stufe 2: Feasibility Pump mit Neustarts", lambda: pump(m, root, params, seed=seed, trace=trace)),
    ]
    ctr, x0 = begin(root)
    lps, iters = 1, root[1]
    steps, ran = [], []
    for index, (name, title, run) in enumerate(stages):
        r = run()
        ran.append(r)
        lps += r.lps - 1
        iters += max(r.iters - root[1], 0)
        if trace:
            steps.append(Step("stage", title, x0.copy(), m.lb.copy(), m.ub.copy(), lps, {"stage": index, "found": r.found, "method": name}))
            steps.extend(r.steps)
        if r.found:
            return Result("K1", True, r.x, r.obj, lps, iters, None, {"stage": index, "stage_results": ran}, steps)
    return Result("K1", False, None, None, lps, iters, "keine Stufe fand eine Lösung", {"stage": len(stages), "stage_results": ran}, steps)


def run_all(m, seed=0, names=METHODS[:-1]):
    """Alle Einzelverfahren auf einer Instanz (Wurzel einmal gelöst); None, wenn die Relaxation unzulässig ist. Für Messungen und die Landkarte."""
    root = solve_root(m)
    if root is None:
        return None
    return {name: run_named(m, root, name, seed) for name in names}


def has_fractional_root(m, root):
    """Hat die Wurzel-LP gebrochene ganzzahlige Variablen? Nur dort gibt es etwas zu runden."""
    return len(frac_indices(m, root[0])) > 0


def to_array(values):
    return np.asarray(values, float)
