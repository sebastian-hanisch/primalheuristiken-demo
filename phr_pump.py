"""Feasibility Pump (Fischetti, Glover, Lodi 2005) mit Neustarts, optional mit Zielgewicht (Objective Feasibility Pump).

Abwechselnd: (1) das LP über den Abstand zur gerundeten Lösung x~ minimieren (Zielfunktion: Σ x_j für x~_j = 0, Σ (1 − x_j) für x~_j = 1, jeweils über die
ganzzahligen Variablen) und (2) die LP-Lösung x* wieder runden. Ist x* ganzzahlig, ist es eine zulässige Lösung. Bleibt die Rundung unverändert (Stillstand),
werden die T Einträge mit dem größten Abstand umgekehrt; wiederholt sie sich nach höchstens drei Schritten (Zyklus), kommt eine Zufallsstörung; findet
der Pump `stall` Iterationen lang keinen kleineren Abstand, folgt ein Neustart (starke Zufallsrundung der Wurzellösung, Zielgewicht zurückgesetzt).
Höchstens `restarts` Neustarts und `maxit` Iterationen (= LPs) insgesamt. Zufall nur über einen SplitMix64-Strom aus dem übergebenen Zufalls-Seed."""

from __future__ import annotations

import math

import numpy as np

from phr_lp import TOL, solve_lp
from phr_result import Params, Step, de, make_result
from phr_rounding import begin
from phr_scenario import SplitMix64


def distance_objective(m, xt, alpha=0.0):
    """Zielfunktion des Pump-LPs: Abstand zur gerundeten Lösung xt über die ganzzahligen Variablen; mit alpha > 0 gemischt mit der Originalzielfunktion
    ((1 − α) · Abstand + α · √|I| / ‖c‖ · c)."""
    cd = np.zeros(m.n)
    cd[m.ints] = np.where(xt < 0.5, 1.0, -1.0)
    if alpha > 0:
        cnorm = np.linalg.norm(m.c) or 1.0
        cd = (1 - alpha) * cd + alpha * math.sqrt(len(m.ints)) / cnorm * m.c
    return cd


def flip_on_stall(xt, xi, flips):
    """Stillstand: die `flips` Einträge mit dem größten Abstand |x*_j − x~_j| umkehren."""
    d = np.abs(xi - xt)
    idx = np.argsort(-d, kind="stable")[:flips]
    new = xt.copy()
    new[idx] = 1 - new[idx]
    return new


def perturb_on_cycle(xi, new, rng):
    """Zyklus: Eintrag j wird umgekehrt, wenn |x*_j − x~_j| + max(ρ_j, 0) > 0,5 mit ρ_j gleichverteilt in [−0,3; 0,7]."""
    rho = np.array([rng.uniform(-0.3, 0.7) for _ in range(len(xi))])
    flip = np.abs(xi - new) + np.maximum(rho, 0) > 0.5
    out = new.copy()
    out[flip] = 1 - out[flip]
    return out


def restart_point(x0i, rng):
    """Neustart: Wurzellösung mit starkem Rauschen (±0,5) auf 0/1 runden."""
    x1 = x0i + np.array([rng.uniform(-0.5, 0.5) for _ in range(len(x0i))])
    return (x1 >= 0.5).astype(float)


def _full(m, xi, x):
    out = x.copy()
    out[m.ints] = xi
    return out


def pump(m, root, params=Params(), seed=0, trace=False):
    """Feasibility Pump mit Neustarts; `params.restarts`, `.objective_weight`, `.flips`, `.stall`, `.maxit`, `.phi`."""
    name = "FP_objrs" if params.objective_weight > 0 else "FP_rs"
    rng = SplitMix64(seed + 7919)
    ctr, x0 = begin(root)
    ints = m.ints
    x0i = x0[ints]
    xt = np.round(x0i).astype(float)
    hist = [tuple(xt)]
    alpha = params.objective_weight
    n_flip = n_cycle = restarts = its = 0
    best_d, since = np.inf, 0
    steps = []
    dists = []
    info = lambda: {"iterations": its, "flip_steps": n_flip, "cycle_steps": n_cycle, "restarts": restarts, "distances": list(dists)}  # noqa: E731
    if trace:
        steps.append(Step("root", f"Wurzel-LP gelöst (1 LP). Erste Rundung x~ der ganzzahligen Variablen: {int((np.abs(x0i - xt) > TOL).sum())} Einträge weichen von der LP-Lösung ab.",
                          x0.copy(), m.lb.copy(), m.ub.copy(), 1, {"target": xt.copy(), "dist": None}))
    while its < params.maxit:
        x = solve_lp(m, m.lb, m.ub, ctr, c=distance_objective(m, xt, alpha))
        its += 1
        if x is None:
            if trace:
                steps.append(Step("fail", "Das Pump-LP ist unzulässig.", x0.copy(), m.lb.copy(), m.ub.copy(), ctr.lps, {"target": xt.copy()}))
            return make_result(m, name, ctr, None, "LP unzulässig", info(), steps)
        xi = x[ints]
        dist = np.abs(xi - np.round(xi))
        if (dist <= TOL).all():
            xs = x.copy()
            xs[ints] = np.round(xi)
            res = make_result(m, name, ctr, xs, "Prüfer", info(), steps)
            if trace:
                steps.append(Step("done", f"Das Pump-LP liefert eine ganzzahlige Lösung (Iteration {its}, {ctr.lps} LPs): " + ("zulässig." if res.found else "der Prüfer lehnt sie ab."),
                                  xs.copy(), m.lb.copy(), m.ub.copy(), ctr.lps, {"target": xt.copy(), "dist": 0.0}))
            return res
        d = float(np.abs(xi - xt).sum())
        dists.append(d)
        if d < best_d - 1e-9:
            best_d, since = d, 0
        else:
            since += 1
        event, note = "pump", ""
        if since >= params.stall and restarts < params.restarts:
            restarts += 1
            event, note = "restart", f" {params.stall} Iterationen ohne kleineren Abstand: Neustart {restarts} (Wurzellösung mit starkem Rauschen gerundet, Zielgewicht zurückgesetzt)."
            xt_new = restart_point(x0i, rng)
            alpha = params.objective_weight
            best_d, since = np.inf, 0
            hist = [tuple(xt_new)]
            if trace:
                steps.append(Step(event, f"Iteration {its}: Abstand {de(d)}." + note, x.copy(), m.lb.copy(), m.ub.copy(), ctr.lps, {"target": xt_new.copy(), "dist": d, "previous": xt.copy()}))
            xt = xt_new
            continue
        new = np.round(xi).astype(float)
        if (new == xt).all():
            new = flip_on_stall(xt, xi, params.flips)
            n_flip += 1
            event, note = "stall", f" Die Rundung ändert sich nicht (Stillstand): die {params.flips} Einträge mit dem größten Abstand werden umgekehrt."
        elif tuple(new) in hist[-3:]:
            new = perturb_on_cycle(xi, new, rng)
            n_cycle += 1
            event, note = "cycle", " Die Rundung wiederholt sich (Zyklus): Zufallsstörung."
        if trace:
            changed = int((new != xt).sum())
            steps.append(Step(event, f"Iteration {its}: Abstand {de(d)} zur Rundung x~, {changed} Einträge der nächsten Rundung ändern sich." + note,
                              x.copy(), m.lb.copy(), m.ub.copy(), ctr.lps, {"target": new.copy(), "dist": d, "previous": xt.copy()}))
        xt = new
        hist.append(tuple(xt))
        alpha *= params.phi
    if trace:
        steps.append(Step("fail", f"Abbruch: {params.maxit} Iterationen erreicht, keine ganzzahlige Lösung gefunden.", x0.copy(), m.lb.copy(), m.ub.copy(), ctr.lps, {"target": xt.copy()}))
    return make_result(m, name, ctr, None, "Iterationsgrenze", info(), steps)
