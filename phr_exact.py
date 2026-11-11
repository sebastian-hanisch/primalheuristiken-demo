"""Exaktes Verfahren als Referenz: HiGHS-MILP (scipy.optimize.milp) und Aufzählung für kleine Instanzen.

Beides ist NICHT der Code der Heuristiken; die Orakel-Tests vergleichen zusätzlich mit CP-SAT (nur Test-Umgebung)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def solve_exact(m, time_limit=60):
    """Optimum per HiGHS-MILP: (Zielwert, x, Ergebnisobjekt); bei Misserfolg (None, None, res)."""
    cons = []
    if m.A.shape[0]:
        cons.append(LinearConstraint(m.A, -np.inf, m.b))
    if m.Aeq.shape[0]:
        cons.append(LinearConstraint(m.Aeq, m.beq, m.beq))
    res = milp(m.c, constraints=cons, integrality=m.is_int.astype(int), bounds=Bounds(m.lb, m.ub),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status != 0:
        return None, None, res
    x = res.x.copy()
    x[m.ints] = np.round(x[m.ints])
    return float(res.fun), x, res


def enumerate_optimum(m, limit=2 ** 20):
    """Optimum durch Aufzählung aller 0/1-Belegungen der ganzzahligen Variablen (nur reine 0/1-Instanzen ohne stetige Variablen)."""
    if m.has_cont:
        raise ValueError("Aufzählung nur ohne stetige Variablen")
    k = len(m.ints)
    if 2 ** k > limit:
        raise ValueError(f"2^{k} Belegungen sind zu viele")
    best, best_x = None, None
    chunk = 1 << min(k, 16)
    for start in range(0, 2 ** k, chunk):
        idx = np.arange(start, min(start + chunk, 2 ** k))
        X = ((idx[:, None] >> np.arange(k)) & 1).astype(float)
        ok = np.ones(len(idx), bool)
        if m.A.shape[0]:
            ok &= (X @ m.A.T <= m.b + 1e-9).all(axis=1)
        if m.Aeq.shape[0]:
            ok &= (np.abs(X @ m.Aeq.T - m.beq) <= 1e-9).all(axis=1)
        if not ok.any():
            continue
        vals = X[ok] @ m.c
        j = int(np.argmin(vals))
        if best is None or vals[j] < best - 1e-9:
            best, best_x = float(vals[j]), X[ok][j]
    return best, best_x


def exact_value(m, time_limit=60):
    """Optimalwert (oder None, wenn nicht in der Zeit bewiesen); die kurze Form für Messungen."""
    return solve_exact(m, time_limit)[0]
