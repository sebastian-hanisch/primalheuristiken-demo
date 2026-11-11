"""LP-Löser mit Zähler (HiGHS über scipy.optimize.linprog) und Schranken-Propagation.

Zähleinheit der ganzen Demo: **gelöste LPs** (die Wurzel wird bei jedem Verfahren mitgezählt; ein Knoten des Suchbaums ist genau ein gelöstes LP).
Simplex-Iterationen werden mitgeschrieben, aber nie mit LPs vermischt (ein Knoten des Suchbaums ist ein LP)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

TOL = 1e-6


class LpCounter:
    """Zählt gelöste LPs, deren Simplex-Iterationen und die unzulässigen unter ihnen."""

    def __init__(self):
        self.lps = 0
        self.iters = 0
        self.infeasible = 0


def solve_lp(m, lb, ub, ctr, c=None):
    """LP-Relaxation mit den Schranken lb/ub (und optional anderer Zielfunktion c); None, wenn unzulässig. Zählt genau ein LP."""
    res = linprog(m.c if c is None else c, A_ub=m.A if m.A.shape[0] else None, b_ub=m.b if m.A.shape[0] else None,
                  A_eq=m.Aeq if m.Aeq.shape[0] else None, b_eq=m.beq if m.Aeq.shape[0] else None,
                  bounds=np.column_stack([lb, ub]), method="highs")
    ctr.lps += 1
    ctr.iters += int(getattr(res, "nit", 0) or 0)
    if res.status != 0:
        ctr.infeasible += 1
        return None
    return res.x


def frac_indices(m, x):
    """Ganzzahlige Variablen mit gebrochenem LP-Wert (Rauschgrenze TOL)."""
    xi = x[m.ints]
    return m.ints[np.abs(xi - np.round(xi)) > TOL]


def _triples(m):
    """Alle Zeilen als G x ≤ h (eine Gleichung als zwei Zeilen), als Koordinatenliste für die Propagation; einmal je Instanz berechnet."""
    if m._g is None:
        G = np.vstack([m.A, m.Aeq, -m.Aeq])
        r, cidx = np.nonzero(G)
        m._g = (r, cidx, G[r, cidx], np.concatenate([m.b, m.beq, -m.beq]), G.shape[0])
    return m._g


def propagate(m, lb, ub, max_pass=60):
    """Aktivitätsbasierte Schranken-Propagation: aus der kleinsten möglichen Aktivität jeder Zeile folgt für jede Variable eine schärfere Grenze;
    ganzzahlige Grenzen werden gerundet. Rückgabe (ok, lb, ub) mit neuen Arrays; ok ist falsch, wenn eine Zeile unerfüllbar wird."""
    r, cidx, v, h, m_rows = _triples(m)
    lb = lb.copy()
    ub = ub.copy()
    pos = v > 0
    for _ in range(max_pass):
        lo_term = np.where(pos, v * lb[cidx], v * ub[cidx])
        minact = np.bincount(r, weights=lo_term, minlength=m_rows)
        if (minact > h + 1e-7).any():
            return False, lb, ub
        resid = h[r] - (minact[r] - lo_term)          # Platz für a_j · x_j
        newb = resid / v
        upd_ub = np.full(m.n, np.inf)
        np.minimum.at(upd_ub, cidx[pos], newb[pos])    # a > 0: x_j ≤ newb
        upd_lb = np.full(m.n, -np.inf)
        np.maximum.at(upd_lb, cidx[~pos], newb[~pos])  # a < 0: x_j ≥ newb
        upd_ub = np.where(m.is_int, np.floor(upd_ub + 1e-7), upd_ub)
        upd_lb = np.where(m.is_int, np.ceil(upd_lb - 1e-7), upd_lb)
        nub = np.minimum(ub, upd_ub)
        nlb = np.maximum(lb, upd_lb)
        changed = bool((nub < ub - 1e-9).any() or (nlb > lb + 1e-9).any())
        ub, lb = nub, nlb
        if (lb > ub + 1e-7).any():
            return False, lb, ub
        if not changed:
            break
    return True, lb, ub
