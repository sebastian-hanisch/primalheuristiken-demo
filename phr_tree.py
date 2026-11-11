"""Einfacher Suchbaum als Vergleich: Tiefensuche und Suche nach bester Schranke zuerst, ohne Schnitte und ohne Presolve.

Knoten = ein gelöstes LP (Wurzel eingeschlossen; bei der Suche nach bester Schranke zählt auch jedes unzulässige Kind-LP). Verzweigt wird auf die Variable, die am
nächsten an 0,5 liegt. `incumbent` ist der Zielwert einer bekannten Lösung (Startlösung); `node_limit` begrenzt die LPs. Der Beweis (`proved`) gilt nur, wenn der
Baum im Limit vollständig abgesucht wurde."""

from __future__ import annotations

import heapq
import math

import numpy as np

from phr_lp import LpCounter, frac_indices, solve_lp


def _pruned(m, z, best):
    return math.ceil(z - 1e-6) >= best if m.obj_integral else z >= best - 1e-6


def _branch_var(m, x):
    fr = frac_indices(m, x)
    d = np.abs(x[fr] - np.floor(x[fr]) - 0.5)
    return fr[int(np.argmin(d))]


def depth_first(m, incumbent=math.inf, node_limit=1000):
    """Tiefensuche; zuerst in Richtung der nächsten ganzen Zahl. Rückgabe: nodes (LPs), best (Zielwert oder None), proved (Baum vollständig abgesucht)."""
    best = incumbent
    ctr = LpCounter()
    stack = [(m.lb.copy(), m.ub.copy())]
    while stack and ctr.lps < node_limit:
        lb, ub = stack.pop()
        x = solve_lp(m, lb, ub, ctr)
        if x is None:
            continue
        z = float(m.c @ x)
        if _pruned(m, z, best):
            continue
        if not len(frac_indices(m, x)):
            best = z
            continue
        j = _branch_var(m, x)
        fl = math.floor(x[j])
        lo_l, lo_u, hi_l, hi_u = lb.copy(), ub.copy(), lb.copy(), ub.copy()
        lo_u[j] = fl
        hi_l[j] = fl + 1
        if (x[j] - fl) >= 0.5:
            stack += [(lo_l, lo_u), (hi_l, hi_u)]
        else:
            stack += [(hi_l, hi_u), (lo_l, lo_u)]
    return {"nodes": ctr.lps, "best": best if best < math.inf else None, "proved": not stack}


def best_bound(m, incumbent=math.inf, node_limit=1000):
    """Suche nach bester Schranke zuerst (Halde). Rückgabe wie `depth_first`."""
    best = incumbent
    ctr = LpCounter()
    heap, count = [], 0

    def push(x, lb, ub):
        nonlocal best, count
        z = float(m.c @ x)
        if _pruned(m, z, best):
            return
        if not len(frac_indices(m, x)):
            best = z
            return
        count += 1
        heapq.heappush(heap, (z, count, lb, ub, x))

    x = solve_lp(m, m.lb, m.ub, ctr)
    if x is not None:
        push(x, m.lb.copy(), m.ub.copy())
    while heap and ctr.lps < node_limit:
        z, _, lb, ub, x = heapq.heappop(heap)
        if _pruned(m, z, best):
            continue
        j = _branch_var(m, x)
        fl = math.floor(x[j])
        for lo, hi in ((lb[j], fl), (fl + 1, ub[j])):
            l2, u2 = lb.copy(), ub.copy()
            l2[j], u2[j] = lo, hi
            xc = solve_lp(m, l2, u2, ctr)
            if xc is not None:
                push(xc, l2, u2)
    proved = all(_pruned(m, h[0], best) for h in heap)
    return {"nodes": ctr.lps, "best": best if best < math.inf else None, "proved": proved}


STRATEGIES = {"Tiefensuche": depth_first, "Beste Schranke zuerst": best_bound}
