"""Runden einer LP-Lösung zu einer ganzzahligen: drei Verfahren, jedes eine kleine Einheit.

* `round_lock`  Rundung nach Sperren: jede gebrochene Variable nur in eine Richtung, in der keine Zeile verletzt werden kann.
* `round_near`  Rundung mit Schwelle θ mit Schwelle θ (kein Blick auf die Zeilen).
* `round_prop`  Sequentielles Runden mit Propagation: eine Variable nach der anderen runden, Schranken propagieren, bei Widerspruch die andere Richtung.

Alle geben ein `Result` zurück; die Wurzel-LP wird vom Aufrufer gelöst und als `root = (x, Iterationen)` übergeben (gezählt als ein LP)."""

from __future__ import annotations

import math

import numpy as np

import phr_check as K
from phr_lp import LpCounter, TOL, frac_indices, propagate, solve_lp
from phr_result import Step, de, make_result


def begin(root):
    """Zähler nach der Wurzel (ein LP) und eine Kopie der Wurzellösung."""
    ctr = LpCounter()
    ctr.lps = 1
    ctr.iters = root[1]
    return ctr, root[0].copy()


def solve_root(m):
    """Wurzel-LP: (x, Simplex-Iterationen) oder None, wenn die Relaxation unzulässig ist."""
    ctr = LpCounter()
    x = solve_lp(m, m.lb, m.ub, ctr)
    return None if x is None else (x, ctr.iters)


def rounding_direction(m, j):
    """Welche Richtung die Sperren der Variable j erlauben: 'frei' (beide), 'ab', 'auf' oder 'keine'."""
    down_ok, up_ok = m.downlocks[j] == 0, m.uplocks[j] == 0
    if down_ok and up_ok:
        return "frei"
    if down_ok:
        return "ab"
    if up_ok:
        return "auf"
    return "keine"


def lock_table(m, x):
    """Je gebrochener ganzzahliger Variable: Name, LP-Wert, Sperren abwärts und aufwärts, erlaubte Richtung (für die Anzeige und den Test)."""
    return [{"j": int(j), "name": m.names[j], "value": float(x[j]), "down": int(m.downlocks[j]), "up": int(m.uplocks[j]),
             "direction": rounding_direction(m, j)} for j in frac_indices(m, x)]


def cover_theta(m):
    """Klassische Überdeckungs-Schwelle 1/f (f = Länge der längsten ≥-Zeile); ohne ≥-Zeile 0,5."""
    f = 0
    for i in range(m.A.shape[0]):
        nz = m.A[i][m.A[i] != 0]
        if len(nz) and (nz < 0).all():
            f = max(f, len(nz))
    return 1.0 / f if f else 0.5


def _finish_fixed(m, ctr, vals):
    """Ganzzahlige Variablen auf vals festsetzen; stetige Variablen per LP (1 LP), ohne stetige Variablen ohne LP."""
    if not m.has_cont:
        x = np.zeros(m.n)
        x[m.ints] = vals
        return x
    lb, ub = m.lb.copy(), m.ub.copy()
    lb[m.ints] = vals
    ub[m.ints] = vals
    return solve_lp(m, lb, ub, ctr)


def _snap(kind, text, x, lb, ub, lps, **data):
    return Step(kind, text, x.copy(), lb.copy(), ub.copy(), lps, data)


# ------------------------------------------------------------------------------------------------------------------ Sperren-Rundung
def round_lock(m, root, trace=False):
    """Rundung nach Sperren: gebrochene Variablen nur in eine Richtung runden, die keine Zeile verletzen kann (kein LP über die Wurzel hinaus)."""
    ctr, x = begin(root)
    steps = []
    table = lock_table(m, x)
    if trace:
        steps.append(_snap("root", f"Wurzel-LP gelöst (1 LP): {len(table)} der {len(m.ints)} ganzzahligen Variablen sind gebrochen.", x, m.lb, m.ub, 1, table=table))
        free = sum(1 for t in table if t["direction"] != "keine")
        steps.append(_snap("locks", f"Sperren: {free} von {len(table)} gebrochenen Variablen haben eine Richtung ohne Sperre, "
                                    f"{len(table) - free} sind in beiden Richtungen gesperrt.", x, m.lb, m.ub, 1, table=table))
    for t in table:
        j = t["j"]
        direction = t["direction"]
        if direction == "keine":
            if trace:
                steps.append(_snap("fail", f"Abbruch bei {m.names[j]} (LP-Wert {de(x[j])}): {t['down']} Zeilen sperren das Abrunden, {t['up']} das Aufrunden. "
                                           "Jede Rundrichtung kann eine Zeile verletzen.", x, m.lb, m.ub, 1, blocked=j))
            return make_result(m, "R_lock", ctr, None, "Sperre", {"blocked": j}, steps)
        if direction == "frei":
            x[j] = math.floor(x[j]) if m.c[j] >= 0 else math.ceil(x[j])
        elif direction == "ab":
            x[j] = math.floor(x[j])
        else:
            x[j] = math.ceil(x[j])
    x[m.ints] = np.round(x[m.ints])
    res = make_result(m, "R_lock", ctr, x, "unzulässig", steps=steps)
    if trace:
        steps.append(_snap("done", "Alle gebrochenen Variablen in ihre freie Richtung gerundet; der Prüfer " + ("nimmt die Lösung an." if res.found else "lehnt sie ab."),
                           x, m.lb, m.ub, 1))
    return res


# ------------------------------------------------------------------------------------------------------------------ nächste ganze Zahl
def round_near(m, root, theta=0.5, trace=False):
    """Rundung mit Schwelle θ: ab dem Nachkommaanteil θ wird aufgerundet, sonst abgerundet; die Zeilen werden nicht angesehen."""
    ctr, x0 = begin(root)
    xi = x0[m.ints]
    vals = np.where(xi - np.floor(xi + TOL) >= theta - 1e-12, np.ceil(xi - TOL), np.floor(xi + TOL))
    vals = np.clip(vals, m.lb[m.ints], m.ub[m.ints])
    xs = _finish_fixed(m, ctr, vals)
    res = make_result(m, "R_near", ctr, xs, "LP unzulässig" if xs is None else "Zeile verletzt")
    if trace:
        steps = [_snap("root", f"Wurzel-LP gelöst (1 LP): {len(frac_indices(m, x0))} gebrochene Variablen.", x0, m.lb, m.ub, 1)]
        shown = xs
        if xs is None:
            shown = x0.copy()
            shown[m.ints] = vals
        viol = K.violations(m, shown) if xs is not None else ["das LP mit den gerundeten Werten ist unzulässig"]
        txt = f"Alle Variablen mit Nachkommaanteil ≥ {de(theta)} aufgerundet, die übrigen abgerundet. "
        txt += "Der Prüfer nimmt die Lösung an." if res.found else f"Der Prüfer lehnt sie ab ({len(viol)} Verletzungen, erste: {viol[0] if viol else '-'})."
        steps.append(_snap("done", txt, shown, m.lb, m.ub, ctr.lps, violations=viol))
        res.steps = steps
    return res


# ------------------------------------------------------------------------------------------------------------------ Runden mit Propagation
def _newly_fixed(m, lb_old, ub_old, lb, ub):
    was = (ub_old - lb_old) <= 1e-9
    now = (ub - lb) <= 1e-9
    return [int(j) for j in m.ints if now[j] and not was[j]]


def round_prop(m, root, theta=0.5, trace=False):
    """Sequentielles Runden mit Propagation: die Variable mit dem kleinsten Abstand zur nächsten ganzen Zahl zuerst runden, Schranken propagieren,
    bei Widerspruch die andere Richtung, sonst Abbruch (kein Rückschritt)."""
    ctr, x = begin(root)
    lb, ub = m.lb.copy(), m.ub.copy()
    steps = []
    ok, lb2, ub2 = propagate(m, lb, ub)
    if not ok:
        if trace:
            steps.append(_snap("fail", "Schon die Propagation an der Wurzel zeigt einen Widerspruch.", x, lb, ub, 1))
        return make_result(m, "R_prop", ctr, None, "Widerspruch an der Wurzel", steps=steps)
    start_fixed = _newly_fixed(m, lb, ub, lb2, ub2)
    lb, ub = lb2, ub2
    if trace:
        steps.append(_snap("root", f"Wurzel-LP gelöst (1 LP): {len(frac_indices(m, x))} gebrochene Variablen. Die Propagation der Anfangsschranken "
                                   f"fixiert schon {len(start_fixed)} Variablen.", x, lb, ub, 1, new_fixed=start_fixed))
    ints = m.ints
    dist = np.abs(x[ints] - np.round(x[ints]))
    order = ints[np.argsort(dist, kind="stable")]
    flips = 0
    for j in order:
        if lb[j] >= ub[j]:
            continue
        fl = math.floor(x[j] + TOL)
        up = (x[j] - fl) >= theta - 1e-12
        first = min(max(fl + 1 if up else fl, lb[j]), ub[j])
        tried = False
        for val in (first, (fl if up else fl + 1)):
            if val < lb[j] - 1e-9 or val > ub[j] + 1e-9:
                continue
            l2, u2 = lb.copy(), ub.copy()
            l2[j] = u2[j] = val
            ok2, l2, u2 = propagate(m, l2, u2)
            if ok2:
                new = _newly_fixed(m, lb, ub, l2, u2)
                if trace:
                    flipped = val != first
                    steps.append(_snap("round", f"{m.names[j]} (LP-Wert {de(x[j])}) wird auf {int(val)} " + ("gesetzt (die erste Richtung widersprach, andere Richtung)" if flipped else "gerundet")
                                       + f"; die Propagation fixiert {len([k for k in new if k != j])} weitere Variablen.", x, l2, u2, 1,
                                       chosen=int(j), value=float(val), flipped=flipped, new_fixed=[k for k in new if k != j]))
                lb, ub = l2, u2
                tried = True
                if val != first:
                    flips += 1
                break
        if not tried:
            if trace:
                steps.append(_snap("fail", f"Sackgasse bei {m.names[j]} (LP-Wert {de(x[j])}): beide Rundrichtungen führen nach der Propagation zu einem Widerspruch. "
                                           "Das Verfahren kennt keinen Rückschritt und bricht ab.", x, lb, ub, 1, blocked=int(j)))
            return make_result(m, "R_prop", ctr, None, "Sackgasse", {"flips": flips, "blocked": int(j)}, steps)
    xs = _finish_fixed(m, ctr, lb[ints])
    res = make_result(m, "R_prop", ctr, xs, "LP unzulässig" if xs is None else "Zeile verletzt", {"flips": flips}, steps)
    if trace:
        shown = xs if xs is not None else x
        res.steps.append(_snap("done", "Alle Variablen fixiert; " + ("der Prüfer nimmt die Lösung an" + (f" ({ctr.lps} LPs, da stetige Variablen zu bestimmen waren)." if m.has_cont else " (ohne ein weiteres LP).")
                                                                       if res.found else "der Prüfer lehnt sie ab."), shown, lb, ub, ctr.lps))
    return res
