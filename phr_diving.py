"""LP-Tauchen: eine Variable festsetzen, das LP neu lösen, wiederholen, bis die LP-Lösung ganzzahlig ist.

Wird das LP unzulässig, gibt es einen chronologischen Rückschritt (die zuletzt festgesetzte, noch nicht umgekehrte Variable bekommt die andere Richtung;
das ist zuerst die eben gescheiterte Fixierung selbst, danach frühere), höchstens `backtracks` Mal. Mit `propagate=True` werden nach jedem Festsetzen die Schranken propagiert (Widerspruch = unzulässig, ohne ein LP zu lösen).

Tauchregeln (`rule`):
* `frac`    Fractional Diving: die Variable mit dem kleinsten Abstand zur nächsten ganzen Zahl, in diese Richtung.
* `coef`    Coefficient Diving: die Variable mit den wenigsten Sperren in der gewählten Richtung (die Richtung mit weniger Sperren), bei Gleichstand frac.
* `guided`  Guided Diving: die Variable, die am nächsten an einer Referenzlösung `ref` liegt, auf deren Wert."""

from __future__ import annotations

import math

import numpy as np

from phr_lp import TOL, frac_indices, propagate, solve_lp
from phr_result import Params, Step, de, make_result
from phr_rounding import begin


def dive_candidates(m, x, rule, ref=None):
    """Kandidaten (Rang, Variable, Zielwert), sortiert; der erste wird festgesetzt. Rang als Tupel, danach Variablennummer (Gleichstand)."""
    out = []
    for j in frac_indices(m, x):
        fl = math.floor(x[j])
        dn, up = x[j] - fl, fl + 1 - x[j]
        if rule == "frac":
            val = fl + 1 if up <= dn else fl
            score = (min(dn, up),)
        elif rule == "coef":
            ul, dl = m.uplocks[j], m.downlocks[j]
            if ul < dl:
                val, lk = fl + 1, ul
            elif dl < ul:
                val, lk = fl, dl
            else:
                val, lk = (fl + 1, ul) if up <= dn else (fl, dl)
            score = (lk, up if val == fl + 1 else dn)
        elif rule == "guided":
            if ref is None:
                raise ValueError("Guided Diving braucht eine Referenzlösung")
            val = ref[j]
            score = (abs(x[j] - ref[j]),)
        else:
            raise ValueError(f"unbekannte Tauchregel: {rule}")
        out.append((score, int(j), float(val)))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _snap(kind, text, x, lb, ub, lps, **data):
    return Step(kind, text, x.copy(), lb.copy(), ub.copy(), lps, data)


def dive(m, root, params=Params(), trace=False, ref=None):
    """LP-Tauchen mit Rückschritten (siehe Kopf der Datei); `params.rule`, `.backtracks`, `.propagate`, `.max_lps`."""
    name = {"frac": "D_frac", "coef": "D_coef", "guided": "D_guided"}[params.rule] + ("P" if params.propagate else "")
    ctr, x = begin(root)
    lb, ub = m.lb.copy(), m.ub.copy()
    steps = []
    if trace:
        steps.append(_snap("root", f"Wurzel-LP gelöst (1 LP): {len(frac_indices(m, x))} gebrochene Variablen.", x, lb, ub, 1))
    stack = []           # (lb_vor, ub_vor, Variable, Wert, umgekehrt)
    backtracks = 0
    depth_max = 0
    while True:
        if ctr.lps >= params.max_lps:
            if trace:
                steps.append(_snap("fail", f"Abbruch: {params.max_lps} LPs erreicht.", x, lb, ub, ctr.lps))
            return make_result(m, name, ctr, None, "LP-Grenze", {"backtracks": backtracks, "depth": depth_max}, steps)
        cand = dive_candidates(m, x, params.rule, ref)
        if not cand:
            xs = x.copy()
            xs[m.ints] = np.round(xs[m.ints])
            res = make_result(m, name, ctr, xs, "Zeile verletzt", {"backtracks": backtracks, "depth": depth_max}, steps)
            if trace:
                steps.append(_snap("done", f"Die LP-Lösung ist ganzzahlig: fertig nach {ctr.lps} LPs und {backtracks} Rückschritten.", xs, lb, ub, ctr.lps))
            return res
        _, j, val = cand[0]
        pre_l, pre_u = lb.copy(), ub.copy()
        l2, u2 = lb.copy(), ub.copy()
        l2[j] = u2[j] = val
        okp = True
        if params.propagate:
            okp, l2, u2 = propagate(m, l2, u2)
        x2 = solve_lp(m, l2, u2, ctr) if okp else None
        if x2 is not None:
            stack.append((pre_l, pre_u, j, val, False))
            extra = [int(k) for k in m.ints if (u2[k] - l2[k]) <= 1e-9 and (pre_u[k] - pre_l[k]) > 1e-9 and k != j]
            old_value = float(x[j])
            lb, ub, x = l2, u2, x2
            depth_max = max(depth_max, len(stack))
            if trace:
                text = f"{m.names[j]} (LP-Wert {de(old_value)}) wird auf {int(val)} festgesetzt"
                text += f"; die Propagation fixiert {len(extra)} weitere Variablen" if params.propagate else ""
                steps.append(_snap("fix", f"{text}. LP {ctr.lps} gelöst: {len(frac_indices(m, x))} gebrochene Variablen übrig.", x, lb, ub, ctr.lps,
                                   chosen=j, value=float(val), new_fixed=extra))
            continue
        if trace:
            steps.append(_snap("infeasible", f"{m.names[j]} = {int(val)}: " + ("die Propagation zeigt einen Widerspruch (ohne LP)." if not okp else "das LP ist unzulässig."),
                               x, pre_l, pre_u, ctr.lps, chosen=j, value=float(val)))
        if params.flip_failed:                       # die gescheiterte Fixierung selbst kommt zuerst an die Reihe (Abweichung von der Vormessung, siehe README)
            stack.append((pre_l, pre_u, j, val, False))
        while True:                                  # Rückschritt
            if not stack or backtracks >= params.backtracks:
                if trace:
                    steps.append(_snap("fail", "Sackgasse: " + ("kein Rückschritt mehr möglich." if not stack else f"das Budget von {params.backtracks} Rückschritten ist aufgebraucht."),
                                       x, lb, ub, ctr.lps))
                return make_result(m, name, ctr, None, "Sackgasse", {"backtracks": backtracks, "depth": depth_max}, steps)
            lb_p, ub_p, jj, vv, flipped = stack.pop()
            if flipped:
                continue
            backtracks += 1
            other = 1.0 - vv if (m.ub[jj] - m.lb[jj] == 1) else vv
            l2, u2 = lb_p.copy(), ub_p.copy()
            l2[jj] = u2[jj] = other
            okp = True
            if params.propagate:
                okp, l2, u2 = propagate(m, l2, u2)
            x2 = solve_lp(m, l2, u2, ctr) if okp else None
            if x2 is None:
                if trace:
                    steps.append(_snap("backtrack", f"Rückschritt {backtracks}: {m.names[jj]} auf {int(other)} ist ebenfalls unzulässig, noch eine Ebene zurück.", x, lb_p, ub_p, ctr.lps,
                                       chosen=jj, value=float(other)))
                continue
            stack.append((lb_p, ub_p, jj, other, True))
            lb, ub, x = l2, u2, x2
            if trace:
                steps.append(_snap("backtrack", f"Rückschritt {backtracks}: {m.names[jj]} bekommt stattdessen den Wert {int(other)}. LP {ctr.lps} gelöst: "
                                                f"{len(frac_indices(m, x))} gebrochene Variablen.", x, lb, ub, ctr.lps, chosen=jj, value=float(other)))
            break
