"""Zulässigkeitsprüfer: kennt keine Heuristik, keine Propagation und keine Sperren, nur die Zeilen der Instanz (numpy).

Jede Heuristik der Demo meldet eine Lösung erst, wenn dieser Prüfer sie annimmt; die Orakel-Tests prüfen ihn zusätzlich gegen die Definition der
Familien (Zeilen aus den Rohdaten der Instanz, nicht aus den Matrizen)."""

from __future__ import annotations

import numpy as np

TOL = 1e-6


def violations(m, x, tol=TOL):
    """Alle verletzten Bedingungen von x als Texte; leer heißt zulässig und ganzzahlig an den ganzzahligen Variablen."""
    x = np.asarray(x, float)
    out = []
    low = np.where(x < m.lb - tol)[0]
    high = np.where(x > m.ub + tol)[0]
    for j in low:
        out.append(f"{m.names[j]} = {x[j]:.3f} liegt unter der unteren Schranke {m.lb[j]:g}")
    for j in high:
        out.append(f"{m.names[j]} = {x[j]:.3f} liegt über der oberen Schranke {m.ub[j]:g}")
    if m.A.shape[0]:
        act = m.A @ x
        for i in np.where(act - m.b > tol)[0]:
            out.append(f"Zeile {i + 1} ({m.senses[i]}): Aktivität {act[i]:.3f} verletzt die Grenze {m.b[i]:g}")
    if m.Aeq.shape[0]:
        act = m.Aeq @ x
        for i in np.where(np.abs(act - m.beq) > tol)[0]:
            out.append(f"Gleichung {i + 1}: Aktivität {act[i]:.3f} statt {m.beq[i]:g}")
    xi = x[m.ints]
    for k in np.where(np.abs(xi - np.round(xi)) > tol)[0]:
        out.append(f"{m.names[m.ints[k]]} = {xi[k]:.3f} ist nicht ganzzahlig")
    return out


def is_feasible(m, x, tol=TOL):
    """Wahr genau dann, wenn x alle Zeilen, Schranken und die Ganzzahligkeit erfüllt."""
    return not violations(m, x, tol)


def objective(m, x):
    return float(m.c @ np.asarray(x, float))
