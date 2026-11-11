"""Darstellungshilfen der App (Streamlit): Schrittansicht, Legende, Tabellen und der Ablauf einer Messung auf Knopfdruck.

Die Rechnung steht nicht hier, sondern in phr_evaluation.py und den Verfahrensmodulen; dieses Modul formatiert nur."""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

import phr_visualization as V
from phr_result import de

LEGEND = (":gray[■] frei, Wert 0 · :green[■] frei, Wert 1 · :orange[■] gebrochen · :blue[■] fixiert (heller 0, dunkler 1) · :violet[■] in diesem Schritt fixiert bzw. blockiert")


def show_legend():
    st.caption(LEGEND)


def step_slider(key, n):
    """Schrittregler 0 … n − 1 mit begrenztem Wert (Instanzwechsel kann n verkleinern); bei nur einem Schritt gibt es keinen Regler."""
    if n <= 1:
        return 0
    state_key = f"step_{key}"
    st.session_state[state_key] = min(max(int(st.session_state.get(state_key, 0)), 0), n - 1)
    return st.slider("Schritt", 0, n - 1, key=state_key, help="Zeigt Schritt für Schritt, was das Verfahren tut; die Pfeiltasten bewegen den Regler.")


def reset_steps(keys):
    for k in keys:
        st.session_state[f"step_{k}"] = 0


def values_and_codes(m, step):
    """Wert und Statuscode je ganzzahlige Variable für die Gitteransicht eines Schritts."""
    ints = m.ints
    fixed = (step.ub[ints] - step.lb[ints]) <= 1e-9
    values = np.where(fixed, step.lb[ints], step.x[ints])
    highlight = set(step.data.get("new_fixed", []))
    for key in ("chosen", "blocked"):
        if step.data.get(key) is not None:
            highlight.add(int(step.data[key]))
    return values, V.status_codes(m, values, step.lb, step.ub, highlight)


def show_step(m, steps, key, idx, pump_distances=None):
    """Ein Schritt: Text (je nach Art), Kennzahlen und Gitteransicht (beim Pump LP-Lösung x* und Rundung x~ nebeneinander, darunter der Abstand je Iteration)."""
    step = steps[idx]
    values, codes = values_and_codes(m, step)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Schritt", f"{idx + 1} von {len(steps)}")
    c2.metric("Gelöste LPs bis hier", step.lps)
    c3.metric("Gebrochene Variablen", sum(1 for c in codes if c == 2))
    c4.metric("Fixierte Variablen", sum(1 for c in codes if c in (3, 4, 5)))
    text = step.text
    if step.kind == "fail":
        st.error(text)
    elif step.kind == "done":
        st.success(text)
    elif step.kind == "stage":
        st.markdown(f"**{text}**")
    else:
        st.info(text)
    if pump_distances is not None and "target" in step.data:
        target = np.asarray(step.data["target"])
        left, right = st.columns(2)
        left.plotly_chart(V.build_grid(m, values, codes, "LP-Lösung x*"), width="stretch", key=f"grid_{key}_{idx}_a")
        right.plotly_chart(V.build_grid(m, target, V.status_codes(m, target), "Rundung x~ (Ziel des nächsten LPs)"), width="stretch", key=f"grid_{key}_{idx}_b")
        its = sum(1 for s in steps[: idx + 1] if s.kind in ("pump", "stall", "cycle", "restart"))
        restarts_at = [i + 1 for i, s in enumerate([s for s in steps if s.kind in ("pump", "stall", "cycle", "restart")]) if s.kind == "restart"]
        if pump_distances:
            st.plotly_chart(V.build_pump_distance(pump_distances, its, restarts_at), width="stretch", key=f"dist_{key}_{idx}")
    else:
        st.plotly_chart(V.build_grid(m, values, codes), width="stretch", key=f"grid_{key}_{idx}")


def result_line(res, gap):
    """Ein Satz zum Ergebnis eines Verfahrens: zulässig (mit Lücke und LPs) oder Grund des Scheiterns."""
    if res.found:
        g = "" if gap is None else f", Lücke {de(gap, 1)} %"
        return f"✅ zulässige Lösung{g}, {res.lps} {'LP' if res.lps == 1 else 'LPs'}"
    return f"❌ scheitert ({res.fail}), {res.lps} {'LP' if res.lps == 1 else 'LPs'}"


def run_on_click(key, label, args, fn, hint):
    """Teure Messung auf Knopfdruck: Ergebnis im Session State, solange die Argumente (Familie, Größe, …) gleich bleiben; zeigt Spinner und gemessene Zeit.
    Gibt das Ergebnis (oder None) zurück."""
    state_key = f"exp_{key}"
    if st.button(label, key=f"run_{key}", help=hint):
        with st.spinner(f"Messung läuft ({hint}) …"):
            t0 = time.time()
            result = fn(*args)
            st.session_state[state_key] = {"args": args, "result": result, "seconds": time.time() - t0}
    stored = st.session_state.get(state_key)
    if stored and stored["args"] == args:
        st.caption(f"Gerechnet in {de(stored['seconds'], 1)} s.")
        return stored["result"]
    if stored:
        st.caption("Die zuletzt gezeigte Messung gehörte zu anderen Einstellungen: erneut starten.")
    else:
        st.caption(f"Noch nicht gemessen ({hint}).")
    return None


def percent(v, digits=0):
    return "–" if v is None else de(100 * v, digits)


def number(v, digits=1):
    return "–" if v is None else de(v, digits)
