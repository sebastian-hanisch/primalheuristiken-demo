"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.

Gitteransicht: jede ganzzahlige Variable ist eine Zelle (Zeilen und Spalten der Familie, z. B. Maschinen × Aufträge), die Farbe sagt, was mit ihr ist
(frei und ganzzahlig, gebrochen, fixiert, in diesem Schritt fixiert); die Zahl in der Zelle ist der Wert. Beschriftungen sind Text, damit Plotly keine lineare Achse anlegt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import phr_constants as C
from phr_result import de

# Statuscodes der Gitteransicht (Reihenfolge = Farbskala) und ihre Namen für die Legende
STATUS = ("frei, Wert 0", "frei, Wert 1", "gebrochen", "fixiert auf 0", "fixiert auf 1", "in diesem Schritt fixiert")
WRAP = 20                                          # Zellen je Zeile der Gitteransicht bei langen Einzelzeilen
STATUS_COLORS = ("#d9dce3", "#21c354", "#ff8700", "#8fb7e8", "#1c6fc4", "#a259d6")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def status_codes(m, values, lb=None, ub=None, highlight=(), tol=1e-6):
    """Statuscode je ganzzahliger Variable (Reihenfolge m.ints): fixiert (lb = ub) auf 0/1, in diesem Schritt fixiert (`highlight`), gebrochen oder frei auf 0/1."""
    codes = []
    hl = set(int(j) for j in highlight)
    for k, j in enumerate(m.ints):
        v = values[k]
        fixed = lb is not None and ub is not None and ub[j] - lb[j] <= 1e-9
        if int(j) in hl:
            codes.append(5)
        elif fixed:
            codes.append(4 if v >= 0.5 else 3)
        elif abs(v - round(v)) > tol:
            codes.append(2)
        else:
            codes.append(1 if round(v) == 1 else 0)
    return codes


def build_grid(m, values, codes, title=None, key_height=None):
    """Gitteransicht: `values` und `codes` je ganzzahlige Variable in der Reihenfolge von m.ints."""
    rows, cols = m.grid
    wrap = rows == 1 and cols > WRAP                  # eine lange Zeile (z. B. 60 Mengen) bricht in Blöcke zu je WRAP Zellen um, sonst überlappen die Zahlen
    shown_rows, shown_cols = (-(-cols // WRAP), WRAP) if wrap else (rows, cols)
    z = np.full((shown_rows, shown_cols), np.nan)
    text = [["" for _ in range(shown_cols)] for _ in range(shown_rows)]
    hover = [["" for _ in range(shown_cols)] for _ in range(shown_rows)]
    for k, j in enumerate(m.ints):
        r, c = m.grid_pos(j)
        if wrap:
            r, c = c // WRAP, c % WRAP
        z[r, c] = codes[k]
        v = values[k]
        text[r][c] = de(v) if abs(v - round(v)) > 1e-6 else str(int(round(v)))
        hover[r][c] = f"{m.names[j]}: {de(v, 3)} ({STATUS[codes[k]]})"
    if wrap:
        xs = [str(c + 1) for c in range(shown_cols)]
        ys = [f"{m.col_labels[r * WRAP]}–{m.col_labels[min((r + 1) * WRAP, cols) - 1]}" for r in range(shown_rows)]
    else:
        xs = m.col_labels if len(m.col_labels) == cols else [str(c + 1) for c in range(cols)]
        ys = m.row_labels if len(m.row_labels) == rows else [str(r + 1) for r in range(rows)]
        ys = [y if y else " " for y in ys]
    scale = []
    for i, color in enumerate(STATUS_COLORS):
        scale += [[i / len(STATUS_COLORS), color], [(i + 1) / len(STATUS_COLORS), color]]
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ys, text=text, customdata=hover, texttemplate="%{text}", hovertemplate="%{customdata}<extra></extra>", colorscale=scale, zmin=-0.5,
                               zmax=len(STATUS_COLORS) - 0.5, showscale=False, xgap=2, ygap=2))
    fig.update_xaxes(side="top", type="category")
    fig.update_yaxes(autorange="reversed", type="category")
    height = key_height or int(min(420, max(110, 38 * shown_rows + 60)))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.0, y=0.98, yanchor="top"))
        height += 40
    fig = _base(fig, height)
    fig.update_layout(margin=dict(l=10, r=10, t=66 if title else 28, b=6))
    return fig


def build_pump_distance(distances, current, restarts_at=()):
    """Abstand zur Rundung je Pump-Iteration; senkrechte Markierung beim gezeigten Schritt, Punkte bei den Neustarts."""
    xs = list(range(1, len(distances) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=distances, mode="lines+markers", line=dict(color=C.COLOR_LINE, width=2), marker=dict(size=5), name="Abstand",
                             hovertemplate="Iteration %{x}: Abstand %{y:.2f}<extra></extra>"))
    if restarts_at:
        fig.add_trace(go.Scatter(x=list(restarts_at), y=[distances[i - 1] for i in restarts_at], mode="markers", marker=dict(color=C.COLOR_BAD, size=10, symbol="diamond"), name="Neustart"))
    if current:
        fig.add_vline(x=current, line_dash="dot", line_color="#888")
    fig.update_xaxes(title_text="Iteration (LP)")
    fig.update_yaxes(title_text="Abstand Σ|x* − x~|")
    return _base(fig, 260)


def _color(metric, value):
    """Zellfarbe der Landkarte: Erfolg 0…1 (rot → grün), Lücke 0…100 % und LPs (grün → rot, LPs logarithmisch)."""
    if value is None:
        return 0.5
    if metric == "rate":
        return float(value)
    if metric == "gap":
        return 1 - min(float(value), 100.0) / 100.0
    return 1 - min(np.log10(max(float(value), 1.0)) / 2.0, 1.0)


def build_landkarte(rows, columns, metric, labels, row_labels):
    """Landkarte als Heatmap: Zeilen = Familien (nach Strukturklasse), Spalten = Verfahren; `rows[i][name]` ist ein Wörterbuch mit rate, gap, lps, n.
    metric: 'rate' (Erfolg in %), 'gap' (Lücke in %) oder 'lps' (gelöste LPs)."""
    z, text = [], []
    for row in rows:
        zr, tr = [], []
        for name in columns:
            s = row[name]
            v = s[metric]
            zr.append(_color(metric, v))
            if v is None:
                tr.append("–")
            elif metric == "rate":
                tr.append(f"{de(100 * v, 0)}")
            elif metric == "gap":
                tr.append(de(v, 1))
            else:
                tr.append(de(v, 1))
        z.append(zr)
        text.append(tr)
    fig = go.Figure(go.Heatmap(z=z, x=[labels[c] for c in columns], y=row_labels, text=text, texttemplate="%{text}", colorscale=[[0, "#e8837c"], [0.5, "#f3e6a8"], [1, "#7fc99a"]], zmin=0, zmax=1,
                               showscale=False, xgap=2, ygap=2, hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
    fig.update_xaxes(side="top", type="category", tickangle=-30)
    fig.update_yaxes(autorange="reversed", type="category")
    fig = _base(fig, 60 + 30 * len(rows))
    fig.update_layout(margin=dict(l=10, r=10, t=110, b=6))
    return fig


def build_bars(names, values, title_y, percent=True, color=C.COLOR_LINE):
    """Balken je Variante (Erfolg in % o. ä.)."""
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=color, text=[f"{de(v, 0 if percent else 1)}" if v is not None else "–" for v in values], textposition="outside",
                           hovertemplate="%{x}: %{y:.1f}<extra></extra>"))
    fig.update_xaxes(type="category", tickangle=-30)
    fig.update_yaxes(title_text=title_y, range=[0, max([v or 0 for v in values] + [1]) * 1.2])
    return _base(fig, 300)
