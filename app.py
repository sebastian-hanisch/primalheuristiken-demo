"""Ohne Suchbaum zur ganzzahligen Lösung: Runden, Tauchen, Pump - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwölf Familien ganzzahliger Programme (Strukturklassen ≤, ≥, Gleichungen, gemischt), drei Rundungen, Tauchen mit Rückschritten, der Feasibility Pump und eine Kaskade,
alles selbst gebaut; LP-Relaxationen per HiGHS (scipy), Referenzoptimum per HiGHS-MILP. Aufbau nach DEMO-PLAYBOOK §2. Zähleinheit überall: gelöste LPs.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import phr_constants as C
import phr_evaluation as E
import phr_landkarte as LK
import phr_tree as T
import phr_ui as UI
import phr_visualization as V
from phr_exact import solve_exact
from phr_layout import dunkles_design, kopf_css
from phr_presets import SETTING_SPECS, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from phr_result import de
from phr_rounding import lock_table

st.set_page_config(page_title="Primalheuristiken – Sebastian Hanisch", layout="wide")


# --- zwischengespeicherte Rechnungen -------------------------------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _analysis(settings):
    return E.analyse(settings)


@st.cache_data(show_spinner=False)
def _exact_solution(family, size_idx, seed):
    from phr_scenario import make_instance
    m = make_instance(family, size_idx, seed)
    opt, x, _ = solve_exact(m, C.EXACT_TIME_LIMIT)
    return opt, x


@st.cache_data(show_spinner=False)
def _live_row(family, size_idx):
    return E.live_row(family, size_idx)


@st.cache_data(show_spinner=False)
def _exp_theta(family, size_idx):
    return E.sweep(family, size_idx, E.theta_variants())


@st.cache_data(show_spinner=False)
def _exp_dive(family, size_idx, rule):
    return {"grid": E.sweep(family, size_idx, E.dive_variants(rule)), "rules": E.sweep(family, size_idx, E.rule_variants())}


@st.cache_data(show_spinner=False)
def _exp_pump(family, size_idx):
    return E.sweep(family, size_idx, E.pump_variants())


@st.cache_data(show_spinner=False)
def _tree(family, size_idx, seed, strategy, start_obj):
    """Suchbaum auf der Instanz ohne Startlösung, mit der Startlösung der Kaskade (falls vorhanden) und mit dem bekannten Optimum."""
    import math
    from phr_scenario import make_instance
    m = make_instance(family, size_idx, seed)
    run = T.STRATEGIES[strategy]
    opt = solve_exact(m, C.EXACT_TIME_LIMIT)[0]
    none = run(m, math.inf, C.BB_NODE_LIMIT)
    start = run(m, start_obj, C.BB_NODE_LIMIT) if start_obj is not None else None
    best = run(m, opt, C.BB_NODE_LIMIT)
    return {"none": (none["nodes"], none["proved"]), "start": None if start is None else (start["nodes"], start["proved"]), "opt": (best["nodes"], best["proved"])}


def _fam_label(key):
    return f"{C.FAMILIES[key]['label']} [{C.FAMILIES[key]['klass']}]"


# --- Kopf und Einleitung -------------------------------------------------------------------------------------------------------------------------

# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in phr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Ohne Suchbaum zur ganzzahligen Lösung: Runden, Tauchen, Pump")
st.markdown(
    """
Wie kommt man zu einer zulässigen ganzzahligen Lösung, **ohne einen Suchbaum aufzuspannen**? Man löst die LP-Relaxation und **rundet** sie, man **taucht** (eine Variable festsetzen, das LP neu lösen)
oder lässt den **Feasibility Pump** zwischen LP-Lösung und Rundung pendeln. **Ob Runden zulässig ist, steht in den Sperren der Zeilen**: Bei nichtnegativen Koeffizienten erlaubt eine ≤-Zeile nur Abrunden und eine ≥-Zeile nur Aufrunden (allgemein entscheidet das Vorzeichen des Koeffizienten),
eine Gleichung oder eine gegenläufige Zeile erlaubt keins von beiden. Die **Landkarte** zeigt für zwölf Strukturen, welches Verfahren wann gelingt und was es an gelösten LPs kostet.
Nachfolger von [branch-cut-demo](https://github.com/sebastian-hanisch/branch-cut-demo): dort gibt es nur den Fall einer einzigen ≤-Zeile, in dem Runden trivial zulässig ist; hier fehlt der Baum, und die Struktur der Zeilen entscheidet.

Gleich darunter erklärt der Expander **💬 In einfachen Worten** die Idee ohne Fachwörter. Weiter unten: die **Kernfragen** zur Instanz, die **Landkarte** und drei Experimente, der Abschnitt **🚧 Wo die Annahmen enden**, der Vergleich mit dem exakten Verfahren und dem Suchbaum und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/runden-oder-tauchen-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Ein Planungsprogramm rechnet zuerst so, als dürfte man alles teilen, und liefert Bruchteile: Eine Maschine bekommt zu einem Teil diesen Auftrag, zu einem Teil jenen. Gebraucht wird aber eine ganze Lösung, und zwar **ohne alle Möglichkeiten in einem Suchbaum durchzuprobieren**. Die Demo zeigt drei Wege: **Runden**, **Tauchen** (eine Entscheidung festlegen, neu rechnen, wiederholen) und den **Feasibility Pump** (zwischen Lösung und Rundung pendeln). Ob Runden erlaubt ist, steht in den Regeln der Aufgabe: Bei nicht negativen Zahlen in den Regeln erlauben Obergrenzen nur Abrunden, Untergrenzen nur Aufrunden, Gleichungen und gegenläufige Regeln keins von beiden (allgemein entscheidet das Vorzeichen der Zahl, deshalb gibt es Ausnahmen). Wo Runden scheitert, helfen Nachrechnen, Tauchen und Pump, aber auch sie scheitern messbar, und eine gefundene Lösung ist noch nicht die beste. Eine einzelne Instanz belegt nichts, die vorberechnete Landkarte zählt.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/runden-oder-tauchen-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die Suche nach einer Lösung ohne Baum", expanded=True):
    st.markdown(
        """
1. **LP-Relaxation lösen:** Die Ganzzahligkeit wird fallen gelassen; das LP liefert einen Punkt $x^*$, bei dem manche ganzzahligen Variablen gebrochen sind.
2. **Runden:** *nach Sperren* (nur in eine Richtung, in der keine Zeile verletzt werden kann), *mit Schwelle θ* (ab dem Nachkommaanteil θ aufrunden, sonst abrunden; bei θ = 0,5 ist das die nächste ganze Zahl) oder *mit Propagation* (eine Variable nach der anderen runden und nach jedem Schritt die Schranken der übrigen aus den Zeilen verschärfen).
3. **Tauchen:** eine gebrochene Variable festsetzen, das LP neu lösen, wiederholen; wird das LP unzulässig, gibt es einen Rückschritt (die andere Richtung). Mit Propagation schon vor dem LP.
4. **Feasibility Pump:** abwechselnd die LP-Lösung runden und ein LP lösen, das den Abstand zur Rundung minimiert; bei Stillstand oder Zyklus gibt es Störungen und Neustarts.
5. **Kaskade:** erst Runden mit Propagation (kein weiteres LP), nur bei Scheitern Tauchen, nur dann der Pump.

**Zähleinheit überall: gelöste LPs** (die Wurzel mitgezählt). Simplex-Iterationen werden nie mit LPs vermischt; ein Knoten des Vergleichsbaums ist ein gelöstes LP.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# --- Seitenleiste --------------------------------------------------------------------------------------------------------------------------------

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Instanz**")
    st.selectbox("Familie der Zeilen (Struktur)", C.FAMILY_KEYS, key="family", format_func=_fam_label,
                 help="Zwölf Familien in vier Strukturklassen: nur ≤, nur ≥, Gleichungen, gemischt. Die Klasse steht in eckigen Klammern.")
    fam_info = C.FAMILIES[st.session_state["family"]]
    st.caption(f"{fam_info['story']} Zeilen: {fam_info['rows']}.")
    st.radio("Größe", C.SIZE_LABELS, key="size", horizontal=True, help="klein: höchstens 25 ganzzahlige Variablen (bei der Standortplanung 6); mittel: 12 bis 120.")
    st.number_input("Zufalls-Seed der Instanz", 0, C.SEED_MAX, key="seed", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)

settings = E.Settings(**{k: st.session_state[k] for k in SETTING_SPECS})
sync_query_params({k: st.session_state[k] for k in SETTING_SPECS})
instance_id = (settings.family, settings.size, settings.seed)
if st.session_state.get("_instance_id") != instance_id:
    UI.reset_steps(("runden", "tauchen", "pump", "kaskade"))
    st.session_state["_instance_id"] = instance_id

a = _analysis(settings)
m = a.m
size_idx = E.size_index(settings.size)
opt = a.opt
grid_cols = f"{m.grid[0]} × {m.grid[1]}" if m.grid[0] > 1 else f"{m.grid[1]}"

# --- Die Instanz ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Instanz und ihre LP-Relaxation")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ganzzahlige Variablen", len(m.ints), delta=f"Gitter {grid_cols}", delta_color="off")
k2.metric("Zeilen", m.n_rows, delta=C.FAMILIES[settings.family]["klass"] + "-Klasse", delta_color="off")
k3.metric("Optimum (HiGHS-MILP)", "–" if opt is None else de(opt, 1), delta="Referenz", delta_color="off")
k4.metric("LP-Wert der Wurzel", de(a.root_obj, 1), delta=("–" if opt is None else f"{a.root_obj - opt:+.1f}".replace(".", ",") + " gegen das Optimum"), delta_color="off")
k5.metric("Gebrochene Variablen", a.n_frac, delta="in der Wurzel-LP", delta_color="off")
root_vals = a.root[0][m.ints]
st.plotly_chart(V.build_grid(m, root_vals, V.status_codes(m, root_vals)), width="stretch", key="root_grid")
UI.show_legend()
rows_text = f"Zeilen: {', '.join(m.row_labels[:3])} …; " if m.grid[0] > 1 else ""
st.caption(f"Gitteransicht: {rows_text}Spalten: {', '.join(m.col_labels[:3])} …; jede Zelle ist eine ganzzahlige Variable (z. B. {m.names[m.ints[0]]}). "
           "Gibt es gebrochene (orange) Zellen, ist die LP-Lösung nicht zulässig ganzzahlig.")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen an diese Instanz")
table = lock_table(m, a.root[0])
blocked = sum(1 for t in table if t["direction"] == "keine")
lock_gap = E.gap_percent(a.lock.obj, opt)

st.markdown("**1. Darf man diese LP-Lösung runden?** (Sperren der Zeilen)")
kind, text = E.verdict_locks(a.n_frac, blocked, a.lock.found, lock_gap)
getattr(st, kind)(text)

st.markdown("**2. Rettet die Propagation das Runden?**")
kind, text = E.verdict_rounding(a.lock.found, a.near.found, a.prop.found, a.prop.lps)
getattr(st, kind)(text)
data = LK.load()
cell = LK.cell(data, settings.family, size_idx) if data else None
if cell:
    st.caption(E.landkarte_sentence(cell))

st.markdown("**3. Tauchen oder Pumpen?** (Einstellungen der Reiter weiter unten)")
dive_gap = E.gap_percent(a.dive.obj, opt) if a.dive is not None and a.dive.found else None
pump_gap = E.gap_percent(a.pump.obj, opt) if a.pump.found else None
kind, text = E.verdict_dive_pump(bool(a.dive and a.dive.found), a.dive.lps if a.dive else 0, dive_gap, a.pump.found, a.pump.lps, pump_gap, a.ref_missing)
getattr(st, kind)(text)

st.markdown("**4. Was kostet die Kaskade Runden → Tauchen → Pump?**")
kind, text = E.verdict_cascade(a.casc.info["stage"], a.casc.lps, a.casc.found, a.base_dive.found, a.base_dive.lps)
getattr(st, kind)(text)

st.markdown("---")

# --- Schritt für Schritt -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Schritt für Schritt: die Verfahren an dieser Instanz")
tab_round, tab_dive, tab_pump, tab_casc = st.tabs(["Runden", "Tauchen", "Pump", "Kaskade"])

with tab_round:
    st.markdown("**Sperren:** Je Variable zählt die Tabelle, wie viele Zeilen ein Verkleinern (abwärts) bzw. Vergrößern (aufwärts) verletzen kann; eine Gleichung sperrt beides. "
                "Bei einer ≤-Zeile sperrt ein positiver Koeffizient das Vergrößern und ein negativer das Verkleinern (eine ≥-Zeile steht negiert in derselben Form, ihre Koeffizienten sperren also umgekehrt). "
                "Frei ist eine Richtung ohne Sperre. Gleichungen, die nur stetige Variablen enthalten (wie bei `ufl`), sperren keine ganzzahlige Variable.")
    if table:
        st.dataframe([{"Variable": t["name"], "LP-Wert": de(t["value"], 3), "Sperren abwärts": t["down"], "Sperren aufwärts": t["up"],
                       "Erlaubte Rundrichtung": {"frei": "beide", "ab": "nur abrunden", "auf": "nur aufrunden", "keine": "keine"}[t["direction"]]} for t in table], hide_index=True)
    else:
        st.caption("Die Wurzel-LP ist ganzzahlig: keine gebrochene Variable, keine Tabelle.")
    st.radio("Rundungsschwelle θ (ab diesem Nachkommaanteil wird aufgerundet)", C.THETA_OPTIONS, key="theta", horizontal=True,
             help="Wirkt auf „mit Schwelle θ“ und „mit Propagation“; Runden nach Sperren braucht keine Schwelle. „1/f“: eins durch die Länge der längsten ≥-Zeile, ohne ≥-Zeile 0,5.")
    st.caption(f"Für diese Instanz ist θ = {de(a.theta, 3)} ({'1/f = ' + de(a.theta, 3) if settings.theta == '1/f' else 'gewählter Wert'}).")
    cols = st.columns(3)
    for col, (title, res) in zip(cols, (("Nach Sperren", a.lock), ("Mit Schwelle θ", a.near), ("Mit Propagation", a.prop))):
        with col:
            st.markdown(f"**{title}**")
            st.write(UI.result_line(res, E.gap_percent(res.obj, opt)))
    view = st.radio("Schritt für Schritt zeigen", ["Nach Sperren", "Mit Schwelle θ", "Mit Propagation"], key="runden_view", horizontal=True, index=2)
    chosen = {"Nach Sperren": a.lock, "Mit Schwelle θ": a.near, "Mit Propagation": a.prop}[view]
    idx = UI.step_slider("runden", len(chosen.steps))
    UI.show_step(m, chosen.steps, f"runden_{view}", idx)
    UI.show_legend()

with tab_dive:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.radio("Tauchregel", tuple(C.RULE_LABELS), key="rule", format_func=lambda k: C.RULE_LABELS[k],
                 help="Welche Variable zuerst festgesetzt wird. Guided braucht eine Referenzlösung: hier die von „Runden mit Propagation“; findet sie keine, läuft das Tauchen nicht.")
    with c2:
        st.radio("Rückschritte (höchstens)", C.BACKTRACK_OPTIONS, key="backtracks", horizontal=True, help="Wie oft nach einem unzulässigen LP die andere Richtung der Variable versucht wird.")
    with c3:
        st.toggle("Propagation nach jedem Festsetzen", key="prop", help="Schärft nach jedem Festsetzen die Schranken aller Variablen aus den Zeilen; ein Widerspruch zählt als unzulässig, ohne ein LP zu lösen.")
    if a.dive is None:
        st.warning("Guided Tauchen braucht eine Referenzlösung, und Runden mit Propagation hat auf dieser Instanz keine gefunden: Das Tauchen läuft deshalb nicht. Eine andere Regel oder Instanz wählen.")
    else:
        d = a.dive
        st.write(UI.result_line(d, E.gap_percent(d.obj, opt)) + f" · Rückschritte: {d.info.get('backtracks', 0)}")
        idx = UI.step_slider("tauchen", len(d.steps))
        UI.show_step(m, d.steps, "tauchen", idx)
        UI.show_legend()

with tab_pump:
    c1, c2 = st.columns(2)
    with c1:
        st.radio("Neustarts (höchstens)", C.RESTART_OPTIONS, key="restarts", horizontal=True, help="Findet der Pump 15 Iterationen lang keinen kleineren Abstand, beginnt er neu (Wurzellösung mit starkem Rauschen gerundet).")
    with c2:
        st.toggle("Zielgewicht (Objective Feasibility Pump)", key="objw", help="Mischt die Zielfunktion des Problems in das Abstands-LP (Gewicht 1, je Iteration × 0,9). Ohne: reiner Pump.")
    p = a.pump
    info = p.info
    st.write(UI.result_line(p, E.gap_percent(p.obj, opt)) + f" · Iterationen: {info['iterations']}, Stillstände: {info['flip_steps']}, Zyklen: {info['cycle_steps']}, Neustarts: {info['restarts']}")
    idx = UI.step_slider("pump", len(p.steps))
    UI.show_step(m, p.steps, "pump", idx, pump_distances=info["distances"])
    UI.show_legend()

with tab_casc:
    st.markdown("Standardeinstellungen der Vormessung: θ = 0,5, Tauchen mit Propagation und 20 Rückschritten, Pump mit 10 Neustarts ohne Zielgewicht. Die Wurzel wird einmal gezählt.")
    casc = a.casc
    stage_rows = []
    names = ["Stufe 0: Runden mit Propagation", "Stufe 1: Tauchen mit Propagation", "Stufe 2: Pump mit Neustarts"]
    for i, r in enumerate(casc.info["stage_results"]):
        stage_rows.append({"Stufe": names[i], "Ergebnis": "✅ Lösung" if r.found else "❌ keine Lösung", "Zusätzliche LPs": r.lps - 1})
    for i in range(len(casc.info["stage_results"]), 3):
        stage_rows.append({"Stufe": names[i], "Ergebnis": "– nicht nötig", "Zusätzliche LPs": 0})
    st.dataframe(stage_rows, hide_index=True)
    st.write(UI.result_line(casc, E.gap_percent(casc.obj, opt)))
    idx = UI.step_slider("kaskade", len(casc.steps))
    UI.show_step(m, casc.steps, "kaskade", idx, pump_distances=None)
    UI.show_legend()

st.markdown("---")

# --- Die Landkarte -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die Landkarte: Struktur der Zeilen gegen Verfahren")
if data is None:
    st.warning("Die vorberechnete Landkarte (phr_landkarte.json) fehlt.")
else:
    n_per = data["meta"]["map_n"]
    st.caption(f"Vorberechnet mit tools/landkarte_messen.py: je Familie und Größe die ersten {n_per} Instanzen mit gebrochener Wurzel-LP (Zufalls-Seeds ab 0, SplitMix64), jede mit allen Verfahren "
               "und dem exakten Optimum; Einheit gelöste LPs (Wurzel mitgezählt). Erfolg = Anteil der Instanzen mit zulässiger Lösung; Lücke = mittlere Abweichung vom Optimum in Prozent des Betrags "
               "des Optimums, nur über die erfolgreichen Läufe; LPs sind das Mittel über alle Läufe der Zelle (Zelle = Familie in einer Größe).")
    mc1, mc2 = st.columns(2)
    with mc1:
        metric_label = st.radio("Zahl in der Zelle", ["Erfolg (%)", "Lücke (%)", "LPs"], key="map_metric", horizontal=True)
    with mc2:
        size_label = st.radio("Größe der Instanzen", ["klein", "mittel", "beide zusammen"], key="map_size", horizontal=True)
    metric = {"Erfolg (%)": "rate", "Lücke (%)": "gap", "LPs": "lps"}[metric_label]
    sizes = (0,) if size_label == "klein" else ((1,) if size_label == "mittel" else (0, 1))
    columns = LK.MAP_METHODS
    short = {"R_lock": "Runden: Sperren", "R_near": "Runden: θ = 0,5", "R_prop": "Runden: Propagation", "D_frac": "Tauchen", "D_fracP": "Tauchen + Prop.", "D_coefP": "Tauchen (Coef.) + Prop.",
             "FP_rs": "Pump", "FP_objrs": "Pump + Zielgewicht", "K1": "Kaskade"}
    rows, row_labels, selected_cells = [], [], []
    for klass, fams in LK.classes().items():
        for f in fams:
            rows.append({name: LK.pooled(data, name, [f], sizes) for name in columns})
            row_labels.append(f"{klass}  {f}")
            selected_cells += [LK.cell(data, f, s) for s in sizes]
    st.plotly_chart(V.build_landkarte(rows, columns, metric, short, row_labels), width="stretch", key="landkarte")
    st.caption("Zeilen nach Strukturklasse (≤, ≥, =, gemischt); Farbe: grün = gut (hoher Erfolg, kleine Lücke, wenige LPs), rot = schlecht. Dünne Zellen stehen in der Tabelle unten (n).")
    kind, text = E.verdict_map(selected_cells)
    getattr(st, kind)(text)
    stages = LK.stage_shares(data, None, sizes)
    st.markdown(f"**Kaskade, alle zwölf Familien ({size_label}):** Stufe 0 (Runden mit Propagation) löst {de(100 * stages[0], 1)} % der Instanzen, Stufe 1 (Tauchen) {de(100 * stages[1], 1)} %, "
                f"Stufe 2 (Pump) {de(100 * stages[2], 1)} %, keine Stufe {de(100 * stages[3], 1)} %.")
    with st.expander("Alle Zahlen der Landkarte und die zwölf Familien"):
        detail = []
        for klass, fams in LK.classes().items():
            for f in fams:
                for s in sizes:
                    c = LK.cell(data, f, s)
                    row = {"Klasse": klass, "Familie": f, "Größe": C.SIZE_LABELS[s], "Instanzen (n)": c["n"], "Seeds angesehen": c["scanned"]}
                    for name in columns:
                        st_ = LK.stat(c, name)
                        row[short[name]] = f"{UI.percent(st_['rate'])} % · {UI.number(st_['gap'])} % · {UI.number(st_['lps'])} LP"
                    detail.append(row)
        st.markdown("Zelle: Erfolg · mittlere Lücke · mittlere LPs.")
        st.dataframe(detail, hide_index=True)
        st.dataframe([{"Familie": k, "Name": v["label"], "Klasse": v["klass"], "Zeilenarten": v["rows"], "Beschreibung": v["story"]} for k, v in C.FAMILIES.items()], hide_index=True)
    st.markdown("**Diese Zeile selbst nachmessen** (andere Instanzen als die vorberechneten):")
    live = UI.run_on_click("map", f"📊 {settings.family} ({settings.size}) mit {C.LIVE_N} Instanzen live messen", (settings.family, size_idx, C.LIVE_N), lambda f, s, n: _live_row(f, s),
                           f"{C.LIVE_N} Instanzen ab Seed {C.LIVE_SEED0}, rund 1 bis 40 Sekunden, je nach Familie und Größe")
    if live:
        st.dataframe([{"Verfahren": short[n], "Erfolg (%)": UI.percent(live["rows"][n]["rate"]), "Lücke (%)": UI.number(live["rows"][n]["gap"]), "LPs": UI.number(live["rows"][n]["lps"])} for n in columns],
                     hide_index=True)
        st.caption(f"{live['instances']} Instanzen mit gebrochener Wurzel (von {live['scanned']} angesehenen Seeds); Kaskade: Stufe 0 {de(100 * live['stages'][0], 0)} %, "
                   f"Stufe 1 {de(100 * live['stages'][1], 0)} %, Stufe 2 {de(100 * live['stages'][2], 0)} %, keine {de(100 * live['stages'][3], 0)} %.")

st.markdown("---")

# --- Experiment: Rundungsschwelle ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Gibt es eine beste Rundungsschwelle θ?")
st.caption(f"Misst auf Knopfdruck {C.LIVE_N} Instanzen der gewählten Familie und Größe mit gebrochener Wurzel (feste Seeds ab {C.LIVE_SEED0}): Runden mit Schwelle θ und Runden mit Propagation bei θ = 0,3 / 0,5 / 0,7 / 1/f.")
res = UI.run_on_click("theta", f"📊 Schwelle messen ({settings.family}, {settings.size})", (settings.family, size_idx), _exp_theta, f"{C.LIVE_N} Instanzen, rund 1 bis 30 Sekunden, je nach Familie und Größe")
if res and res["instances"]:
    rows_ = res["rows"]
    st.dataframe([{"θ": lab, "Mit Schwelle θ: Erfolg (%)": UI.percent(rows_[f"R_near θ={lab}"]["rate"]), "Lücke (%)": UI.number(rows_[f"R_near θ={lab}"]["gap"]),
                   "Mit Propagation: Erfolg (%)": UI.percent(rows_[f"R_prop θ={lab}"]["rate"]), "Lücke (%) ": UI.number(rows_[f"R_prop θ={lab}"]["gap"])} for lab in C.THETA_OPTIONS], hide_index=True)
    st.plotly_chart(V.build_bars([f"θ = {lab}" for lab in C.THETA_OPTIONS], [100 * rows_[f"R_near θ={lab}"]["rate"] for lab in C.THETA_OPTIONS], "Erfolg, Runden mit Schwelle θ (%)"), width="stretch", key="theta_bars")
    kind, text = E.sweep_verdict_theta(rows_)
    getattr(st, kind)(text + f" ({res['instances']} Instanzen.)")

st.markdown("---")

# --- Experiment: Tauchen -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Was bringen Propagation und Rückschritte beim Tauchen?")
st.caption(f"Misst {C.LIVE_N} Instanzen: Tauchen mit Regel „{C.RULE_LABELS[settings.rule]}“ bei 0 / 1 / 5 / 20 Rückschritten, ohne und mit Propagation, dazu die drei Regeln bei 20 Rückschritten mit Propagation.")
res = UI.run_on_click("dive", f"📊 Tauchen messen ({settings.family}, {settings.size})", (settings.family, size_idx, settings.rule), _exp_dive, f"{C.LIVE_N} Instanzen, rund 1 bis 70 Sekunden, je nach Familie und Größe")
if res and res["grid"]["instances"]:
    g = res["grid"]["rows"]
    st.dataframe([{"Variante": n, "Erfolg (%)": UI.percent(v["rate"]), "Lücke (%)": UI.number(v["gap"]), "LPs im Mittel": UI.number(v["lps"])} for n, v in g.items()], hide_index=True)
    kind, text = E.verdict_dive_sweep(g)
    getattr(st, kind)(text + f" ({res['grid']['instances']} Instanzen.)")
    r_ = res["rules"]["rows"]
    st.markdown("**Die drei Tauchregeln** (20 Rückschritte, mit Propagation):")
    st.dataframe([{"Regel": C.RULE_LABELS[n], "Instanzen mit Referenz": v["n"], "Erfolg (%)": UI.percent(v["rate"]), "Lücke (%)": UI.number(v["gap"]), "LPs im Mittel": UI.number(v["lps"])}
                  for n, v in r_.items()], hide_index=True)
    st.caption("Guided läuft nur, wo Runden mit Propagation eine Referenzlösung findet (Spalte „Instanzen mit Referenz“); Erfolg und LPs beziehen sich auf diese Instanzen.")

st.markdown("---")

# --- Experiment: Pump ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Was bringen Neustarts und Zielgewicht beim Pump?")
st.caption(f"Misst {C.LIVE_N} Instanzen: Pump mit 0 / 3 / 10 Neustarts, ohne und mit Zielgewicht (höchstens 300 Iterationen).")
res = UI.run_on_click("pump", f"📊 Pump messen ({settings.family}, {settings.size})", (settings.family, size_idx), _exp_pump, f"{C.LIVE_N} Instanzen, rund 1 bis 40 Sekunden, je nach Familie und Größe")
if res and res["instances"]:
    g = res["rows"]
    st.dataframe([{"Variante": n, "Erfolg (%)": UI.percent(v["rate"]), "Lücke (%)": UI.number(v["gap"]), "LPs im Mittel": UI.number(v["lps"])} for n, v in g.items()], hide_index=True)
    st.plotly_chart(V.build_bars(list(g), [100 * v["rate"] for v in g.values()], "Erfolg (%)"), width="stretch", key="pump_bars")
    kind, text = E.verdict_pump_sweep(g)
    getattr(st, kind)(text + f" ({res['instances']} Instanzen.)")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Instanzen sind erzeugt, nicht aus der Praxis** (zwölf Familien, 6 bis 120 ganzzahlige Variablen, ganzzahlige Variablen nur 0/1; bei der Standortplanung kommen stetige Zuordnungsvariablen hinzu) | Die Landkarte zeigt, welches Verfahren bei welcher *Struktur* gelingt; ihre Zahlen sind für andere Größen und Daten nicht übertragbar (die Erfolgsraten verschieben sich zwischen klein und mittel, siehe Landkarte). Allgemeine ganzzahlige Variablen sind nicht gemessen. | – |
| **Der Suchbaum zum Vergleich ist einfach** (ohne Schnitte, Presolve und Warmstart, höchstens 1 000 LPs) | Aussagen über den Wert einer Startlösung gelten nur für diesen Baum; ein Löser mit Schnitten und eigenen Heuristiken verhält sich anders. | [branch-cut-demo](https://github.com/sebastian-hanisch/branch-cut-demo) für die Schnitte |
| **Ein gefundener Punkt beweist nichts** | Eine Heuristik allein beweist keine Optimalität: Sie liefert eine zulässige Lösung und damit eine obere Schranke; ein Beweis entsteht nur, wenn diese die untere Schranke (z. B. den aufgerundeten LP-Wert) erreicht. Die Lücke zum Optimum kennt nur, wer das Optimum (HiGHS-MILP) separat berechnet hat. | [branch-bound-demo](https://github.com/sebastian-hanisch/branch-bound-demo) |
| **Die Kaskade hört bei der ersten zulässigen Lösung auf, nicht bei der besten** | Sie versucht die Stufen in fester Reihenfolge (Runden, Tauchen, Pump); ihre Lücke ist in 15 von 24 Zellen der Landkarte größer als die des Tauchens, in 3 gleich und in 6 kleiner, und sie braucht in 2 Zellen mehr LPs als das Tauchen allein (siehe Landkarte, Spalten Lücke und LPs). | – |
| **Jedes LP wird von Null an gelöst** (kein Warmstart in scipy) | Gezählt werden gelöste LPs, nicht Rechenzeit; Zeiten und Simplex-Iterationen sind nicht verglichen. | – |
"""
)

st.markdown("---")

# --- Vergleich: exakt und Suchbaum ---------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Vergleich: exaktes Optimum und Suchbaum"):
    tab_exact, tab_tree = st.tabs(["🧮 Exakt (HiGHS-MILP)", "🌳 Suchbaum und Startlösung"])
    with tab_exact:
        st.markdown("Das Optimum der gewählten Instanz per HiGHS-MILP (`scipy.optimize.milp`), verglichen mit der Lösung der Kaskade.")
        if st.button("🧮 Optimum zeigen", key="run_exact"):
            st.session_state["exact_args"] = (settings.family, size_idx, settings.seed)
        if st.session_state.get("exact_args") == (settings.family, size_idx, settings.seed):
            value, x = _exact_solution(settings.family, size_idx, settings.seed)
            if value is None:
                st.warning(f"Das Optimum wurde nicht in {C.EXACT_TIME_LIMIT} Sekunden bewiesen.")
            else:
                st.metric("Optimum", de(value, 1), delta=("Kaskade: " + (de(a.casc.obj, 1) + f" ({de(E.gap_percent(a.casc.obj, value), 1)} % Lücke)" if a.casc.found else "keine Lösung")), delta_color="off")
                vals = x[m.ints]
                st.plotly_chart(V.build_grid(m, vals, V.status_codes(m, vals), "Optimale Lösung"), width="stretch", key="exact_grid")
                if a.casc.found:
                    cv = a.casc.x[m.ints]
                    st.plotly_chart(V.build_grid(m, cv, V.status_codes(m, cv), "Lösung der Kaskade"), width="stretch", key="casc_grid")
        else:
            st.caption("Noch nicht berechnet.")
    with tab_tree:
        st.markdown(f"Ein einfacher Suchbaum (ohne Schnitte und Presolve, höchstens {C.BB_NODE_LIMIT} LPs) beweist das Optimum: ohne Startlösung, mit der Lösung der Kaskade und, als Obergrenze des Nutzens, mit dem bekannten Optimum. "
                    "Die LPs der Kaskade sind nicht eingerechnet.")
        strategy = st.selectbox("Suchstrategie", tuple(T.STRATEGIES), key="tree_strategy", help="Tiefensuche: immer das zuletzt erzeugte Kind; beste Schranke zuerst: der Knoten mit dem kleinsten LP-Wert.")
        start_obj = a.casc.obj if a.casc.found else None
        stats = UI.run_on_click("tree", f"🌳 Suchbaum starten ({strategy})", (settings.family, size_idx, settings.seed, strategy, start_obj),
                                lambda f, s, sd, st_, so: _tree(f, s, sd, st_, so), f"bis {3 * C.BB_NODE_LIMIT} LPs, rund 1 bis 10 Sekunden")
        if stats:
            kind, text = E.verdict_tree(stats, strategy)
            getattr(st, kind)(text)

st.markdown("---")

# --- Formulierung --------------------------------------------------------------------------------------------------------------------------------

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/runden-oder-tauchen-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist ein ganzzahliges lineares Programm (MIP) in Minimierungsform; gesucht ist eine zulässige ganzzahlige Lösung ohne Suchbaum, möglichst nahe am Optimum.

| Symbol | Bedeutung |
|---|---|
| $x \in \mathbb{R}^n$ | Variablen; $I \subseteq \{1,\dots,n\}$ sind die ganzzahligen (hier immer 0/1), die übrigen stetig |
| $c$ | Kostenvektor (Maximierung steht negiert); die Zielgröße ist $c^\top x$ |
| $A x \le b$ | die Ungleichungszeilen; eine ≥-Zeile steht mit umgekehrtem Vorzeichen |
| $A_{eq}\, x = b_{eq}$ | die Gleichungszeilen |
| $\ell \le x \le u$ | Schranken (hier $0 \le x \le 1$) |
| $x^*$ | Lösung der LP-Relaxation (ohne Ganzzahligkeit) |
| $\tilde{x}$ | gerundeter Punkt (0/1 auf den Variablen in $I$) |
| $\mathrm{down}(j)$, $\mathrm{up}(j)$ | Zahl der Zeilen, die ein Verkleinern bzw. Vergrößern von $x_j$ verletzen können (die Sperren) |
| $\theta$ | Rundungsschwelle: ab diesem Nachkommaanteil wird aufgerundet |
| $\Delta$ | Abstand einer LP-Lösung zur Rundung im Pump |
| $\alpha$ | Gewicht der Zielfunktion im Pump (Zielgewicht) |
| $z^{opt}$, $z^{inc}$ | Wert des Optimums bzw. der besten bekannten Lösung |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/runden-oder-tauchen-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Problem.** $\min\; c^\top x$ unter $A x \le b,\; A_{eq}\, x = b_{eq},\; \ell \le x \le u,\; x_j \in \mathbb{Z}$ für $j \in I$.
Die LP-Relaxation lässt $x_j \in \mathbb{Z}$ fallen; ihr Wert $c^\top x^*$ ist eine untere Schranke des Optimums. Die Lücke einer Lösung $\hat{x}$ ist $100 \cdot (c^\top \hat{x} - z^{opt}) / |z^{opt}|$ Prozent.

**Sperren.** Für $j \in I$ zählen $\mathrm{down}(j) = |\{i : a_{ij} < 0\}| + |\{i : a^{eq}_{ij} \ne 0\}|$ die Zeilen, die ein Verkleinern, und
$\mathrm{up}(j) = |\{i : a_{ij} > 0\}| + |\{i : a^{eq}_{ij} \ne 0\}|$ die Zeilen, die ein Vergrößern von $x_j$ verletzen können. Gilt $\mathrm{down}(j) = 0$, kann Abrunden von $x_j$ keine Zeile verletzen
(unabhängig von allen anderen Variablen); gilt $\mathrm{up}(j) = 0$, gilt das für Aufrunden. **Runden nach Sperren** gelingt genau dann, wenn jede gebrochene Variable $\mathrm{down}(j) = 0$ oder $\mathrm{up}(j) = 0$ hat.

**Rundung mit Schwelle $\theta$** (bei $\theta = 0{,}5$ die nächste ganze Zahl): $\tilde{x}_j = \lceil x^*_j \rceil$, falls $x^*_j - \lfloor x^*_j \rfloor \ge \theta$, sonst $\lfloor x^*_j \rfloor$.

**Propagation** (aktivitätsbasiert). Für eine Zeile $\sum_k a_k x_k \le b$ ist die kleinste mögliche Aktivität $\mathrm{minact} = \sum_{a_k > 0} a_k \ell_k + \sum_{a_k < 0} a_k u_k$.
Für $a_j > 0$ folgt $x_j \le \ell_j + (b - \mathrm{minact})/a_j$, für $a_j < 0$ folgt $x_j \ge u_j + (b - \mathrm{minact})/a_j$; bei ganzzahligen Variablen wird abgerundet bzw. aufgerundet.
Gilt $\mathrm{minact} > b$ oder $\ell_j > u_j$, ist die Teilbelegung unzulässig. Gleichungen zählen als zwei Ungleichungen.

**Feasibility Pump.** Zu einem gerundeten Punkt $\tilde{x}$ wird das LP
$\min\; \Delta(x, \tilde{x}) = \sum_{j \in I:\, \tilde{x}_j = 0} x_j + \sum_{j \in I:\, \tilde{x}_j = 1} (1 - x_j)$ unter den Zeilen gelöst; ist die Lösung $x^*$ ganzzahlig, ist sie zulässig, sonst wird neu gerundet.
Mit Zielgewicht $\alpha$ lautet die Zielfunktion $(1-\alpha)\,\Delta + \alpha\,\sqrt{|I|}\, c^\top x / \lVert c \rVert$, $\alpha$ sinkt je Iteration um den Faktor 0,9.

**Suchbaum.** Knoten mit $\lceil c^\top x^* \rceil \ge z^{inc}$ werden abgeschnitten (bei ganzzahligen Zielwerten; $z^{inc}$ = Wert der besten bekannten Lösung).

**Literatur.** Fischetti, M., Glover, F., Lodi, A.: The feasibility pump. *Mathematical Programming* 104 (2005), 91–104. Das Zielgewicht (Objective Feasibility Pump): Achterberg, T., Berthold, T.: Improving the feasibility pump. *Discrete Optimization* 4 (2007), 77–86.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Exakte Suche erklärt: Rucksack und Cutting Stock](https://sebastianhanisch.net/konzepte-exakte-suche.html)."
)
