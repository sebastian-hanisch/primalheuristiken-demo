"""Auswertung: Einstellungen, Analyse einer Instanz, Messungen über feste Instanzen und die Verdikte (reine Funktionen, DEMO-PLAYBOOK §12).

Zähleinheit überall: gelöste LPs (Wurzel mitgezählt). Lücke = (Wert − Optimum) / |Optimum| in Prozent, über die erfolgreichen Läufe;
Erfolg = Anteil der Instanzen, für die das Verfahren eine vom Prüfer angenommene Lösung liefert."""

from __future__ import annotations

from dataclasses import dataclass

import phr_algorithm as A
import phr_constants as C
from phr_diving import dive
from phr_exact import solve_exact
from phr_lp import frac_indices
from phr_pump import pump
from phr_result import Params, de
from phr_rounding import cover_theta, lock_table, round_lock, round_near, round_prop, solve_root
from phr_scenario import make_instance


@dataclass(frozen=True)
class Settings:
    """Alle Regler der App. Immer mit Schlüsselwörtern aufrufen."""
    family: str = C.DEFAULT_FAMILY
    size: str = C.DEFAULT_SIZE
    seed: int = C.DEFAULT_SEED
    theta: str = C.DEFAULTS["theta"]
    backtracks: int = C.DEFAULTS["backtracks"]
    prop: bool = C.DEFAULTS["prop"]
    rule: str = C.DEFAULTS["rule"]
    restarts: int = C.DEFAULTS["restarts"]
    objw: bool = C.DEFAULTS["objw"]


def size_index(size):
    return C.SIZE_LABELS.index(size)


def theta_value(m, label):
    """Rundungsschwelle aus der Auswahl: „0,3“ → 0,3 usw.; „1/f“ → 1 / Länge der längsten ≥-Zeile (ohne ≥-Zeile 0,5)."""
    return cover_theta(m) if label == "1/f" else float(label.replace(",", "."))


def params_of(settings, m):
    return Params(theta=theta_value(m, settings.theta), backtracks=settings.backtracks, propagate=settings.prop, rule=settings.rule,
                  restarts=settings.restarts, objective_weight=1.0 if settings.objw else 0.0)


def gap_percent(value, optimum):
    """Lücke eines Zielwerts zum Optimum in Prozent von |Optimum| (Minimierung); ohne Optimum None."""
    if value is None or optimum is None:
        return None
    return 100.0 * (value - optimum) / max(abs(optimum), 1e-9)


@dataclass
class Analysis:
    m: object
    root: tuple
    root_obj: float
    opt: float | None
    lock: object
    near: object
    prop: object
    dive: object | None
    pump: object
    casc: object
    base_dive: object          # Tauchen mit Propagation, Standardeinstellungen: Vergleich für die Kaskade
    n_frac: int
    ref_missing: bool          # Guided Diving ohne Referenzlösung (Runden mit Propagation fand keine)
    theta: float


def analyse(settings):
    """Eine Instanz mit allen Verfahren (mit Aufzeichnung der Schritte). Die Instanz hat nur dann etwas zu runden, wenn n_frac > 0."""
    m = make_instance(settings.family, size_index(settings.size), settings.seed)
    root = solve_root(m)
    if root is None:
        raise ValueError("LP-Relaxation unzulässig")
    params = params_of(settings, m)
    opt = solve_exact(m, C.EXACT_TIME_LIMIT)[0]
    lock = round_lock(m, root, trace=True)
    near = round_near(m, root, params.theta, trace=True)
    prop = round_prop(m, root, params.theta, trace=True)
    ref_missing = False
    d = None
    if params.rule == "guided":
        if prop.found:
            d = dive(m, root, params, trace=True, ref=prop.x)
        else:
            ref_missing = True
    else:
        d = dive(m, root, params, trace=True)
    p = pump(m, root, params, seed=settings.seed, trace=True)
    casc = A.cascade(m, root, settings.seed, Params(theta=0.5), trace=True)
    base = dive(m, root, Params(rule="frac", propagate=True))
    return Analysis(m, root, float(m.c @ root[0]), opt, lock, near, prop, d, p, casc, base, len(frac_indices(m, root[0])), ref_missing, params.theta)


# ------------------------------------------------------------------------------------------------------------------ Messungen
def measure_instance(m, seed):
    """Alle Einzelverfahren der Landkarte plus Kaskade K1 auf einer Instanz: {Name: {found, gap, lps}} (None, wenn die Relaxation unzulässig ist).
    Rückgabe zusätzlich: Stufe der Kaskade, Zahl der gebrochenen Variablen, Wurzellücke."""
    root = solve_root(m)
    if root is None:
        return None
    opt = solve_exact(m, C.EXACT_TIME_LIMIT)[0]
    if opt is None:
        return None
    results = {name: A.run_named(m, root, name, seed) for name in A.METHODS[:-1]}
    found, value, lps, stage = A.combine_cascade(results)
    out = {name: {"found": r.found, "gap": gap_percent(r.obj, opt), "lps": r.lps} for name, r in results.items()}
    out["K1"] = {"found": found, "gap": gap_percent(value, opt), "lps": lps}
    return {"methods": out, "stage": stage, "n_frac": len(frac_indices(m, root[0])), "root_gap": gap_percent(opt, float(m.c @ root[0]))}


def fractional_seeds(family, size_idx, count, start=0, scan_limit=C.MAP_SCAN):
    """Die ersten `count` Seeds ab `start`, deren Wurzel-LP gebrochen ist, und wie viele Seeds dafür angesehen wurden: (Seeds, angesehen)."""
    seeds, scanned = [], 0
    seed = start
    while len(seeds) < count and scanned < scan_limit:
        m = make_instance(family, size_idx, seed)
        root = solve_root(m)
        scanned += 1
        if root is not None and len(frac_indices(m, root[0])) > 0:
            seeds.append(seed)
        seed += 1
    return seeds, scanned


def aggregate(records):
    """Je Verfahren: Zahl der Läufe, Erfolge, mittlere Lücke über die Erfolgreichen, mittlere LPs über alle Läufe. records: Liste von {Name: {found, gap, lps}}."""
    names = records[0].keys() if records else []
    out = {}
    for name in names:
        runs = [r[name] for r in records if r.get(name) is not None]
        ok = [r for r in runs if r["found"]]
        out[name] = {"n": len(runs), "ok": len(ok), "rate": len(ok) / len(runs) if runs else None,
                     "gap": sum(r["gap"] for r in ok) / len(ok) if ok else None,
                     "lps": sum(r["lps"] for r in runs) / len(runs) if runs else None}
    return out


def sweep(family, size_idx, variants, count=C.LIVE_N, start=C.LIVE_SEED0):
    """Live-Messung: `variants` = {Name: f(m, root, seed) → Result oder None (nicht anwendbar)} auf den ersten `count` Instanzen mit gebrochener Wurzel ab Seed `start`.
    Gibt aggregate(...) zurück plus die Zahl der angesehenen Seeds."""
    seeds, scanned = fractional_seeds(family, size_idx, count, start, scan_limit=40 * count)
    records = []
    for seed in seeds:
        m = make_instance(family, size_idx, seed)
        root = solve_root(m)
        opt = solve_exact(m, C.EXACT_TIME_LIMIT)[0]
        if opt is None:
            continue
        rec = {}
        for name, fn in variants.items():
            r = fn(m, root, seed)
            rec[name] = None if r is None else {"found": r.found, "gap": gap_percent(r.obj, opt), "lps": r.lps}
        records.append(rec)
    out = aggregate(records)
    return {"rows": out, "instances": len(records), "scanned": scanned, "seeds": seeds}


def theta_variants():
    """Rundung mit Schwelle θ und Rundung mit Propagation bei θ = 0,3 / 0,5 / 0,7 / 1/f."""
    v = {}
    for label in C.THETA_OPTIONS:
        v[f"R_near θ={label}"] = lambda m, root, seed, lab=label: round_near(m, root, theta_value(m, lab))
        v[f"R_prop θ={label}"] = lambda m, root, seed, lab=label: round_prop(m, root, theta_value(m, lab))
    return v


def dive_variants(rule):
    """Tauchen mit Rückschritten 0 / 1 / 5 / 20, ohne und mit Propagation (Regel `rule`); Guided braucht die Referenzlösung aus Runden mit Propagation."""
    v = {}
    for prop in (False, True):
        for bt in C.BACKTRACK_OPTIONS:
            def run(m, root, seed, prop=prop, bt=bt):
                p = Params(rule=rule, propagate=prop, backtracks=bt)
                if rule == "guided":
                    ref = round_prop(m, root, 0.5)
                    return dive(m, root, p, ref=ref.x) if ref.found else None
                return dive(m, root, p)
            v[f"{'mit' if prop else 'ohne'} Propagation, {bt} Rückschritte"] = run
    return v


def rule_variants():
    """Die drei Tauchregeln bei 20 Rückschritten und Propagation."""
    v = {}
    for rule in ("frac", "coef", "guided"):
        def run(m, root, seed, rule=rule):
            p = Params(rule=rule, propagate=True, backtracks=20)
            if rule == "guided":
                ref = round_prop(m, root, 0.5)
                return dive(m, root, p, ref=ref.x) if ref.found else None
            return dive(m, root, p)
        v[rule] = run
    return v


def pump_variants():
    """Pump mit 0 / 3 / 10 Neustarts, ohne und mit Zielgewicht."""
    v = {}
    for w in (0.0, 1.0):
        for r in C.RESTART_OPTIONS:
            v[f"{'mit' if w else 'ohne'} Zielgewicht, {r} Neustarts"] = lambda m, root, seed, w=w, r=r: pump(m, root, Params(restarts=r, objective_weight=w), seed=seed)
    return v


# ------------------------------------------------------------------------------------------------------------------ Verdikte
def verdict_locks(n_frac, n_blocked, lock_found, lock_gap):
    """Kernfrage 1: Welche Rundrichtung erlauben die Sperren? Rein: Zahl der gebrochenen Variablen, der in beiden Richtungen gesperrten, Ergebnis der Rundung nach Sperren."""
    if n_frac == 0:
        return "info", "Die Wurzel-LP dieser Instanz ist schon ganzzahlig: es gibt nichts zu runden. Eine andere Instanz (🎲) oder Größe zeigt mehr."
    if n_blocked == 0:
        gap = f" Ihr Wert liegt {de(lock_gap, 1)} % über dem Optimum." if lock_gap is not None else ""
        return "success", (f"Alle {n_frac} gebrochenen Variablen haben eine Richtung, in der keine Zeile sie sperrt: Runden nach Sperren liefert eine zulässige Lösung (1 LP).{gap}")
    return "warning", (f"{n_blocked} von {n_frac} gebrochenen Variablen sind in beiden Richtungen gesperrt: Jede Rundrichtung kann eine Zeile verletzen, "
                       f"Runden nach Sperren liefert hier keine Lösung." + (" (Das Ergebnis des Verfahrens bestätigt es.)" if not lock_found else ""))


def verdict_rounding(lock_found, near_found, prop_found, prop_lps):
    """Kernfrage 2: Rettet die Propagation das Runden? Rein: Erfolg der drei Rundungen auf dieser Instanz."""
    names = [n for n, f in (("nach Sperren", lock_found), ("mit Schwelle θ", near_found), ("mit Propagation", prop_found)) if f]
    if prop_found and not lock_found:
        return "success", f"Runden nach Sperren scheitert, **Runden mit Propagation gelingt** ({prop_lps} {'LP' if prop_lps == 1 else 'LPs'}): Die Propagation prüft nach jedem Runden, welche Werte die Zeilen noch zulassen."
    if prop_found and near_found:
        return "info", "Auf dieser Instanz gelingen alle drei Rundungen, auch nach Sperren: Es gibt keinen Unterschied zu zeigen."
    if prop_found:
        return "info", "Auf dieser Instanz gelingen Runden nach Sperren und Runden mit Propagation; nur Runden mit Schwelle θ scheitert."
    if names:
        return "warning", f"Runden mit Propagation scheitert; gelungen ist nur: {', '.join(names)}."
    return "error", "Keine der drei Rundungen liefert eine zulässige Lösung: Hier braucht es Tauchen oder den Pump."


def _g(gap):
    """Lücke als Text mit Komma; ohne bewiesenes Optimum fehlt sie."""
    return "–" if gap is None else de(gap, 1)


def verdict_dive_pump(dive_found, dive_lps, dive_gap, pump_found, pump_lps, pump_gap, ref_missing=False):
    """Kernfrage 3: Tauchen oder Pumpen? Rein: je Erfolg, LPs und Lücke der beiden Läufe auf dieser Instanz (die Einstellungen der Tabs)."""
    if ref_missing:
        return "warning", "Guided Tauchen braucht eine Referenzlösung; Runden mit Propagation hat keine gefunden, also läuft das Tauchen nicht."
    if dive_found and pump_found:
        cheaper = "Tauchen" if dive_lps < pump_lps else ("Pump" if pump_lps < dive_lps else "beide gleich")
        better = "Tauchen" if (dive_gap or 0) < (pump_gap or 0) else ("Pump" if (pump_gap or 0) < (dive_gap or 0) else "beide gleich")
        return "info", (f"Beide finden eine Lösung: Tauchen mit {dive_lps} LPs und {_g(dive_gap)} % Lücke, Pump mit {pump_lps} LPs und {_g(pump_gap)} % Lücke. "
                        f"Weniger LPs: {cheaper}; kleinere Lücke: {better} (diese eine Instanz).")
    if dive_found:
        return "info", f"Nur das Tauchen findet eine Lösung ({dive_lps} LPs, {_g(dive_gap)} % Lücke); der Pump scheitert nach {pump_lps} LPs."
    if pump_found:
        return "info", f"Nur der Pump findet eine Lösung ({pump_lps} LPs, {_g(pump_gap)} % Lücke); das Tauchen scheitert nach {dive_lps} LPs."
    return "error", "Weder Tauchen noch Pump finden mit diesen Einstellungen eine Lösung."


def verdict_cascade(stage, lps, found, dive_p_found, dive_p_lps):
    """Kernfrage 4: Kaskade. Rein: Stufe, die gelöst hat (0, 1, 2; 3 = keine), gesamte LPs und das Ergebnis des reinen Tauchens mit Propagation zum Vergleich."""
    if not found:
        return "error", f"Keine der drei Stufen findet eine Lösung ({lps} LPs)."
    stage_name = {0: "Runden mit Propagation (Stufe 0)", 1: "Tauchen (Stufe 1)", 2: "der Pump (Stufe 2)"}[stage]
    vs = (f"Das reine Tauchen mit Propagation braucht {dive_p_lps} LPs." if dive_p_found else "Das reine Tauchen mit Propagation findet hier keine Lösung.")
    return "success", f"Die Kaskade findet die Lösung mit {stage_name}, insgesamt {lps} {'LP' if lps == 1 else 'LPs'}. {vs}"


def sweep_verdict_theta(rows):
    """Verdikt der θ-Messung: wie viele der vier Schwellen für Runden mit Schwelle θ wie viele Erfolge haben; rein auf Zeilen aus `sweep`."""
    best = max(C.THETA_OPTIONS, key=lambda lab: (rows[f"R_near θ={lab}"]["rate"] or 0))
    r = {lab: rows[f"R_near θ={lab}"]["rate"] for lab in C.THETA_OPTIONS}
    spread = max(r.values()) - min(r.values())
    best_rate = rows[f"R_near θ={best}"]["rate"]
    prop_min = min(rows[f"R_prop θ={lab}"]["rate"] for lab in C.THETA_OPTIONS)
    prop_max = max(rows[f"R_prop θ={lab}"]["rate"] for lab in C.THETA_OPTIONS)
    if best_rate == 0:
        prop = "gelingt bei keiner Schwelle" if prop_max == 0 else f"liegt je nach θ zwischen {de(100 * prop_min, 0)} % und {de(100 * prop_max, 0)} %"
        return "info", f"Runden mit Schwelle θ gelingt bei keiner der vier Schwellen (0 % der Instanzen). Runden mit Propagation {prop}."
    return "info", (f"Runden mit Schwelle θ gelingt bei θ = {best} am häufigsten ({de(100 * best_rate, 0)} % der Instanzen); zwischen der besten und der schlechtesten Schwelle "
                    f"liegen {de(100 * spread, 0)} Prozentpunkte. Runden mit Propagation liegt je nach θ zwischen {de(100 * prop_min, 0)} % und {de(100 * prop_max, 0)} %.")


# ------------------------------------------------------------------------------------------------------------------ weitere Messungen und Verdikte
def live_row(family, size_idx, count=C.LIVE_N, start=C.LIVE_SEED0):
    """Eine Zeile der Landkarte live nachgemessen: die ersten `count` Instanzen mit gebrochener Wurzel ab Seed `start`, alle Verfahren und die Kaskade.
    Rückgabe: {rows: aggregate(...), stages: Anteile der Kaskadenstufen 0/1/2/keine, instances, scanned}."""
    seeds, scanned = fractional_seeds(family, size_idx, count, start, scan_limit=40 * count)
    recs, stages = [], [0, 0, 0, 0]
    for seed in seeds:
        rec = measure_instance(make_instance(family, size_idx, seed), seed)
        if rec is None:
            continue
        recs.append(rec["methods"])
        stages[rec["stage"]] += 1
    total = max(len(recs), 1)
    return {"rows": aggregate(recs), "stages": [x / total for x in stages], "instances": len(recs), "scanned": scanned}


def landkarte_sentence(c):
    """Ein Satz über die gespeicherte Zelle der Landkarte (Familie und Größe der aktuellen Instanz): Erfolg von Runden nach Sperren, Runden mit Propagation und der Kaskade.
    c ist eine Zelle aus phr_landkarte.json; rein."""
    n = c["n"]

    def pct(name):
        return de(100 * c["methods"][name]["ok"] / c["methods"][name]["n"], 0)

    return (f"In der vorberechneten Landkarte ({n} Instanzen dieser Familie und Größe mit gebrochener Wurzel) gelang Runden nach Sperren in {pct('R_lock')} %, Runden mit Propagation "
            f"in {pct('R_prop')} % und die Kaskade in {pct('K1')} % der Instanzen.")


def verdict_map(cells):
    """Verdikt der Landkarte über die gezeigten Zellen (Liste von Zellen aus phr_landkarte.json): in wie vielen Zellen Runden nach Sperren und Runden mit Propagation
    immer bzw. nie gelingen. rein."""
    def always(name):
        return sum(1 for c in cells if c["methods"][name]["ok"] == c["methods"][name]["n"])

    def never(name):
        return sum(1 for c in cells if c["methods"][name]["ok"] == 0)

    total = len(cells)
    return "info", (f"Von {total} Zellen (Familie × Größe) gelingt Runden nach Sperren in {always('R_lock')} bei jeder Instanz und in {never('R_lock')} bei keiner; "
                    f"Runden mit Propagation gelingt in {always('R_prop')} bei jeder und in {never('R_prop')} bei keiner Instanz. "
                    f"Die Kaskade löst jede Instanz in {always('K1')} Zellen.")


def verdict_dive_sweep(rows):
    """Verdikt der Tauchmessung (Zeilen aus `sweep` mit dive_variants): Propagation und Rückschritte, Erfolg und LPs bei 0 und 20 Rückschritten."""
    a, b = rows["ohne Propagation, 0 Rückschritte"], rows["mit Propagation, 0 Rückschritte"]
    c, d = rows["ohne Propagation, 20 Rückschritte"], rows["mit Propagation, 20 Rückschritte"]
    return "info", (f"Ohne Rückschritt gelingt das Tauchen in {de(100 * a['rate'], 0)} % der Instanzen ohne und {de(100 * b['rate'], 0)} % mit Propagation; mit 20 Rückschritten in "
                    f"{de(100 * c['rate'], 0)} % ohne ({de(c['lps'], 1)} LPs im Mittel) und {de(100 * d['rate'], 0)} % mit Propagation ({de(d['lps'], 1)} LPs).")


def verdict_pump_sweep(rows):
    """Verdikt der Pump-Messung (Zeilen aus `sweep` mit pump_variants): Erfolg und LPs ohne und mit 10 Neustarts, jeweils ohne Zielgewicht, und Lücke mit Zielgewicht."""
    a, b = rows["ohne Zielgewicht, 0 Neustarts"], rows["ohne Zielgewicht, 10 Neustarts"]
    w = rows["mit Zielgewicht, 10 Neustarts"]
    gap_a = "–" if a["gap"] is None else de(a["gap"], 1)
    gap_w = "–" if w["gap"] is None else de(w["gap"], 1)
    return "info", (f"Ohne Zielgewicht gelingt der Pump ohne Neustarts in {de(100 * a['rate'], 0)} % ({de(a['lps'], 1)} LPs), mit 10 Neustarts in {de(100 * b['rate'], 0)} % ({de(b['lps'], 1)} LPs). "
                    f"Die Lücke zum Optimum beträgt ohne Zielgewicht {gap_a} % (0 Neustarts), mit Zielgewicht {gap_w} % (10 Neustarts).")


def verdict_tree(stats, strategy):
    """Verdikt des Suchbaum-Vergleichs auf der Instanz: stats = {'none': (LPs, bewiesen), 'start': (LPs, bewiesen) oder None, 'opt': (LPs, bewiesen)}."""
    n0, p0 = stats["none"]
    if stats["start"] is None:
        return "warning", f"{strategy}: ohne Startlösung {n0} LPs ({'bewiesen' if p0 else 'Limit erreicht'}); die Kaskade hat keine Startlösung geliefert."
    n1, p1 = stats["start"]
    no, po = stats["opt"]
    diff = n0 - n1
    word = "spart" if diff > 0 else ("kostet" if diff < 0 else "ändert")
    amount = f" {abs(diff)} LPs" if diff else " nichts"
    return "info", (f"{strategy}: ohne Startlösung {n0} LPs; mit der Startlösung der Kaskade {n1} LPs (Beweis {'gelungen' if p1 else 'nicht im Limit'}); "
                    f"sie {word}{amount}. Mit dem bekannten Optimum als Startlösung wären es {no} LPs (Obergrenze des Nutzens). Die LPs der Kaskade selbst sind nicht eingerechnet.")
