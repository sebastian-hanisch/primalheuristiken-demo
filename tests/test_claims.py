"""Aussagenprüfung (DEMO-PLAYBOOK §4 und §11a): jede Zahl und jede Zählung der README und der Preset-Hilfen, in der Form, in der sie dort steht.

Drei Arten von Belegen:
* **Gespeicherte Messungen** (`phr_landkarte.json`, `phr_theta.json`, `phr_baum.json`, erzeugt von tools/landkarte_messen.py, tools/theta_messen.py, tools/baum_messen.py):
  hier gelesen und gezählt, exakt. Dass die Dateien zum heutigen Code passen, prüfen die Nachmessungen unten mit Bändern (LP-Wahl von HiGHS und Rundung können sich je Plattform
  um wenige Instanzen verschieben, Playbook §4).
* **Preset-Instanzen**: die Aussage der Hilfe steht als Test an genau dieser Instanz.
* **Gegenmessung Rückschritt** (tools/rueckschritt_messen.py, Zahlen nur in der README): hier nur die Richtung an einer Stichprobe."""

import json
from pathlib import Path

import pytest

import phr_constants as C
import phr_evaluation as ev
import phr_landkarte as LK
from phr_diving import dive
from phr_result import Params
from phr_rounding import solve_root
from phr_scenario import make_instance

ROOT = Path(__file__).resolve().parent.parent
DATA = LK.load()
THETA = json.loads((ROOT / "phr_theta.json").read_text(encoding="utf-8"))
BAUM = json.loads((ROOT / "phr_baum.json").read_text(encoding="utf-8"))


def rate(family, size, method):
    c = LK.cell(DATA, family, size)
    return c["methods"][method]["ok"], c["methods"][method]["n"]


# ------------------------------------------------------------------------------------------------------------------ Umfang der Landkarte
def test_landkarte_has_24_cells_with_100_instances_each_and_names_its_unit():
    assert len(DATA["cells"]) == 24 and all(c["n"] == 100 for c in DATA["cells"])
    assert DATA["meta"]["map_n"] == 100 and "LPs" in DATA["meta"]["einheit"]
    assert sum(LK.pooled(DATA, "R_lock")["n"] for _ in [0]) == 2400


def test_thin_cells_needed_many_more_seeds_than_the_others():
    scanned = {(c["family"], c["size"]): c["scanned"] for c in DATA["cells"]}
    assert scanned[("setcover", 0)] == 1639 and scanned[("cover_card", 0)] == 1269 and scanned[("ufl", 0)] == 1591
    assert scanned[("assign_side", 0)] == 328 and scanned[("setpart", 0)] == 178
    assert max(v for k, v in scanned.items() if k not in {("setcover", 0), ("cover_card", 0), ("ufl", 0)}) <= 328


# ------------------------------------------------------------------------------------------------------------------ Hook: Sperren
@pytest.mark.parametrize("family", ["mkp", "stable", "mknap", "setcover", "vertexcover", "ufl"])
def test_lock_rounding_always_succeeds_for_pure_le_pure_ge_and_the_mixed_family(family):
    for size in (0, 1):
        ok, n = rate(family, size, "R_lock")
        assert ok == n == 100, (family, size)


@pytest.mark.parametrize("family", ["cover_card", "gap", "assign_side", "setpart", "seqprec"])
def test_lock_rounding_never_succeeds_with_equalities_or_an_opposing_row(family):
    for size in (0, 1):
        assert rate(family, size, "R_lock")[0] == 0, (family, size)


def test_lock_rounding_on_bin_packing_depends_on_the_size():
    assert [rate("binpack", s, "R_lock")[0] for s in (0, 1)] == [43, 17]


def test_class_counts_of_lock_rounding():
    classes = LK.classes()
    assert [LK.pooled(DATA, "R_lock", classes[k])["ok"] for k in C.CLASS_ORDER] == [600, 400, 60, 200]
    assert [LK.pooled(DATA, "R_lock", classes[k])["n"] for k in C.CLASS_ORDER] == [600, 600, 1000, 200]


def test_propagation_rescues_equations_with_slack_but_not_pure_equation_systems():
    assert [rate("gap", s, "R_prop")[0] for s in (0, 1)] == [83, 94]
    assert [rate("binpack", s, "R_prop")[0] for s in (0, 1)] == [99, 98]
    assert [rate("seqprec", s, "R_prop")[0] for s in (0, 1)] == [77, 47]
    for family in ("assign_side", "setpart"):
        assert [rate(family, s, "R_prop")[0] for s in (0, 1)] == [0, 0]
    assert [rate("cover_card", s, "R_prop")[0] for s in (0, 1)] == [77, 4]


def test_propagation_always_succeeds_for_le_and_pure_ge_and_mixed():
    for family in ("mkp", "stable", "mknap", "setcover", "vertexcover", "ufl"):
        for size in (0, 1):
            ok, n = rate(family, size, "R_prop")
            assert ok == n, (family, size)


def test_nearest_integer_rounding_pooled_and_for_vertex_cover():
    assert LK.pooled(DATA, "R_near")["ok"] == 665
    assert [rate("vertexcover", s, "R_near")[0] for s in (0, 1)] == [100, 100]          # Halbganzzahligkeit: x ≥ 1/2 aufrunden
    assert [rate("setcover", s, "R_near")[0] for s in (0, 1)] == [93, 27]
    assert [rate("stable", s, "R_near")[0] for s in (0, 1)] == [0, 0]


def test_rounding_gap_on_knapsack_shrinks_with_size():
    gaps = [LK.stat(LK.cell(DATA, "mkp", s), "R_lock")["gap"] for s in (0, 1)]
    assert round(gaps[0], 1) == 23.4 and round(gaps[1], 1) == 15.5


# ------------------------------------------------------------------------------------------------------------------ Tauchen und Pump
def test_pooled_diving_and_pump_numbers():
    pooled = {n: LK.pooled(DATA, n) for n in LK.MAP_METHODS}
    assert (pooled["D_frac"]["ok"], pooled["D_fracP"]["ok"], pooled["D_coefP"]["ok"]) == (2319, 2342, 2369)
    assert [round(pooled[n]["lps"], 1) for n in ("D_frac", "D_fracP", "D_coefP")] == [13.4, 9.3, 8.0]
    assert (pooled["FP_rs"]["ok"], pooled["FP_objrs"]["ok"]) == (2294, 2193)
    assert [round(pooled[n]["gap"], 1) for n in ("FP_rs", "FP_objrs")] == [33.1, 12.6]
    assert [round(pooled[n]["lps"], 1) for n in ("FP_rs", "FP_objrs")] == [22.0, 38.0]


def test_dead_ends_exist_for_diving_with_propagation_and_for_the_pump():
    assert rate("cover_card", 1, "D_fracP")[0] == 64 and rate("setpart", 1, "D_fracP")[0] == 91 and rate("seqprec", 1, "D_fracP")[0] == 90
    assert rate("binpack", 1, "FP_rs")[0] == 79 and rate("mknap", 1, "FP_rs")[0] == 63 and rate("mknap", 1, "FP_objrs")[0] == 29


def test_diving_with_propagation_is_cheaper_than_the_pump_in_seven_of_ten_equation_cells():
    expensive = [(f, s) for f in ("gap", "assign_side", "setpart", "binpack", "seqprec") for s in (0, 1)
                 if LK.stat(LK.cell(DATA, f, s), "D_fracP")["lps"] >= LK.stat(LK.cell(DATA, f, s), "FP_rs")["lps"]]
    assert expensive == [("gap", 1), ("seqprec", 0), ("seqprec", 1)]


def test_pump_with_objective_weight_has_the_smaller_gap_in_most_cells():
    smaller = sum(1 for c in DATA["cells"] if LK.stat(c, "FP_objrs")["gap"] is not None and LK.stat(c, "FP_rs")["gap"] is not None
                  and LK.stat(c, "FP_objrs")["gap"] < LK.stat(c, "FP_rs")["gap"])
    assert smaller == 22


# ------------------------------------------------------------------------------------------------------------------ Kaskade
def test_cascade_pooled_numbers_and_stage_counts():
    k1, d = LK.pooled(DATA, "K1"), LK.pooled(DATA, "D_fracP")
    assert (k1["ok"], d["ok"], k1["n"]) == (2399, 2342, 2400)
    assert [round(k1["lps"], 2), round(d["lps"], 2)] == [5.79, 9.27]
    assert [round(k1["gap"], 1), round(d["gap"], 1)] == [22.0, 7.1]
    assert [sum(c["stages"][k] for c in DATA["cells"]) for k in range(4)] == [1779, 566, 54, 1]


def test_cascade_stages_by_class_and_family():
    stages = {k: LK.stage_shares(DATA, fams) for k, fams in LK.classes().items()}
    assert stages["≤"][0] == 1.0 and stages["gemischt"][0] == 1.0
    assert round(stages["="][1], 3) == 0.483 and round(stages["≥"][2], 3) == 0.06 and round(stages["="][0], 3) == 0.498
    for s in (0, 1):
        assert LK.cell(DATA, "assign_side", s)["stages"] == [0, 100, 0, 0]
        assert LK.cell(DATA, "setpart", s)["stages"][0] == 0
    assert LK.cell(DATA, "cover_card", 1)["stages"] == [4, 60, 36, 0] and LK.cell(DATA, "setpart", 1)["stages"] == [0, 91, 8, 1]


def test_cascade_takes_the_rounding_solution_where_stage_zero_suffices():
    for family in ("mkp", "stable", "mknap", "setcover", "vertexcover", "ufl"):
        for s in (0, 1):
            c = LK.cell(DATA, family, s)
            assert c["stages"][0] == 100 and LK.stat(c, "K1")["gap"] == pytest.approx(LK.stat(c, "R_prop")["gap"])      # Stufe 0 genügt: Lösung und Lücke sind die der Rundung


def test_cascade_gap_is_larger_than_that_of_diving_in_15_of_24_cells_not_in_the_knapsack_cells():
    worse = [(c["family"], c["size"]) for c in DATA["cells"] if LK.stat(c, "K1")["gap"] > LK.stat(c, "D_fracP")["gap"]]
    assert len(worse) == 15
    assert ("mkp", 0) not in worse and ("mkp", 1) not in worse and ("gap", 0) not in worse
    assert set(LK.stat(LK.cell(DATA, f, 0), "K1")["gap"] > LK.stat(LK.cell(DATA, f, 0), "D_fracP")["gap"] for f in ("vertexcover", "setcover", "stable", "ufl")) == {True}


# ------------------------------------------------------------------------------------------------------------------ Rundungsschwelle
def _theta(kind, family, label):
    acc = [c["acc"][f"{kind}|{label}"] for c in THETA["cells"] if c["family"] == family]
    n, ok = sum(a["n"] for a in acc), sum(a["ok"] for a in acc)
    return ok, n, (sum(a["gap_sum"] for a in acc) / ok if ok else None)


def test_no_single_threshold_is_best_for_every_family():
    best_sets = []
    for family in C.FAMILY_KEYS:
        rates = {lab: _theta("near", family, lab)[0] for lab in C.THETA_OPTIONS}
        top = max(rates.values())
        if top > 0:
            best_sets.append({lab for lab, r in rates.items() if r == top})
    assert len(best_sets) == 10                                      # assign_side und setpart: bei keiner Schwelle ein Erfolg
    assert not set.intersection(*best_sets)


def test_threshold_numbers_quoted_in_the_readme():
    assert [_theta("near", "stable", lab)[0] for lab in C.THETA_OPTIONS] == [0, 0, 200, 0]
    assert [_theta("near", "setcover", lab)[0] for lab in C.THETA_OPTIONS] == [184, 120, 0, 200]
    assert [_theta("near", "vertexcover", lab)[0] for lab in C.THETA_OPTIONS] == [200, 200, 0, 200]
    assert [_theta("near", "ufl", lab)[0] for lab in C.THETA_OPTIONS] == [200, 177, 1, 177]
    assert [_theta("near", "mkp", lab)[0] for lab in C.THETA_OPTIONS] == [4, 59, 160, 59]
    for family in ("assign_side", "setpart"):
        assert [_theta("near", family, lab)[0] for lab in C.THETA_OPTIONS] == [0, 0, 0, 0]


def test_more_aggressive_threshold_buys_success_with_gap_on_set_cover():
    ok3, _, gap3 = _theta("near", "setcover", "0,3")
    okf, _, gapf = _theta("near", "setcover", "1/f")
    assert ok3 < okf == 200 and gap3 < gapf


def test_rounding_with_propagation_hardly_depends_on_the_threshold_except_for_cover_card():
    for family in ("gap", "binpack", "seqprec"):
        assert len({_theta("prop", family, lab)[0] for lab in C.THETA_OPTIONS}) == 1, family
    rates = [_theta("prop", "cover_card", lab)[0] for lab in C.THETA_OPTIONS]
    assert rates == [77, 81, 103, 73]


# ------------------------------------------------------------------------------------------------------------------ Suchbaum
def _tree(strategy, family):
    acc = [c["strategies"][strategy] for c in BAUM["cells"] if c["family"] == family]
    t = {k: sum(a[k] for a in acc) for k in ("paired", "none", "start", "casc", "opt", "censored")}
    return t, 100 * (t["none"] - t["start"]) / t["none"], 100 * (t["none"] - t["start"] - t["casc"]) / t["none"]


def _band_sign(strategy, family):
    lo, hi = BAUM["netto_band_95"][f"{strategy}|{family}"]
    return "positiv" if lo > 0 else ("negativ" if hi < 0 else "offen")


def test_start_solution_net_gain_for_depth_first_search_is_positive_in_three_families_and_open_in_three():
    signs = {f: _band_sign("Tiefensuche", f) for f in C.FAMILY_KEYS}
    assert [f for f, v in signs.items() if v == "positiv"] == ["gap", "binpack", "mkp"]
    assert [f for f, v in signs.items() if v == "offen"] == ["seqprec", "setcover", "cover_card"]
    assert [f for f, v in signs.items() if v == "negativ"] == ["assign_side", "setpart", "stable", "mknap", "vertexcover", "ufl"]
    assert [round(_tree("Tiefensuche", f)[1], 1) for f in ("gap", "binpack", "mkp")] == [45.9, 16.1, 16.3]
    assert [round(_tree("Tiefensuche", f)[2], 1) for f in ("gap", "binpack", "mkp")] == [43.9, 12.9, 15.9]
    assert [[round(x, 1) for x in BAUM["netto_band_95"][f"Tiefensuche|{f}"]] for f in ("gap", "binpack", "mkp")] == [[33.4, 55.3], [7.0, 20.8], [11.5, 20.8]]


def test_start_solution_hardly_helps_best_bound_search():
    signs = {f: _band_sign("Beste Schranke zuerst", f) for f in C.FAMILY_KEYS}
    assert [f for f, v in signs.items() if v == "positiv"] == ["binpack"] and [f for f, v in signs.items() if v == "offen"] == ["gap"]
    assert round(_tree("Beste Schranke zuerst", "binpack")[1], 1) == 22.8 and round(_tree("Beste Schranke zuerst", "binpack")[2], 1) == 21.9
    assert round(_tree("Beste Schranke zuerst", "gap")[1], 1) == 4.3 and round(_tree("Beste Schranke zuerst", "gap")[2], 1) == -1.6
    assert max(_tree("Beste Schranke zuerst", f)[1] for f in C.FAMILY_KEYS if f != "binpack") < 5


def test_tree_measurement_pairs_and_censoring():
    assert BAUM["limit"] == 1000 and BAUM["count"] == 60
    assert _tree("Tiefensuche", "gap")[0]["paired"] == 120
    assert {f: _tree("Tiefensuche", f)[0]["censored"] for f in ("mkp", "cover_card", "mknap", "binpack")} == {"mkp": 7, "cover_card": 6, "mknap": 4, "binpack": 4}
    assert _tree("Beste Schranke zuerst", "binpack")[0]["censored"] == 22 and _tree("Beste Schranke zuerst", "mknap")[0]["censored"] == 4


# ------------------------------------------------------------------------------------------------------------------ Gegenmessung Rückschritt
def test_flipping_the_failed_fixing_first_helps_on_the_multiple_knapsack_sample():
    ok = {True: 0, False: 0}
    seeds, _ = ev.fractional_seeds("mknap", 0, 25, 0)
    for seed in seeds:
        m = make_instance("mknap", 0, seed)
        root = solve_root(m)
        for flip in (True, False):
            ok[flip] += dive(m, root, Params(rule="frac", propagate=False, flip_failed=flip)).found
    assert ok[True] >= 23 and ok[False] <= 14, ok            # lokal 25 gegen 8 von 25; Band wegen der LP-Wahl je Plattform


# ------------------------------------------------------------------------------------------------------------------ Nachmessung: passen die gespeicherten Dateien zum Code?
@pytest.mark.parametrize("family,size", [("stable", 0), ("vertexcover", 0), ("gap", 0)])
def test_stored_landkarte_cells_are_reproduced_within_bands(family, size):
    c = LK.cell(DATA, family, size)
    seeds, scanned = ev.fractional_seeds(family, size, 100, 0)
    assert scanned == c["scanned"]
    recs = [ev.measure_instance(make_instance(family, size, s), s) for s in seeds]
    for name in LK.MAP_METHODS:
        ok = sum(r["methods"][name]["found"] for r in recs)
        lps = sum(r["methods"][name]["lps"] for r in recs) / len(recs)
        stored = c["methods"][name]
        assert abs(ok - stored["ok"]) <= 6, (name, ok, stored["ok"])
        assert abs(lps - stored["lps_sum"] / stored["n"]) <= 0.25 * lps + 1.0, (name, lps)
    stages = [sum(1 for r in recs if r["stage"] == k) for k in range(4)]
    assert all(abs(a - b) <= 6 for a, b in zip(stages, c["stages"]))


# ------------------------------------------------------------------------------------------------------------------ Presets
def _analysis(name):
    return ev.analyse(ev.Settings(**C.PRESETS[name]))


def test_preset_standard_case_gap():
    a = _analysis("Aufträge auf Maschinen (Standardfall)")
    assert len(a.m.ints) == 18 and a.n_frac == 6
    assert not a.lock.found and not a.near.found and a.prop.found and a.prop.lps == 1 and ev.gap_percent(a.prop.obj, a.opt) == pytest.approx(0.0, abs=1e-9)
    assert a.casc.info["stage"] == 0 and a.base_dive.found and 4 <= a.base_dive.lps <= 7


def test_preset_dead_end_of_rounding_with_propagation():
    a = _analysis("Sackgasse beim Runden mit Propagation")
    assert not a.prop.found and a.prop.fail == "Sackgasse" and a.base_dive.found and a.base_dive.lps == 5 and a.casc.info["stage"] == 1


def test_preset_pump_needed():
    a = _analysis("Der Pump wird gebraucht")
    assert len(a.m.ints) == 60 and a.m.Aeq.shape[0] == 20 and a.m.A.shape[0] == 0
    assert a.prop.fail == "Sackgasse" and not a.base_dive.found and a.base_dive.fail == "Sackgasse" and 30 <= a.base_dive.lps <= 45
    assert a.casc.info["stage"] == 2 and a.casc.found and a.casc.info["stage_results"][2].found and 10 <= a.casc.info["stage_results"][2].lps <= 20


def test_preset_assign_side_pump_with_cycles_and_restarts_but_diving_is_cheaper():
    a = _analysis("Zuordnung mit Zusatzgleichungen")
    assert len(a.m.ints) == 25 and a.m.A.shape[0] == 0
    assert not (a.lock.found or a.near.found or a.prop.found)
    assert a.pump.found and a.pump.info["cycle_steps"] >= 1 and a.pump.info["restarts"] == 2 and 30 <= a.pump.lps <= 48
    assert a.base_dive.found and a.base_dive.lps == 3


def test_preset_knapsack_rounding_down_is_feasible_but_far_from_the_optimum():
    a = _analysis("Rucksack: Abrunden genügt")
    assert len(a.m.ints) == 12 and a.m.A.shape[0] == 5 and a.m.Aeq.shape[0] == 0
    assert a.lock.found and round(ev.gap_percent(a.lock.obj, a.opt), 1) == 26.1
    assert a.prop.found and ev.gap_percent(a.prop.obj, a.opt) == pytest.approx(0.0, abs=1e-9)


def test_preset_opposing_row():
    a = _analysis("Eine gegenläufige Zeile")
    assert a.m.senses.count(">=") == 12 and a.m.senses.count("<=") == 1
    assert not a.lock.found and a.prop.found and round(ev.gap_percent(a.prop.obj, a.opt), 1) == 8.6


# ------------------------------------------------------------------------------------------------------------------ Verdikte der Kernfragen je Preset
@pytest.mark.parametrize("name,kind_locks", [("Aufträge auf Maschinen (Standardfall)", "warning"), ("Rucksack: Abrunden genügt", "success"), ("Eine gegenläufige Zeile", "warning")])
def test_lock_verdict_matches_the_structure_of_each_preset(name, kind_locks):
    from phr_rounding import lock_table
    a = _analysis(name)
    blocked = sum(1 for t in lock_table(a.m, a.root[0]) if t["direction"] == "keine")
    assert ev.verdict_locks(a.n_frac, blocked, a.lock.found, ev.gap_percent(a.lock.obj, a.opt))[0] == kind_locks
    assert (blocked == 0) == a.lock.found


# ------------------------------------------------------------------------------------------------------------------ Zusätzliche Zahlen der README
def test_readme_totals_of_the_single_methods():
    assert LK.pooled(DATA, "R_lock")["ok"] == 1260 and LK.pooled(DATA, "R_prop")["ok"] == 1779
    assert LK.pooled(DATA, "R_near")["ok"] == 665


def test_gap_of_lock_rounding_by_class_and_for_the_mixed_family():
    classes = LK.classes()
    assert [round(LK.pooled(DATA, "R_lock", classes[k])["gap"], 1) for k in ("≤", "≥")] == [49.8, 74.1]
    assert [round(LK.stat(LK.cell(DATA, "ufl", s), "R_lock")["gap"], 1) for s in (0, 1)] == [52.3, 60.4]


def test_gap_family_pump_and_diving_numbers():
    k, m = LK.cell(DATA, "gap", 0), LK.cell(DATA, "gap", 1)
    assert (LK.stat(k, "FP_rs")["ok"], round(LK.stat(k, "FP_rs")["lps"], 1)) == (93, 35.9)
    assert (LK.stat(m, "FP_rs")["ok"], round(LK.stat(m, "FP_rs")["lps"], 1)) == (100, 5.6)
    assert (LK.stat(k, "D_fracP")["ok"], round(LK.stat(k, "D_fracP")["lps"], 1)) == (100, 4.4)
    assert rate("cover_card", 0, "D_fracP")[0] == 100


def test_set_partition_start_solution_costs_more_than_it_saves():
    t, brutto, netto = _tree("Tiefensuche", "setpart")
    assert round(brutto, 1) == 12.8 and round(netto, 1) == -76.5


# ------------------------------------------------------------------------------------------------------------------ Gegenmessung Rückschritt (gespeichert)
RUECK = json.loads((ROOT / "phr_rueckschritt.json").read_text(encoding="utf-8"))


def _rueck(name, flip, family=None, size=None):
    cells = [c for c in RUECK["cells"] if (family is None or c["family"] == family) and (size is None or c["size"] == size)]
    n = sum(c["results"][f"{name}|{flip}"]["n"] for c in cells)
    ok = sum(c["results"][f"{name}|{flip}"]["ok"] for c in cells)
    lps = sum(c["results"][f"{name}|{flip}"]["lps"] for c in cells)
    return ok, n, lps / n


def test_stored_backtracking_counter_measurement_pooled_numbers():
    assert [_rueck(n, True)[:2] for n in ("D_frac", "D_fracP", "D_coefP")] == [(2319, 2400), (2342, 2400), (2369, 2400)]
    assert [_rueck(n, False)[:2] for n in ("D_frac", "D_fracP", "D_coefP")] == [(2091, 2400), (2256, 2400), (2297, 2400)]
    assert [round(_rueck(n, flip)[2], 2) for n in ("D_frac", "D_fracP") for flip in (True, False)] == [13.43, 14.30, 9.27, 9.48]


def test_stored_backtracking_counter_measurement_cells_quoted_in_the_readme():
    assert [(_rueck("D_frac", flip, "mknap", s)[0]) for s in (0, 1) for flip in (True, False)] == [100, 31, 100, 10]
    assert [_rueck("D_frac", flip, "setpart", 0)[0] for flip in (True, False)] == [100, 65]
    assert [_rueck("D_frac", flip, "assign_side", 0)[0] for flip in (True, False)] == [100, 83]
    assert [_rueck("D_frac", flip, "gap", 1)[0] for flip in (True, False)] == [89, 92]
    assert [_rueck("D_fracP", flip, "seqprec", 1)[0] for flip in (True, False)] == [90, 95]
    # dieselben Instanzen wie die Landkarte: das Tauchen mit Rückschritt der Landkarte ist die Variante flip=True
    assert _rueck("D_fracP", True)[0] == LK.pooled(DATA, "D_fracP")["ok"] and _rueck("D_frac", True)[0] == LK.pooled(DATA, "D_frac")["ok"]


# ------------------------------------------------------------------------------------------------------------------ Verdikt „Rundung“ je Preset und Seed (Playbook §11a Punkt 7)
@pytest.mark.parametrize("name,fragment", [
    ("Aufträge auf Maschinen (Standardfall)", "Runden mit Propagation gelingt"),
    ("Sackgasse beim Runden mit Propagation", "Keine der drei Rundungen"),
    ("Der Pump wird gebraucht", "Keine der drei Rundungen"),
    ("Zuordnung mit Zusatzgleichungen", "Keine der drei Rundungen"),
    ("Rucksack: Abrunden genügt", "gelingen alle drei Rundungen"),
    ("Eine gegenläufige Zeile", "Runden mit Propagation gelingt"),
])
def test_rounding_verdict_line_of_every_preset(name, fragment):
    a = _analysis(name)
    assert fragment in ev.verdict_rounding(a.lock.found, a.near.found, a.prop.found, a.prop.lps)[1]


@pytest.mark.parametrize("seed,fragment", [(0, "gelingen alle drei Rundungen"), (1, "nur Runden mit Schwelle θ scheitert"), (2, "nur Runden mit Schwelle θ scheitert")])
def test_rounding_verdict_never_claims_all_three_when_nearest_integer_rounding_fails(seed, fragment):
    a = ev.analyse(ev.Settings(family="mkp", size="klein", seed=seed))
    assert a.lock.found and a.prop.found and a.near.found == (seed == 0)
    assert fragment in ev.verdict_rounding(a.lock.found, a.near.found, a.prop.found, a.prop.lps)[1]


# ------------------------------------------------------------------------------------------------------------------ Nachträge nach der unabhängigen Aussagenprüfung
def test_pump_versus_diving_per_cell_counts():
    cells = DATA["cells"]
    assert sum(1 for c in cells if LK.stat(c, "FP_rs")["lps"] > LK.stat(c, "D_fracP")["lps"]) == 11
    assert sum(1 for c in cells if LK.stat(c, "FP_rs")["lps"] == LK.stat(c, "D_fracP")["lps"]) == 0
    assert sum(1 for c in cells if LK.stat(c, "FP_rs")["gap"] > LK.stat(c, "D_fracP")["gap"]) == 24
    assert round(100 * LK.pooled(DATA, "FP_rs")["rate"], 1) == 95.6


def test_cascade_gap_and_lps_compared_with_diving_per_cell():
    cells = DATA["cells"]
    gap = lambda c, m: LK.stat(c, m)["gap"]                        # noqa: E731
    larger = [c for c in cells if gap(c, "K1") > gap(c, "D_fracP") + 1e-9]
    equal = [(c["family"], c["size"]) for c in cells if abs(gap(c, "K1") - gap(c, "D_fracP")) <= 1e-9]
    smaller = [(c["family"], c["size"]) for c in cells if gap(c, "K1") < gap(c, "D_fracP") - 1e-9]
    assert len(larger) == 15
    assert equal == [("assign_side", 0), ("assign_side", 1), ("setpart", 0)]
    assert smaller == [("gap", 0), ("gap", 1), ("setpart", 1), ("seqprec", 1), ("mkp", 0), ("mkp", 1)]
    more_lps = [(c["family"], c["size"]) for c in cells if LK.stat(c, "K1")["lps"] > LK.stat(c, "D_fracP")["lps"]]
    assert more_lps == [("setpart", 1), ("cover_card", 1)]
    assert [round(LK.stat(LK.cell(DATA, "setpart", 1), m)["lps"], 1) for m in ("K1", "D_fracP")] == [28.6, 21.4]
    assert [round(LK.stat(LK.cell(DATA, "cover_card", 1), m)["lps"], 1) for m in ("K1", "D_fracP")] == [56.7, 54.5]


def test_objective_weight_lowers_the_gap_but_also_the_success_rate_on_knapsack_cells():
    assert [rate("mknap", s, "FP_rs")[0] for s in (0, 1)] == [75, 63] and [rate("mknap", s, "FP_objrs")[0] for s in (0, 1)] == [28, 29]
    assert [rate("mkp", 0, "FP_rs")[0], rate("mkp", 0, "FP_objrs")[0]] == [100, 87]


def test_failed_pump_runs_hit_the_iteration_limit_and_failed_dives_end_in_dead_ends():
    from phr_algorithm import run_named
    reasons = set()
    seeds, _ = ev.fractional_seeds("binpack", 1, 40, 0)
    for seed in seeds:
        m = make_instance("binpack", 1, seed)
        r = run_named(m, solve_root(m), "FP_rs", seed)
        if not r.found:
            reasons.add(r.fail)
    assert reasons == {"Iterationsgrenze"}
    reasons, failures = set(), 0
    seeds, _ = ev.fractional_seeds("cover_card", 1, 20, 0)
    for seed in seeds:
        m = make_instance("cover_card", 1, seed)
        r = run_named(m, solve_root(m), "D_fracP", seed)
        if not r.found:
            failures += 1
            reasons.add(r.fail)
    assert failures >= 3 and reasons == {"Sackgasse"}


def test_threshold_gap_of_set_cover_by_size():
    def gap(size, label):
        e = next(c for c in THETA["cells"] if c["family"] == "setcover" and c["size"] == size)["acc"][f"near|{label}"]
        return e["ok"], round(e["gap_sum"] / e["ok"], 1)
    assert [gap(0, "0,3"), gap(0, "1/f"), gap(1, "0,3"), gap(1, "1/f")] == [(99, 45.9), (100, 47.6), (85, 46.0), (100, 97.6)]


def test_backtracking_numbers_quoted_in_the_readme():
    four = [("mknap", 0), ("mknap", 1), ("setpart", 0), ("assign_side", 0)]
    diff = sum(_rueck("D_frac", True, f, s)[0] - _rueck("D_frac", False, f, s)[0] for f, s in four)
    assert diff == 211 and _rueck("D_frac", True)[0] - _rueck("D_frac", False)[0] == 228
    assert round(100 * _rueck("D_fracP", False)[0] / 2400, 1) == 94.0 and round(100 * _rueck("D_fracP", True)[0] / 2400, 1) == 97.6
    assert round(100 * (1 - LK.pooled(DATA, "D_fracP")["lps"] / LK.pooled(DATA, "D_frac")["lps"])) == 31


# ------------------------------------------------------------------------------------------------------------------ Zielgewicht gepaart (Folgekorrektur nach der Erklärseite)
ZIEL = json.loads((ROOT / "phr_zielgewicht.json").read_text(encoding="utf-8"))


def test_paired_objective_weight_lowers_the_gap_in_21_of_24_cells_and_not_on_three():
    cells = ZIEL["cells"]
    assert len(cells) == 24
    lower = [c for c in cells if c["gap_with_both"] < c["gap_without_both"]]
    assert len(lower) == 21
    assert [(c["family"], c["size"]) for c in cells if c not in lower] == [("binpack", 1), ("seqprec", 0), ("mknap", 1)]


def test_paired_objective_weight_numbers_quoted_in_the_readme():
    def cell(f, s):
        return next(c for c in ZIEL["cells"] if c["family"] == f and c["size"] == s)
    assert [(cell("binpack", 1)["both"], round(cell("binpack", 1)["gap_without_both"], 1), round(cell("binpack", 1)["gap_with_both"], 1))] == [(67, 20.8, 21.9)]
    assert [round(cell("seqprec", 0)[k], 1) for k in ("gap_without_both", "gap_with_both")] == [2.2, 2.9]
    assert [(cell("mknap", 1)["both"], round(cell("mknap", 1)["gap_without_both"], 1), round(cell("mknap", 1)["gap_with_both"], 1))] == [(28, 14.4, 21.3)]
    assert [round(cell("mknap", 0)[k], 1) for k in ("gap_without_both", "gap_with_both")] == [11.6, 10.3]
    unpaired = [round(LK.stat(LK.cell(DATA, "mknap", s), m)["gap"], 1) for s in (0, 1) for m in ("FP_rs", "FP_objrs")]
    assert unpaired == [20.3, 10.3, 20.7, 20.5]
    # dieselben Instanzen wie die Landkarte: Erfolgszahlen stimmen überein
    assert all(cell(c["family"], c["size"])["ok_without"] == LK.stat(LK.cell(DATA, c["family"], c["size"]), "FP_rs")["ok"] for c in DATA["cells"])


def test_lock_rounding_counts_for_the_six_families_with_equations():
    assert sum(rate(f, s, "R_lock")[0] for f in ("gap", "assign_side", "setpart", "seqprec") for s in (0, 1)) == 0
    assert sum(rate("binpack", s, "R_lock")[0] for s in (0, 1)) == 60
    assert sum(rate("ufl", s, "R_lock")[0] for s in (0, 1)) == 200


def test_failed_pump_runs_on_the_small_knapsack_cell_also_hit_the_iteration_limit():
    from phr_algorithm import run_named
    reasons = set()
    seeds, _ = ev.fractional_seeds("mknap", 0, 25, 0)
    for seed in seeds:
        m = make_instance("mknap", 0, seed)
        r = run_named(m, solve_root(m), "FP_rs", seed)
        if not r.found:
            reasons.add(r.fail)
    assert reasons == {"Iterationsgrenze"}
