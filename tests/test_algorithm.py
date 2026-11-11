"""Einheiten an den Mini-Instanzen, von Hand gerechnet (DEMO-PLAYBOOK §1b): Sperren, drei Rundungen, Propagation, Tauchen, Pumpen, Kaskade.

mini_pack (conftest): LP x = (1; 0,5; 0), Wert −7, Optimum −5. mini_gap: LP x = (1; 0,5; 0; 0,5), Wert 3,5, Optimum 4 mit x = (1, 0, 0, 1).
Handrechnung zu mini_gap: Zeile Maschine 1: 2·x(M1,A1) + 2·x(M1,A2) ≤ 3, Maschine 2 entsprechend, Auftrag 1: x(M1,A1) + x(M2,A1) = 1, Auftrag 2 entsprechend.
Nach der Substitution der Gleichungen bleibt: maximiere 2·x0 + x1 unter 0,5 ≤ x0 + x1 ≤ 1,5, also x0 = 1, x1 = 0,5 (eindeutig)."""

import numpy as np
import pytest

import phr_algorithm as A
import phr_check as K
import phr_diving as D
import phr_pump as P
import phr_rounding as R
from phr_lp import LpCounter, propagate, solve_lp
from phr_result import Params


def _root(m):
    return R.solve_root(m)


# ------------------------------------------------------------------------------------------------------------------ LP und Sperren
def test_root_lp_values(mini_pack, mini_gap):
    assert _root(mini_pack)[0] == pytest.approx([1, 0.5, 0])
    assert _root(mini_gap)[0] == pytest.approx([1, 0.5, 0, 0.5])
    assert float(mini_gap.c @ _root(mini_gap)[0]) == pytest.approx(3.5)


def test_locks_counted_per_row(mini_pack, mini_gap):
    assert list(mini_pack.uplocks) == [1, 1, 1] and list(mini_pack.downlocks) == [0, 0, 0]       # eine ≤-Zeile, positive Koeffizienten
    assert list(mini_gap.uplocks) == [2, 2, 2, 2]     # je eine ≤-Zeile (Koeffizient 2 > 0) und eine Gleichung
    assert list(mini_gap.downlocks) == [1, 1, 1, 1]   # nur die Gleichung sperrt das Verkleinern


def test_negated_row_gives_down_locks():
    from phr_scenario import gen_vertexcover
    m = gen_vertexcover(6, 3)                       # nur ≥-Zeilen (negiert gespeichert)
    assert (m.uplocks == 0).all() and (m.downlocks > 0).all()


def test_rounding_direction_and_lock_table(mini_pack, mini_gap):
    assert R.rounding_direction(mini_pack, 1) == "ab"
    t = R.lock_table(mini_gap, _root(mini_gap)[0])
    assert [(r["j"], r["down"], r["up"], r["direction"]) for r in t] == [(1, 1, 2, "keine"), (3, 1, 2, "keine")]
    free = type("M", (), {})()                       # beide Sperren 0: frei
    free.downlocks, free.uplocks = np.array([0]), np.array([0])
    assert R.rounding_direction(free, 0) == "frei"


def test_cover_theta_is_one_over_longest_ge_row():
    from phr_scenario import gen_cover_card, gen_gap, gen_setcover
    m = gen_setcover(12, 0)
    f = max(int((m.A[i] != 0).sum()) for i in range(m.A.shape[0]))
    assert R.cover_theta(m) == pytest.approx(1 / f)
    assert R.cover_theta(gen_gap(3, 6, 0)) == 0.5                       # keine ≥-Zeile
    mc = gen_cover_card(12, 1)
    assert R.cover_theta(mc) == pytest.approx(1 / max(int((mc.A[i] != 0).sum()) for i in range(mc.A.shape[0] - 1)))   # die Kardinalitätszeile zählt nicht


# ------------------------------------------------------------------------------------------------------------------ Runden
def test_round_lock_pack_rounds_down_to_the_optimum(mini_pack):
    r = R.round_lock(mini_pack, _root(mini_pack))
    assert r.found and r.obj == pytest.approx(-5) and r.lps == 1 and list(r.x) == [1, 0, 0]


def test_round_lock_gap_is_blocked(mini_gap):
    r = R.round_lock(mini_gap, _root(mini_gap))
    assert not r.found and r.fail == "Sperre" and r.info["blocked"] == 1 and r.lps == 1


def test_round_near_threshold(mini_pack, mini_gap):
    assert not R.round_near(mini_pack, _root(mini_pack), 0.5).found     # 0,5 wird aufgerundet: 2+2 > 3
    r = R.round_near(mini_pack, _root(mini_pack), 0.7)
    assert r.found and r.obj == pytest.approx(-5)
    g = R.round_near(mini_gap, _root(mini_gap), 0.5)                     # (1, 1, 0, 1): Auftrag 2 doppelt
    assert not g.found and any("Gleichung" in v for v in K.violations(mini_gap, [1, 1, 0, 1]))


def test_round_prop_gap_finds_the_optimum_with_one_lp(mini_gap):
    r = R.round_prop(mini_gap, _root(mini_gap), 0.5, trace=True)
    assert r.found and list(r.x) == [1, 0, 0, 1] and r.obj == pytest.approx(4) and r.lps == 1
    rounded = [s for s in r.steps if s.kind == "round"]
    assert len(rounded) == 1 and rounded[0].data["chosen"] == 0 and sorted(rounded[0].data["new_fixed"]) == [1, 2, 3]   # Propagation fixiert die drei übrigen


def test_round_prop_pack_equals_lock_here(mini_pack):
    r = R.round_prop(mini_pack, _root(mini_pack))
    assert r.found and list(r.x) == [1, 0, 0] and r.lps == 1


# ------------------------------------------------------------------------------------------------------------------ Propagation
def test_propagate_fixing_x1_forces_everything(mini_gap):
    lb, ub = mini_gap.lb.copy(), mini_gap.ub.copy()
    lb[1] = ub[1] = 1                                  # Auftrag 2 auf Maschine 1: dort bleibt Platz 1 → Auftrag 1 passt nicht dazu
    ok, l2, u2 = propagate(mini_gap, lb, ub)
    assert ok and list(l2) == [0, 1, 1, 0] and list(u2) == [0, 1, 1, 0]


def test_propagate_detects_a_contradiction(mini_gap):
    lb, ub = mini_gap.lb.copy(), mini_gap.ub.copy()
    lb[0] = lb[1] = 1                                  # beide Aufträge auf Maschine 1: 4 > 3
    assert not propagate(mini_gap, lb, ub)[0]


def test_propagate_leaves_free_bounds_alone(mini_pack):
    ok, lb, ub = propagate(mini_pack, mini_pack.lb, mini_pack.ub)
    assert ok and list(lb) == [0, 0, 0] and list(ub) == [1, 1, 1]      # 2 ≤ 3: jeder Gegenstand passt einzeln


# ------------------------------------------------------------------------------------------------------------------ Tauchen
def test_dive_without_backtracking_ends_in_a_dead_end(mini_gap):
    r = D.dive(mini_gap, _root(mini_gap), Params(rule="frac", propagate=False, backtracks=0), trace=True)
    assert not r.found and r.fail == "Sackgasse" and r.lps == 3             # Wurzel, x(M1,A2) = 1, dann x(M1,A1) = 1 unzulässig
    assert [s.kind for s in r.steps] == ["root", "fix", "infeasible", "fail"]


def test_dive_with_one_backtrack_flips_the_failed_fixing_and_finds_cost_5_in_four_lps(mini_gap):
    r = D.dive(mini_gap, _root(mini_gap), Params(rule="frac", propagate=False, backtracks=1), trace=True)
    assert r.found and list(r.x) == [0, 1, 1, 0] and r.obj == pytest.approx(5) and r.lps == 4 and r.info["backtracks"] == 1
    assert [s.kind for s in r.steps] == ["root", "fix", "infeasible", "backtrack", "done"]


def test_dive_with_propagation_needs_two_lps(mini_gap):
    r = D.dive(mini_gap, _root(mini_gap), Params(rule="frac", propagate=True, backtracks=0))
    assert r.found and list(r.x) == [0, 1, 1, 0] and r.lps == 2


def test_dive_candidates_order_and_rules(mini_gap):
    x = _root(mini_gap)[0]
    assert [(j, v) for _, j, v in D.dive_candidates(mini_gap, x, "frac")] == [(1, 1.0), (3, 1.0)]          # Gleichstand: kleinere Nummer zuerst
    coef = [(j, v) for _, j, v in D.dive_candidates(mini_gap, x, "coef")]
    assert coef == [(1, 0.0), (3, 0.0)]       # Abrunden hat 1 Sperre (die Gleichung), Aufrunden 2: Coefficient rundet ab, wo Fractional aufrundet
    ref = np.array([0, 1, 1, 0.0])
    guided = D.dive_candidates(mini_gap, x, "guided", ref)
    assert [(j, v) for _, j, v in guided] == [(1, 1.0), (3, 0.0)]
    with pytest.raises(ValueError):
        D.dive_candidates(mini_gap, x, "guided")
    with pytest.raises(ValueError):
        D.dive_candidates(mini_gap, x, "nope")


def test_dive_candidates_coef_prefers_the_less_locked_direction(mini_pack):
    x = np.array([1, 0.5, 0.0])
    (_, j, val), = D.dive_candidates(mini_pack, x, "coef")
    assert j == 1 and val == 0.0                                                                           # Abrunden hat keine Sperren, Aufrunden eine


# ------------------------------------------------------------------------------------------------------------------ Pump (Einheiten)
def test_distance_objective_signs(mini_gap):
    c = P.distance_objective(mini_gap, np.array([1.0, 0.0, 0.0, 1.0]))
    assert list(c) == [-1, 1, 1, -1]                                    # Wert 1 belohnt (−1), Wert 0 bestraft (+1)


def test_distance_objective_blends_in_the_cost_vector(mini_gap):
    c = P.distance_objective(mini_gap, np.array([1.0, 0.0, 0.0, 1.0]), alpha=1.0)
    assert list(c) == pytest.approx(np.sqrt(4) / np.linalg.norm(mini_gap.c) * mini_gap.c)         # α = 1: nur die Kosten, auf √|I| skaliert


def test_flip_on_stall_flips_the_largest_distances():
    xt = np.array([1.0, 0.0, 0.0, 1.0])
    xi = np.array([0.9, 0.5, 0.0, 0.7])
    assert list(P.flip_on_stall(xt, xi, 2)) == [1.0, 1.0, 0.0, 0.0]     # Abstände 0,1 / 0,5 / 0 / 0,3: die beiden größten (Position 1 und 3)


def test_perturb_on_cycle_flips_every_entry_farther_than_half_and_is_reproducible():
    from phr_scenario import SplitMix64
    xi = np.array([0.0, 0.2, 0.9])
    new = np.array([1.0, 1.0, 0.0])                       # Abstände 1,0 / 0,8 / 0,9: jeder > 0,5, also flippt jeder, egal wie ρ ausfällt
    assert list(P.perturb_on_cycle(xi, new, SplitMix64(1))) == [0.0, 0.0, 1.0]
    close = P.perturb_on_cycle(np.array([0.0, 0.0]), np.array([0.0, 0.0]), SplitMix64(5))
    assert list(close) == list(P.perturb_on_cycle(np.array([0.0, 0.0]), np.array([0.0, 0.0]), SplitMix64(5))) and set(close) <= {0.0, 1.0}


def test_restart_point_is_binary_and_reproducible():
    from phr_scenario import SplitMix64
    x0 = np.array([0.0, 1.0, 0.5, 0.5])
    a = P.restart_point(x0, SplitMix64(3))
    assert set(a) <= {0.0, 1.0} and list(a) == list(P.restart_point(x0, SplitMix64(3)))
    assert a[0] == 0.0 and a[1] == 1.0     # Rauschen höchstens ±0,5: 0 + (−0,5…0,5) < 0,5 und 1 − 0,5 ≥ 0,5 bei Gleichheit


def test_pump_on_pack_returns_a_feasible_solution(mini_pack):
    r = P.pump(mini_pack, _root(mini_pack), Params(), seed=0, trace=True)
    assert r.found and K.is_feasible(mini_pack, r.x) and r.steps[0].kind == "root" and r.steps[-1].kind == "done"


# ------------------------------------------------------------------------------------------------------------------ Kaskade
def test_cascade_stage_zero_on_mini_gap(mini_gap):
    r = A.cascade(mini_gap, _root(mini_gap), 0, trace=True)
    assert r.found and r.info["stage"] == 0 and r.lps == 1 and r.obj == pytest.approx(4)
    assert [s.kind for s in r.steps][0] == "stage"


def test_combine_cascade_counts_the_root_once():
    from phr_result import Result
    fail = lambda lps: Result("x", False, None, None, lps, 0)                    # noqa: E731
    ok = Result("x", True, np.zeros(1), 7.0, 5, 0)
    res = {"R_prop": fail(1), "D_fracP": fail(9), "FP_rs": ok}
    assert A.combine_cascade(res) == (True, 7.0, 1 + 0 + 8 + 4, 2)
    res["FP_rs"] = fail(11)
    assert A.combine_cascade(res) == (False, None, 1 + 0 + 8 + 10, 3)


def test_unknown_method_name_is_refused(mini_pack):
    with pytest.raises(ValueError):
        A.run_named(mini_pack, _root(mini_pack), "R_nothing")


# ------------------------------------------------------------------------------------------------------------------ Aufzeichnung ändert nichts
@pytest.mark.parametrize("name", A.METHODS[:-1])
def test_trace_does_not_change_the_result(mini_gap, name):
    root = _root(mini_gap)
    a = A.run_named(mini_gap, root, name, 3, trace=False)
    b = A.run_named(mini_gap, root, name, 3, trace=True)
    assert (a.found, a.lps, a.obj) == (b.found, b.lps, b.obj) and b.steps and not a.steps


def test_solve_lp_counts_one_lp_per_call(mini_gap):
    ctr = LpCounter()
    solve_lp(mini_gap, mini_gap.lb, mini_gap.ub, ctr)
    solve_lp(mini_gap, mini_gap.lb, mini_gap.ub, ctr)
    assert ctr.lps == 2 and ctr.infeasible == 0
