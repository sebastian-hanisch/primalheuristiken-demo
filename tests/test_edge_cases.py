"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): ganzzahlige Wurzel, genau zwei Variablen, Grenzen inklusive, unzulässige Relaxation,
Obergrenzen der LPs und Iterationen, Gleichstände."""

import numpy as np
import pytest

import phr_algorithm as A
import phr_diving as D
import phr_pump as P
import phr_rounding as R
from phr_result import Params
from phr_scenario import Mip, make_instance


def _pack(cap, n=3):
    return Mip("pack", [-5, -4, -3][:n], [[2] * n], [cap], None, None, np.zeros(n), np.ones(n), np.arange(n), True, [f"G{i + 1}" for i in range(n)], (1, n), [""],
               [f"G{i + 1}" for i in range(n)])


def test_integral_root_needs_one_lp_for_rounding_and_diving_and_two_for_the_pump():
    m = _pack(6)                                          # alle drei passen: LP-Lösung (1, 1, 1) ist ganzzahlig
    root = R.solve_root(m)
    assert list(root[0]) == [1, 1, 1]
    for name in ("R_lock", "R_near", "R_prop", "D_frac", "D_fracP", "D_coefP"):
        r = A.run_named(m, root, name)
        assert r.found and r.lps == 1 and r.obj == pytest.approx(-12), name
    r = A.run_named(m, root, "FP_rs")
    assert r.found and r.lps == 2 and r.info["iterations"] == 1                 # das erste Pump-LP liefert schon die ganzzahlige Lösung
    assert A.cascade(m, root, 0).info["stage"] == 0


def test_integral_root_with_continuous_variables_needs_the_extra_lp_for_rounding():
    m = make_instance("ufl", 0, 0)                        # Wurzel dieser Instanz ist ganzzahlig; die stetige Zuordnung braucht nach dem Runden ein LP
    root = R.solve_root(m)
    assert not len(R.frac_indices(m, root[0]))
    assert A.run_named(m, root, "R_near").lps == 2 and A.run_named(m, root, "R_lock").lps == 1


def test_exactly_two_variables():
    m = _pack(3, n=2)                                      # x1 + x2 ≤ 1,5: LP x = (1; 0,5)
    root = R.solve_root(m)
    assert list(root[0]) == pytest.approx([1, 0.5])
    for name in A.METHODS[:-1]:
        r = A.run_named(m, root, name, 1)
        if name == "R_near":
            assert not r.found                          # 0,5 wird aufgerundet: (1, 1) braucht Kapazität 4
            continue
        assert r.found and r.obj in (-5, -4), name      # nur ein Gegenstand passt: höchstens −5
    assert A.run_named(m, root, "R_lock").obj == -5


def test_infeasible_relaxation_returns_none():
    m = Mip("bad", [1], [[-1]], [-2], None, None, [0], [1], [0], True, ["x"], (1, 1), [""], ["x"])    # −x ≤ −2, also x ≥ 2 > 1
    assert R.solve_root(m) is None


def test_dive_lp_limit_is_inclusive_and_reports_the_reason():
    m = make_instance("setpart", 1, 20)
    root = R.solve_root(m)
    r = D.dive(m, root, Params(rule="frac", propagate=False, max_lps=2))
    assert not r.found and r.fail == "LP-Grenze" and r.lps == 2


def test_pump_iteration_limit_is_inclusive_and_reports_the_reason():
    m = make_instance("setpart", 1, 20)
    root = R.solve_root(m)
    r = P.pump(m, root, Params(maxit=3, restarts=0), seed=0)
    assert not r.found and r.fail == "Iterationsgrenze" and r.info["iterations"] == 3 and r.lps == 4      # Wurzel + 3 Pump-LPs


def test_pump_without_restarts_never_restarts_and_with_restarts_does_on_a_hard_instance():
    m = make_instance("mknap", 0, 0)
    root = R.solve_root(m)
    assert P.pump(m, root, Params(restarts=0, maxit=60), seed=1).info["restarts"] == 0
    assert P.pump(m, root, Params(restarts=3, maxit=60), seed=1).info["restarts"] >= 1


def test_ties_in_candidates_are_broken_by_variable_number():
    x = np.array([0.5, 0.5, 0.5])
    m = _pack(1.0)
    assert [j for _, j, _ in D.dive_candidates(m, x, "frac")] == [0, 1, 2]


def test_guided_dive_needs_its_reference():
    m = make_instance("gap", 0, 0)
    root = R.solve_root(m)
    with pytest.raises(ValueError):
        D.dive(m, root, Params(rule="guided"))


def test_unknown_rule_is_refused():
    m = _pack(3)
    with pytest.raises(KeyError):
        D.dive(m, R.solve_root(m), Params(rule="nope"))


def test_theta_value_one_over_f_and_fixed_values():
    import phr_evaluation as ev
    assert ev.theta_value(make_instance("gap", 0, 0), "1/f") == 0.5               # keine ≥-Zeile
    assert ev.theta_value(make_instance("gap", 0, 0), "0,3") == 0.3
    sc = make_instance("setcover", 1, 0)
    assert ev.theta_value(sc, "1/f") < 0.5
