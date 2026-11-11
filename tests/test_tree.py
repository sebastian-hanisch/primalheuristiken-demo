"""Suchbaum (Tiefensuche, beste Schranke) gegen das Optimum: Orakel HiGHS-MILP und Aufzählung; Startlösung und Beweis."""

import math

import pytest

import phr_algorithm as A
import phr_rounding as R
import phr_tree as T
from phr_exact import enumerate_optimum, solve_exact
from phr_scenario import make_instance

FAMILIES = ["gap", "mkp", "stable", "vertexcover", "setpart", "binpack", "mknap", "seqprec", "setcover", "ufl", "assign_side", "cover_card"]


@pytest.mark.parametrize("strategy", list(T.STRATEGIES))
def test_search_proves_the_optimum_of_small_instances(strategy):
    run = T.STRATEGIES[strategy]
    proved = 0
    for fam in FAMILIES:
        for seed in range(4):
            m = make_instance(fam, 0, seed)
            r = run(m, math.inf, 3000)
            opt = solve_exact(m)[0]
            if r["proved"]:
                proved += 1
                assert r["best"] == pytest.approx(opt, abs=1e-6), (fam, seed)
    assert proved >= 40


def test_search_equals_enumeration_on_tiny_instances():
    for seed in range(5):
        m = make_instance("gap", 0, seed)
        assert T.depth_first(m, math.inf, 3000)["best"] == pytest.approx(enumerate_optimum(m)[0])
        assert T.best_bound(m, math.inf, 3000)["best"] == pytest.approx(enumerate_optimum(m)[0])


def test_a_start_solution_never_adds_nodes_to_the_depth_first_search():
    """Mit einer bekannten Lösung als obere Grenze ist der Baum der Tiefensuche ein Teilbaum des Baums ohne sie: nie mehr LPs. Mit dem Optimum als Grenze bleibt der Wert erhalten."""
    fewer = 0
    for fam in ("gap", "mkp", "setpart", "binpack"):
        for seed in range(6):
            m = make_instance(fam, 0, seed)
            casc = A.cascade(m, R.solve_root(m), seed)
            base = T.depth_first(m, math.inf, 3000)
            if not casc.found or not base["proved"]:
                continue
            with_start = T.depth_first(m, casc.obj, 3000)
            assert with_start["nodes"] <= base["nodes"] and with_start["best"] == pytest.approx(base["best"]), (fam, seed)
            with_opt = T.depth_first(m, base["best"], 3000)
            assert with_opt["nodes"] <= with_start["nodes"] and with_opt["best"] == pytest.approx(base["best"]), (fam, seed)
            fewer += with_start["nodes"] < base["nodes"]
    assert fewer >= 3


def test_node_limit_stops_the_search_without_proof():
    m = make_instance("mknap", 1, 0)
    r = T.depth_first(m, math.inf, 5)
    assert r["nodes"] <= 6 and not r["proved"]
    assert not T.best_bound(m, math.inf, 5)["proved"]


def test_integral_root_is_proved_with_one_node():
    m = make_instance("setcover", 0, 0)                      # Wurzel ganzzahlig
    assert T.depth_first(m, math.inf, 100) == {"nodes": 1, "best": pytest.approx(solve_exact(m)[0]), "proved": True}
