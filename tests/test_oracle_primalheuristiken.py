"""Orakel-Tests (DEMO-PLAYBOOK §12): jede Kernroutine gegen einen ANDEREN Rechenweg.

* Zulässigkeitsprüfer und Zielfunktion gegen die Definition jeder Familie, gelesen aus den Rohdaten der Instanz (`meta`), nicht aus den Matrizen.
* Optimum (HiGHS-MILP) gegen Aufzählung aller 0/1-Belegungen, gegen die geschlossene Formel der Standortplanung (Teilmengen der Standorte) und gegen CP-SAT (OR-Tools).
* Jede von einer Heuristik gemeldete Lösung: nach der Definition zulässig, Wert ≥ Optimum, LP-Wert ≤ Optimum.
* Sperren-Satz: Runden nach Sperren gelingt genau dann, wenn jede gebrochene Variable in mindestens einer Richtung ungesperrt ist.
* Propagation ist korrekt: sie schneidet nie eine bekannte zulässige Lösung ab, und ein gemeldeter Widerspruch stimmt mit der Aufzählung überein.
* LP-Zählung: die gemeldete Zahl gelöster LPs gleich der Zahl der echten HiGHS-Aufrufe (Wurzel + Rest)."""

import itertools

import numpy as np
import pytest

import phr_algorithm as A
import phr_check as K
import phr_constants as C
import phr_lp
import phr_rounding as R
from phr_exact import enumerate_optimum, solve_exact
from phr_lp import frac_indices, propagate
from phr_scenario import SplitMix64, make_instance

FAMILIES = list(C.FAMILIES)


# ------------------------------------------------------------------------------------------------------------------ Definition der Familien
def sem_feasible(m, x, tol=1e-6):
    """Zulässigkeit nach der Definition der Familie (aus meta), unabhängig von den Matrizen und vom Prüfer."""
    x = np.asarray(x, float)
    meta, fam = m.meta, m.family
    if (x < -tol).any() or (x > 1 + tol).any():
        return False
    if (np.abs(x[m.ints] - np.round(x[m.ints])) > tol).any():
        return False
    ok = True
    if fam == "mkp":
        ok = all(sum(meta["w"][r][j] * x[j] for j in range(m.n)) <= meta["cap"][r] + tol for r in range(len(meta["cap"])))
    elif fam == "stable":
        ok = all(x[u] + x[v] <= 1 + tol for u, v in meta["edges"])
    elif fam == "vertexcover":
        ok = all(x[u] + x[v] >= 1 - tol for u, v in meta["edges"])
    elif fam == "mknap":
        n, k = meta["n"], meta["K"]
        ok = all(sum(x[i * k + q] for q in range(k)) <= 1 + tol for i in range(n))
        ok = ok and all(sum(meta["w"][i] * x[i * k + q] for i in range(n)) <= meta["cap"] + tol for q in range(k))
    elif fam in ("setcover", "cover_card"):
        ok = all(sum(row[j] * x[j] for j in range(m.n)) >= 1 - tol for row in meta["M"])
        if fam == "cover_card":
            ok = ok and x.sum() <= meta["K"] + tol
    elif fam == "gap":
        M, J = meta["M"], meta["J"]
        ok = all(abs(sum(x[i * J + j] for i in range(M)) - 1) <= tol for j in range(J))
        ok = ok and all(sum(meta["a"][i][j] * x[i * J + j] for j in range(J)) <= meta["cap"] + tol for i in range(M))
    elif fam == "assign_side":
        N = meta["N"]
        g = x.reshape(N, N)
        ok = np.allclose(g.sum(axis=0), 1, atol=tol) and np.allclose(g.sum(axis=1), 1, atol=tol)
        ok = ok and all(abs(x[S].sum() - r) <= tol for S, r in zip(meta["groups"], meta["rhs"]))
    elif fam == "setpart":
        ok = all(abs(sum(x[j] for j, c in enumerate(meta["cols"]) if e in c) - 1) <= tol for e in range(meta["E"]))
    elif fam == "binpack":
        n, B = meta["n"], meta["B"]
        ok = all(abs(sum(x[i * B + b] for b in range(B)) - 1) <= tol for i in range(n))
        ok = ok and all(sum(meta["a"][i] * x[i * B + b] for i in range(n)) <= meta["C"] * x[n * B + b] + tol for b in range(B))
    elif fam == "seqprec":
        n = meta["n"]
        g = x.reshape(n, n)
        ok = np.allclose(g.sum(axis=0), 1, atol=tol) and np.allclose(g.sum(axis=1), 1, atol=tol)
        pos = [sum(p * x[j * n + p] for p in range(n)) for j in range(n)]
        ok = ok and all(pos[a] + 1 <= pos[b] + tol for a, b in meta["pairs"])
    elif fam == "ufl":
        F, Cn = meta["F"], meta["C"]
        ok = all(abs(sum(x[F + i * F + j] for j in range(F)) - 1) <= tol for i in range(Cn))
        ok = ok and all(x[F + i * F + j] <= x[j] + tol for i in range(Cn) for j in range(F))
    return bool(ok)


def sem_objective(m, x):
    meta, fam = m.meta, m.family
    if fam == "mkp":
        return -sum(p * x[j] for j, p in enumerate(meta["profit"]))
    if fam in ("stable", "vertexcover"):
        return (-1 if fam == "stable" else 1) * sum(w * x[j] for j, w in enumerate(meta["w"]))
    if fam == "mknap":
        return -sum(meta["val"][i] * x[i * meta["K"] + q] for i in range(meta["n"]) for q in range(meta["K"]))
    if fam in ("setcover", "cover_card", "setpart"):
        return sum(c * x[j] for j, c in enumerate(meta["cost"]))
    if fam == "gap":
        return sum(meta["cost"][i][j] * x[i * meta["J"] + j] for i in range(meta["M"]) for j in range(meta["J"]))
    if fam == "assign_side":
        N = meta["N"]
        return sum(meta["cost"][i][j] * x[i * N + j] for i in range(N) for j in range(N))
    if fam == "binpack":
        return sum(x[meta["n"] * meta["B"] + b] for b in range(meta["B"]))
    if fam == "seqprec":
        n = meta["n"]
        return sum(meta["cost"][j][p] * x[j * n + p] for j in range(n) for p in range(n))
    if fam == "ufl":
        F, Cn = meta["F"], meta["C"]
        return sum(meta["f"][j] * x[j] for j in range(F)) + sum(meta["d"][j][i] * x[F + i * F + j] for i in range(Cn) for j in range(F))
    raise AssertionError(fam)


def _vectors(m, rng, opt_x, count=24):
    """Prüfvektoren: das Optimum, Ein-Bit-Änderungen des Optimums und Zufallsbelegungen (stetige Variablen zufällig in [0, 1])."""
    out = [opt_x.copy()]
    for _ in range(count // 2):
        v = opt_x.copy()
        j = int(m.ints[rng.below(len(m.ints))])
        v[j] = 1 - round(v[j])
        out.append(v)
    for _ in range(count // 2):
        v = np.array([rng.rand() for _ in range(m.n)])
        v[m.ints] = [rng.below(2) for _ in m.ints]
        out.append(v)
    return out


@pytest.mark.parametrize("fam", FAMILIES)
def test_checker_and_objective_agree_with_the_definition_of_the_family(fam):
    rng = SplitMix64(17)
    seen = {True: 0, False: 0}
    for seed in range(6):
        m = make_instance(fam, 0, seed)
        opt, opt_x, _ = solve_exact(m)
        assert sem_feasible(m, opt_x), "das Optimum muss nach der Definition zulässig sein"
        assert opt == pytest.approx(sem_objective(m, opt_x), abs=1e-6)
        for v in _vectors(m, rng, opt_x):
            assert K.is_feasible(m, v) == sem_feasible(m, v), (fam, seed)
            seen[K.is_feasible(m, v)] += 1
    assert seen[True] >= 6 and seen[False] >= 6, seen                    # nicht leer: beide Ausgänge kommen vor


# ------------------------------------------------------------------------------------------------------------------ Optimum
@pytest.mark.parametrize("fam,seeds", [("mkp", range(8)), ("stable", range(6)), ("vertexcover", range(6)), ("mknap", range(3)), ("gap", range(4)),
                                       ("binpack", range(1)), ("setpart", range(1))])
def test_exact_optimum_equals_enumeration(fam, seeds):
    for seed in seeds:
        m = make_instance(fam, 0, seed)
        value, x = enumerate_optimum(m, limit=2 ** 21)
        assert value == pytest.approx(solve_exact(m)[0], abs=1e-6), (fam, seed)
        assert sem_feasible(m, x) and sem_objective(m, x) == pytest.approx(value, abs=1e-6)


def test_ufl_optimum_equals_the_closed_formula_over_all_location_subsets():
    for seed in range(8):
        m = make_instance("ufl", 0, seed)
        F, Cn, f, d = m.meta["F"], m.meta["C"], m.meta["f"], m.meta["d"]
        best = min(sum(f[j] for j in S) + sum(min(d[j][i] for j in S) for i in range(Cn))
                   for r in range(1, F + 1) for S in itertools.combinations(range(F), r))
        assert solve_exact(m)[0] == pytest.approx(best, abs=1e-6), seed


def _cpsat_optimum(m):
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{j}") for j in range(m.n)]
    meta, fam = m.meta, m.family
    if fam == "gap":
        M, J = meta["M"], meta["J"]
        for j in range(J):
            model.Add(sum(x[i * J + j] for i in range(M)) == 1)
        for i in range(M):
            model.Add(sum(meta["a"][i][j] * x[i * J + j] for j in range(J)) <= meta["cap"])
    elif fam == "mkp":
        for r in range(len(meta["cap"])):
            model.Add(sum(meta["w"][r][j] * x[j] for j in range(m.n)) <= meta["cap"][r])
    elif fam == "setcover":
        for row in meta["M"]:
            model.Add(sum(x[j] for j in range(m.n) if row[j]) >= 1)
    elif fam == "setpart":
        for e in range(meta["E"]):
            model.Add(sum(x[j] for j, c in enumerate(meta["cols"]) if e in c) == 1)
    elif fam == "stable":
        for u, v in meta["edges"]:
            model.Add(x[u] + x[v] <= 1)
    model.Minimize(sum(int(sem_objective_coeff) * x[j] for j, sem_objective_coeff in enumerate(m.c)))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = 30
    assert solver.Solve(model) == cp_model.OPTIMAL
    return solver.ObjectiveValue()


@pytest.mark.parametrize("fam", ["gap", "mkp", "setcover", "setpart", "stable"])
def test_exact_optimum_equals_cpsat_on_small_and_medium_instances(fam):
    pytest.importorskip("ortools")
    for si, seeds in ((0, range(5)), (1, range(2))):
        for seed in seeds:
            m = make_instance(fam, si, seed)
            assert solve_exact(m)[0] == pytest.approx(_cpsat_optimum(m), abs=1e-6), (fam, si, seed)


# ------------------------------------------------------------------------------------------------------------------ Heuristiken
@pytest.mark.parametrize("fam", FAMILIES)
def test_every_reported_solution_is_feasible_by_definition_and_not_below_the_optimum(fam):
    found_any = 0
    for si, seeds in ((0, range(5)), (1, range(1))):
        for seed in seeds:
            m = make_instance(fam, si, seed)
            opt = solve_exact(m)[0]
            root = R.solve_root(m)
            assert float(m.c @ root[0]) <= opt + 1e-6                          # die LP-Schranke liegt unter dem Optimum
            for name, r in A.run_all(m, seed).items():
                if r.found:
                    found_any += 1
                    assert sem_feasible(m, r.x), (fam, name, seed)
                    assert r.obj >= opt - 1e-6 and r.obj == pytest.approx(sem_objective(m, r.x), abs=1e-6), (fam, name, seed)
                else:
                    assert r.x is None and r.obj is None and r.fail
    assert found_any > 0


def test_lock_rounding_succeeds_exactly_when_no_fractional_variable_is_blocked():
    checked = {"blocked": 0, "free": 0}
    for fam in FAMILIES:
        for seed in range(12):
            m = make_instance(fam, 0, seed)
            root = R.solve_root(m)
            if not len(frac_indices(m, root[0])):
                continue
            blocked = any(t["direction"] == "keine" for t in R.lock_table(m, root[0]))
            r = R.round_lock(m, root)
            assert r.found == (not blocked), (fam, seed)
            if not blocked:
                assert sem_feasible(m, r.x)
            checked["blocked" if blocked else "free"] += 1
    assert checked["blocked"] >= 20 and checked["free"] >= 20 and checked["blocked"] + checked["free"] == 91, checked


def test_propagation_never_cuts_off_a_known_feasible_point():
    rng = SplitMix64(5)
    tested = 0
    for fam in FAMILIES:
        for seed in range(5):
            m = make_instance(fam, 0, seed)
            opt_x = solve_exact(m)[1]
            ok, lb, ub = propagate(m, m.lb, m.ub)
            assert ok and (lb <= opt_x + 1e-6).all() and (opt_x <= ub + 1e-6).all(), (fam, seed)
            for _ in range(3):
                lo, hi = m.lb.copy(), m.ub.copy()
                for j in m.ints:
                    if rng.below(3) == 0:                  # ein Drittel der ganzzahligen Variablen auf den Wert der bekannten Lösung festsetzen
                        lo[j] = hi[j] = round(opt_x[j])
                ok, lb, ub = propagate(m, lo, hi)
                assert ok and (lb <= opt_x + 1e-6).all() and (opt_x <= ub + 1e-6).all(), (fam, seed)
                tested += 1
    assert tested >= 150


def test_a_reported_contradiction_agrees_with_enumeration():
    """Auf gap klein: 12 von 18 Variablen auf die Werte einer bekannten Lösung festsetzen, bei einem Drittel der Fälle zusätzlich eine davon umkehren. Meldet die
    Propagation einen Widerspruch, darf keine der 64 Vervollständigungen zulässig sein (ein Widerspruch bei zulässiger Vervollständigung wäre ein Fehler)."""
    rng = SplitMix64(11)
    contradictions = consistent = feasible_cases = 0
    for seed in range(60):
        m = make_instance("gap", 0, seed)
        opt_x = solve_exact(m)[1]
        fixed = rng.sample(m.n, 12)
        free = [j for j in range(m.n) if j not in fixed]
        values = {j: round(opt_x[j]) for j in fixed}
        if rng.below(3) == 0:
            j = fixed[0]
            values[j] = 1 - values[j]
        lo, hi = m.lb.copy(), m.ub.copy()
        for j, v in values.items():
            lo[j] = hi[j] = v
        ok = propagate(m, lo, hi)[0]
        feasible = False
        for bits in range(2 ** len(free)):
            x = np.zeros(m.n)
            for j, v in values.items():
                x[j] = v
            for k, j in enumerate(free):
                x[j] = (bits >> k) & 1
            if sem_feasible(m, x):
                feasible = True
                break
        assert ok or not feasible, seed
        contradictions += not ok
        consistent += ok
        feasible_cases += feasible
    assert contradictions >= 10 and consistent >= 10 and feasible_cases >= 30, (contradictions, consistent, feasible_cases)


# ------------------------------------------------------------------------------------------------------------------ Zählung der LPs
@pytest.fixture
def counted_linprog(monkeypatch):
    calls = {"n": 0}
    real = phr_lp.linprog

    def wrapper(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(phr_lp, "linprog", wrapper)
    return calls


@pytest.mark.parametrize("fam", ["gap", "setpart", "mkp", "ufl", "binpack"])
def test_reported_lp_count_equals_the_real_number_of_highs_calls(fam, counted_linprog):
    for seed in range(4):
        m = make_instance(fam, 0, seed)
        root = R.solve_root(m)
        for name in A.METHODS[:-1]:
            counted_linprog["n"] = 0
            r = A.run_named(m, root, name, seed)
            assert r.lps == counted_linprog["n"] + 1, (fam, name, seed)          # + die Wurzel, die der Aufrufer löste
        counted_linprog["n"] = 0
        c = A.cascade(m, root, seed)
        assert c.lps == counted_linprog["n"] + 1, (fam, "K1", seed)


def test_cascade_equals_the_combination_of_the_separate_runs():
    checked = 0
    for fam in FAMILIES:
        for seed in range(6):
            m = make_instance(fam, 0, seed)
            root = R.solve_root(m)
            sep = {n: A.run_named(m, root, n, seed) for n in A.CASCADE_ORDER}
            found, value, lps, stage = A.combine_cascade(sep)
            c = A.cascade(m, root, seed)
            assert (c.found, c.lps, c.info["stage"]) == (found, lps, stage), (fam, seed)
            assert c.obj == value
            checked += 1
    assert checked == 72
