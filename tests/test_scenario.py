"""Szenario: Zufallsgenerator gegen veröffentlichte Referenzwerte, Determinismus, Größen der zwölf Familien, Zulässigkeit der Instanzen."""

import pytest

import phr_constants as C
import phr_scenario as S
from phr_exact import solve_exact

# Die ganzzahligen Variablen je Familie und Größe, wie in der Landkarte genannt (klein / mittel)
N_INT = {"mkp": (12, 20), "stable": (16, 30), "mknap": (18, 32), "setcover": (24, 120), "vertexcover": (16, 50), "cover_card": (24, 120),
         "gap": (18, 64), "assign_side": (25, 100), "setpart": (21, 60), "binpack": (21, 42), "seqprec": (25, 81), "ufl": (6, 12)}


def test_splitmix64_matches_the_published_reference_values():
    r = S.SplitMix64(0)                      # Vigna, splitmix64.c: Ausgabe für Seed 0
    assert [r.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    r = S.SplitMix64(1234567)
    assert [r.next() for _ in range(2)] == [6457827717110365317, 3203168211198807973]


def test_rng_helpers_stay_in_range_and_permute():
    r = S.SplitMix64(9)
    assert all(3 <= r.randint(3, 5) <= 5 for _ in range(200))
    assert all(0.0 <= r.rand() < 1.0 for _ in range(200))
    assert sorted(r.permutation(12)) == list(range(12)) and len(set(r.sample(10, 4))) == 4


def test_every_family_is_listed_and_sizes_match_the_map():
    assert set(S.GENERATORS) == set(C.FAMILIES) == set(N_INT)
    for fam in C.FAMILIES:
        for si in (0, 1):
            if fam == "binpack":      # die Zahl der Behälter B hängt von den Gegenstandsgrößen ab: n·B + B Variablen, im Mittel 21 bzw. 42
                counts = [len(S.make_instance(fam, si, seed).ints) for seed in range(40)]
                assert abs(sum(counts) / len(counts) - N_INT[fam][si]) <= 3, (si, counts)
                continue
            assert len(S.make_instance(fam, si, 3).ints) == N_INT[fam][si], (fam, si)


@pytest.mark.parametrize("fam", list(C.FAMILIES))
def test_instances_are_deterministic_and_seed_dependent(fam):
    a, b, c = S.make_instance(fam, 0, 4), S.make_instance(fam, 0, 4), S.make_instance(fam, 0, 5)

    def same(p, q):
        return all(u.shape == v.shape and (u == v).all() for u, v in ((p.c, q.c), (p.A, q.A), (p.b, q.b), (p.Aeq, q.Aeq), (p.beq, q.beq)))

    assert same(a, b)
    assert not same(a, c)


@pytest.mark.parametrize("fam", list(C.FAMILIES))
def test_small_instances_are_feasible_and_have_the_stated_row_kinds(fam):
    m = S.make_instance(fam, 0, 11)
    assert solve_exact(m)[0] is not None
    assert len(m.senses) == m.A.shape[0] and set(m.senses) <= {"<=", ">="}
    klass = C.FAMILIES[fam]["klass"]
    has_eq, has_le, has_ge = m.Aeq.shape[0] > 0, "<=" in m.senses, ">=" in m.senses
    if klass == "≤":
        assert has_le and not has_eq and not has_ge
    if klass == "≥":
        assert has_ge and not has_eq
    if fam == "cover_card":
        assert has_le and has_ge                       # die eine gegenläufige Zeile
    if fam in ("assign_side", "setpart"):
        assert has_eq and not has_le and not has_ge


def test_unknown_family_and_size_are_refused():
    with pytest.raises(ValueError):
        S.make_instance("nope", 0, 1)
    with pytest.raises(ValueError):
        S.make_instance("gap", 2, 1)


def test_gap_instance_matches_its_description():
    m = S.make_instance("gap", 0, 5)
    assert (m.A.shape[0], m.Aeq.shape[0], m.n) == (3, 6, 18)                    # 3 Maschinen, 6 Aufträge
    assert m.grid == (3, 6) and m.names[0] == "A1→M1" and m.names[6] == "A1→M2"
    assert all(5 <= v <= 25 for row in m.meta["a"] for v in row)


def test_assign_side_is_feasible_by_construction_with_the_hidden_permutation():
    import numpy as np
    from phr_check import is_feasible
    for seed in range(8):
        m = S.make_instance("assign_side", 0, seed)
        N = m.meta["N"]
        x = np.zeros(N * N)
        for i, j in enumerate(m.meta["perm"]):
            x[i * N + j] = 1
        assert is_feasible(m, x)


def test_setpart_hidden_partition_is_feasible():
    import numpy as np
    from phr_check import is_feasible
    m = S.make_instance("setpart", 0, 2)
    assert solve_exact(m)[0] is not None and np.isfinite(m.c).all()
    assert is_feasible(m, solve_exact(m)[1])
