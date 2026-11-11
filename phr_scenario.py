"""Szenario: ganzzahlige lineare Programme (MIPs) in zwölf Familien aus vier Strukturklassen (nur ≤, ≥, Gleichungen, gemischt).

Alles ist Minimierung: min c·x unter A x ≤ b (eine ≥-Zeile steht negiert in A), Aeq x = beq, lb ≤ x ≤ ub, x[ints] ganzzahlig (0/1).
Die Instanzen laufen über einen eigenen Zufallsgenerator (SplitMix64 auf Python-Ints) statt über `numpy.random`: numpy garantiert keine
über Versionen stabilen Zufallsströme, die CI installiert aber wöchentlich die neueste Version. So sind Voreinstellungen, Zufalls-Seeds und
jede im Text genannte Zahl auf Windows und Linux dieselben. Der Generator zieht die Zufallszahlen in fester Reihenfolge je Familie.

Die Familien (und was sie zeigen) stehen in `phr_constants.FAMILIES`; hier steht nur, wie eine Instanz gebaut wird.
"""

from __future__ import annotations

import math

import numpy as np

_MASK = (1 << 64) - 1


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n):
        """Ganzzahl in 0..n-1 (die Modulo-Verzerrung bei n <= 1000 liegt unter 1e-16)."""
        return self.next() % n

    def randint(self, lo, hi):
        """Ganzzahl in lo..hi (beide inklusive)."""
        return lo + self.below(hi - lo + 1)

    def rand(self):
        """Gleitkommazahl in [0, 1) mit 53 Zufallsbits."""
        return (self.next() >> 11) / float(1 << 53)

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self.rand()

    def permutation(self, n):
        """Zufällige Reihenfolge von 0..n-1 (Fisher-Yates)."""
        p = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.below(i + 1)
            p[i], p[j] = p[j], p[i]
        return p

    def sample(self, n, k):
        """k verschiedene Zahlen aus 0..n-1 (die ersten k einer Zufallsreihenfolge)."""
        return self.permutation(n)[:k]


class Mip:
    """Ein ganzzahliges Programm in Minimierungsform mit allem, was die Anzeige braucht (Namen, Gitterform, Zeilenart)."""

    def __init__(self, family, c, A, b, Aeq, beq, lb, ub, ints, obj_integral, names, grid, row_labels, col_labels, senses=None, meta=None):
        self.family = family
        n = len(c)
        self.c = np.asarray(c, float)
        self.A = np.asarray(A, float).reshape(-1, n) if A is not None and len(A) else np.zeros((0, n))
        self.b = np.asarray(b, float) if b is not None and len(b) else np.zeros(0)
        self.Aeq = np.asarray(Aeq, float).reshape(-1, n) if Aeq is not None and len(Aeq) else np.zeros((0, n))
        self.beq = np.asarray(beq, float) if beq is not None and len(beq) else np.zeros(0)
        self.lb = np.asarray(lb, float)
        self.ub = np.asarray(ub, float)
        self.ints = np.asarray(ints, int)
        self.n = n
        self.obj_integral = obj_integral      # Zielwerte sind für ganzzahlige x ganze Zahlen (nur für die Schranke im Suchbaum)
        self.names = list(names)              # Anzeigename je Variable
        self.grid = grid                      # (Zeilen, Spalten) für die Gitteransicht der ganzzahligen Variablen
        self.row_labels = list(row_labels)
        self.col_labels = list(col_labels)
        self.senses = tuple(senses) if senses is not None else ("<=",) * self.A.shape[0]   # Anzeige: A-Zeile war ≤ oder ≥ (negiert gespeichert)
        self.meta = meta or {}
        self.is_int = np.zeros(n, bool)
        self.is_int[self.ints] = True
        self.has_cont = bool((~self.is_int).any())
        self._g = None

    # --- Sperren: wie viele Zeilen verbieten eine Änderung der Variable nach unten (down) bzw. nach oben (up)? --------------------------------
    @property
    def uplocks(self):
        """Zeilen, die ein Erhöhen der Variable verletzen kann: ≤-Zeilen mit positivem Koeffizienten, jede Gleichung mit der Variable."""
        return (self.A > 0).sum(axis=0) + (self.Aeq != 0).sum(axis=0)

    @property
    def downlocks(self):
        """Zeilen, die ein Verkleinern der Variable verletzen kann: ≤-Zeilen mit negativem Koeffizienten, jede Gleichung mit der Variable."""
        return (self.A < 0).sum(axis=0) + (self.Aeq != 0).sum(axis=0)

    @property
    def n_rows(self):
        return self.A.shape[0] + self.Aeq.shape[0]

    def grid_pos(self, j):
        """Zeile und Spalte der Variable j in der Gitteransicht."""
        return divmod(int(j), self.grid[1])

    def row_text(self, i):
        """Kurzer Text der i-ten Zeile (A-Zeilen zuerst, dann Gleichungen): Art und Zahl der Variablen."""
        if i < self.A.shape[0]:
            return f"{self.senses[i]} ({int((self.A[i] != 0).sum())} Variablen)"
        return f"= ({int((self.Aeq[i - self.A.shape[0]] != 0).sum())} Variablen)"


def _graph(n, p, rng):
    edges = {(i, j) for i in range(n) for j in range(i + 1, n) if rng.rand() < p}
    for i in range(n):                    # keine isolierten Knoten
        if not any(i in e for e in edges):
            j = (i + 1 + rng.below(n - 1)) % n
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def _row_grid(count, prefix):
    return [f"{prefix}{k + 1}" for k in range(count)]


# ------------------------------------------------------------------------------------------------------------------ Familien: nur ≤
def gen_mkp(n, seed, rows=5, tight=0.5):
    """Mehrdimensionaler Rucksack: höchstens `tight` der Gesamtgewichte je Zeile; max Gewinn (negiert: min -Gewinn). Nur ≤-Zeilen."""
    rng = SplitMix64(seed)
    w = [[rng.randint(10, 100) for _ in range(n)] for _ in range(rows)]
    cap = [math.floor(tight * sum(r)) for r in w]
    profit = [round(sum(w[r][j] for r in range(rows)) / rows + 100 * rng.rand()) for j in range(n)]
    return Mip("mkp", [-p for p in profit], w, cap, None, None, np.zeros(n), np.ones(n), np.arange(n), True,
               [f"G{j + 1}" for j in range(n)], (1, n), [""], _row_grid(n, "G"), meta={"w": w, "cap": cap, "profit": profit})


def gen_stable(n, seed, p=0.3):
    """Gewichtete stabile Menge: max Gewicht, je Kante höchstens ein Endpunkt. Nur ≤-Zeilen (Kantenzeilen x_u + x_v ≤ 1)."""
    rng = SplitMix64(seed)
    w = [rng.randint(1, 10) for _ in range(n)]
    edges = _graph(n, p, rng)
    A = np.zeros((len(edges), n))
    for r, (u, v) in enumerate(edges):
        A[r, u] = A[r, v] = 1
    return Mip("stable", [-x for x in w], A, np.ones(len(edges)), None, None, np.zeros(n), np.ones(n), np.arange(n), True,
               [f"v{j + 1}" for j in range(n)], (1, n), [""], _row_grid(n, "v"), meta={"w": w, "edges": edges})


def gen_mknap(n, K, seed, tight=0.45):
    """Mehrfach-Rucksack: jeder Gegenstand höchstens einmal (≤) und jede Kapazität ≤; max Wert. Nur ≤-Zeilen, zwei Arten."""
    rng = SplitMix64(seed)
    w = [rng.randint(10, 50) for _ in range(n)]
    val = [wi + rng.randint(0, 10) for wi in w]
    cap = math.floor(tight * sum(w) / K)
    nv = n * K
    A = np.zeros((n + K, nv))
    for i in range(n):
        for k in range(K):
            A[i, i * K + k] = 1
            A[n + k, i * K + k] = w[i]
    c = [-val[i] for i in range(n) for _ in range(K)]
    names = [f"G{i + 1}→R{k + 1}" for i in range(n) for k in range(K)]
    return Mip("mknap", c, A, [1] * n + [cap] * K, None, None, np.zeros(nv), np.ones(nv), np.arange(nv), True, names, (n, K),
               _row_grid(n, "G"), _row_grid(K, "R"), meta={"w": w, "val": val, "cap": cap, "n": n, "K": K})


# ------------------------------------------------------------------------------------------------------------------ Familien: ≥
def gen_setcover(rows, seed, cols=None, dens=0.1):
    """Mengenüberdeckung: jedes Element von mindestens einer gewählten Menge überdeckt; min Kosten. Nur ≥-Zeilen."""
    cols = cols or 2 * rows
    rng = SplitMix64(seed)
    M = [[1 if rng.rand() < dens else 0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):                 # jede Zeile mindestens 2 Spalten
        while sum(M[i]) < 2:
            M[i][rng.below(cols)] = 1
    cost = [rng.randint(1, 20) for _ in range(cols)]
    return Mip("setcover", cost, -np.array(M, float), -np.ones(rows), None, None, np.zeros(cols), np.ones(cols), np.arange(cols), True,
               [f"S{j + 1}" for j in range(cols)], (1, cols), [""], _row_grid(cols, "S"), senses=(">=",) * rows, meta={"M": M, "cost": cost})


def gen_vertexcover(n, seed, p=0.3):
    """Knotenüberdeckung: je Kante mindestens ein Endpunkt gewählt; min Gewicht. Nur ≥-Zeilen (Kantenzeilen x_u + x_v ≥ 1)."""
    rng = SplitMix64(seed)
    w = [rng.randint(1, 10) for _ in range(n)]
    edges = _graph(n, p, rng)
    A = np.zeros((len(edges), n))
    for r, (u, v) in enumerate(edges):
        A[r, u] = A[r, v] = -1
    return Mip("vertexcover", w, A, -np.ones(len(edges)), None, None, np.zeros(n), np.ones(n), np.arange(n), True,
               [f"v{j + 1}" for j in range(n)], (1, n), [""], _row_grid(n, "v"), senses=(">=",) * len(edges), meta={"w": w, "edges": edges})


def gen_cover_card(rows, seed, dens=0.1):
    """Überdeckung plus Kardinalitätsgrenze: ≥-Zeilen wie bei der Mengenüberdeckung und eine ≤-Zeile (höchstens K Mengen,
    K = kleinste mögliche Zahl + 1). Eine einzige gegenläufige Zeile."""
    from phr_exact import solve_exact   # Importzyklus vermeiden: phr_exact kennt nur Mip
    base = gen_setcover(rows, seed, dens=dens)
    n = base.n
    card = Mip("tmp", np.ones(n), base.A, base.b, None, None, np.zeros(n), np.ones(n), np.arange(n), True, base.names, base.grid, base.row_labels, base.col_labels)
    kmin = solve_exact(card)[0]
    K = int(round(kmin)) + 1
    A = np.vstack([base.A, np.ones((1, n))])
    b = np.concatenate([base.b, [K]])
    return Mip("cover_card", base.c, A, b, None, None, base.lb, base.ub, base.ints, True, base.names, base.grid, base.row_labels, base.col_labels,
               senses=(">=",) * rows + ("<=",), meta={"M": base.meta["M"], "cost": base.meta["cost"], "K": K})


# ------------------------------------------------------------------------------------------------------------------ Familien: Gleichungen
def gen_gap(M, J, seed, tight=0.85):
    """Aufträge auf Maschinen: jeder Auftrag genau einer Maschine (Gleichung), Kapazität je Maschine ≤; min Kosten.
    Variable x[i, j] = Auftrag j auf Maschine i, Index i·J + j."""
    rng = SplitMix64(seed)
    cost = [[rng.randint(10, 50) for _ in range(J)] for _ in range(M)]
    a = [[rng.randint(5, 25) for _ in range(J)] for _ in range(M)]
    cap = math.floor(sum(sum(a[i][j] for i in range(M)) / M for j in range(J)) / M / tight)
    n = M * J
    A = np.zeros((M, n))
    Aeq = np.zeros((J, n))
    for i in range(M):
        for j in range(J):
            A[i, i * J + j] = a[i][j]
            Aeq[j, i * J + j] = 1
    names = [f"A{j + 1}→M{i + 1}" for i in range(M) for j in range(J)]
    return Mip("gap", [cost[i][j] for i in range(M) for j in range(J)], A, [cap] * M, Aeq, np.ones(J), np.zeros(n), np.ones(n), np.arange(n), True,
               names, (M, J), _row_grid(M, "M"), _row_grid(J, "A"), meta={"cost": cost, "a": a, "cap": cap, "M": M, "J": J})


def gen_assign_side(N, seed, K=None, cells=None):
    """Zuordnung N × N (jede Zeile und Spalte genau einmal) plus K Zusatzgleichungen Σ x = r über je `cells` Zellen; r stammt aus einer
    verborgenen Permutation, die Instanz ist also zulässig. Nur Gleichungen."""
    K = K or max(2, N // 2)
    cells = cells or max(3, N)
    rng = SplitMix64(seed)
    cost = [[rng.randint(1, 50) for _ in range(N)] for _ in range(N)]
    perm = rng.permutation(N)
    n = N * N
    rows, rhs, groups = [], [], []
    for i in range(N):
        row = np.zeros(n)
        row[i * N:(i + 1) * N] = 1
        rows.append(row)
        rhs.append(1)
    for j in range(N):
        row = np.zeros(n)
        row[j::N] = 1
        rows.append(row)
        rhs.append(1)
    for _ in range(K):
        S = rng.sample(n, cells)
        row = np.zeros(n)
        row[S] = 1
        rows.append(row)
        rhs.append(sum(1 for s in S if perm[s // N] == s % N))
        groups.append(sorted(S))
    names = [f"({i + 1},{j + 1})" for i in range(N) for j in range(N)]
    return Mip("assign_side", [cost[i][j] for i in range(N) for j in range(N)], None, None, np.array(rows), np.array(rhs, float),
               np.zeros(n), np.ones(n), np.arange(n), True, names, (N, N), _row_grid(N, "i"), _row_grid(N, "j"),
               meta={"cost": cost, "perm": perm, "groups": groups, "rhs": rhs[2 * N:], "N": N})


def gen_setpart(E, seed, extra=3):
    """Mengenpartitionierung: jedes Element genau einmal überdeckt (Gleichungen); eine verborgene Partition macht die Instanz zulässig,
    dazu Zufallsspalten der Größe 2 bis 4. min Kosten."""
    rng = SplitMix64(seed)
    elems = rng.permutation(E)
    cols = []
    i = 0
    while i < E:
        s = min(rng.randint(2, 4), E - i)
        cols.append(sorted(elems[i:i + s]))
        i += s
    ncols = E * extra
    while len(cols) < ncols:
        cols.append(sorted(rng.sample(E, rng.randint(2, 4))))
    order = rng.permutation(len(cols))
    cols = [cols[j] for j in order]
    cost = [len(c) * rng.randint(2, 7) + rng.randint(0, 5) for c in cols]
    Aeq = np.zeros((E, len(cols)))
    for j, c in enumerate(cols):
        for e in c:
            Aeq[e, j] = 1
    return Mip("setpart", cost, None, None, Aeq, np.ones(E), np.zeros(len(cols)), np.ones(len(cols)), np.arange(len(cols)), True,
               [f"S{j + 1}" for j in range(len(cols))], (1, len(cols)), [""], _row_grid(len(cols), "S"), meta={"cols": cols, "cost": cost, "E": E})


def gen_binpack(n, seed, C=20):
    """Bin-Packing-Zuordnung: jeder Gegenstand genau einem Behälter (Gleichung), Behälterkapazität ≤ C·y_b; min Zahl benutzter Behälter.
    Variablen: x[i, b] (Index i·B + b), danach y[b] (Index n·B + b)."""
    rng = SplitMix64(seed)
    a = [rng.randint(4, 12) for _ in range(n)]
    B = int(math.ceil(sum(a) / C)) + 1
    nv = n * B + B
    Aeq = np.zeros((n, nv))
    A = np.zeros((B, nv))
    for i in range(n):
        for b in range(B):
            Aeq[i, i * B + b] = 1
            A[b, i * B + b] = a[i]
    for b in range(B):
        A[b, n * B + b] = -C
    c = np.concatenate([np.zeros(n * B), np.ones(B)])
    names = [f"G{i + 1}→B{b + 1}" for i in range(n) for b in range(B)] + [f"y{b + 1}" for b in range(B)]
    return Mip("binpack", c, A, np.zeros(B), Aeq, np.ones(n), np.zeros(nv), np.ones(nv), np.arange(nv), True, names, (n + 1, B),
               _row_grid(n, "G") + ["offen"], _row_grid(B, "B"), meta={"a": a, "C": C, "B": B, "n": n})


def gen_seqprec(n, seed, nprec=None):
    """Auftragsreihenfolge als Zuordnung Auftrag → Position (Gleichungen) mit Vorrangzeilen Σ p·x[a,p] + 1 ≤ Σ p·x[b,p] (≤); die Kosten sind
    Gewicht × Position plus Rauschen; eine verborgene Reihenfolge macht die Instanz zulässig. Variable x[j, p], Index j·n + p."""
    nprec = nprec or n
    rng = SplitMix64(seed)
    perm = rng.permutation(n)                # perm[j] = Position von Auftrag j
    w = [rng.randint(1, 9) for _ in range(n)]
    cost = [[w[j] * (p + 1) + rng.randint(0, 14) for p in range(n)] for j in range(n)]
    nv = n * n
    Aeq = np.zeros((2 * n, nv))
    for j in range(n):
        for p in range(n):
            Aeq[j, j * n + p] = 1
            Aeq[n + p, j * n + p] = 1
    pairs = set()
    while len(pairs) < nprec:
        a, b = rng.sample(n, 2)
        pairs.add((a, b) if perm[a] < perm[b] else (b, a))
    pairs = sorted(pairs)
    A = np.zeros((len(pairs), nv))
    for r, (a, b) in enumerate(pairs):
        for p in range(n):
            A[r, a * n + p] = p
            A[r, b * n + p] = -p
    names = [f"A{j + 1}@P{p + 1}" for j in range(n) for p in range(n)]
    return Mip("seqprec", [cost[j][p] for j in range(n) for p in range(n)], A, -np.ones(len(pairs)), Aeq, np.ones(2 * n), np.zeros(nv),
               np.ones(nv), np.arange(nv), True, names, (n, n), _row_grid(n, "A"), _row_grid(n, "P"), meta={"cost": cost, "pairs": pairs, "n": n})


# ------------------------------------------------------------------------------------------------------------------ Familie: gemischt
def gen_ufl(F, C, seed, fmin=150, fmax=400, dmax=50):
    """Standortplanung mit Fixkosten (starke Formulierung x[i, j] ≤ y_j): y_j ganzzahlig, x[i, j] stetig in [0, 1]; Gleichungen (jeder Kunde
    zu 100 % bedient) und ≤-Zeilen. Die Zuordnung ist stetig, nur die Standortwahl ganzzahlig. Variablen: y_0 … y_{F-1}, dann x[i, j] (Kunde-major)."""
    rng = SplitMix64(seed)
    d = [[rng.randint(1, dmax) for _ in range(C)] for _ in range(F)]     # d[j][i]: Kosten Standort j → Kunde i
    f = [rng.randint(fmin, fmax) for _ in range(F)]
    nx = C * F
    n = F + nx
    c = np.array(f + [d[j][i] for i in range(C) for j in range(F)], float)
    Aeq = np.zeros((C, n))
    A = np.zeros((nx, n))
    for i in range(C):
        for j in range(F):
            col = F + i * F + j
            Aeq[i, col] = 1
            A[i * F + j, col] = 1
            A[i * F + j, j] = -1
    names = [f"F{j + 1}" for j in range(F)] + [f"x{i + 1},{j + 1}" for i in range(C) for j in range(F)]
    return Mip("ufl", c, A, np.zeros(nx), Aeq, np.ones(C), np.zeros(n), np.ones(n), np.arange(F), False, names, (1, F), [""], _row_grid(F, "F"),
               meta={"f": f, "d": d, "F": F, "C": C})


# ------------------------------------------------------------------------------------------------------------------ Zugriff
GENERATORS = {
    "mkp": lambda s, seed: gen_mkp(s[0], seed),
    "stable": lambda s, seed: gen_stable(s[0], seed, s[1]),
    "mknap": lambda s, seed: gen_mknap(s[0], s[1], seed),
    "setcover": lambda s, seed: gen_setcover(s[0], seed),
    "vertexcover": lambda s, seed: gen_vertexcover(s[0], seed, s[1]),
    "cover_card": lambda s, seed: gen_cover_card(s[0], seed),
    "gap": lambda s, seed: gen_gap(s[0], s[1], seed),
    "assign_side": lambda s, seed: gen_assign_side(s[0], seed, s[1], s[2]),
    "setpart": lambda s, seed: gen_setpart(s[0], seed),
    "binpack": lambda s, seed: gen_binpack(s[0], seed),
    "seqprec": lambda s, seed: gen_seqprec(s[0], seed),
    "ufl": lambda s, seed: gen_ufl(s[0], s[1], seed),
}


def make_instance(family, size_index, seed):
    """Deterministisch aus (Familie, Größe 0 = klein / 1 = mittel, Zufalls-Seed): gleiche Eingabe, gleiche Instanz."""
    import phr_constants as C
    if family not in GENERATORS:
        raise ValueError(f"unbekannte Familie: {family}")
    if size_index not in (0, 1):
        raise ValueError("Größe muss 0 (klein) oder 1 (mittel) sein")
    return GENERATORS[family](C.FAMILIES[family]["sizes"][size_index], seed)
