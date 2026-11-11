"""Gemeinsame Fixtures: Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

mini_pack: 3 Gegenstände, eine ≤-Zeile. Handrechnung: LP-Optimum x = (1; 0,5; 0), Wert −7; Abrunden gibt (1, 0, 0) mit Wert −5 = Optimum.
mini_gap: 2 Maschinen × 2 Aufträge, Variablen in der Reihenfolge (M1,A1), (M1,A2), (M2,A1), (M2,A2), Kosten 1, 2, 3, 3, Auftragsgröße überall 2,
Kapazität 3 je Maschine. Handrechnung siehe tests/test_algorithm.py."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Live-Messungen der App (je Knopf 20 Instanzen) laufen in den Tests mit 3 Instanzen: spart Minuten, die Logik ist dieselbe.
os.environ.setdefault("PHR_LIVE_N", "3")

from phr_scenario import Mip  # noqa: E402


@pytest.fixture
def mini_pack():
    return Mip("mini_pack", [-5, -4, -3], [[2, 2, 2]], [3], None, None, np.zeros(3), np.ones(3), np.arange(3), True,
               ["G1", "G2", "G3"], (1, 3), [""], ["G1", "G2", "G3"])


@pytest.fixture
def mini_gap():
    A = [[2, 2, 0, 0], [0, 0, 2, 2]]
    Aeq = [[1, 0, 1, 0], [0, 1, 0, 1]]
    names = ["A1→M1", "A2→M1", "A1→M2", "A2→M2"]
    return Mip("mini_gap", [1, 2, 3, 3], A, [3, 3], Aeq, [1, 1], np.zeros(4), np.ones(4), np.arange(4), True, names, (2, 2), ["M1", "M2"], ["A1", "A2"])
