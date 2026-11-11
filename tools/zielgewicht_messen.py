"""Gepaarte Messung zum Zielgewicht des Pumps: Lücke des Pumps ohne und mit Zielgewicht (je 10 Neustarts) auf denselben Instanzen wie die Landkarte (je Familie und Größe die
ersten 100 mit gebrochener Wurzel, Seeds ab 0). Die ungepaarte Lücke der Landkarte mittelt über die jeweils erfolgreichen Läufe; hier wird zusätzlich nur über die Instanzen
gemittelt, die BEIDE Varianten lösen. Schreibt phr_zielgewicht.json.

    python tools/zielgewicht_messen.py [--procs 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import phr_constants as C  # noqa: E402
from phr_evaluation import fractional_seeds, gap_percent  # noqa: E402
from phr_exact import solve_exact  # noqa: E402
from phr_pump import pump  # noqa: E402
from phr_result import Params  # noqa: E402
from phr_rounding import solve_root  # noqa: E402
from phr_scenario import make_instance  # noqa: E402


def cell(args):
    family, si = args
    seeds, _ = fractional_seeds(family, si, C.MAP_N, 0, C.MAP_SCAN)
    rows = []
    for seed in seeds:
        m = make_instance(family, si, seed)
        root = solve_root(m)
        opt = solve_exact(m, C.EXACT_TIME_LIMIT)[0]
        a = pump(m, root, Params(objective_weight=0.0), seed=seed)
        b = pump(m, root, Params(objective_weight=1.0), seed=seed)
        rows.append([seed, gap_percent(a.obj, opt) if a.found else None, gap_percent(b.obj, opt) if b.found else None])
    both = [r for r in rows if r[1] is not None and r[2] is not None]
    return {"family": family, "size": si, "n": len(rows), "ok_without": sum(r[1] is not None for r in rows), "ok_with": sum(r[2] is not None for r in rows),
            "both": len(both), "gap_without_both": sum(r[1] for r in both) / len(both) if both else None,
            "gap_with_both": sum(r[2] for r in both) / len(both) if both else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "phr_zielgewicht.json"))
    a = ap.parse_args()
    cells = [(f, si) for f in C.FAMILY_KEYS for si in (0, 1)]
    cells.sort(key=lambda c: -(c[0] in ("mknap", "binpack", "setpart")))
    with Pool(a.procs) as pool:
        res = pool.map(cell, cells)
    res.sort(key=lambda r: (C.FAMILY_KEYS.index(r["family"]), r["size"]))
    Path(a.out).write_text(json.dumps({"cells": res}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("fertig")


if __name__ == "__main__":
    main()
