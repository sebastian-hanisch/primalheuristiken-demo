"""Messung der Rundungsschwelle θ: Runden auf die nächste ganze Zahl und Runden mit Propagation bei θ = 0,3 / 0,5 / 0,7 / 1/f über dieselben Instanzen wie die Landkarte
(je Familie und Größe die ersten 100 mit gebrochener Wurzel, Seeds ab 0). Erfolg und mittlere Lücke über die Erfolgreichen. Schreibt phr_theta.json.

    python tools/theta_messen.py [--procs 6]
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
from phr_evaluation import fractional_seeds, gap_percent, theta_value  # noqa: E402
from phr_exact import solve_exact  # noqa: E402
from phr_rounding import round_near, round_prop, solve_root  # noqa: E402
from phr_scenario import make_instance  # noqa: E402


def cell(args):
    family, si = args
    seeds, _ = fractional_seeds(family, si, C.MAP_N, 0, C.MAP_SCAN)
    acc = {f"{kind}|{lab}": {"n": 0, "ok": 0, "gap_sum": 0.0} for kind in ("near", "prop") for lab in C.THETA_OPTIONS}
    for seed in seeds:
        m = make_instance(family, si, seed)
        root = solve_root(m)
        opt = solve_exact(m, C.EXACT_TIME_LIMIT)[0]
        for lab in C.THETA_OPTIONS:
            th = theta_value(m, lab)
            for kind, fn in (("near", round_near), ("prop", round_prop)):
                r = fn(m, root, th)
                e = acc[f"{kind}|{lab}"]
                e["n"] += 1
                if r.found:
                    e["ok"] += 1
                    e["gap_sum"] += gap_percent(r.obj, opt)
    return {"family": family, "size": si, "acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "phr_theta.json"))
    a = ap.parse_args()
    cells = [(f, si) for f in C.FAMILY_KEYS for si in (0, 1)]
    with Pool(a.procs) as pool:
        res = pool.map(cell, cells)
    Path(a.out).write_text(json.dumps({"cells": res}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("fertig")


if __name__ == "__main__":
    main()
