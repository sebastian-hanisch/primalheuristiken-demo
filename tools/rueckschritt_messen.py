"""Gegenmessung zum Rückschritt beim Tauchen: Die Vormessung kehrte bei einem unzulässigen LP die vorletzte erfolgreiche Fixierung um (die gescheiterte stand nicht auf dem Stapel);
dieser Bau kehrt zuerst die gescheiterte Fixierung selbst um (`Params.flip_failed`). Gleiche Instanzen (die der Landkarte), gleiche Tauchregeln, 20 Rückschritte.

    python tools/rueckschritt_messen.py [--count 100] [--procs 6]

Zähleinheit: gelöste LPs (Wurzel mitgezählt); Erfolg = Anteil der Instanzen mit zulässiger Lösung. Schreibt phr_rueckschritt.json."""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import phr_constants as C  # noqa: E402
from phr_diving import dive  # noqa: E402
from phr_evaluation import fractional_seeds  # noqa: E402
from phr_result import Params  # noqa: E402
from phr_rounding import solve_root  # noqa: E402
from phr_scenario import make_instance  # noqa: E402

VARIANTS = {"D_frac": ("frac", False), "D_fracP": ("frac", True), "D_coefP": ("coef", True)}


def cell(args):
    family, si, count = args
    seeds, _ = fractional_seeds(family, si, count, 0, C.MAP_SCAN)
    out = {}
    for name, (rule, prop) in VARIANTS.items():
        for flip in (True, False):
            ok = lps = 0
            for seed in seeds:
                m = make_instance(family, si, seed)
                r = dive(m, solve_root(m), Params(rule=rule, propagate=prop, flip_failed=flip))
                ok += r.found
                lps += r.lps
            out[f"{name}|{flip}"] = {"n": len(seeds), "ok": ok, "lps": lps}
    return family, si, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=C.MAP_N)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "phr_rueckschritt.json"))
    a = ap.parse_args()
    tasks = [(f, si, a.count) for f in C.FAMILY_KEYS for si in (0, 1)]
    total, cells = {}, []
    with Pool(a.procs) as pool:
        for family, si, out in pool.imap_unordered(cell, tasks):
            cells.append({"family": family, "size": si, "results": out})
            for k, v in out.items():
                t = total.setdefault(k, {"n": 0, "ok": 0, "lps": 0})
                for key in t:
                    t[key] += v[key]
            print(family, C.SIZE_LABELS[si], {k: f"{v['ok']}/{v['n']}" for k, v in out.items() if k.startswith("D_frac|")}, flush=True)
    cells.sort(key=lambda c: (C.FAMILY_KEYS.index(c["family"]), c["size"]))
    Path(a.out).write_text(json.dumps({"cells": cells}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("GESAMT")
    for k, t in sorted(total.items()):
        print(f"{k:16s} Erfolg {t['ok']}/{t['n']} = {t['ok'] / t['n']:.3f}  LPs im Mittel {t['lps'] / t['n']:.2f}")


if __name__ == "__main__":
    main()
