"""Messung der Landkarte „Struktur der Zeilen gegen Verfahren“: schreibt phr_landkarte.json (von der App angezeigt, von tests/test_claims.py gegengeprüft).

Je Familie und Größe: die ersten MAP_N Instanzen mit gebrochener Wurzel-LP, gesucht ab Zufalls-Seed 0 (höchstens MAP_SCAN Seeds), jede mit allen Verfahren
und dem exakten Optimum (HiGHS-MILP). Zähleinheit: gelöste LPs (Wurzel mitgezählt). Aufruf (im Test-venv):

    python tools/landkarte_messen.py [--procs 6] [--out phr_landkarte.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import phr_constants as C  # noqa: E402
from phr_evaluation import fractional_seeds, measure_instance  # noqa: E402
from phr_scenario import make_instance  # noqa: E402


def measure_cell(args):
    family, si = args
    t0 = time.time()
    seeds, scanned = fractional_seeds(family, si, C.MAP_N, 0, C.MAP_SCAN)
    methods = {}
    stages = [0, 0, 0, 0]
    used = 0
    frac_sum = rootgap_sum = 0.0
    for seed in seeds:
        rec = measure_instance(make_instance(family, si, seed), seed)
        if rec is None:
            continue
        used += 1
        frac_sum += rec["n_frac"]
        rootgap_sum += rec["root_gap"]
        stages[rec["stage"]] += 1
        for name, r in rec["methods"].items():
            e = methods.setdefault(name, {"n": 0, "ok": 0, "gap_sum": 0.0, "gap_n": 0, "lps_sum": 0})
            e["n"] += 1
            e["lps_sum"] += r["lps"]
            if r["found"]:
                e["ok"] += 1
                e["gap_sum"] += r["gap"]
                e["gap_n"] += 1
    return {"family": family, "size": si, "scanned": scanned, "n": used, "stages": stages, "frac_mean": frac_sum / max(used, 1),
            "root_gap_mean": rootgap_sum / max(used, 1), "methods": methods, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "phr_landkarte.json"))
    a = ap.parse_args()
    cells = [(f, si) for f in C.FAMILY_KEYS for si in (0, 1)]
    cells.sort(key=lambda c: -(c[0] in ("setpart", "cover_card", "mknap", "seqprec", "binpack")))     # die schweren zuerst
    t0 = time.time()
    out = []
    with Pool(a.procs) as pool:
        for rec in pool.imap_unordered(measure_cell, cells):
            out.append(rec)
            print(f"{rec['family']:12s} {C.SIZE_LABELS[rec['size']]:7s} n={rec['n']:3d} gesehen={rec['scanned']:5d} {rec['seconds']:6.1f} s", flush=True)
    out.sort(key=lambda r: (C.FAMILY_KEYS.index(r["family"]), r["size"]))
    meta = {"map_n": C.MAP_N, "map_scan": C.MAP_SCAN, "einheit": "gelöste LPs (Wurzel mitgezählt)", "sekunden": round(time.time() - t0)}
    Path(a.out).write_text(json.dumps({"meta": meta, "cells": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("fertig", meta["sekunden"], "s ->", a.out)


if __name__ == "__main__":
    main()
