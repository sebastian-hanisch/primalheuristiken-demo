"""Messung: Was spart die Startlösung der Kaskade dem einfachen Suchbaum? (Tiefensuche und beste Schranke zuerst, ohne Schnitte und Presolve, Limit 1 000 LPs.)

Je Familie und Größe die ersten COUNT Instanzen mit gebrochener Wurzel (Seeds ab 0), je Strategie drei Läufe: ohne Startlösung, mit dem Zielwert der Kaskade, mit dem
bekannten Optimum. Gepaart über Instanzen, bei denen ohne Startlösung und mit Kaskaden-Startlösung der Beweis im Limit gelang. Einheit: gelöste LPs (Knoten).
Brutto = (LPs ohne − LPs mit) / LPs ohne; netto zieht die LPs der Kaskade (alle, auch die Wurzel) von der Ersparnis ab. Je Familie (beide Größen zusammen) zusätzlich ein
95-%-Band für die Nettoersparnis per Bootstrap über die gepaarten Instanzen (2 000 Ziehungen, SplitMix64, Verhältnis der Summen). Schreibt phr_baum.json.

    python tools/baum_messen.py [--count 60] [--procs 6]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import phr_algorithm as A  # noqa: E402
import phr_constants as C  # noqa: E402
import phr_tree as T  # noqa: E402
from phr_evaluation import fractional_seeds  # noqa: E402
from phr_exact import solve_exact  # noqa: E402
from phr_rounding import solve_root  # noqa: E402
from phr_scenario import SplitMix64, make_instance  # noqa: E402


def cell(args):
    family, si, count = args
    seeds, _ = fractional_seeds(family, si, count, 0, C.MAP_SCAN)
    out = {"family": family, "size": si, "instances": 0, "no_start": 0, "strategies": {}}
    for sname, run in T.STRATEGIES.items():
        acc = {"paired": 0, "none": 0, "start": 0, "casc": 0, "opt": 0, "opt_n": 0, "censored": 0, "records": []}
        for seed in seeds:
            m = make_instance(family, si, seed)
            root = solve_root(m)
            casc = A.cascade(m, root, seed)
            opt = solve_exact(m, C.EXACT_TIME_LIMIT)[0]
            none = run(m, math.inf, C.BB_NODE_LIMIT)
            if not none["proved"]:
                acc["censored"] += 1
                continue
            if not casc.found:
                continue
            start = run(m, casc.obj, C.BB_NODE_LIMIT)
            if not start["proved"]:
                acc["censored"] += 1
                continue
            best = run(m, opt, C.BB_NODE_LIMIT)
            acc["paired"] += 1
            acc["none"] += none["nodes"]
            acc["start"] += start["nodes"]
            acc["casc"] += casc.lps
            acc["opt"] += best["nodes"]
            acc["records"].append([seed, none["nodes"], start["nodes"], casc.lps, best["nodes"]])
        out["strategies"][sname] = acc
    out["instances"] = len(seeds)
    return out


def bootstrap_band(records, draws=2000, seed=12345):
    """95-%-Band der Nettoersparnis (Verhältnis der Summen) über gepaarte Instanzen; records: [seed, ohne, mit, Kaskade, Optimum]."""
    rng = SplitMix64(seed)
    n = len(records)
    values = []
    for _ in range(draws):
        pick = [records[rng.below(n)] for _ in range(n)]
        none = sum(r[1] for r in pick)
        values.append(100 * (none - sum(r[2] for r in pick) - sum(r[3] for r in pick)) / none)
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws) - 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "phr_baum.json"))
    a = ap.parse_args()
    cells = [(f, si, a.count) for f in C.FAMILY_KEYS for si in (0, 1)]
    cells.sort(key=lambda c: -(c[0] in ("mknap", "mkp", "cover_card", "binpack", "setpart", "seqprec")))
    res = []
    with Pool(a.procs) as pool:
        for rec in pool.imap_unordered(cell, cells):
            res.append(rec)
            print(rec["family"], C.SIZE_LABELS[rec["size"]], {k: (v["paired"], v["censored"]) for k, v in rec["strategies"].items()}, flush=True)
    res.sort(key=lambda r: (C.FAMILY_KEYS.index(r["family"]), r["size"]))
    bands = {}
    for strat in T.STRATEGIES:
        for fam in C.FAMILY_KEYS:
            recs = [r for c in res if c["family"] == fam for r in c["strategies"][strat]["records"]]
            bands[f"{strat}|{fam}"] = bootstrap_band(recs) if recs else None
    Path(a.out).write_text(json.dumps({"count": a.count, "limit": C.BB_NODE_LIMIT, "netto_band_95": bands, "cells": res}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("fertig")


if __name__ == "__main__":
    main()
