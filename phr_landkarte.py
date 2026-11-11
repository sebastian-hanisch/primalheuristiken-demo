"""Die vorberechnete Landkarte (phr_landkarte.json, erzeugt von tools/landkarte_messen.py): Zugriff und Zusammenfassungen.

Einheit überall: gelöste LPs (Wurzel mitgezählt). Je Zelle (Familie × Größe) die ersten 100 Instanzen mit gebrochener Wurzel-LP ab Zufalls-Seed 0 (oder weniger,
wenn in den angesehenen Seeds nicht genug gebrochene vorkamen; `n` und `scanned` stehen in der Zelle)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import phr_constants as C

PATH = Path(__file__).resolve().parent / "phr_landkarte.json"
MAP_METHODS = ("R_lock", "R_near", "R_prop", "D_frac", "D_fracP", "D_coefP", "FP_rs", "FP_objrs", "K1")


@lru_cache(maxsize=1)
def load():
    """Die gespeicherte Messung (Wörterbuch mit meta und cells) oder None, wenn die Datei fehlt."""
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cell(data, family, size_idx):
    for c in data["cells"]:
        if c["family"] == family and c["size"] == size_idx:
            return c
    return None


def stat(c, method):
    """Erfolg (Anteil), mittlere Lücke in % über die Erfolgreichen, mittlere LPs über alle Läufe einer Zelle und Methode."""
    e = c["methods"][method]
    return {"n": e["n"], "ok": e["ok"], "rate": e["ok"] / e["n"] if e["n"] else None,
            "gap": e["gap_sum"] / e["gap_n"] if e["gap_n"] else None, "lps": e["lps_sum"] / e["n"] if e["n"] else None}


def pooled(data, method, families=None, sizes=(0, 1)):
    """Über mehrere Zellen zusammengefasst (jede Instanz zählt gleich): n, Erfolge, Rate, mittlere Lücke über die Erfolgreichen, mittlere LPs."""
    n = ok = gap_n = 0
    gap_sum = lps_sum = 0.0
    for c in data["cells"]:
        if (families is not None and c["family"] not in families) or c["size"] not in sizes:
            continue
        e = c["methods"][method]
        n += e["n"]
        ok += e["ok"]
        gap_n += e["gap_n"]
        gap_sum += e["gap_sum"]
        lps_sum += e["lps_sum"]
    return {"n": n, "ok": ok, "rate": ok / n if n else None, "gap": gap_sum / gap_n if gap_n else None, "lps": lps_sum / n if n else None}


def stage_shares(data, families=None, sizes=(0, 1)):
    """Anteil der Instanzen, die die Kaskade in Stufe 0 / 1 / 2 gelöst hat oder in keiner (Summe 1)."""
    total = 0
    stages = [0, 0, 0, 0]
    for c in data["cells"]:
        if (families is not None and c["family"] not in families) or c["size"] not in sizes:
            continue
        total += c["n"]
        for k in range(4):
            stages[k] += c["stages"][k]
    return [s / total for s in stages] if total else [0.0] * 4


def classes():
    """Familien je Strukturklasse in der Reihenfolge der Landkarte."""
    return {k: [f for f in C.FAMILY_KEYS if C.FAMILIES[f]["klass"] == k] for k in C.CLASS_ORDER}
