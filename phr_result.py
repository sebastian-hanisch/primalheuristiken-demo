"""Gemeinsame Typen der Heuristiken: Ergebnis, Einzelschritt (für die Schritt-für-Schritt-Ansicht) und Einstellungen.

Ein Schritt ist nur eine Aufzeichnung (`trace=True`); das Ergebnis eines Verfahrens ist mit und ohne Aufzeichnung gleich (tests/test_algorithm.py)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import phr_check as K


def de(value, digits=2):
    """Zahl mit Dezimalkomma für Texte."""
    return f"{value:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Params:
    """Einstellungen der Heuristiken. Standardwerte = die der Vormessung (Rückschritte 20, höchstens 10 Neustarts, 15 Iterationen Wartezeit,
    300 Iterationen insgesamt, T = 10 Umkehrungen bei Stillstand, φ = 0,9). Immer mit Schlüsselwörtern aufrufen (ein eingefügtes Feld würde sonst still Werte verschieben)."""
    theta: float = 0.5               # Rundungsschwelle: ab diesem Nachkommaanteil wird aufgerundet
    backtracks: int = 20             # Rückschritte beim Tauchen
    propagate: bool = True           # Schranken-Propagation nach jedem Fixieren
    rule: str = "frac"               # Tauchregel: frac (am nächsten an ganz), coef (wenigste Sperren), guided (zur Referenzlösung)
    restarts: int = 10               # Neustarts des Pumps
    objective_weight: float = 0.0    # Zielgewicht α₀ des Pumps (0 = reiner Pump)
    flips: int = 10                  # T: Umkehrungen bei Stillstand
    stall: int = 15                  # Iterationen ohne kleineren Abstand bis zum Neustart
    maxit: int = 300                 # Iterationen des Pumps insgesamt
    phi: float = 0.9                 # Abnahme des Zielgewichts je Iteration
    max_lps: int = 400               # Obergrenze der LPs eines Tauchgangs
    flip_failed: bool = True         # Rückschritt kehrt zuerst die gescheiterte Fixierung um (False: wie in der Vormessung die vorletzte; nur für die Gegenmessung)


@dataclass
class Step:
    """Ein sichtbarer Schritt: Text, aktueller Punkt (LP-Lösung oder Rundung), aktuelle Schranken und die bis hier gelösten LPs."""
    kind: str
    text: str
    x: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    lps: int
    data: dict = field(default_factory=dict)


@dataclass
class Result:
    method: str
    found: bool
    x: np.ndarray | None
    obj: float | None
    lps: int
    iters: int
    fail: str | None = None
    info: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)


def make_result(m, method, ctr, x, fail=None, info=None, steps=None):
    """Ergebnis bauen; gefunden gilt nur, wenn der unabhängige Prüfer (phr_check) x annimmt."""
    ok = x is not None and K.is_feasible(m, x)
    return Result(method, bool(ok), x.copy() if ok else None, K.objective(m, x) if ok else None, ctr.lps, ctr.iters,
                  None if ok else (fail or "unzulässig"), info or {}, steps if steps is not None else [])
