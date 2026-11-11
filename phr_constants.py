"""Konstanten: Familien der Instanzen, Regler-Optionen, Messgrenzen, Farben und Presets."""

import os

# Zwölf Familien in vier Strukturklassen. `sizes`: Parameter des Erzeugers (klein, mittel), siehe phr_scenario.GENERATORS.
# `klass`: welche Zeilenarten die Familie hat; `rows`: Zeilen in Worten; `story`: wofür das Modell steht.
FAMILIES = {
    "gap": dict(label="Aufträge auf Maschinen (Zuordnung mit Kapazität)", klass="=", sizes=[(3, 6), (4, 16)], rows="Gleichungen (jeder Auftrag genau einmal) und ≤ (Kapazität je Maschine)",
                story="Jeder Auftrag kommt auf genau eine Maschine, jede Maschine hat eine Kapazität; minimiere die Kosten."),
    "assign_side": dict(label="Zuordnung mit Zusatzgleichungen", klass="=", sizes=[(5, 3, 5), (10, 5, 10)], rows="nur Gleichungen",
                        story="Eine Zuordnung N × N (jede Zeile und Spalte genau einmal) plus Gleichungen, die Gruppen von Zellen zählen."),
    "setpart": dict(label="Mengenpartitionierung", klass="=", sizes=[(7,), (20,)], rows="nur Gleichungen",
                    story="Jedes Element wird von genau einer gewählten Menge überdeckt; minimiere die Kosten der Mengen."),
    "binpack": dict(label="Bin Packing (Zuordnung zu Behältern)", klass="=", sizes=[(5,), (8,)], rows="Gleichungen (jeder Gegenstand genau einmal) und ≤ (Behälterkapazität)",
                    story="Gegenstände in möglichst wenige Behälter der Größe 20; die Kapazitätszeile hängt an der Behältervariable."),
    "seqprec": dict(label="Reihenfolge mit Vorrang (Auftrag → Position)", klass="=", sizes=[(5,), (9,)], rows="Gleichungen (Zuordnung) und ≤ (Vorrang)",
                    story="Jeder Auftrag bekommt eine Position, jede Position einen Auftrag; für einige Paare muss der eine vor dem anderen kommen."),
    "mkp": dict(label="Mehrdimensionaler Rucksack", klass="≤", sizes=[(12,), (20,)], rows="nur ≤ (fünf Kapazitätszeilen)",
                story="Gegenstände in einen Rucksack mit fünf Kapazitäten; maximiere den Gewinn."),
    "stable": dict(label="Gewichtete stabile Menge", klass="≤", sizes=[(16, 0.3), (30, 0.15)], rows="nur ≤ (eine Zeile je Kante)",
                   story="Knoten wählen, von denen keine zwei durch eine Kante verbunden sind; maximiere das Gewicht."),
    "mknap": dict(label="Mehrfach-Rucksack", klass="≤", sizes=[(6, 3), (8, 4)], rows="nur ≤ (jeder Gegenstand höchstens einmal, Kapazität je Rucksack)",
                  story="Gegenstände auf mehrere Rucksäcke verteilen, jeder höchstens einmal; maximiere den Wert."),
    "setcover": dict(label="Mengenüberdeckung", klass="≥", sizes=[(12,), (60,)], rows="nur ≥ (jedes Element mindestens einmal)",
                     story="Jedes Element muss von mindestens einer gewählten Menge überdeckt sein; minimiere die Kosten."),
    "vertexcover": dict(label="Knotenüberdeckung", klass="≥", sizes=[(16, 0.3), (50, 0.12)], rows="nur ≥ (eine Zeile je Kante)",
                        story="Knoten so wählen, dass jede Kante mindestens einen gewählten Endpunkt hat; minimiere das Gewicht."),
    "cover_card": dict(label="Überdeckung mit Kardinalitätsgrenze", klass="≥", sizes=[(12,), (60,)], rows="≥ (Überdeckung) und eine gegenläufige ≤-Zeile (höchstens K Mengen)",
                       story="Wie die Mengenüberdeckung, aber es dürfen höchstens K Mengen gewählt werden (K = kleinste mögliche Zahl + 1)."),
    "ufl": dict(label="Standortplanung (gemischt)", klass="gemischt", sizes=[(6, 18), (12, 36)], rows="Gleichungen und ≤; Standorte ganzzahlig, Zuordnung stetig",
                story="Standorte öffnen (ganzzahlig) und Kunden stetig zuordnen; minimiere Fixkosten plus Wegekosten."),
}
FAMILY_KEYS = tuple(FAMILIES)
CLASS_ORDER = ("≤", "≥", "=", "gemischt")
SIZE_LABELS = ("klein", "mittel")

# Regler-Optionen. Nur diese Werte wurden in der Vormessung gemessen (Rundungsschwelle, Rückschritte, Neustarts).
THETA_OPTIONS = ("0,3", "0,5", "0,7", "1/f")
BACKTRACK_OPTIONS = (0, 1, 5, 20)
RESTART_OPTIONS = (0, 3, 10)
RULE_LABELS = {"frac": "Fractional (nächste ganze Zahl)", "coef": "Coefficient (wenigste Sperren)", "guided": "Guided (nahe an der Referenzlösung)"}

DEFAULT_FAMILY, DEFAULT_SIZE, DEFAULT_SEED = "gap", "klein", 0
SEED_MAX = 999999
DEFAULTS = dict(theta="0,5", backtracks=20, prop=True, rule="frac", restarts=10, objw=False)

# Messung: feste Seeds, nie ein einzelner Seed als Befund. Landkarte: je Familie und Größe die ersten MAP_N Instanzen mit gebrochener Wurzel-LP
# (gesucht ab Seed 0, höchstens MAP_SCAN Seeds); Live-Messungen in der App: LIVE_N Instanzen ab LIVE_SEED0.
MAP_N, MAP_SCAN = 100, 8000
LIVE_N, LIVE_SEED0 = int(os.environ.get("PHR_LIVE_N", "20")), 100000      # PHR_LIVE_N: nur die Tests verkleinern die Live-Messungen
BB_NODE_LIMIT = 1000
EXACT_TIME_LIMIT = 60

COLOR_FRAC = "#e8a33d"        # gebrochene LP-Variable
COLOR_ONE = "#2F6B65"         # Wert 1 (gewählt)
COLOR_ZERO = "#dcdcdc"        # Wert 0
COLOR_FIXED = "#4c78a8"       # fixiert (Rundung, Propagation)
COLOR_LINE = "#4c78a8"
COLOR_BAD = "#c0504d"
COLOR_OK = "#2F6B65"

# Presets: Familie, Größe und Zufalls-Seed; alle anderen Regler stehen bei jedem Preset auf ihren Standardwerten. Die Seeds sind so gewählt, dass die Wurzel-LP
# gebrochen ist und die Aussage des Presets an dieser Instanz gilt (tests/test_claims.py prüft die Aussage je Preset; LP-Zahlen als Bänder).
PRESETS = {
    "Aufträge auf Maschinen (Standardfall)": dict(family="gap", size="klein", seed=0),
    "Sackgasse beim Runden mit Propagation": dict(family="gap", size="klein", seed=5),
    "Der Pump wird gebraucht": dict(family="setpart", size="mittel", seed=20),
    "Zuordnung mit Zusatzgleichungen": dict(family="assign_side", size="klein", seed=10),
    "Rucksack: Abrunden genügt": dict(family="mkp", size="klein", seed=0),
    "Eine gegenläufige Zeile": dict(family="cover_card", size="klein", seed=3),
}
PRESET_HELP = {
    "Aufträge auf Maschinen (Standardfall)": "18 Variablen, 6 davon im LP gebrochen: Runden nach Sperren und mit Schwelle θ scheitern, Runden mit Propagation gelingt mit einem LP (Lücke 0,0 %).",
    "Sackgasse beim Runden mit Propagation": "Dieselbe Familie, anderer Zufalls-Seed: Runden mit Propagation endet in einer Sackgasse, Tauchen löst die Instanz mit 5 LPs.",
    "Der Pump wird gebraucht": "Mengenpartitionierung mit 60 Variablen (nur Gleichungen): Runden mit Propagation und Tauchen (20 Rückschritte, 38 LPs) enden in Sackgassen, erst der Pump findet eine Lösung (14 LPs).",
    "Zuordnung mit Zusatzgleichungen": "Nur Gleichungen, 25 Variablen: Runden scheitert in jeder Form. Der Pump braucht 39 LPs mit Zyklusstörungen und 2 Neustarts, Tauchen löst dieselbe Instanz mit 3 LPs.",
    "Rucksack: Abrunden genügt": "Nur ≤-Zeilen, 12 Variablen: Runden nach Sperren gelingt, liegt aber 26,1 % über dem Optimum; Runden mit Propagation trifft das Optimum.",
    "Eine gegenläufige Zeile": "Zwölf ≥-Zeilen und eine ≤-Zeile (höchstens K Mengen): Runden nach Sperren scheitert, Runden mit Propagation gelingt (Lücke 8,6 %).",
}
