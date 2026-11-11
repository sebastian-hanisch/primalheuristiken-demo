# Ohne Suchbaum zur ganzzahligen Lösung: Runden, Tauchen, Pump – Streamlit-Demo

<!-- Nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-primalheuristiken-demo.streamlit.app/)** -->

Teil der Reihe **Exakte Suche erklärt: Rucksack und Cutting Stock** der Website „Sebastian Hanisch – Operations Research und Machine Learning“, Nachfolger von
[branch-cut-demo](../branch-cut-demo) (gleicher Zweck, andere Bauweise: eine zulässige ganzzahlige Lösung ohne den Suchbaum). Die Schwäche des Vorgängers, die dieses Stück sichtbar macht:
`branch-cut-demo` und `branch-bound-demo` rechnen am Rucksack mit **einer** ≤-Zeile, wo Runden trivial zulässig ist; sobald Gleichungen oder gegenläufige Zeilen dazukommen, ist das nicht mehr so, und die Demo
sagt nichts darüber, wann ein LP-Wert gerundet werden darf. Hier sind es zwölf Familien in vier Strukturklassen, alle Heuristiken selbst gebaut (Runden, Tauchen, Feasibility Pump, Kaskade),
die LP-Relaxationen löst HiGHS über `scipy.optimize.linprog`, das Referenzoptimum `scipy.optimize.milp`.

## Kernfrage

**Ob man eine LP-Lösung runden darf, steht in den Sperren der Zeilen:** Bei nichtnegativen Koeffizienten erlaubt eine ≤-Zeile nur Abrunden und eine ≥-Zeile nur Aufrunden (allgemein entscheidet das Vorzeichen des Koeffizienten:
bei `ufl` sperren die ≤-Zeilen das Abrunden), eine Gleichung oder eine zweite, gegenläufige Zeile erlaubt keins von beiden.
Wo das Runden scheitert, hilft die Schranken-Propagation bei Gleichungen mit Kapazitätsschlupf (`gap` 83 und 94, `binpack` 99 und 98, `seqprec` 77 und 47 von je 100 Instanzen klein und mittel), nicht bei reinen
Gleichungssystemen (`assign_side` und `setpart`: 0 von 400); dort bleiben Tauchen mit Rückschritt und der Feasibility Pump; beide scheitern messbar oft, das Tauchen in Sackgassen, der Pump an seiner Iterationsgrenze von 300 Iterationen. Die **Landkarte** in der App zeigt für jede Familie und Größe Erfolg, Lücke und gelöste LPs je Verfahren.

**Zähleinheit überall: gelöste LPs** (die Wurzel mitgezählt; bei der Kaskade einmal). Simplex-Iterationen werden nie mit LPs vermischt; ein Knoten des einfachen Vergleichsbaums ist ein gelöstes LP und zählt als solches.
Erfolg = Anteil der Instanzen, für die der unabhängige Prüfer (`phr_check.py`) die Lösung annimmt; Lücke = (Wert − Optimum) / |Optimum| in Prozent (bei einem Optimum von null wäre sie nicht definiert; der Code setzt dann den Nenner auf 10⁻⁹), Mittel über die erfolgreichen Läufe; LPs = Mittel über alle Läufe.
Eine **Zelle** der Landkarte ist eine Familie in einer Größe: 12 Familien × 2 Größen = 24 Zellen, je 100 Instanzen mit gebrochener Wurzel (2 400 Instanzen).

**Nicht der Aufhänger:** „Eine Startlösung spart dem Suchbaum Knoten“ (netto gesichert positiv nur bei drei von zwölf Familien der Tiefensuche und bei einer der Suche nach bester Schranke; 60 Instanzen je Familie und Größe, siehe Befund unten) und „der Pump rettet“ (er findet in 95,6 % der Instanzen etwas, aber in 24 von 24 Zellen der Landkarte mit größerer Lücke als das Tauchen mit Propagation; mehr LPs braucht er in 11 von 24 Zellen, im Mittel über
alle 2 400 Instanzen 22,0 gegen 9,3). Beides steht unten als Befund.

## Befunde und Korrekturen gegenüber der Vormessung

Die Vormessung (zwei Runden, 12 Familien, Zufallsgenerator `numpy.random`) war die Erwartung; gemessen gilt der Bau mit dem Zufallsgenerator der Demo (SplitMix64), je Familie und Größe die ersten 100 Instanzen mit
gebrochener Wurzel-LP (2 400 Instanzen), Messung in `phr_landkarte.json`.

| Aussage der Vormessung (Zahlen aus dem Planungsordner `primalheuristik-planung`, nicht im Repo und hier nicht neu geprüft) | Im Bau |
|---|---|
| Aufrunden/Abrunden nach Sperren gelingt bei ≤, ≥ und scheitert bei Gleichungen und gegenläufiger Zeile | **bestätigt** (Tabelle unten); Bin Packing zwischen den Klassen (43 % klein, 17 % mittel), siehe Grenzen |
| Runden mit Propagation rettet `gap` (250 von 271), nicht `assign_side` und `setpart` | **bestätigt:** `gap` 83 und 94 von 100 (klein/mittel), `assign_side` und `setpart` 0 von 400 |
| **Tauchen mit Propagation (20 Rückschritte): Erfolg 92,4 %, 11,7 LPs; Kaskade 99,7 % mit 8,05 LPs** | **verschoben:** Tauchen mit Propagation 97,6 % mit 9,3 LPs, Kaskade 99,96 % (2 399 von 2 400) mit 5,8 LPs. Ursache zum größeren Teil der Rückschritt (siehe unten) |
| **Propagation im Tauchen hebt den Erfolg (20 Rückschritte) von 0,837 auf 0,924, die LPs von 18,1 auf 11,7** | **verschoben:** Erfolg 96,6 % → 97,6 %, LPs 13,4 → 9,3: Die Propagation spart knapp ein Drittel der LPs (31 %), hebt den Erfolg aber nur um einen Punkt |
| Kaskade kauft die wenigen LPs mit Güte (Lücke 19 % gegen 5 % bei Stufe 0) | **bestätigt über alle Instanzen, nicht in jeder Zelle:** 22,0 % gegen 7,1 % für das Tauchen (Mittel über die jeweils gelösten Instanzen); in 15 von 24 Zellen ist die Lücke der Kaskade größer, in 3 gleich, in 6 kleiner (siehe Befunde) |
| `gap`: Pump 268/271 mit 6,1 LPs | Größenabhängig: klein 93 von 100 mit 35,9 LPs, mittel 100 von 100 mit 5,6 LPs; die Vormessung hatte größere Instanzen, siehe Grenzen |
| Rundungsschwelle θ hat kein einheitliches Optimum | **bestätigt** (Befund unten, Regler in der App) |
| Startlösung spart netto nur bei 3 von 12 Familien (Tiefensuche), der Best-Bound-Suche fast nichts | **bestätigt, jetzt mit Band:** Tiefensuche netto positiv (95-%-Band über je 120 Instanzen) bei `gap`, `binpack`, `mkp`; bei sechs Familien negativ, bei dreien offen; Best-Bound nur `binpack` positiv, `gap` offen |
| Rucksack ab n ≈ 80–120: Abrunden + Propagation unter 1 % Lücke | **nicht neu gemessen** (die Größen dieser Demo enden bei 12 bis 120 ganzzahligen Variablen) |
| drei dünne Zellen (setcover klein 5, cover_card klein 16, ufl klein 11 Instanzen mit gebrochener Wurzel) | **behoben:** Die Messung sucht Seeds, bis 100 Instanzen mit gebrochener Wurzel beisammen sind (1 639, 1 269 und 1 591 Seeds angesehen) |

**Rückschritt beim Tauchen (Korrektur am eigenen Messcode).** Die Vormessung kehrte bei einem unzulässigen LP die *vorletzte* erfolgreiche Fixierung um, weil die gescheiterte selbst nie auf dem Stapel stand. Dieser Bau kehrt zuerst
die gescheiterte Fixierung selbst um (`Params.flip_failed`, Standard wahr; falsch ist das Verhalten der Vormessung). Gegenmessung mit `tools/rueckschritt_messen.py` auf denselben 2 400 Instanzen, 20 Rückschritte:

| Variante | mit Rückschritt der gescheiterten Fixierung (dieser Bau) | wie in der Vormessung |
|---|---|---|
| Fractional, ohne Propagation | 2 319 von 2 400 (13,4 LPs im Mittel) | 2 091 (14,3 LPs) |
| Fractional, mit Propagation | 2 342 (9,3 LPs) | 2 256 (9,5 LPs) |
| Coefficient, mit Propagation | 2 369 (8,0 LPs) | 2 297 |

Der Unterschied kommt bei Fractional ohne Propagation vor allem aus vier Zellen (211 der 228 Instanzen Unterschied): `mknap` (Fractional ohne Propagation klein 100 gegen 31, mittel 100 gegen 10 von 100), `setpart` klein (100 gegen 65) und `assign_side` klein (100 gegen 83). Mit dem Verhalten der Vormessung
liegen die Gesamtwerte (87,1 % und 94,0 %) näher an den Zahlen der Vormessung (83,7 % und 92,4 %, aus dem Planungsordner `primalheuristik-planung`, nicht im Repo): Der Rückschritt erklärt beim Tauchen mit Propagation 3,6 von 5,2 Punkten Abstand,
woher der Rest kommt, ist nicht gemessen (Vermutung: anderer Zufallsgenerator und andere Instanzen). Bei Coefficient mit Propagation spielt `mknap` keine Rolle, dort sind es vor allem `assign_side` klein und `setpart` klein.
Der Rückschritt ist nicht überall ein Vorteil: Fractional ohne Propagation bei `gap` mittel 89 gegen 92 und Fractional mit Propagation bei `seqprec` mittel 90 gegen 95 von 100 (Tests: `test_stored_backtracking_counter_measurement_…`, `test_backtracking_numbers_quoted_in_the_readme`).

## Modell

Ganzzahlige lineare Programme in Minimierungsform, $\min c^\top x$ unter $Ax \le b$, $A_{eq}x = b_{eq}$, $0 \le x \le 1$, $x_j \in \mathbb{Z}$ für $j \in I$ (die ganzzahligen Variablen sind hier immer 0/1, bei `ufl` kommen stetige Zuordnungsvariablen in [0, 1] hinzu; eine ≥-Zeile steht negiert in $A$; die **Länge** einer Zeile ist die Zahl ihrer Koeffizienten ungleich null).
Zwölf Familien in vier Klassen, je zwei Größen (klein: höchstens 25 ganzzahlige Variablen, bei der Standortplanung 6; mittel: 12 bis 120):

| Klasse | Familie | Zeilen | ganzzahlige Variablen klein / mittel |
|---|---|---|---|
| ≤ | `mkp` Mehrdimensionaler Rucksack | nur ≤ (5) | 12 / 20 |
| ≤ | `stable` gewichtete stabile Menge | nur ≤ (je Kante) | 16 / 30 |
| ≤ | `mknap` Mehrfach-Rucksack | nur ≤ | 18 / 32 |
| ≥ | `setcover` Mengenüberdeckung | nur ≥ | 24 / 120 |
| ≥ | `vertexcover` Knotenüberdeckung | nur ≥ | 16 / 50 |
| ≥ | `cover_card` Überdeckung mit höchstens K Mengen | ≥ und eine ≤-Zeile | 24 / 120 |
| = | `gap` Aufträge auf Maschinen (**Hauptfamilie**) | = und ≤ | 18 / 64 |
| = | `assign_side` Zuordnung mit Zusatzgleichungen | nur = | 25 / 100 |
| = | `setpart` Mengenpartitionierung | nur = | 21 / 60 |
| = | `binpack` Bin Packing | = und ≤ | im Mittel 21 / 42 |
| = | `seqprec` Reihenfolge mit Vorrang | = und ≤ | 25 / 81 |
| gemischt | `ufl` Standortplanung (Zuordnung stetig) | = und ≤ | 6 / 12 |

**Erzeuger der Instanzen** (`phr_scenario.py`; ganzzahlige Zufallswerte gleichverteilt in den genannten Bereichen, jeweils aus dem Zufalls-Seed der Instanz):

| Familie | Parameter (klein / mittel) und Zufallswerte |
|---|---|
| `mkp` | 12 / 20 Gegenstände, 5 Zeilen; Gewichte 10 bis 100; Kapazität je Zeile = abgerundete Hälfte der Zeilensumme; Gewinn = gerundetes mittleres Gewicht über die 5 Zeilen + 100 · u, u gleichverteilt in [0, 1) |
| `stable` | 16 / 30 Knoten, Kante je Knotenpaar mit Wahrscheinlichkeit 0,3 / 0,15 (keine isolierten Knoten); Gewichte 1 bis 10 |
| `mknap` | 6 × 3 / 8 × 4 (Gegenstände × Rucksäcke); Gewicht 10 bis 50, Wert = Gewicht + 0 bis 10; Kapazität je Rucksack = abgerundete 0,45 × Gesamtgewicht / Rucksäcke |
| `setcover` | 12 / 60 Elemente, doppelt so viele Mengen; Element in Menge mit Wahrscheinlichkeit 0,1, jedes Element in mindestens 2 Mengen; Kosten 1 bis 20 |
| `vertexcover` | 16 / 50 Knoten, Kantenwahrscheinlichkeit 0,3 / 0,12 (keine isolierten Knoten); Gewichte 1 bis 10 |
| `cover_card` | wie `setcover`, dazu eine Zeile „höchstens K Mengen“ mit K = kleinste Zahl von Mengen, die überdecken, plus 1 |
| `gap` | 3 × 6 / 4 × 16 (Maschinen × Aufträge); Kosten 10 bis 50, Größe 5 bis 25; Kapazität je Maschine = abgerundete (Summe der über die Maschinen gemittelten Größen) / (Maschinen · 0,85) |
| `assign_side` | N = 5 / 10; 3 / 5 Zusatzgleichungen über je 5 / 10 Zellen; Kosten 1 bis 50; rechte Seiten aus einer verborgenen Permutation (die Instanz ist zulässig) |
| `setpart` | 7 / 20 Elemente, dreimal so viele Mengen der Größe 2 bis 4; Kosten = Größe · (2 bis 7) + (0 bis 5); eine verborgene Partition macht die Instanz zulässig |
| `binpack` | 5 / 8 Gegenstände der Größe 4 bis 12, Behälterkapazität 20, Behälterzahl = aufgerundete (Gesamtgröße / 20) + 1 |
| `seqprec` | 5 / 9 Aufträge und ebenso viele Vorrangpaare; Kosten = Gewicht (1 bis 9) · Position + 0 bis 14; eine verborgene Reihenfolge macht die Instanz zulässig |
| `ufl` | 6 / 12 Standorte, 18 / 36 Kunden; Fixkosten 150 bis 400, Wegekosten 1 bis 50 |

Formeln (Sperren, Rundung, Propagation, Pump, Suchbaum) stehen im Expander „📐 Mathematische Formulierung“ der App. Die Instanzen kommen aus SplitMix64 (`phr_scenario.py`), nicht aus `numpy.random`: gleiche
Zufalls-Seeds, gleiche Instanzen auf Windows und Linux (Test gegen die veröffentlichten Referenzwerte).

## Methodik

* **Sperren:** `phr_scenario.Mip.uplocks/downlocks` zählen je Variable die Zeilen, die ein Vergrößern bzw. Verkleinern verletzen können. **Runden nach Sperren** (`phr_rounding.round_lock`) gelingt genau dann, wenn jede gebrochene
  Variable in mindestens einer Richtung ungesperrt ist (Satz, im Orakel-Test an 91 Instanzen mit gebrochener Wurzel geprüft, beide Ausgänge).
* **Runden mit Schwelle θ** (`round_near`; bei θ = 0,5 die nächste ganze Zahl; Schwelle θ ∈ {0,3; 0,5; 0,7; 1/f}, f = längste ≥-Zeile) und **Runden mit Propagation** (`round_prop`: die Variable mit dem kleinsten Abstand zur nächsten ganzen Zahl zuerst, bei Gleichstand die kleinere Nummer; zuerst die Richtung, die die Schwelle θ vorgibt, Schranken propagieren, bei Widerspruch die andere Richtung, sonst Abbruch ohne Rückschritt). Mit „1/f“ ist f die Länge der längsten reinen ≥-Zeile, ohne eine solche Zeile ist die Schwelle 0,5.
* **Tauchen** (`phr_diving.dive`): Fractional (kleinster Abstand zur nächsten ganzen Zahl, bei Gleichstand die kleinere Nummer), Coefficient (die Richtung mit weniger Sperren, die Variable mit den wenigsten Sperren zuerst) und Guided (die Variable mit dem kleinsten Abstand zum Wert der Referenzlösung, auf diesen Wert; Referenz ist hier die Lösung von „Runden mit Propagation“, ohne sie läuft das Tauchen nicht), 0 / 1 / 5 / 20
  Rückschritte, mit oder ohne Propagation nach jedem Festsetzen. Ein Rückschritt ist ein Versuch, die zuletzt festgesetzte, noch nicht umgekehrte Variable auf die andere Richtung zu setzen (auch ein erfolgloser zählt); „Sackgasse“ heißt: kein Rückschritt mehr möglich oder das Budget aufgebraucht.
* **Feasibility Pump** (`phr_pump.pump`, Fischetti, Glover, Lodi 2005; das Zielgewicht ist der Objective Feasibility Pump von Achterberg und Berthold 2007, *Discrete Optimization* 4, 77–86): T = 10 Umkehrungen bei Stillstand, Zufallsstörung bei Zyklus, Neustart nach 15 Iterationen ohne kleineren Abstand (0 / 3 / 10 Neustarts), höchstens 300
  Iterationen (jede ein LP), optional Zielgewicht: Startwert α₀ = 1, nach jedem Neustart wieder 1, sonst je Iteration × 0,9; die Zielfunktion mischt den Abstand mit den Kosten, skaliert mit √(Zahl ganzzahliger Variablen) / ‖c‖ (euklidische Norm der Kosten).
* **Kaskade** (`phr_algorithm.cascade`): Runden mit Propagation, nur bei Scheitern Tauchen mit Propagation (20 Rückschritte), nur dann der Pump mit 10 Neustarts.
* **Vergleich:** exaktes Optimum per HiGHS-MILP; ein einfacher Suchbaum (Tiefensuche und beste Schranke zuerst, ohne Schnitte und Presolve, Limit 1 000 LPs, `phr_tree.py`).
* **Unabhängiger Prüfer** (`phr_check.py`): kennt keine Heuristik; jede gemeldete Lösung muss ihn passieren.

Jede Heuristik ist eine kleine Einheit mit eigenem Test; die Schrittansicht der App zeichnet dieselben Läufe auf, mit und ohne Aufzeichnung ist das Ergebnis gleich (Test).

## Befunde (gemessen, keine Behauptungen)

Die Zahlen dieses Abschnitts stammen aus den gespeicherten Messungen `phr_landkarte.json` (2 400 Instanzen, je Zelle 100), `phr_theta.json`, `phr_baum.json`, `phr_rueckschritt.json` und `phr_zielgewicht.json`; `tests/test_claims.py` liest sie und prüft die genannten Zahlen und Zählungen (Verweise im Text).
Die Dateien erzeugen `tools/landkarte_messen.py`, `tools/theta_messen.py`, `tools/baum_messen.py`, `tools/rueckschritt_messen.py` und `tools/zielgewicht_messen.py`; drei Zellen der Landkarte werden im Test nachgemessen (Bänder, weil die LP-Wahl von HiGHS sich je Plattform um wenige
Instanzen verschieben kann). Zahlen der Vormessung stehen nur in der Tabelle „Befunde und Korrekturen“ und sind dort als solche gekennzeichnet.

**Landkarte, Runden nach Sperren (Instanzen mit gebrochener Wurzel, je Zelle 100, beide Größen zusammen 200):**

| Klasse | Familien | Erfolg von Runden nach Sperren | Test |
|---|---|---|---|
| ≤ | `mkp`, `stable`, `mknap` | 600 von 600 | `test_lock_rounding_always_succeeds_…`, `test_class_counts_of_lock_rounding` |
| ≥ | `setcover`, `vertexcover` | 400 von 400 | dieselben |
| ≥ und eine ≤-Zeile | `cover_card` | 0 von 200 | `test_lock_rounding_never_succeeds_…` |
| = | `gap`, `assign_side`, `setpart`, `seqprec` | 0 von 800 | dieselben |
| = | `binpack` | 43 von 100 (klein), 17 von 100 (mittel) | `test_lock_rounding_on_bin_packing_…` |
| gemischt | `ufl` | 200 von 200 | `test_lock_rounding_always_succeeds_…` |

Klassenweise (Instanzen): ≤ 600 von 600, ≥ 400 von 600, = 60 von 1 000, gemischt 200 von 200. Die Lücke der erfolgreichen Rundungen ist groß: im Mittel 49,8 % bei den ≤-Familien, 74,1 % bei den ≥-Familien (Landkarte, Spalte Lücke).
Bei `ufl` gelingt das Runden, weil die Gleichungen nur die stetigen Variablen enthalten und die ganzzahligen Standortvariablen nach oben ungesperrt sind; die Lücke liegt trotzdem bei 52,3 % (klein) und 60,4 % (mittel).

**Runden mit Propagation und mit Schwelle θ** (Test: `test_propagation_rescues_…`, `test_propagation_always_succeeds_…`, `test_nearest_integer_rounding_…`):
Runden mit Propagation gelingt bei allen ≤-Familien, `setcover`, `vertexcover` und `ufl` in jeder Instanz, bei `gap` in 83 (klein) und 94 (mittel) von 100, bei `binpack` in 99 und 98, bei `seqprec` in 77 und 47,
bei `cover_card` in 77 und 4, bei `assign_side` und `setpart` in 0 von 400. Über alle 2 400 Instanzen: Runden nach Sperren 1 260, mit Schwelle θ = 0,5 665, mit Propagation 1 779 Erfolge. Bei `vertexcover` gelingt θ = 0,5
in 200 von 200 (Deutung, nicht gemessen: die LP-Ecken sind halbganzzahlig), bei `stable` in 0 von 200.
Zählt man `ufl` mit, haben sechs Familien Gleichungen: Runden nach Sperren gelingt bei `gap`, `assign_side`, `setpart` und `seqprec` in 0 von 800 Instanzen, bei `binpack` in 60 von 200 und bei `ufl` in 200 von 200 (dort stehen die Gleichungen nur über den stetigen Zuordnungsvariablen und sperren die ganzzahligen Standortvariablen nicht). Warum `binpack` dazwischen liegt, ist nicht gemessen (Vermutung: die gebrochenen Variablen sind oft nur Behältervariablen, die nach oben ungesperrt sind).

**Tauchen und Pump** (Test: `test_pooled_diving_and_pump_numbers`, `test_dead_ends_exist_…`): Erfolge über 2 400 Instanzen: Fractional ohne Propagation 2 319 (13,4 LPs im Mittel), mit Propagation 2 342 (9,3 LPs), Coefficient mit
Propagation 2 369 (8,0 LPs); Pump mit Neustarts 2 294 (22,0 LPs, Lücke 33,1 %), mit Zielgewicht 2 193 (38,0 LPs, Lücke 12,6 %). Das Tauchen mit Propagation braucht bei sieben der zehn Gleichungszellen weniger LPs als der Pump;
teurer ist es bei `gap` mittel und bei `seqprec` klein und mittel (`test_diving_with_propagation_is_cheaper_…`). **Das Tauchen scheitert in Sackgassen (Ergebnis „Sackgasse“, nicht an der LP-Grenze):** Tauchen mit Propagation löst `cover_card` mittel nur in 64, `setpart` mittel in 91, `seqprec` mittel
in 90 von 100 Instanzen; der Pump bei `binpack` mittel in 79, bei `mknap` mittel in 63 (mit Zielgewicht 29) von 100. Das Zielgewicht senkt die Lücke der erfolgreichen Läufe in 22 von 24 Zellen, bei teils deutlich geringerer Erfolgsrate (`mknap` klein 75 → 28, mittel 63 → 29 von 100, `mkp` klein 100 → 87); dieser Vergleich mittelt über verschiedene Instanzen und enthält deshalb einen Auswahleffekt. **Gepaart** (`tools/zielgewicht_messen.py`, nur Instanzen, die beide Varianten lösen) senkt es die Lücke in 21 von 24 Zellen; nicht bei `binpack` mittel (20,8 → 21,9 % auf 67 gemeinsamen Instanzen), `seqprec` klein (2,2 → 2,9 %) und `mknap` mittel (14,4 → 21,3 % auf 28 gemeinsamen Instanzen, ungepaart 20,7 → 20,5 %). Bei `mknap` klein sinkt die Lücke gepaart von 11,6 auf 10,3 %, ungepaart von 20,3 auf 10,3 %: dort ist der größte Teil des scheinbaren Vorteils Auswahl der leichteren Instanzen (Test `test_paired_objective_weight_…`). Der Pump scheitert an der Iterationsgrenze, nicht in Sackgassen (Tests `test_objective_weight_…`, `test_failed_pump_runs_…`, geprüft an `binpack` mittel mit 40 und `mknap` klein mit 25 Instanzen).

**Kaskade** (Test: `test_cascade_pooled_numbers_and_stage_counts`, `test_cascade_stages_by_class_and_family`): löst 2 399 von 2 400 Instanzen mit im Mittel 5,79 LPs (Tauchen mit Propagation allein: 2 342, 9,27 LPs). Stufe 0 löst 1 779 Instanzen, Stufe 1
(Tauchen) 566, Stufe 2 (Pump) 54, keine Stufe 1 (`setpart` mittel). Die Stufenverteilung ist die Landkarte selbst: bei den ≤-Familien und `ufl` löst Stufe 0 jede Instanz, bei `assign_side` jede Instanz erst Stufe 1, bei `cover_card`
mittel lösen Stufe 0, 1 und 2 vier, 60 und 36 Instanzen. **Sie hört bei der ersten zulässigen Lösung auf, nicht bei der besten:** über die jeweils gelösten Instanzen (Kaskade 2 399, Tauchen 2 342) 22,0 % Lücke gegen 7,1 % beim Tauchen; in 15 von 24 Zellen ist die Lücke der Kaskade größer als die des Tauchens, in 3 gleich (`assign_side` klein und mittel,
`setpart` klein: Stufe 0 gelingt dort nie, die Kaskade ist dort das Tauchen) und in 6 kleiner (`mkp` und `gap` klein und mittel, `setpart` mittel, `seqprec` mittel). Mehr LPs als das Tauchen allein braucht sie in 2 Zellen (`setpart` mittel 28,6 gegen 21,4, `cover_card` mittel 56,7
gegen 54,5) (`test_cascade_gap_and_lps_compared_with_diving_per_cell`, `test_pump_versus_diving_per_cell_counts`).

**Rundungsschwelle θ** (Test: `test_no_single_threshold_is_best_…`, `test_threshold_numbers_quoted_in_the_readme`): Es gibt keine beste Schwelle für alle Familien. Erfolge von Runden mit Schwelle θ bei θ = 0,3 / 0,5 / 0,7 / 1/f
(beide Größen, 200 Instanzen): `stable` 0 / 0 / 200 / 0, `setcover` 184 / 120 / 0 / 200, `vertexcover` 200 / 200 / 0 / 200, `ufl` 200 / 177 / 1 / 177, `mkp` 4 / 59 / 160 / 59, `assign_side` und `setpart` bei keiner Schwelle ein Erfolg.
Bei `setcover` steigt von θ = 0,3 auf 1/f der Erfolg von 184 auf 200 von 200 und die Lücke der Erfolgreichen von 45,9 % auf 72,6 %, vor allem bei mittel (46,0 % → 97,6 %; klein 45,9 % → 47,6 %; Tests `test_more_aggressive_threshold_…`, `test_threshold_gap_of_set_cover_by_size`). Runden mit Propagation hängt bei `gap`, `binpack` und `seqprec` nicht an θ,
bei `cover_card` ja (Erfolge 77 / 81 / 103 / 73 von 200).

**Startlösung und Suchbaum** (60 Instanzen je Familie und Größe, also 120 je Familie, ab Seed 0, Limit 1 000 LPs, gepaart über Instanzen, in denen der Beweis ohne und mit Startlösung im Limit gelang; Test: `test_start_solution_…`): Ersparnis der Startlösung der
Kaskade in Prozent der LPs ohne Startlösung, brutto / netto nach Abzug aller LPs der Kaskade, dazu das 95-%-Band der Nettoersparnis (Bootstrap über die gepaarten Instanzen, 2 000 Ziehungen, SplitMix64). **Tiefensuche:** netto gesichert positiv bei `gap` (45,9 / 43,9, Band 33,4 bis 55,3),
`binpack` (16,1 / 12,9, Band 7,0 bis 20,8) und `mkp` (16,3 / 15,9, Band 11,5 bis 20,8); gesichert negativ bei `assign_side`, `setpart` (12,8 brutto, −76,5 netto), `stable`, `mknap`, `vertexcover` und `ufl`; **offen** (Band enthält null) bei `seqprec` (−0,8, Band −16,1 bis 13,7),
`setcover` (−4,2; −9,3 bis 1,7) und `cover_card` (−4,1; −14,8 bis 8,2). **Suche nach bester Schranke zuerst:** gesichert positiv nur bei `binpack` (22,8 / 21,9, Band 11,9 bis 33,7), `gap` offen (brutto 4,3, netto −1,6, Band −4,6 bis 1,6), bei den übrigen zehn Familien netto gesichert negativ,
brutto jeweils unter 5. Mit 30 Instanzen je Familie und Größe (frühere Fassung dieser Messung, die Datei wurde überschrieben und liegt nicht im Repo) waren es netto `gap` 34,7, `binpack` 21,2 und `mkp` 15,5: Die Zahlen schwanken mit der Stichprobe, das positive Vorzeichen dieser drei blieb.

## Ehrliche Grenzen

* **Erzeugte Instanzen, 6 bis 120 ganzzahlige Variablen, ganzzahlige Variablen nur 0/1.** Die Landkarte zeigt, *welches* Verfahren bei welcher Struktur gelingt; ihre Zahlen verschieben sich mit der Größe (z. B. `seqprec` Runden mit Propagation 77 → 47 von 100,
  `cover_card` 77 → 4, Tauchen bei `cover_card` 100 → 64). Allgemeine ganzzahlige Variablen sind nicht gemessen.
* **Der Suchbaum ist einfach:** ohne Schnitte, Presolve und Warmstart (scipy bietet keinen), höchstens 1 000 LPs. Instanzen ohne Beweis im Limit fehlen in den gepaarten Zahlen: bei der Tiefensuche `mkp` 7, `cover_card` 6, `mknap` 4 und `binpack` 4 von je 120, bei der Suche nach bester Schranke `binpack` 22 und `mknap` 4 von 120; gerade die schweren Instanzen fehlen also, wo eine Startlösung am meisten bringen könnte.
* **Aussagen über die gemessene Regel, nicht über die Methodenklasse:** „Tauchen“ ist hier Fractional/Coefficient/Guided mit chronologischem Rückschritt, „Pump“ der Pump nach Fischetti/Glover/Lodi mit den genannten Einstellungen (Zielgewicht nach Achterberg/Berthold).
* **Tauchgänge sind auf 400 LPs begrenzt** (`Params.max_lps`); in den untersuchten Zellen (`cover_card`, `setpart`, `seqprec` mittel) war jeder Fehlschlag eine Sackgasse, in anderen Zellen ist das nicht untersucht.
* **Abstimmung:** Die Einstellungen (20 Rückschritte, T = 10, φ = 0,9, 15 Iterationen Wartezeit, 10 Neustarts) stammen aus der Vormessung (Neustarts dort auf getrennten Zufalls-Seeds abgestimmt) und wurden im Bau nicht neu abgestimmt;
  Rundungsschwelle, Rückschritte und Neustarts sind als Regler über alle Werte gemessen, nicht abgestimmt.
* **Eine Heuristik allein beweist keine Optimalität:** ein Beweis entsteht nur, wenn ihre obere Schranke die untere (z. B. den aufgerundeten LP-Wert) erreicht; die Lücke kennt, wer das Optimum separat berechnet (HiGHS-MILP, in der Landkarte nur Instanzen mit bewiesenem Optimum).
* **Nicht gemessen:** Rechenzeit (gezählt werden LPs), Familien mit mehr als 150 ganzzahligen Variablen, Schnitte und Presolve im Vergleichsbaum, Relax-and-Fix, RINS und Local Branching.
* **Plattform:** Welche LP-Ecke HiGHS bei Mehrdeutigkeit liefert, kann zwischen Windows und Linux abweichen; einzelne LP-Zahlen (z. B. „5 LPs“ in den Preset-Hilfen) können sich um wenige verschieben. Die Tests prüfen deshalb Bänder.
* **Plotly-Auswahl** wird nicht benutzt; die Schrittansicht ist ein Regler (`st.slider`), im Browser geprüft, nicht mit echtem Mausklick auf Zellen.

## Verifikation

* **Kern:** jede Einheit (Sperren, die drei Rundungen, Propagation, Tauchen mit Rückschritt, Pumpschritte, Kaskade) gegen eine von Hand gerechnete Mini-Instanz (`tests/conftest.py`, `tests/test_algorithm.py`).
* **Orakel** (`tests/test_oracle_primalheuristiken.py`): Zulässigkeitsprüfer und Zielfunktion gegen die Definition jeder der zwölf Familien aus den Rohdaten; Optimum (HiGHS-MILP) gegen Aufzählung aller 0/1-Belegungen, gegen die
  geschlossene Formel der Standortplanung und gegen CP-SAT; jede gemeldete Heuristiklösung nach der Definition zulässig und nicht unter dem Optimum; der Sperren-Satz; Propagation schneidet nie eine bekannte Lösung ab und
  meldet Widersprüche im Einklang mit der Aufzählung; die gemeldete LP-Zahl gleich der Zahl der echten HiGHS-Aufrufe; Kaskade gleich der Zusammensetzung der Einzelläufe. Ein Scratch-Lauf (nicht im Repo, die Zahlen der Läufe sind deshalb hier nicht genannt) über 480 Instanzen
  (12 Familien × 2 Größen × 20 Seeds) fand keinen Fehler.
* **Suchbaum** (`tests/test_tree.py`): beweist das Optimum kleiner Instanzen (HiGHS, Aufzählung); eine Startlösung fügt der Tiefensuche in den geprüften Instanzen nie LPs hinzu.
* **Verdikte** als reine Funktionen (`tests/test_verdicts.py`), **Randfälle** (`tests/test_edge_cases.py`), **Szenario** (`tests/test_scenario.py`, SplitMix64 gegen veröffentlichte Werte), **Presets** und **Layout** (aktives Preset
  hervorgehoben, jeder Regler hebt es auf), **AppTest** (jeder Button, jeder Regler, jede Familie und Größe, jede Option, alle Messknöpfe), Footer und Laufzeit-Importe.
* **Aussagen:** `tests/test_claims.py` (siehe Befunde); dort auch das Verdikt „Rundung“ je Preset und für `mkp` klein mit den Seeds 0, 1 und 2 (`test_rounding_verdict_…`), nach einem Fehler der Anzeige, den die unabhängige Aussagenprüfung fand: Der Text behauptete „jede Rundung gelingt“, ohne die Rundung mit Schwelle θ zu prüfen.

## Tests

277 Tests (`pytest --collect-only -q tests`), lokal mit `-n 8` in rund 30 Sekunden. Die Live-Messungen der App laufen in den Tests mit 3 statt 20 Instanzen (Umgebungsvariable `PHR_LIVE_N`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Kopf, Kernfragen, Schrittansicht, Landkarte, Experimente, Vergleich, Formeln) |
| `phr_scenario.py` | SplitMix64, `Mip`, zwölf Instanzfamilien |
| `phr_lp.py` | LP-Löser mit Zähler (HiGHS), Schranken-Propagation |
| `phr_check.py` | unabhängiger Zulässigkeitsprüfer |
| `phr_exact.py` | HiGHS-MILP und Aufzählung als Referenz |
| `phr_rounding.py` | Sperren, Runden nach Sperren, mit Schwelle θ, mit Propagation |
| `phr_diving.py` | Tauchen (Fractional, Coefficient, Guided) mit Rückschritt |
| `phr_pump.py` | Feasibility Pump mit Neustarts |
| `phr_algorithm.py` | benannte Varianten und Kaskade (dünn) |
| `phr_result.py` | `Params`, `Result`, `Step` |
| `phr_tree.py` | einfacher Suchbaum (Tiefensuche, beste Schranke) |
| `phr_evaluation.py` | Analyse einer Instanz, Messungen, Verdikte (reine Funktionen) |
| `phr_landkarte.py` | Zugriff auf `phr_landkarte.json` |
| `phr_landkarte.json`, `phr_theta.json`, `phr_baum.json`, `phr_rueckschritt.json`, `phr_zielgewicht.json` | gespeicherte Messungen |
| `phr_constants.py` | Familien, Regler-Optionen, Messgrenzen, Farben, Presets |
| `phr_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `phr_layout.py` | Fester Kopf (Stylesheet) |
| `phr_ui.py`, `phr_visualization.py` | Darstellung (Schrittansicht, Landkarte, Gitter; Achsen gesperrt) |
| `tools/` | Messskripte (`landkarte_messen.py`, `theta_messen.py`, `baum_messen.py`, `rueckschritt_messen.py`, `zielgewicht_messen.py`) |
| `requirements.txt` / `requirements-dev.txt` | Laufzeit (streamlit, numpy, scipy, plotly) / plus pytest, ortools (Gegenprobe) |
| `tests/` | siehe Verifikation |

## Bewusst nicht enthalten

* **Aufrunden einzelner Modelle:** gibt es schon in [column-generation-demo](../column-generation-demo) (Spaltenmodell), [fixkosten-netzdesign-demo](../fixkosten-netzdesign-demo) (Startlösung des Baums) und
  [vertex-cover-demo](../vertex-cover-demo) (LP-Rundung der Knotenüberdeckung); hier sind sie Spezialfälle der Landkarte. Neu sind das Sperren-Argument, Propagation, Tauchen, der Pump und die Landkarte.
* Relax-and-Fix, RINS, Local Branching, Schnitte und Presolve im Vergleichsbaum, allgemeine ganzzahlige Variablen, Rechenzeitvergleiche, ein Pump mit Rückfallstufe (Zielfunktion → reiner Pump).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Messungen neu erzeugen: `python tools/landkarte_messen.py`, `python tools/theta_messen.py`, `python tools/baum_messen.py`, `python tools/rueckschritt_messen.py`, `python tools/zielgewicht_messen.py` (`landkarte_messen.py` brauchte lokal mit 6 Prozessen rund 200 bis 230 Sekunden).

Gebaut mit Streamlit, Plotly, NumPy und SciPy (HiGHS).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Exakte Suche erklärt: Rucksack und Cutting Stock](https://sebastianhanisch.net/konzepte-exakte-suche.html).
