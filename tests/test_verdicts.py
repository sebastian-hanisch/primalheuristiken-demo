"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle der Art und jede Zahl im Text steht hier. Die Verdikte sind reine Funktionen
(Art, Text); die App ruft st.<Art>(Text)."""

import phr_constants as C
import phr_evaluation as ev


def test_verdict_locks_nothing_to_round():
    kind, text = ev.verdict_locks(0, 0, True, None)
    assert kind == "info" and "schon ganzzahlig" in text


def test_verdict_locks_all_free_names_count_and_gap():
    kind, text = ev.verdict_locks(5, 0, True, 12.5)
    assert kind == "success" and "Alle 5 gebrochenen Variablen" in text and "12,5 % über dem Optimum" in text
    assert "über dem Optimum" not in ev.verdict_locks(5, 0, True, None)[1]


def test_verdict_locks_blocked_names_both_counts_and_confirmation():
    kind, text = ev.verdict_locks(6, 2, False, None)
    assert kind == "warning" and "2 von 6 gebrochenen Variablen" in text and "bestätigt" in text
    assert "bestätigt" not in ev.verdict_locks(6, 2, True, None)[1]


def test_verdict_rounding_four_cases():
    kind, text = ev.verdict_rounding(False, False, True, 1)
    assert kind == "success" and "Runden mit Propagation gelingt" in text and "(1 LP)" in text
    assert "(3 LPs)" in ev.verdict_rounding(False, False, True, 3)[1]
    kind, text = ev.verdict_rounding(True, True, True, 1)
    assert kind == "info" and "gelingen alle drei Rundungen" in text
    kind, text = ev.verdict_rounding(True, False, True, 1)                 # mkp klein, Seed 1: nur die Rundung mit Schwelle θ scheitert
    assert kind == "info" and "gelingen Runden nach Sperren und Runden mit Propagation" in text and "nur Runden mit Schwelle θ scheitert" in text
    assert "alle drei" not in text
    kind, text = ev.verdict_rounding(False, True, False, 1)
    assert kind == "warning" and "nur: mit Schwelle θ" in text
    assert ev.verdict_rounding(False, False, False, 1)[0] == "error"


def test_verdict_dive_pump_cases():
    kind, text = ev.verdict_dive_pump(True, 5, 12.1, True, 9, 26.8, False)
    assert kind == "info" and "Tauchen mit 5 LPs und 12,1 % Lücke" in text and "Pump mit 9 LPs und 26,8 % Lücke" in text and "Weniger LPs: Tauchen" in text and "kleinere Lücke: Tauchen" in text
    assert "Weniger LPs: beide gleich" in ev.verdict_dive_pump(True, 5, 12.1, True, 5, 3.0, False)[1]
    assert "kleinere Lücke: Pump" in ev.verdict_dive_pump(True, 5, 12.1, True, 5, 3.0, False)[1]
    assert "Nur das Tauchen findet eine Lösung (5 LPs, 12,1 % Lücke)" in ev.verdict_dive_pump(True, 5, 12.1, False, 301, None, False)[1]
    assert "Nur der Pump findet eine Lösung (9 LPs, 4,0 % Lücke); das Tauchen scheitert nach 40 LPs" in ev.verdict_dive_pump(False, 40, None, True, 9, 4.0, False)[1]
    assert ev.verdict_dive_pump(False, 40, None, False, 301, None, False)[0] == "error"
    assert ev.verdict_dive_pump(False, 0, None, True, 9, 4.0, True)[0] == "warning"


def test_verdict_cascade_names_stage_and_compares_with_diving():
    kind, text = ev.verdict_cascade(0, 1, True, True, 5)
    assert kind == "success" and "Runden mit Propagation (Stufe 0)" in text and "insgesamt 1 LP." in text and "braucht 5 LPs" in text
    assert "Tauchen (Stufe 1)" in ev.verdict_cascade(1, 7, True, False, 0)[1] and "findet hier keine Lösung" in ev.verdict_cascade(1, 7, True, False, 0)[1]
    assert "der Pump (Stufe 2)" in ev.verdict_cascade(2, 30, True, True, 12)[1]
    assert ev.verdict_cascade(3, 80, False, False, 0)[0] == "error"


def _theta_rows(near, prop):
    rows = {}
    for lab, r in zip(C.THETA_OPTIONS, near):
        rows[f"R_near θ={lab}"] = {"rate": r}
    for lab, r in zip(C.THETA_OPTIONS, prop):
        rows[f"R_prop θ={lab}"] = {"rate": r}
    return rows


def test_sweep_verdict_theta_names_the_best_and_the_spread():
    kind, text = ev.sweep_verdict_theta(_theta_rows([0.1, 0.5, 0.25, 0.0], [0.8, 0.9, 0.7, 0.85]))
    assert kind == "info" and "bei θ = 0,5 am häufigsten (50 % der Instanzen)" in text and "50 Prozentpunkte" in text and "zwischen 70 % und 90 %" in text


def test_sweep_verdict_theta_when_nothing_succeeds():
    _, text = ev.sweep_verdict_theta(_theta_rows([0, 0, 0, 0], [0, 0, 0, 0]))
    assert "bei keiner der vier Schwellen (0 % der Instanzen)" in text and "Runden mit Propagation gelingt bei keiner Schwelle" in text
    assert "liegt je nach θ zwischen 0 % und 50 %" in ev.sweep_verdict_theta(_theta_rows([0, 0, 0, 0], [0, 0.5, 0.25, 0]))[1]


def test_landkarte_sentence_uses_the_stored_counts():
    c = {"n": 100, "methods": {"R_lock": {"ok": 0, "n": 100}, "R_prop": {"ok": 83, "n": 100}, "K1": {"ok": 100, "n": 100}}}
    s = ev.landkarte_sentence(c)
    assert "100 Instanzen" in s and "Runden nach Sperren in 0 %" in s and "Runden mit Propagation in 83 %" in s and "die Kaskade in 100 %" in s


def test_verdict_map_counts_always_and_never():
    def cell(lock, prop, k1):
        return {"methods": {"R_lock": {"ok": lock, "n": 10}, "R_prop": {"ok": prop, "n": 10}, "K1": {"ok": k1, "n": 10}}}
    kind, text = ev.verdict_map([cell(10, 10, 10), cell(0, 10, 10), cell(0, 5, 9), cell(10, 0, 10)])
    assert kind == "info" and "Von 4 Zellen" in text
    assert "Runden nach Sperren in 2 bei jeder Instanz und in 2 bei keiner" in text
    assert "Runden mit Propagation gelingt in 2 bei jeder und in 1 bei keiner" in text and "Kaskade löst jede Instanz in 3 Zellen" in text


def test_verdict_dive_sweep_and_pump_sweep_name_their_numbers():
    rows = {"ohne Propagation, 0 Rückschritte": {"rate": 0.5, "lps": 3.0}, "mit Propagation, 0 Rückschritte": {"rate": 0.75, "lps": 3.0},
            "ohne Propagation, 20 Rückschritte": {"rate": 0.9, "lps": 14.25}, "mit Propagation, 20 Rückschritte": {"rate": 0.95, "lps": 9.0}}
    _, text = ev.verdict_dive_sweep(rows)
    assert "50 % der Instanzen ohne und 75 % mit Propagation" in text and "90 % ohne (14,2 LPs" in text and "95 % mit Propagation (9,0 LPs" in text
    prows = {"ohne Zielgewicht, 0 Neustarts": {"rate": 0.8, "lps": 20.0, "gap": 25.0}, "ohne Zielgewicht, 10 Neustarts": {"rate": 0.95, "lps": 30.0, "gap": 24.0},
             "mit Zielgewicht, 10 Neustarts": {"rate": 0.9, "lps": 40.0, "gap": 10.0}}
    _, text = ev.verdict_pump_sweep(prows)
    assert "ohne Neustarts in 80 % (20,0 LPs)" in text and "10 Neustarts in 95 % (30,0 LPs)" in text and "ohne Zielgewicht 25,0 %" in text and "mit Zielgewicht 10,0 %" in text


def test_verdict_tree_cases():
    kind, text = ev.verdict_tree({"none": (100, True), "start": (60, True), "opt": (40, True)}, "Tiefensuche")
    assert kind == "info" and "ohne Startlösung 100 LPs" in text and "60 LPs (Beweis gelungen)" in text and "spart 40 LPs" in text and "wären es 40 LPs" in text
    assert "kostet 10 LPs" in ev.verdict_tree({"none": (50, True), "start": (60, False), "opt": (40, True)}, "Tiefensuche")[1]
    assert "ändert nichts" in ev.verdict_tree({"none": (50, True), "start": (50, True), "opt": (40, True)}, "Tiefensuche")[1]
    assert ev.verdict_tree({"none": (50, False), "start": None, "opt": (40, True)}, "Tiefensuche")[0] == "warning"


def test_gap_percent_uses_the_absolute_optimum_and_handles_missing_values():
    assert ev.gap_percent(-700.0, -742.0) == 100 * 42 / 742
    assert ev.gap_percent(110.0, 100.0) == 10.0 and ev.gap_percent(None, 100.0) is None and ev.gap_percent(1.0, None) is None
