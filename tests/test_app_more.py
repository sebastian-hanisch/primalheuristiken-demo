"""Weitere AppTests (der Rauchtest nach Vorlage steht in test_app.py): jede Familie und Größe, jede Option der Methodenregler, letzter Schritt jedes Reiters, alle Messknöpfe."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import phr_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))   # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_has_charts_and_a_title():
    at = _fresh_app()
    assert at.title and len(at.get("plotly_chart")) >= 3


def test_all_sliders_at_once_at_min_and_max():
    for edge in ("min", "max"):
        at = _fresh_app()
        for slider in at.slider:
            slider.set_value(int(slider.min if edge == "min" else slider.max))
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("family", list(C.FAMILIES))
@pytest.mark.parametrize("size", list(C.SIZE_LABELS))
def test_every_family_and_size_renders(family, size):
    at = _fresh_app()
    at.selectbox(key="family").set_value(family)
    at.radio(key="size").set_value(size)
    at.run(timeout=TIMEOUT)
    assert not at.exception, (family, size, [str(e) for e in at.exception])


def test_every_option_of_the_method_controls_runs():
    at = _fresh_app()
    for key, options in (("theta", C.THETA_OPTIONS), ("rule", tuple(C.RULE_LABELS)), ("backtracks", C.BACKTRACK_OPTIONS), ("restarts", C.RESTART_OPTIONS)):
        for value in options:
            at.radio(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])
    for key in ("prop", "objw"):
        for value in (True, False):
            at.toggle(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value)


def test_the_last_step_of_every_tab_renders():
    at = _fresh_app()
    for slider in [s for s in at.slider if s.key.startswith("step_")]:
        slider.set_value(slider.max)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_measurement_buttons_produce_a_verdict_or_table():
    at = _fresh_app()
    for key in ("run_map", "run_theta", "run_dive", "run_pump"):
        at.button(key=key).click().run(timeout=TIMEOUT)
        assert not at.exception, (key, [str(e) for e in at.exception])
    text = " ".join(e.value for e in at.info)
    assert "Runden mit Schwelle θ gelingt" in text and "Ohne Rückschritt gelingt das Tauchen" in text and "Ohne Zielgewicht gelingt der Pump" in text
