"""Permalink (Begrenzen/Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Regler mit festen Optionen (Familie, Größe, Schwelle, Rückschritte, Neustarts, Tauchregel, Schalter) stehen mit `options` in SETTING_SPECS; ein Wert aus der Adresszeile
außerhalb der Optionen wird ignoriert (Standardwert bleibt)."""

import random
from dataclasses import dataclass

import streamlit as st

import phr_constants as C


def _bool(text):
    return str(text).strip().lower() in ("1", "true", "ja")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None


SETTING_SPECS = {
    "family": SettingSpec("fam", str, C.DEFAULT_FAMILY, options=C.FAMILY_KEYS),
    "size": SettingSpec("size", str, C.DEFAULT_SIZE, options=C.SIZE_LABELS),
    "seed": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "theta": SettingSpec("theta", str, C.DEFAULTS["theta"], options=C.THETA_OPTIONS),
    "rule": SettingSpec("rule", str, C.DEFAULTS["rule"], options=tuple(C.RULE_LABELS)),
    "backtracks": SettingSpec("bt", int, C.DEFAULTS["backtracks"], options=C.BACKTRACK_OPTIONS),
    "prop": SettingSpec("prop", _bool, C.DEFAULTS["prop"], options=(False, True)),
    "restarts": SettingSpec("rs", int, C.DEFAULTS["restarts"], options=C.RESTART_OPTIONS),
    "objw": SettingSpec("objw", _bool, C.DEFAULTS["objw"], options=(False, True)),
}
PRESET_KEYS = {"family": "family", "size": "size", "seed": "seed"}      # Schlüssel im Preset → Schlüssel im Session State; alle übrigen Regler stehen auf ihrem Standard


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen bzw. Optionen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.options is not None:
                if value not in spec.options:
                    continue
            else:
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Preset laden: Familie, Größe und Zufalls-Seed aus dem Preset, alle anderen Regler auf ihren Standardwert."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed"] = random.randint(0, C.SEED_MAX)
