"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Optionen der Regler."""

import phr_constants as C
import phr_evaluation as ev
import phr_presets as P
from phr_scenario import make_instance


def test_every_preset_has_help_and_valid_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_options():
    for preset in C.PRESETS.values():
        assert preset["family"] in C.FAMILIES and preset["size"] in C.SIZE_LABELS and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[next(iter(C.PRESETS))]) == ev.Settings()


def test_setting_specs_cover_every_settings_field_and_every_preset_key():
    assert set(P.SETTING_SPECS) == set(ev.Settings.__dataclass_fields__)
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)


def test_defaults_are_valid_options():
    for spec in P.SETTING_SPECS.values():
        if spec.options is not None:
            assert spec.default in spec.options
        else:
            assert spec.lo <= spec.default <= spec.hi


def test_option_values_of_the_regler_match_the_constants():
    assert P.SETTING_SPECS["theta"].options == C.THETA_OPTIONS and P.SETTING_SPECS["backtracks"].options == C.BACKTRACK_OPTIONS
    assert P.SETTING_SPECS["restarts"].options == C.RESTART_OPTIONS and set(P.SETTING_SPECS["rule"].options) == set(C.RULE_LABELS)


def test_permalink_booleans_roundtrip():
    spec = P.SETTING_SPECS["prop"]
    assert spec.caster(str(True)) is True and spec.caster(str(False)) is False


def test_preset_instances_exist():
    for preset in C.PRESETS.values():
        m = make_instance(preset["family"], C.SIZE_LABELS.index(preset["size"]), preset["seed"])
        assert len(m.ints) > 0
