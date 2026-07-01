"""Drift guard for the curated step-settings manifest (``step_settings.py``).

Ensures every flag the friendly "Common settings" widgets emit still exists in
the step's argparse, and that the config-recipe selects resolve to non-empty
lists. Catches the manifest silently drifting from the step scripts.

Run from the ``mamma`` conda env (the per-step ``--help`` scrape imports the
step deps)::

    pytest gui/backend/tests/test_step_settings.py
"""
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]  # gui/backend
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import help_cache  # noqa: E402
import step_settings  # noqa: E402


@pytest.mark.parametrize("step", sorted(step_settings._STEP_SETTINGS))
def test_flag_settings_exist_in_argparse(step):
    """Every flag/flag_pair setting's flag name appears in the step's --help."""
    expected = [flag for s, flag in step_settings.iter_flag_names() if s == step]
    if not expected:
        pytest.skip(f"{step} has no flag-backed settings")
    names = {f["name"] for f in help_cache.get_flags(step)["flags"]}
    missing = [f for f in expected if f not in names]
    assert not missing, f"{step}: settings flags not found in argparse --help: {missing}"


def test_recipe_selects_nonempty():
    """The dynamic config-recipe selects resolve to at least one choice."""
    settings = step_settings.build_settings()
    by_id = {step: {s["id"]: s for s in items} for step, items in settings.items()}
    assert by_id["ma_3d"]["config_file"].get("choices"), "no ma_3d recipes found"
    assert by_id["ma_2d"]["config_path"].get("choices"), "no ma_2d configs found"
