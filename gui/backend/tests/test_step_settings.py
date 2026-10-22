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


def _ma_masks_setting(setting_id):
    return next(s for s in step_settings.build_settings()["ma_masks"] if s["id"] == setting_id)


def test_rig_vote_choices_match_argparse():
    """The rig-vote select offers exactly the values --rig_vote accepts, and
    Auto (the config default) emits no token."""
    s = _ma_masks_setting("rig_vote")
    assert s["target"] == {"kind": "flag", "flag": "--rig_vote", "valued": True}
    assert s["default"] == "auto" and s["omitWhenDefault"] is True
    hint = next(f["valueHint"] for f in help_cache.get_flags("ma_masks")["flags"]
                if f["name"] == "--rig_vote")
    assert hint == "{auto,true,false}"
    assert [c["value"] for c in s["choices"]] == hint.strip("{}").split(",")


def test_help_parser_keeps_choices_metavar_whole():
    text = ("options:\n"
            "  --mode {a,b,c}, --mode_alias {a,b,c}\n"
            "                        Pick one.\n")
    assert help_cache.parse_argparse_help(text) == [
        {"name": "--mode", "valueHint": "{a,b,c}", "help": "Pick one."}]


def _ma_masks_argv(monkeypatch, tmp_path, flags, cell_flags=None):
    """argv the runner builds for one ma_masks cell, through run_cell (the path
    local runs and cluster fan-out nodes share)."""
    repo = str(_BACKEND.parents[1])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from inference import env, runner
    env.bootstrap_env()
    cfg = {"global": {"out_dir": str(tmp_path), "dataset_name": "d",
                      "cam_names": ["IOI_01", "IOI_09", "IOI_17"]},
           "ma_masks": {"script": "run_ma_masks.py", "repo_path": "segmentation",
                        "flags": flags}}
    seen = {}

    def _dispatch(builder, seq, *_a):
        seen["argv"] = builder.python_argv(seq)
        return 0

    class _Sink:
        def update(self, *a, **k):
            pass

    monkeypatch.setattr(runner, "load_task_config", lambda p: cfg)
    monkeypatch.setattr(runner, "_log_paths", lambda *a, **k: ("/dev/null", "/dev/null"))
    monkeypatch.setattr(runner.engines, "dispatch", _dispatch)
    runner.run_cell("t.json", "ma_masks", "seqA", sink=_Sink(), force=True,
                    cell_id="c" if cell_flags else None, cell_flags=cell_flags)
    return seen["argv"]


def _last_value(argv, flag):
    """argparse last-occurrence value of ``flag``."""
    return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]


@pytest.mark.parametrize("value", ["true", "false"])
def test_rig_vote_reaches_local_run_and_fanout_finalize(monkeypatch, tmp_path, value):
    """The GUI's ``--rig_vote <value>`` token reaches a local run and the
    fan-out finalize cell, while per-camera cells keep it off."""
    flags = ["--sam_version sam2", f"--rig_vote {value}"]  # as the settings widget writes them
    assert _last_value(_ma_masks_argv(monkeypatch, tmp_path, flags), "--rig_vote") == value
    cell = _ma_masks_argv(monkeypatch, tmp_path, flags,
                          "--cam_names IOI_01 IOI_09 --cam_init IOI_01 --rig_vote false --skip_collage")
    assert _last_value(cell, "--rig_vote") == "false"
    final = _ma_masks_argv(monkeypatch, tmp_path, flags, "--cam_init IOI_01")
    assert _last_value(final, "--rig_vote") == value
