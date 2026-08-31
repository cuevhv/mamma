"""engine / sif_path / docker_image fall back to the `global` block.

Lets an all-container preset set them once in global instead of on every step,
while a per-step value still overrides (so mixing conda + apptainer stays
possible). Mirrors the existing conda_env precedence.
"""
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from inference import config
from inference.steps import get_builder


def _cap_step(**over):
    base = {"enabled": True, "script": "run_ma_cap.py", "repo_path": "capture",
            "dependencies": [], "submit_cfg": {"gpus": 0}}
    base.update(over)
    return base


def test_engine_and_sif_inherit_from_global():
    g = {"engine": "apptainer", "sif_path": "/abs/mamma.sif",
         "out_dir": "/out", "dataset_name": "d"}
    b = get_builder("ma_cap", _cap_step(), g, "tag")
    assert b.engine == "apptainer"
    assert b.sif_path == "/abs/mamma.sif"


def test_per_step_engine_overrides_global():
    g = {"engine": "apptainer", "sif_path": "/abs/mamma.sif",
         "out_dir": "/out", "dataset_name": "d"}
    b = get_builder("ma_cap", _cap_step(engine="conda"), g, "tag")
    assert b.engine == "conda"  # step wins


def test_engine_defaults_to_conda_without_global():
    b = get_builder("ma_cap", _cap_step(), {"out_dir": "/out", "dataset_name": "d"}, "tag")
    assert b.engine == "conda"


# A real capture file so validate's global.capture_json check passes; the
# fallback behaviour is what these two assert.
_CAPTURE = os.path.join(_REPO, "configs/examples/captures/mamma_example.json")


def _base_global(**over):
    g = {"dataset_name": "d", "out_dir": "o", "capture_json": _CAPTURE}
    g.update(over)
    return g


def test_validate_accepts_global_engine_and_sif():
    cfg = {
        "global": _base_global(engine="apptainer", sif_path="/s.sif"),
        "ma_cap": {"enabled": True, "script": "run_ma_cap.py",
                   "repo_path": "capture", "dependencies": []},
    }
    config.validate(cfg)  # must not raise: sif inherited from global


def test_validate_flags_missing_sif_even_with_global_engine():
    cfg = {
        "global": _base_global(engine="apptainer"),  # no sif anywhere
        "ma_cap": {"enabled": True, "script": "run_ma_cap.py",
                   "repo_path": "capture", "dependencies": []},
    }
    with pytest.raises(config.TaskConfigError, match="sif_path"):
        config.validate(cfg)
