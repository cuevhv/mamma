"""ma_2d --weights precedence: explicit flag > task.json `weights:` > env.

An explicit --weights among the raw task flags must suppress the builder's
own injection, so the final command carries exactly ONE --weights and it is
the user's (with both present, argparse's last-wins tie-break would
silently decide).
"""
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from inference.assets import step_argv_translation
from inference.steps import get_builder


def _weights_positions(argv):
    out = []
    for i, tok in enumerate(argv):
        if tok == "--weights":
            out.append((i, argv[i + 1]))
        elif isinstance(tok, str) and tok.startswith("--weights="):
            out.append((i, tok.split("=", 1)[1]))
    return out


@pytest.fixture()
def _ma2d_env(monkeypatch):
    for env_key, _flag in step_argv_translation("ma_2d"):
        monkeypatch.setenv(env_key, "/x/asset")


def _step(**over):
    base = {"enabled": True, "script": "run_ma_2d.py", "repo_path": "landmarks",
            "config_path": "cfg.yaml", "weights": "data/weights/pinned.ckpt",
            "dependencies": []}
    base.update(over)
    return base


def test_explicit_weights_flag_wins_and_appears_once(_ma2d_env):
    g = {"out_dir": "/out", "dataset_name": "d"}
    b = get_builder("ma_2d", _step(flags=["--weights /custom/commercial.ckpt"]), g, "tag")
    hits = _weights_positions(b.build_argv("seqA"))
    assert len(hits) == 1
    assert hits[0][1] == "/custom/commercial.ckpt"


def test_weights_field_injected_when_no_flag(_ma2d_env):
    g = {"out_dir": "/out", "dataset_name": "d"}
    b = get_builder("ma_2d", _step(flags=[]), g, "tag")
    hits = _weights_positions(b.build_argv("seqA"))
    assert len(hits) == 1
    assert hits[0][1].endswith("data/weights/pinned.ckpt")
