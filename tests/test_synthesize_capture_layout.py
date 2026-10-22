"""CLI ``--footage/--calib/--seq-name`` captures keep their footage layout.

``synthesize_capture`` -> ``materialize_run_config`` -> ma_cap argv must point
ma_cap at image directories for image footage and at the detected videos
subdir for video footage.
"""
import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from inference import config  # noqa: E402
from inference.steps import get_builder  # noqa: E402

_CALIB = _REPO / "configs/examples/calib/140725_Breakdance.yaml"
_PRESET = _REPO / "configs/examples/presets/quick.yaml"
_CAMS = ("IOI_01", "IOI_02")


def _ma_cap_argv(tmp_path):
    synth = config.synthesize_capture(str(tmp_path / "footage"), str(_CALIB), "seq01")
    cap = tmp_path / "cap.json"
    cap.write_text(json.dumps(synth))
    cfg = config.materialize_run_config(str(_PRESET), str(cap), dataset_name="footage")
    return cfg, get_builder("ma_cap", cfg["ma_cap"], cfg["global"], "t").build_argv("seq01")


def test_image_footage_uses_image_dirs(tmp_path):
    for cam in _CAMS:
        d = tmp_path / "footage" / "seq01" / cam
        d.mkdir(parents=True)
        (d / "000000.jpg").write_bytes(b"")
    cfg, argv = _ma_cap_argv(tmp_path)
    assert "videos_dir" not in cfg["ma_cap"]
    assert "--videos_dir" not in argv
    capture = json.loads(Path(argv[argv.index("--json") + 1]).read_text())
    root = capture.get("capture_root") or capture.get("ioi_root")
    assert os.path.isdir(os.path.join(root, "seq01", _CAMS[0]))


@pytest.mark.parametrize("subdir", ["videos", "videos_crf24"])
def test_video_footage_uses_detected_subdir(tmp_path, subdir):
    d = tmp_path / "footage" / "seq01" / subdir
    d.mkdir(parents=True)
    for cam in _CAMS:
        (d / f"{cam}.mp4").write_bytes(b"")
    _, argv = _ma_cap_argv(tmp_path)
    videos_dir = argv[argv.index("--videos_dir") + 1]
    assert Path(videos_dir) == d
