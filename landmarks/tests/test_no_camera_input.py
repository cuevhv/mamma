"""ma_2d exits with an error, before loading any model, when no camera input matches."""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_ma_2d  # noqa: E402


def test_no_camera_files_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(run_ma_2d, "build_model", lambda cfg: pytest.fail("model loaded"))
    args = SimpleNamespace(undistort=False, calibration=None, start=None, end=None,
                           videos_dir=None, images_root_dir=None, cam_names=["IOI_01"])
    with pytest.raises(SystemExit, match="no camera input"):
        run_ma_2d.main(args, str(tmp_path / "out"), None, img_folder=str(tmp_path))
