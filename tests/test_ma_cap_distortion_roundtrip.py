"""Round-trip tests for the ma_cap NPZ distortion contract.

The ma_cap per-camera NPZ must carry **all three** distortion models
(``radtan`` / ``opencv_brown`` / ``vicon_radial_2``) so the downstream
``--undistort`` overlay path works for every lens, not just Vicon. These tests
write a per-camera NPZ via ``capture.run_ma_cap._write_cam_npz`` and read it back
via ``visualization.cameras.load_cameras``, asserting the model + coefficients
survive. A final case proves legacy NPZs (only the ``vicon_radial_2`` key) still
load through the back-compat fallback.

No model weights / GPU needed.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    from capture.calibration import Camera as CalibCamera
    from capture.run_ma_cap import _write_cam_npz
    from capture.frame_source import (
        _camera_from_cam_data,
        frame_source_from_cam_data,
    )
    from visualization.cameras import load_cameras
except Exception as exc:  # pragma: no cover - bare env without deps
    CalibCamera = None
    _IMPORT_ERR = exc


def _load_npz_as_dict(path):
    """Mirror how ma_2d / ma_masks load a per-camera NPZ into a cam_data dict."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


_K = np.array([[1000.0, 0.0, 640.0],
               [0.0, 1000.0, 360.0],
               [0.0, 0.0, 1.0]], dtype=np.float64)
_EYE4 = np.eye(4, dtype=np.float64)

# (model, coeffs) -- one representative per supported distortion model.
_MODELS = [
    ("radtan", (0.1, -0.05, 0.001, 0.002)),                 # 4-param
    ("opencv_brown", (0.1, -0.05, 0.001, 0.002, 0.01)),     # 5-param
    ("vicon_radial_2", (640.0, 360.0, 1e-7, 1e-13, 0.0)),   # pixel-space radial
]


def _make_cam(name, model, coeffs):
    return CalibCamera(
        name=name,
        width=1280,
        height=720,
        intrinsics=_K,
        distortion_model=model,
        distortion_coeffs=tuple(coeffs),
        T_cam_world=_EYE4,
        T_world_cam=_EYE4,
    )


@pytest.mark.skipif(CalibCamera is None, reason="capture/visualization deps unavailable")
class TestDistortionRoundTrip:
    def test_all_models_roundtrip(self, tmp_path):
        gt = tmp_path / "gt"
        gt.mkdir()
        for i, (model, coeffs) in enumerate(_MODELS):
            name = f"cam{i:02d}"
            _write_cam_npz(
                gt / f"{name}.npz",
                cam=_make_cam(name, model, coeffs),
                image_paths=[],
                ioi_seq_dir=tmp_path,
                frames_len=1,
                video_path="dummy.mp4",
            )

        by_name = {c.name: c for c in load_cameras(gt)}
        assert len(by_name) == len(_MODELS)
        for i, (model, coeffs) in enumerate(_MODELS):
            cam = by_name[f"cam{i:02d}"]
            assert cam.distortion_model == model
            assert cam.distortion_coeffs == pytest.approx(tuple(coeffs))

    def test_legacy_vicon_only_npz_fallback(self, tmp_path):
        """An old NPZ carrying only the ``vicon_radial_2`` key still loads."""
        gt = tmp_path / "gt"
        gt.mkdir()
        coeffs = (640.0, 360.0, 1e-7, 1e-13, 0.0)
        np.savez(
            gt / "cam00.npz",
            cam_name=np.array("cam00"),
            cam_int=_K.astype(np.float32),
            cam_ext=_EYE4.astype(np.float32),
            cam_img_w=1280,
            cam_img_h=720,
            vicon_radial_2=np.array(coeffs, dtype=np.float64),
        )
        (cam,) = load_cameras(gt)
        assert cam.distortion_model == "vicon_radial_2"
        assert cam.distortion_coeffs == pytest.approx(coeffs)

    def test_no_distortion_defaults_to_noop(self, tmp_path):
        """An NPZ with no distortion fields defaults to radtan + zero (no-op)."""
        gt = tmp_path / "gt"
        gt.mkdir()
        np.savez(
            gt / "cam00.npz",
            cam_name=np.array("cam00"),
            cam_int=_K.astype(np.float32),
            cam_ext=_EYE4.astype(np.float32),
            cam_img_w=1280,
            cam_img_h=720,
        )
        (cam,) = load_cameras(gt)
        assert cam.distortion_model == "radtan"
        assert all(c == 0.0 for c in cam.distortion_coeffs)


@pytest.mark.skipif(CalibCamera is None, reason="capture/visualization deps unavailable")
class TestCamDataDistortionFallback:
    """frame_source builds a Camera from the NPZ distortion when no explicit
    --calibration camera is supplied (ma_2d / ma_masks --undistort fallback)."""

    def _cam_data(self, tmp_path, model, coeffs):
        gt = tmp_path / "gt"
        gt.mkdir(exist_ok=True)
        path = gt / "cam00.npz"
        _write_cam_npz(
            path,
            cam=_make_cam("cam00", model, coeffs),
            image_paths=["/fake/frame_000.jpg"],
            ioi_seq_dir=tmp_path,
            frames_len=1,
        )
        return _load_npz_as_dict(path)

    def test_camera_from_cam_data_all_models(self, tmp_path):
        for model, coeffs in _MODELS:
            cam_data = self._cam_data(tmp_path, model, coeffs)
            cam = _camera_from_cam_data(cam_data)
            assert cam is not None
            assert cam.distortion_model == model
            assert cam.distortion_coeffs == pytest.approx(tuple(coeffs))
            assert np.asarray(cam.intrinsics).shape == (3, 3)

    def test_camera_from_cam_data_legacy_vicon_only(self):
        coeffs = (640.0, 360.0, 1e-7, 1e-13, 0.0)
        cam_data = {
            "cam_name": np.array("cam00"),
            "cam_int": _K,
            "cam_img_w": 1280,
            "cam_img_h": 720,
            "vicon_radial_2": np.array(coeffs, dtype=np.float64),
        }
        cam = _camera_from_cam_data(cam_data)
        assert cam is not None
        assert cam.distortion_model == "vicon_radial_2"
        assert cam.distortion_coeffs == pytest.approx(coeffs)

    def test_camera_from_cam_data_none_when_absent(self):
        cam_data = {"cam_name": np.array("cam00"), "cam_int": _K}
        assert _camera_from_cam_data(cam_data) is None

    def test_frame_source_uses_npz_distortion_without_explicit_camera(self, tmp_path):
        cam_data = self._cam_data(tmp_path, "radtan", (0.1, -0.05, 0.001, 0.002))
        src = frame_source_from_cam_data(cam_data, undistort=True)  # no camera=
        assert src._undistort is True
        assert src._camera is not None
        assert src._camera.distortion_model == "radtan"

    def test_frame_source_noop_when_no_distortion(self, tmp_path):
        cam_data = self._cam_data(tmp_path, "radtan", (0.1, -0.05, 0.001, 0.002))
        # Drop distortion fields -> undistort should silently disable (no raise).
        for k in ("distortion_model", "distortion_coeffs", "vicon_radial_2"):
            cam_data.pop(k, None)
        src = frame_source_from_cam_data(cam_data, undistort=True)
        assert src._undistort is False
        assert src._camera is None
