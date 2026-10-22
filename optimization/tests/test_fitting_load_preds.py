"""ma_3d 2D-prediction loading: a camera whose 2D file has fewer body slots is
zero-filled (subject not visible there); unreadable or malformed files raise.

Run from the repo root:  python -m pytest optimization/tests -q
"""
import os
import sys

import numpy as np
import pytest

_OPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)

fitting = pytest.importorskip("utils.fitting")

T, N = 4, 512


def _cam(tmp_path):
    fn = tmp_path / "IOI_01.npz"
    np.savez(fn, cam_int=np.eye(3, dtype=np.float32), cam_ext=np.eye(4, dtype=np.float32),
             cam_img_w=2056, cam_img_h=1504)
    return str(fn)


def _load(tmp_path, pred_fn, body_id):
    args = (_cam(tmp_path), [str(pred_fn)], None, None, str(tmp_path), T, [], 0, 1, 1, None, body_id)
    return fitting.load_frames_in_parallel(args)


def _preds(tmp_path, n_people, **drop):
    rng = np.random.default_rng(0)
    data = dict(landmarks=rng.normal(size=(T, n_people, N, 3)).astype(np.float32),
                visibilities=rng.random((T, n_people, N)).astype(np.float32),
                contacts=rng.random((T, n_people, N)).astype(np.float32),
                floor_contacts=rng.random((T, n_people, N)).astype(np.float32))
    for k in drop:
        data.pop(k)
    fn = tmp_path / "pred_IOI_01.npz"
    np.savez(fn, **data)
    return fn, data


def test_present_body_loads_and_converts_uncertainty(tmp_path):
    fn, data = _preds(tmp_path, n_people=2)
    out = _load(tmp_path, fn, body_id=1)
    pts2d, vis = out[1], out[6]
    np.testing.assert_array_equal(pts2d[..., :2], data["landmarks"][:, 1, :, :2])
    np.testing.assert_allclose(pts2d[..., 2], np.sqrt(np.exp(data["landmarks"][:, 1, :, 2])), rtol=1e-6)
    np.testing.assert_array_equal(vis, data["visibilities"][:, 1])


def test_body_missing_from_camera_is_zero_filled(tmp_path):
    fn, _ = _preds(tmp_path, n_people=1)
    out = _load(tmp_path, fn, body_id=1)
    pts2d, vis, contacts, floor_contacts = out[1], out[6], out[9], out[10]
    assert pts2d.shape == (T, N, 3) and not pts2d.any()
    for arr in (vis, contacts, floor_contacts):
        assert arr.shape == (T, N) and not arr.any()


def test_missing_landmarks_key_raises(tmp_path):
    fn, _ = _preds(tmp_path, n_people=2, landmarks=True)
    with pytest.raises(KeyError):
        _load(tmp_path, fn, body_id=0)


def test_corrupt_file_raises(tmp_path):
    fn = tmp_path / "pred_IOI_01.npz"
    fn.write_bytes(b"PK\x03\x04 not a real archive")
    with pytest.raises(Exception) as exc:
        _load(tmp_path, fn, body_id=0)
    assert not isinstance(exc.value, IndexError)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load(tmp_path, tmp_path / "does_not_exist.npz", body_id=0)
