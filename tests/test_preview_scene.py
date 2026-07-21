"""Unit tests for ``visualization.preview_scene`` (GUI 3D mesh-preview blob).

Runs ``build_scene`` against a tiny synthetic SMPL-X model + params npz, so
the blob schema the three.js viewer depends on is guarded without touching
``data/body_models``. No model weights / GPU needed.
"""
import base64
import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from visualization.preview_scene import build_scene  # noqa: E402

V, T, NB = 50, 5, 16


def _decode(b64, dtype):
    return np.frombuffer(base64.b64decode(b64), dtype=dtype)


def _write_model(path, rng):
    parents = np.zeros(55, dtype=np.uint32)
    parents[1:] = np.arange(55, dtype=np.uint32)[:-1]  # simple chain
    np.savez(
        path,
        v_template=rng.standard_normal((V, 3)),
        shapedirs=rng.standard_normal((V, 3, NB)) * 0.01,
        posedirs=np.zeros((V, 3, 486)),
        J_regressor=np.abs(rng.standard_normal((55, V))),
        weights=np.abs(rng.standard_normal((V, 55))) + 1e-3,
        f=rng.integers(0, V, size=(30, 3)).astype(np.uint32),
        kintree_table=np.stack([parents, np.arange(55, dtype=np.uint32)]),
        hands_meanl=rng.standard_normal(45) * 0.1,
        hands_meanr=rng.standard_normal(45) * 0.1,
    )


def _write_params(seq_dir, rng, *, flat_hand=False, v_template_pred=None):
    seq_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        seq_dir / "smplx_params_body_id-00.npz",
        smplx_pose=rng.standard_normal((T, 165)) * 0.2,
        smplx_betas=rng.standard_normal((1, NB)),
        smplx_translation=rng.standard_normal((T, 3)),
        v_template_pred=v_template_pred,
        smplx_export_flat_hand_mean=flat_hand,
    )


@pytest.fixture()
def scene_inputs(tmp_path):
    rng = np.random.default_rng(0)
    model = tmp_path / "SMPLX_NEUTRAL.npz"
    _write_model(model, rng)
    _write_params(tmp_path / "seq", rng)
    return tmp_path, model, rng


def test_blob_schema_and_shapes(scene_inputs):
    root, model, _ = scene_inputs
    scene = build_scene(root, "seq", model_npz=model)

    assert scene["version"] == 2
    assert scene["vertexCount"] == V and scene["jointCount"] == 55
    assert scene["frameCount"] == T
    assert len(scene["parents"]) == 55 and scene["parents"][0] == -1
    assert _decode(scene["faces"], "<u4").size == 30 * 3
    assert _decode(scene["skinIndex"], "<u2").size == V * 4
    sw = _decode(scene["skinWeight"], "<f4").reshape(V, 4)
    np.testing.assert_allclose(sw.sum(axis=1), 1.0, atol=1e-5)

    (p,) = scene["people"]
    assert p["bodyId"] == 0 and p["frameCount"] == T
    assert _decode(p["vShaped"], "<f4").size == V * 3
    assert _decode(p["restJoints"], "<f4").size == 55 * 3
    assert _decode(p["poses"], "<f4").size == T * 55 * 3
    assert _decode(p["transl"], "<f4").size == T * 3
    assert p["color"].startswith("#") and len(p["color"]) == 7
    json.dumps(scene)  # whole blob must be JSON-serializable


def test_no_verts_joints_falls_back_gracefully(scene_inputs):
    """Without verts_joints npz the up-axis vote has no joints: default z, no floor."""
    root, model, _ = scene_inputs
    scene = build_scene(root, "seq", model_npz=model)
    assert scene["upAxis"] == "z"
    assert scene["floor"] is None


def test_fps_recorded_vs_fallback(scene_inputs):
    """The ma_cap recorded rate (gt/global.npz) wins; else the fps arg is used
    and flagged as fallback so the viewer may apply a client-side hint."""
    root, model, _ = scene_inputs
    # no ma_cap dir -> fallback
    scene = build_scene(root, "seq", model_npz=model, fps=24)
    assert scene["fps"] == 24 and scene["fpsSource"] == "fallback"
    # recorded rate in gt/global.npz -> wins over the fps arg
    gt = root / "cap" / "seq" / "gt"
    gt.mkdir(parents=True)
    np.savez(gt / "global.npz", fps=20)
    scene = build_scene(root, "seq", model_npz=model, fps=24, ma_cap_dir=root / "cap")
    assert scene["fps"] == 20 and scene["fpsSource"] == "recorded"


def test_hand_means_baked_only_when_not_flat(tmp_path):
    rng = np.random.default_rng(1)
    model = tmp_path / "SMPLX_NEUTRAL.npz"
    _write_model(model, rng)
    pose = rng.standard_normal((T, 165)) * 0.2
    for name, flat in (("flat", True), ("mean", False)):
        d = tmp_path / name / "seq"
        d.mkdir(parents=True)
        np.savez(d / "smplx_params_body_id-00.npz",
                 smplx_pose=pose, smplx_betas=rng.standard_normal((1, NB)),
                 smplx_translation=np.zeros((T, 3)), v_template_pred=None,
                 smplx_export_flat_hand_mean=flat)
    m = np.load(model)
    flat_p = _decode(build_scene(tmp_path / "flat", "seq", model_npz=model)
                     ["people"][0]["poses"], "<f4").reshape(T, 55, 3)
    mean_p = _decode(build_scene(tmp_path / "mean", "seq", model_npz=model)
                     ["people"][0]["poses"], "<f4").reshape(T, 55, 3)
    np.testing.assert_allclose(flat_p[:, :25], mean_p[:, :25], atol=1e-6)
    np.testing.assert_allclose(
        mean_p[:, 25:40] - flat_p[:, 25:40],
        np.broadcast_to(m["hands_meanl"].reshape(15, 3), (T, 15, 3)), atol=1e-5)
    np.testing.assert_allclose(
        mean_p[:, 40:55] - flat_p[:, 40:55],
        np.broadcast_to(m["hands_meanr"].reshape(15, 3), (T, 15, 3)), atol=1e-5)


def test_v_template_pred_overrides_model_template(tmp_path):
    rng = np.random.default_rng(2)
    model = tmp_path / "SMPLX_NEUTRAL.npz"
    _write_model(model, rng)
    vt = rng.standard_normal((V, 3))
    _write_params(tmp_path / "seq", rng, v_template_pred=vt)
    scene = build_scene(tmp_path, "seq", model_npz=model)
    v_shaped = _decode(scene["people"][0]["vShaped"], "<f4").reshape(V, 3)
    m = np.load(model)
    betas = np.load(tmp_path / "seq" / "smplx_params_body_id-00.npz")["smplx_betas"].reshape(-1)
    expected = vt + np.einsum("vij,j->vi", m["shapedirs"][:, :, :NB], betas)
    np.testing.assert_allclose(v_shaped, expected, atol=1e-5)


def test_missing_params_raises(tmp_path):
    rng = np.random.default_rng(3)
    model = tmp_path / "SMPLX_NEUTRAL.npz"
    _write_model(model, rng)
    (tmp_path / "empty_seq").mkdir()
    with pytest.raises(FileNotFoundError):
        build_scene(tmp_path, "empty_seq", model_npz=model)
