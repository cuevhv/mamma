"""The rig identity metric must reject a constant permutation in one view."""
import numpy as np
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "eval_id_cross_camera_test", Path(__file__).resolve().parents[1] / "scripts/eval_id_continuity.py")
_evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluation)
score_camera = _evaluation.score_camera
score_sequence_records = _evaluation.score_sequence_records


def test_static_camera_permutation_is_invisible_locally_but_not_globally():
    left = np.zeros((10, 30), bool)
    left[2:8, 2:8] = True
    right = np.fliplr(left)
    gt = {f: {0: left, 1: right} for f in range(4)}
    records = {}
    for cam in range(4):
        records[str(cam)] = []
        pred = gt if cam < 3 else {f: {0: right, 1: left} for f in gt}
        local = score_camera(gt, pred, records=records[str(cam)])
        assert local["id_accuracy"] == 1.0
        assert local["id_switches"] == 0
    shared = score_sequence_records(records)
    assert shared["overall"]["id_accuracy"] == 0.75
    assert shared["cameras"]["3"]["id_accuracy"] == 0.0


def test_shared_metric_has_one_gauge_and_penalizes_missing_people():
    records = {"a": [{"frame": 0, "gt_ids": [0, 1], "pred_ids": [7],
                      "matches": [[7, 1, 0.8]]}],
               "b": [{"frame": 0, "gt_ids": [0, 1], "pred_ids": [7, 8],
                      "matches": [[7, 1, 0.8], [8, 0, 0.8]]}]}
    result = score_sequence_records(records)
    assert result["mapping"] == {7: 1, 8: 0}
    assert result["overall"]["id_accuracy"] == 1.0
    assert result["overall"]["id_recall"] == 0.75
    assert score_sequence_records({})["overall"]["id_accuracy"] == 0.0


def test_blank_png_is_not_a_predicted_instance():
    mask = np.ones((4, 4), dtype=bool)
    records = []
    score_camera({0: {0: mask}},
                 {0: {0: mask, 1: np.zeros_like(mask)}}, records=records)
    assert records[0]["pred_ids"] == [0]
    assert score_sequence_records({"A": records})["overall"]["id_precision"] == 1.


def test_pred_masks_read_from_packed_archive_and_loose_files(tmp_path):
    """ma_masks writes masks.zip by default; the evaluator must read it like loose PNGs."""
    import zipfile
    import cv2
    rng = np.random.default_rng(0)
    masks = {(f, p): (rng.random((12, 16)) > 0.5).astype(np.uint8) * 255 for f in range(3) for p in (1, 2)}
    packed, loose = tmp_path / "packed" / "IOI_01" / "masks", tmp_path / "loose" / "IOI_01" / "masks"
    packed.mkdir(parents=True); loose.mkdir(parents=True)
    with zipfile.ZipFile(packed / "masks.zip", "w", zipfile.ZIP_STORED) as z:
        for (f, p), m in masks.items():
            z.writestr(f"mask_{f:04d}_{p:02d}.png", cv2.imencode(".png", m)[1].tobytes())
            cv2.imwrite(str(loose / f"mask_{f:04d}_{p:02d}.png"), m)
    a = _evaluation.load_pred_masks(str(tmp_path / "packed"), "IOI_01", [0, 2])
    b = _evaluation.load_pred_masks(str(tmp_path / "loose"), "IOI_01", [0, 2])
    assert sorted(a) == sorted(b) == [0, 2]
    for f in a:
        assert sorted(a[f]) == sorted(b[f]) == [0, 1]
        for pid in a[f]:
            assert np.array_equal(a[f][pid], masks[(f, pid + 1)] > 127)
            assert np.array_equal(a[f][pid], b[f][pid])
