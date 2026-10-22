"""Person detector factory and the pipeline's detection contract."""
import numpy as np
import pytest
from PIL import Image

from core.person_detector import DFineDetector, build_person_detector
from core.pipeline import SegmentMultipleFrames


class _FixedDetector:
    """Stand-in for a user detector: fixed boxes, built as Class(checkpoint, device)."""

    def __init__(self, checkpoint=None, device="cpu"):
        self.args = (checkpoint, device)
        self.confs = []

    def detect(self, image, conf):
        self.confs.append(conf)
        return [
            (np.array([10, 10, 40, 60], np.float32), 0.9),
            (np.array([11, 10, 40, 61], np.float32), 0.8),  # duplicate of the first
            (np.array([50, 5, 50, 30], np.float32), 0.7),   # zero width, no crop
        ]


def test_import_path_detector():
    det = build_person_detector(f"{__name__}:_FixedDetector", "ckpt", "cpu")
    assert isinstance(det, _FixedDetector) and det.args == ("ckpt", "cpu")


def test_unknown_detector_name():
    with pytest.raises(ValueError, match="Unknown person detector"):
        build_person_detector("nope")


def test_yolo_needs_checkpoint():
    with pytest.raises(FileNotFoundError):
        build_person_detector("yolo", None)


def test_dfine_conf_map():
    xs = np.linspace(0.0, 1.0, 101)
    assert np.all(np.diff(np.interp(xs, *DFineDetector.CONF_MAP)) >= 0)
    # Production threshold and the lowest retry (ultralytics floors YOLO at 0.25).
    assert np.interp(0.5, *DFineDetector.CONF_MAP) == pytest.approx(0.65)
    assert np.interp(0.15, *DFineDetector.CONF_MAP) == pytest.approx(0.5)


def test_collect_detections_crops_and_dedups():
    seg = SegmentMultipleFrames.__new__(SegmentMultipleFrames)
    seg.assignment_config = {}
    seg.sam_version = "sam2"
    seg.person_detector = _FixedDetector()
    seg._log_info = lambda *a, **k: None
    img, dets = seg._collect_yolo_detections(Image.new("RGB", (80, 70)), 0.4, include_clip_features=False)
    assert seg.person_detector.confs == [0.4]
    assert img.size == (80, 70)
    assert [d[3] for d in dets] == [0.9]
    crop, feat, box, _ = dets[0]
    assert crop.shape == (50, 30, 3) and feat is None
    assert box.dtype == np.float32 and box.tolist() == [10, 10, 40, 60]
