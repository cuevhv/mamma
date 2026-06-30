"""Targeted unit tests for SAM3 prompt helper logic.

These tests avoid heavyweight model construction and instead exercise the
helper methods and control flow that were recently added to the pipeline.
"""

import importlib
import sys
import types
from contextlib import contextmanager

import numpy as np

@contextmanager
def _temporary_module_stubs():
    """Temporarily stub optional runtime deps needed only for importing pipeline."""
    original_modules = {}
    added_names = []

    def _set_module(name, module):
        if name in sys.modules:
            original_modules[name] = sys.modules[name]
        else:
            added_names.append(name)
        sys.modules[name] = module

    try:
        import cv2  # noqa: F401
    except Exception:
        _set_module("cv2", types.SimpleNamespace())

    try:
        import open_clip  # noqa: F401
    except Exception:
        _set_module("open_clip", types.SimpleNamespace())

    try:
        import ultralytics  # noqa: F401
    except Exception:
        _set_module("ultralytics", types.SimpleNamespace(YOLO=object))

    try:
        from utils import gui as _gui  # noqa: F401
    except Exception:
        _set_module(
            "utils.gui",
            types.SimpleNamespace(show_images_gui=lambda *args, **kwargs: None),
        )

    try:
        yield
    finally:
        for name, module in original_modules.items():
            sys.modules[name] = module
        for name in added_names:
            sys.modules.pop(name, None)


def _import_segment_multiple_frames():
    with _temporary_module_stubs():
        pipeline_module = importlib.import_module("core.pipeline")
    return pipeline_module.SegmentMultipleFrames


def _make_pipeline(matching_cfg=None):
    SegmentMultipleFrames = _import_segment_multiple_frames()
    pipeline = SegmentMultipleFrames.__new__(SegmentMultipleFrames)
    pipeline.assignment_config = {"matching": matching_cfg or {}}
    pipeline.expected_subjects = None
    pipeline._current_sam_video_source = "dummy_source"
    pipeline._log_info = lambda *args, **kwargs: None
    pipeline._log_warn = lambda *args, **kwargs: None
    pipeline._log_error = lambda *args, **kwargs: None
    return pipeline


class DummyFrames:
    def __init__(self, n_frames=30, image_size=(100, 50), cam_name="IOI_09"):
        self._n_frames = n_frames
        self.image_size = image_size
        self.cam_name = cam_name

    def __len__(self):
        return self._n_frames


class TestSam3PromptHelpers:
    def test_candidate_frames_scale_with_length_and_are_capped(self):
        pipeline = _make_pipeline(
            {
                "start_frame": 10,
                "sam3_redetect_samples": 30,
            }
        )
        candidates = pipeline._sam3_prompt_candidate_frames(800)
        assert len(candidates) == 30
        assert candidates[0] == 10
        assert candidates[-1] == 799

    def test_candidate_frames_include_requested_frame(self):
        pipeline = _make_pipeline(
            {
                "start_frame": 10,
                "sam3_redetect_samples": 30,
            }
        )
        candidates = pipeline._sam3_prompt_candidate_frames(100, frame_id=17)
        assert 17 in candidates
        assert candidates == sorted(set(candidates))

    def test_min_mask_area_uses_frame_size(self):
        pipeline = _make_pipeline({"min_mask_area_ratio": 0.01})
        frames = DummyFrames(image_size=(100, 50))
        assert pipeline._sam3_prompt_min_mask_area(frames) == 50

    def test_size_weight_penalizes_small_masks(self):
        pipeline = _make_pipeline({"min_mask_area_ratio": 0.01})
        image_area = 100 * 100
        small = pipeline._sam3_prompt_size_weight(mask_area=50, image_area=image_area)
        medium = pipeline._sam3_prompt_size_weight(mask_area=200, image_area=image_area)
        large = pipeline._sam3_prompt_size_weight(mask_area=2500, image_area=image_area)
        assert 0.0 < small < medium < large <= 1.0

    def test_score_result_caps_single_dominant_mask(self):
        pipeline = _make_pipeline({"min_mask_area_ratio": 0.01})
        result = {
            "obj_ids": [0, 1, 2],
            "masks": {
                0: np.pad(np.ones((20, 20), dtype=np.uint8), ((0, 0), (0, 0))),
                1: np.ones((10, 10), dtype=np.uint8),
                2: np.ones((10, 10), dtype=np.uint8),
            },
        }
        score_info = pipeline._sam3_prompt_score_result(result, min_area=50, expected_subjects=3)
        assert score_info["valid_count"] == 3
        assert score_info["score"][0] > 0


class TestSam3PromptDelegation:
    def test_process_first_video_passes_frames_into_detection(self, tmp_path):
        pipeline = _make_pipeline()
        frames = DummyFrames()
        called = {}

        def fake_detect(cam_name, sam_source, n_frames, frame_id=None, expected_subjects=None, frames=None):
            called["frames"] = frames
            return [7], {0: {7: np.ones((4, 4), dtype=np.uint8)}}, 0

        pipeline._sam3_prompt_detect_and_propagate = fake_detect
        pipeline._sam3_prompt_save_init_overview = lambda *args, **kwargs: None
        pipeline._build_masks_from_remap = lambda *args, **kwargs: {"ok": True}

        result = pipeline.process_first_video_sam3_prompt(
            frames, frame_id=0, output_path=str(tmp_path), expected_subjects=None
        )

        assert called["frames"] is frames
        assert result == {"ok": True}

    def test_process_new_video_passes_frames_into_detection(self, tmp_path):
        pipeline = _make_pipeline()
        frames = DummyFrames()
        called = {}

        def fake_detect(cam_name, sam_source, n_frames, frame_id=None, expected_subjects=None, frames=None):
            called["frames"] = frames
            return [3], {0: {3: np.ones((4, 4), dtype=np.uint8)}}, 0

        pipeline._sam3_prompt_detect_and_propagate = fake_detect
        pipeline._remap_tracklets_to_init = lambda *args, **kwargs: {3: 5}
        pipeline._build_masks_from_remap = lambda *args, **kwargs: {"ok": True}

        result = pipeline.process_new_video_sam3_prompt(
            frames, mask_data={5: {}}, output_path=str(tmp_path), expected_subjects=None
        )

        assert called["frames"] is frames
        assert result == {"ok": True}
