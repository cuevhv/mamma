"""Integration tests for the streaming mask pipeline.

These exercise the real pipeline methods against a disk-backed MaskStore without
loading SAM/detector/CLIP (the segmenter is built via ``__new__`` with only the
attributes the methods touch). They lock the externally-observable contract:

  * ``save_images_from_video`` writes per-frame mask PNGs with the exact names
    and pixel content downstream ``landmarks/run_ma_2d.py`` reads, while indexing
    SAM's frame tensor one frame at a time (no whole-video copy).
  * ``_postprocess_tracklets`` merges duplicate tracklets and discards tiny ones
    with identical results when operating on the store.
"""
import os

import matplotlib
matplotlib.use("Agg")  # headless render for the crop-summary test
import numpy as np
import torch

from core.mask_store import MaskStore
from core.pipeline import SegmentMultipleFrames


def _make_segmenter(assignment_config):
    """A SegmentMultipleFrames with just the attributes the tested methods use."""
    seg = SegmentMultipleFrames.__new__(SegmentMultipleFrames)
    seg.img_mean_sam = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    seg.img_std_sam = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    seg.assignment_config = assignment_config
    seg._log_info = lambda *a, **k: None
    seg._log_warn = lambda *a, **k: None
    seg._log_error = lambda *a, **k: None
    return seg


def _crop_mask_data(h=20, w=30):
    img = np.random.default_rng(0).integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), bool); mask[5:15, 8:20] = True
    bbox = np.array([8, 5, 20, 15], dtype=np.float32)
    return {0: {"img": [img], "mask": [mask], "frame": [3], "bbox": [bbox], "iou": [1.0]}}


def test_crop_summary_builds_img_bbx_regardless_of_viz(tmp_path):
    """img_bbx (feeds cross-camera matching) is built whether or not the debug
    viz PNG is rendered, and is byte-identical between the two modes."""
    seg = _make_segmenter({"exports": {}})

    d_off = _crop_mask_data(); out_off = tmp_path / "off"; out_off.mkdir()
    seg.save_picked_masks(d_off, str(out_off), render_viz=False)
    assert len(d_off[0]["img_bbx"]) == 1
    assert list(out_off.glob("*.png")) == []          # no viz written

    d_on = _crop_mask_data(); out_on = tmp_path / "on"; out_on.mkdir()
    seg.save_picked_masks(d_on, str(out_on), render_viz=True)
    assert len(d_on[0]["img_bbx"]) == 1
    assert (out_on / "person_00_crop_summary.png").exists()   # viz written

    assert np.array_equal(d_off[0]["img_bbx"][0], d_on[0]["img_bbx"][0])  # matching input identical


def test_compute_centroids_matches_mask_means():
    seg = _make_segmenter({})
    m = np.zeros((10, 10), bool); m[2:6, 4:8] = True          # centroid ~ (5.5, 3.5)
    d = {0: {"mask": [m], "frame": [3]}}
    seg._compute_centroids(d)
    cx, cy = d[0]["centroid_xy"][0]
    ys, xs = np.where(m)
    assert abs(cx - xs.mean()) < 1e-6 and abs(cy - ys.mean()) < 1e-6


def test_slim_mask_record_drops_heavy_keeps_light():
    seg = _make_segmenter({})
    d = {0: {"img": [np.zeros((4, 4, 3), np.uint8)], "mask": [np.zeros((4, 4), bool)],
             "img_bbx": [np.zeros((2, 2, 3), np.uint8)], "features": [1],
             "frame": [0], "bbox": [np.zeros(4)], "centroid_xy": [[1.0, 2.0]]}}
    seg._slim_mask_record(d)
    assert "img" not in d[0] and "mask" not in d[0]            # heavy fields freed
    for fld in ("img_bbx", "features", "frame", "bbox", "centroid_xy"):
        assert fld in d[0]                                     # matching fields retained


def test_reference_point_prefers_centroid_then_mask():
    seg = _make_segmenter({})
    # slim record: centroid_xy present, no mask
    slim = {"frame": [0, 10], "centroid_xy": [[1.0, 2.0], [3.0, 4.0]]}
    p = seg._reference_point_from_mask_data(slim, 9)           # nearest -> idx 1
    assert list(p[:2]) == [3.0, 4.0]
    # legacy record: no centroid, falls back to mask centroid
    m = np.zeros((8, 8), bool); m[0:4, 0:2] = True
    legacy = {"frame": [0], "mask": [m]}
    p2 = seg._reference_point_from_mask_data(legacy, 0)
    ys, xs = np.where(m)
    assert abs(p2[0] - xs.mean()) < 1e-6 and abs(p2[1] - ys.mean()) < 1e-6


def test_compute_clip_features_skips_when_present():
    import torch as _t
    seg = _make_segmenter({})
    feats = _t.ones((3, 8))
    d = {0: {"features": feats, "img_bbx": []}}               # already has features
    seg.compute_clip_features(d)                               # must not overwrite/clear
    assert d[0]["features"] is feats


def test_debug_flags():
    assert _make_segmenter({"exports": {"debug_crop_summary": True}})._debug_crop_summary() is True
    assert _make_segmenter({"exports": {}})._debug_crop_summary() is False
    assert _make_segmenter({"exports": {"debug_full_masks_npy": True}})._debug_full_masks_npy() is True
    assert _make_segmenter({})._debug_full_masks_npy() is False


def test_finalize_writes_slim_npy_by_default(tmp_path):
    seg = _make_segmenter({"exports": {}})  # full-dump off -> slim
    m = np.zeros((10, 10), bool); m[1:5, 1:5] = True
    d = {0: {"img": [np.zeros((10, 10, 3), np.uint8)], "mask": [m],
             "img_bbx": [np.zeros((3, 3, 3), np.uint8)], "features": [1],
             "frame": [0], "bbox": [np.zeros(4)]}}
    out = tmp_path / "cam"; out.mkdir()
    seg._finalize_mask_cache(d, str(out))
    assert "img" not in d[0] and "mask" not in d[0] and "centroid_xy" in d[0]
    rec = np.load(str(out / "masks.npy"), allow_pickle=True)[()]
    assert "img" not in rec[0] and "mask" not in rec[0]
    for fld in ("centroid_xy", "img_bbx", "features", "frame", "bbox"):
        assert fld in rec[0]


def test_finalize_full_dump_keeps_heavy(tmp_path):
    seg = _make_segmenter({"exports": {"debug_full_masks_npy": True}})
    d = {0: {"img": [np.zeros((4, 4, 3), np.uint8)], "mask": [np.zeros((4, 4), bool)],
             "frame": [0]}}
    out = tmp_path / "c"; out.mkdir()
    seg._finalize_mask_cache(d, str(out))
    assert "img" in d[0] and "mask" in d[0]        # heavy kept
    assert "centroid_xy" in d[0]                    # centroid still computed


def test_save_images_writes_expected_mask_pngs(tmp_path):
    seg = _make_segmenter({"exports": {"skip_masked_outputs": True}})
    # bbox pruning is a no-op stub for this test (operates on sampled save_masks).
    seg.get_frames_from_far_bbx = lambda obj_ids, save_masks, fidx: save_masks

    n_frames, sam_size = 4, 32
    w_orig, h_orig = 40, 24
    inference_state = {
        "images": torch.rand(n_frames, 3, sam_size, sam_size),
        "video_width": w_orig,
        "video_height": h_orig,
    }

    store = MaskStore()
    rng = np.random.default_rng(0)
    expected = {}
    for f in range(n_frames):
        masks = {}
        for oid in (0, 1):
            m = rng.integers(0, 2, size=(h_orig, w_orig)).astype(bool)
            masks[oid] = m
            expected[(f, oid)] = m
        store.set_frame(f, masks)

    out = tmp_path / "cam"
    seg.save_images_from_video(inference_state, store, [0, 1], str(out), vis_frame_stride=2)

    # Every mask entry exists with the exact name + content landmarks expects,
    # in the packed masks.zip written by default (the reader handles either layout).
    from core.mask_archive import MaskArchiveReader
    assert (out / "masks" / "masks.zip").is_file()
    with MaskArchiveReader(str(out / "masks")) as reader:
        for (f, oid), m in expected.items():
            name = f"mask_{f:04d}_{oid + 1:02d}.png"
            got = reader.read(name)
            assert got is not None, f"missing {name}"
            assert got.shape == (h_orig, w_orig)
            assert np.array_equal(got > 0, m)

    # The store is closed/cleaned up by save_images_from_video.
    assert not os.path.isdir(store._dir)


def test_postprocess_merges_duplicates_and_discards_tiny():
    seg = _make_segmenter({"masks": {
        "merge_duplicate_tracklets": True,
        "merge_iou_threshold": 0.5,
        "discard_tiny_tracklets": True,
        "tiny_tracklet_min_area_ratio": 0.1,
        "tiny_tracklet_min_frame_ratio": 0.5,
    }})

    h, w = 10, 10
    big = np.zeros((h, w), bool); big[:8, :8] = True   # large mask, id 0
    dup = big.copy()                                   # identical -> merges into id 0
    tiny = np.zeros((h, w), bool); tiny[0, 0] = True   # 1px -> discarded

    store = MaskStore()
    for f in range(4):
        store.set_frame(f, {0: big, 1: dup, 2: tiny})

    store_out, ids = seg._postprocess_tracklets("cam", store, [0, 1, 2], image_size=(w, h))

    assert ids == [0]                       # 1 merged into 0, 2 discarded
    assert store_out.all_obj_ids() == [0]
    # The surviving id keeps a full mask on every frame.
    for f in range(4):
        assert np.array_equal(store_out.frame(f)[0], big)
    store_out.close()


def test_lazy_frame_loading_defaults_to_auto_when_key_missing():
    """A config without ``sam.lazy_frame_loading`` gets the documented "auto":
    eager below ``lazy_min_frames``, lazy from there up."""
    import core.predictor_factory as pf

    class _Predictor:
        def init_state(self, video_path, **kwargs):
            return "state"

    seg = _make_segmenter({"sam": {"propagate_reverse": True}})
    seg.predictor = _Predictor()
    try:
        for n_frames, expected in ((100, False), (1500, True)):
            names = [f"{i:05d}.jpg" for i in range(n_frames)]
            state, _ = seg._init_state_with_fallback("frames_dir", names, "out", "cam")
            assert state == "state"
            assert pf._lazy_frame_loading is expected
    finally:
        pf.set_lazy_frame_loading(False)
