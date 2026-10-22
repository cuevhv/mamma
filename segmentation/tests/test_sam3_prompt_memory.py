"""sam3_prompt memory bounds in predictor_factory: the host-resident img_batch
patch (``_patch_sam3_prompt_host_batch`` / ``_HostImageBatchView``), its gating
in ``start_session``, and the session output-cache prune in ``propagate``.

SAM3 modules and the session predictor are stubbed; the "meta" device stands in
for a GPU so device moves are observable on CPU. No weights are loaded.
"""
import dataclasses
import sys
import types

import numpy as np
import pytest
import torch

from core import predictor_factory as pf


@dataclasses.dataclass
class _Batch:
    img_batch: object
    find_text_batch: list
    targets: object = None


@pytest.fixture
def svi(monkeypatch):
    """Stub ``sam3.model.sam3_video_inference`` whose copy_data_to_device
    records every object it is asked to move."""
    mod = types.ModuleType("sam3.model.sam3_video_inference")
    mod.calls = []

    def copy_data_to_device(data, device, *args, **kwargs):
        mod.calls.append((data, kwargs))
        return data.to(device) if torch.is_tensor(data) else data

    mod.copy_data_to_device = copy_data_to_device
    pkg = types.ModuleType("sam3.model")
    pkg.sam3_video_inference = mod
    root = types.ModuleType("sam3")
    root.model = pkg
    for name, m in (("sam3", root), ("sam3.model", pkg),
                    ("sam3.model.sam3_video_inference", mod)):
        monkeypatch.setitem(sys.modules, name, m)
    return mod


def test_patch_keeps_img_batch_on_host_and_streams_slices(svi):
    pf._patch_sam3_prompt_host_batch()
    video = torch.arange(5 * 3 * 4 * 4, dtype=torch.float32).reshape(5, 3, 4, 4)
    targets = torch.ones(2)
    out = svi.copy_data_to_device(_Batch(video, ["person"], targets), "meta", non_blocking=True)

    assert isinstance(out, _Batch)
    view = out.img_batch
    assert isinstance(view, pf._HostImageBatchView)
    assert view._images.device.type == "cpu" and torch.equal(view._images, video)
    assert all(data is not video for data, _ in svi.calls)  # never bulk-copied
    assert out.targets.device.type == "meta"                # other fields still move
    assert all(kw == {"non_blocking": True} for _, kw in svi.calls)
    assert out.find_text_batch == ["person"]

    sl = view[torch.tensor([1, 3])]
    assert sl.device.type == "meta" and sl.shape == (2, 3, 4, 4)
    assert len(view) == 5 and view.shape == video.shape and view.dtype == video.dtype
    assert view.to("meta") is view


def test_host_view_slices_match_source():
    video = torch.rand(4, 3, 2, 2)
    view = pf._HostImageBatchView(video, "cpu")
    assert torch.equal(view[[2, 0]], video[[2, 0]])
    assert torch.equal(view[torch.tensor([3])], video[[3]])


@pytest.mark.parametrize("case", ["cpu_target", "not_4d", "already_on_device", "no_img_batch"])
def test_patch_passes_other_inputs_to_original(svi, case):
    pf._patch_sam3_prompt_host_batch()
    img = {
        "cpu_target": torch.zeros(2, 3, 4, 4),
        "not_4d": torch.zeros(3, 4, 4),
        "already_on_device": torch.zeros(2, 3, 4, 4, device="meta"),
        "no_img_batch": None,
    }[case]
    data = _Batch(img, [])
    device = "cpu" if case == "cpu_target" else "meta"
    out = svi.copy_data_to_device(data, device)
    assert out is data
    assert len(svi.calls) == 1 and svi.calls[0][0] is data


def test_patch_is_idempotent(svi):
    pf._patch_sam3_prompt_host_batch()
    patched = svi.copy_data_to_device
    pf._patch_sam3_prompt_host_batch()
    assert svi.copy_data_to_device is patched and svi._mamma_host_img_batch


def test_patch_is_a_noop_without_sam3(monkeypatch):
    monkeypatch.setitem(sys.modules, "sam3.model", None)
    assert pf._patch_sam3_prompt_host_batch() is None


class _StubSession:
    """Session predictor stub: a fixed bidirectional sweep that caches every
    emitted frame, like SAM3's ``cached_frame_outputs``."""

    def __init__(self, order):
        self.order = order
        self.requests = []
        self.cache = {}
        self.peak = 0

    def handle_request(self, request):
        self.requests.append(request)
        return {"session_id": "s0"}

    def _get_session(self, _sid):
        return {"state": {"orig_height": 8, "orig_width": 10,
                          "cached_frame_outputs": self.cache}}

    def handle_stream_request(self, request):
        for f in self.order:
            self.cache[f] = object()
            self.peak = max(self.peak, len(self.cache))
            mask = torch.zeros(1, 8, 10, dtype=torch.bool)
            mask[0, 2:4, 3:6] = True
            yield {"frame_index": f, "outputs": {
                "out_obj_ids": torch.tensor([7]), "out_binary_masks": mask,
                "out_boxes_xywh": torch.tensor([[0.3, 0.25, 0.3, 0.25]])}}


def _adapter(session):
    ad = pf._Sam3PromptVideoAdapter.__new__(pf._Sam3PromptVideoAdapter)
    ad._predictor = session
    ad._session_id = None
    ad._video_width = ad._video_height = None
    return ad


@pytest.mark.parametrize("offload", [False, True])
def test_start_session_patches_only_when_offloading(monkeypatch, offload):
    calls = []
    monkeypatch.setattr(pf, "_patch_sam3_prompt_host_batch", lambda: calls.append(1))
    session = _StubSession([])
    ad = _adapter(session)
    assert ad.start_session("video.mp4", offload_video_to_cpu=offload) == "s0"
    assert calls == ([1] if offload else [])
    assert session.requests[0]["offload_video_to_cpu"] is offload
    assert (ad.video_width, ad.video_height) == (10, 8)


SWEEP = [5, 6, 7, 8, 9, 4, 3, 2, 1, 0]  # forward from the prompt frame, then reverse


def test_propagate_prunes_cache_to_window_and_keeps_prompt_frame():
    session = _StubSession(SWEEP)
    ad = _adapter(session)
    ad.start_session("video.mp4")
    got = {}
    out = ad.propagate(sink=lambda f, masks: got.__setitem__(f, masks),
                       prune_cache_window=2, keep_frames=(5,))
    assert out is None
    assert sorted(got) == list(range(10))
    assert isinstance(got[0][7], np.ndarray) and got[0][7].sum() == 6
    # Window [f-2, f+2] around the last emitted frame (0), plus the prompt frame.
    assert sorted(session.cache) == [0, 1, 2, 5]
    assert session.peak <= 2 * 2 + 3  # (2W+1) window + prompt frame + incoming; unpruned: 10


def test_propagate_without_window_keeps_cache_and_returns_sorted_dict():
    session = _StubSession(SWEEP)
    ad = _adapter(session)
    ad.start_session("video.mp4")
    out = ad.propagate()
    assert list(out) == list(range(10))
    assert len(session.cache) == 10


def test_parse_outputs_reads_top_left_xywh_boxes():
    """SAM3 reports normalized top-left xywh boxes; they convert to absolute xyxy."""
    import types

    import torch

    from core.predictor_factory import _Sam3PromptVideoAdapter

    adapter = types.SimpleNamespace(_video_width=200, _video_height=100)
    out = _Sam3PromptVideoAdapter._parse_outputs(adapter, {
        "out_obj_ids": torch.tensor([7]),
        "out_binary_masks": torch.zeros(1, 100, 200, dtype=torch.bool),
        "out_boxes_xywh": torch.tensor([[0.1, 0.2, 0.5, 0.4]]),
    })
    np.testing.assert_allclose(out["boxes"][7], [20.0, 20.0, 120.0, 60.0], atol=1e-4)
