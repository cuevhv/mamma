"""Prefetching lazy frame loaders: identical tensors, GPU-free hot path.

Covers the LRU (image-folder) loader's decode-ahead workers and the mp4
stream loader's chunk-ahead staging, in both sweep directions, against the
eager reference. CPU-only.
"""
import os
import time

import numpy as np
import pytest
import torch

from core.predictor_factory import _LazyEvictingFrameLoader, _VideoStreamFrameLoader

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def _fake_load(counter):
    def load(path, size):
        counter[path] = counter.get(path, 0) + 1
        i = int(path)
        img = torch.full((3, size, size), float(i) / 100.0)
        return img, size, size
    return load


def _lru(counter, n=24, window=8):
    return _LazyEvictingFrameLoader(
        [str(i) for i in range(n)], _fake_load(counter), 16, MEAN.clone(),
        STD.clone(), "cpu", offload_video_to_cpu=True, window=window)


def test_lru_prefetch_identical_and_no_redundant_loads():
    counter = {}
    loader = _lru(counter)
    fwd = [loader[i].clone() for i in range(24)]
    for i in range(24):
        expected = (torch.full((3, 16, 16), i / 100.0) - MEAN) / STD
        assert torch.equal(fwd[i], expected)
    # give in-flight prefetch workers a moment, then check load counts:
    # a frame may be decoded once by a worker plus once by a hot-path race,
    # but redundancy must stay bounded (never thrash-reload the clip)
    time.sleep(0.3)
    assert max(counter.values()) <= 3
    assert sum(counter.values()) <= 24 * 2


def test_lru_prefetch_reverse_sweep():
    counter = {}
    loader = _lru(counter)
    rev = [loader[i].clone() for i in range(23, -1, -1)]
    for k, i in enumerate(range(23, -1, -1)):
        expected = (torch.full((3, 16, 16), i / 100.0) - MEAN) / STD
        assert torch.equal(rev[k], expected)


def _write_test_video(path, n=30, size=64):
    import cv2
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
    assert w.isOpened()
    for i in range(n):
        frame = np.full((size, size, 3), i * 8 % 255, dtype=np.uint8)
        w.write(frame)
    w.release()


def test_stream_loader_chunk_prefetch_matches_direct_decode(tmp_path):
    vid = str(tmp_path / "clip.mp4")
    _write_test_video(vid)
    a = _VideoStreamFrameLoader(vid, 32, MEAN.clone(), STD.clone(), "cpu",
                                offload_video_to_cpu=True, chunk=8)
    b = _VideoStreamFrameLoader(vid, 32, MEAN.clone(), STD.clone(), "cpu",
                                offload_video_to_cpu=True, chunk=8)
    assert len(a) == 30
    # forward sweep on a (staging engages after the first chunk switch);
    # random-ish access on b as the no-staging reference
    fwd = [a[i].clone() for i in range(30)]
    for i in range(29, -1, -1):
        assert torch.equal(fwd[i], b[i])
    # reverse sweep re-reads through staging in the other direction
    rev = [a[i].clone() for i in range(29, -1, -1)]
    for k, i in enumerate(range(29, -1, -1)):
        assert torch.equal(rev[k], b[i])
