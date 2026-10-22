"""Unit tests for the disk-backed MaskStore.

Verifies the store is a faithful, shape-preserving replacement for the old
in-RAM ``{frame_idx: {obj_id: mask}}`` dict, including the merge/discard
mutations the post-processing relies on.
"""
import numpy as np
import pytest

from core.mask_store import MaskStore, _as_stored, _from_stored


def _rand_mask(h=12, w=16, ndim=2, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 2, size=(h, w)).astype(bool)
    return m[None] if ndim == 3 else m


def test_packbits_roundtrip_preserves_shape_and_values():
    for ndim in (2, 3):
        m = _rand_mask(ndim=ndim, seed=ndim)
        bits, shape = _as_stored(m)
        out = _from_stored(bits, shape)
        assert out.shape == m.shape
        assert out.dtype == bool
        assert np.array_equal(out, m)


def test_set_and_get_frame_roundtrip():
    with MaskStore() as store:
        masks = {0: _rand_mask(seed=1), 3: _rand_mask(ndim=3, seed=2)}
        store.set_frame(5, masks)
        got = store.frame(5)
        assert set(got.keys()) == {0, 3}
        assert np.array_equal(got[0], masks[0])
        assert got[3].shape == masks[3].shape  # (1, H, W) preserved
        assert np.array_equal(got[3], masks[3])


def test_unknown_frame_is_empty():
    with MaskStore() as store:
        assert store.frame(99) == {}
        assert store.frame_obj_ids(99) == []
        assert store.obj_area(99, 0) is None
        assert 99 not in store


def test_views_and_image_size():
    with MaskStore() as store:
        store.set_frame(2, {0: _rand_mask(h=8, w=10, seed=3)})
        store.set_frame(0, {0: _rand_mask(h=8, w=10, seed=4), 1: _rand_mask(h=8, w=10, seed=5)})
        assert store.frames() == [0, 2]
        assert len(store) == 2
        assert store.all_obj_ids() == [0, 1]
        assert store.image_size() == (10, 8)  # (W, H)
        assert list(iter(store)) == [0, 2]


def test_obj_area_matches_sum():
    with MaskStore() as store:
        m = _rand_mask(seed=7)
        store.set_frame(0, {4: m})
        assert store.obj_area(0, 4) == int(m.sum())
        assert store.obj_area(0, 999) is None


def test_discard_obj_removes_everywhere():
    with MaskStore() as store:
        store.set_frame(0, {0: _rand_mask(seed=1), 1: _rand_mask(seed=2)})
        store.set_frame(1, {1: _rand_mask(seed=3)})
        store.discard_obj(1)
        assert store.frame_obj_ids(0) == [0]
        assert store.frame_obj_ids(1) == []
        assert store.all_obj_ids() == [0]


def test_merge_obj_transfers_and_keeps_larger():
    with MaskStore() as store:
        keep_mask = _rand_mask(seed=10)
        drop_only = _rand_mask(seed=11)
        # frame 0: both present -> keep's mask retained, drop removed
        store.set_frame(0, {0: keep_mask, 1: _rand_mask(seed=12)})
        # frame 1: only drop present -> transferred to keep
        store.set_frame(1, {1: drop_only})
        store.merge_obj(drop=1, keep=0)
        f0 = store.frame(0)
        assert set(f0.keys()) == {0}
        assert np.array_equal(f0[0], keep_mask)  # unchanged where both existed
        f1 = store.frame(1)
        assert set(f1.keys()) == {0}
        assert np.array_equal(f1[0], drop_only)  # transferred


def test_overwrite_frame_replaces():
    with MaskStore() as store:
        store.set_frame(0, {0: _rand_mask(seed=1), 1: _rand_mask(seed=2)})
        store.set_frame(0, {0: _rand_mask(seed=3)})  # reverse-pass overwrite
        assert store.frame_obj_ids(0) == [0]


def test_close_is_idempotent_and_cleans_up():
    store = MaskStore()
    store.set_frame(0, {0: _rand_mask()})
    d = store._dir
    import os
    assert os.path.isdir(d)
    store.close()
    assert not os.path.isdir(d)
    store.close()  # idempotent
