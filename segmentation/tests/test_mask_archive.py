"""Packed per-camera mask archive: round trip, atomicity, relabel, parity."""
import importlib.util
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from core.consensus import Segment, relabel_masks
from core.mask_archive import ARCHIVE_NAME, MaskArchiveReader, MaskArchiveWriter


def _mask(seed):
    rng = np.random.default_rng(seed)
    m = np.zeros((48, 64), np.uint8)
    m[8:40, 10:50] = (rng.random((32, 40)) > 0.5).astype(np.uint8) * 255
    return m


def _name(frame, pid):
    return f"mask_{frame:04d}_{pid:02d}.png"


def _write_archive(mask_dir, frames=range(4), pids=(1, 2), packed=True):
    masks = {}
    with MaskArchiveWriter(mask_dir, packed=packed) as w:
        for f in frames:
            for p in pids:
                masks[(f, p)] = _mask(f * 10 + p)
                w.write(_name(f, p), masks[(f, p)])
    return masks


def test_packed_round_trip(tmp_path):
    d = str(tmp_path / "masks")
    masks = _write_archive(d)
    assert os.path.isfile(os.path.join(d, ARCHIVE_NAME))
    assert not [f for f in os.listdir(d) if f.endswith(".png")]
    r = MaskArchiveReader(d)
    assert r.packed
    assert r.names() == sorted(_name(f, p) for f, p in masks)
    for (f, p), m in masks.items():
        np.testing.assert_array_equal(r.read(_name(f, p)), m)
    assert r.read(_name(99, 1)) is None
    r.close()


def test_threaded_writes_land_intact(tmp_path):
    d = str(tmp_path / "masks")
    with MaskArchiveWriter(d) as w:
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda f: w.write(_name(f, 1), _mask(f)), range(64)))
    r = MaskArchiveReader(d)
    assert len(r.names()) == 64
    np.testing.assert_array_equal(r.read(_name(17, 1)), _mask(17))
    r.close()


def test_abort_publishes_nothing(tmp_path):
    d = str(tmp_path / "masks")
    w = MaskArchiveWriter(d)
    w.write(_name(0, 1), _mask(0))
    w.abort()
    assert not os.path.exists(os.path.join(d, ARCHIVE_NAME))
    assert not [f for f in os.listdir(d) if f.startswith(ARCHIVE_NAME)]
    assert not MaskArchiveReader(d).names()


def test_exception_in_context_aborts(tmp_path):
    d = str(tmp_path / "masks")
    with pytest.raises(RuntimeError):
        with MaskArchiveWriter(d) as w:
            w.write(_name(0, 1), _mask(0))
            raise RuntimeError("boom")
    assert not os.path.exists(os.path.join(d, ARCHIVE_NAME))


def test_loose_mode_and_auto_detect(tmp_path):
    d = str(tmp_path / "masks")
    masks = _write_archive(d, packed=False)
    assert not os.path.exists(os.path.join(d, ARCHIVE_NAME))
    r = MaskArchiveReader(d)
    assert not r.packed
    assert r.names() == sorted(_name(f, p) for f, p in masks)
    np.testing.assert_array_equal(r.read(_name(2, 1)), masks[(2, 1)])
    assert r.read(_name(99, 1)) is None


def test_packed_wins_over_loose(tmp_path):
    d = str(tmp_path / "masks")
    _write_archive(d, packed=False)
    packed = _write_archive(d, frames=range(2), pids=(1,), packed=True)
    r = MaskArchiveReader(d)
    assert r.packed
    assert r.names() == [_name(0, 1), _name(1, 1)]
    np.testing.assert_array_equal(r.read(_name(0, 1)), packed[(0, 1)])
    r.close()


def test_consensus_relabel_packed(tmp_path):
    d = str(tmp_path / "masks")
    masks = _write_archive(d, frames=range(4), pids=(1, 2))
    seg = Segment(start_frame=1, end_frame=2, permutation=(1, 0), margin_px=50.0)
    renamed = relabel_masks(d, [0, 1], seg)
    assert renamed == 4                      # 2 frames x 2 ids swapped
    r = MaskArchiveReader(d)
    for f in (1, 2):                         # swapped inside the segment
        np.testing.assert_array_equal(r.read(_name(f, 1)), masks[(f, 2)])
        np.testing.assert_array_equal(r.read(_name(f, 2)), masks[(f, 1)])
    for f in (0, 3):                         # untouched outside
        np.testing.assert_array_equal(r.read(_name(f, 1)), masks[(f, 1)])
    r.close()
    with zipfile.ZipFile(os.path.join(d, ARCHIVE_NAME)) as z:
        assert len(z.namelist()) == 8        # no entries lost or duplicated


def test_landmarks_reader_copy_parity(tmp_path):
    """The verbatim reader copy in landmarks/ reads what segmentation wrote."""
    d = str(tmp_path / "masks")
    masks = _write_archive(d)
    path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                        "landmarks", "utils", "mask_archive.py")
    spec = importlib.util.spec_from_file_location("lm_mask_archive", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.MaskArchiveReader(d)
    assert r.packed
    assert r.names() == sorted(_name(f, p) for f, p in masks)
    np.testing.assert_array_equal(r.read(_name(3, 2)), masks[(3, 2)])
    r.close()
