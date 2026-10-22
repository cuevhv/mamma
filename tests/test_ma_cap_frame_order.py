"""ma_cap image-frame order matches the SAM frame loader.

The SAM video predictor (segmentation/core/predictor_factory.py) sorts frame
files by int(stem) when every stem is numeric and lexicographically otherwise.
ma_cap's per-camera list must use the same order, or ma_masks frame i and SAM
frame i point at different images for unpadded names (1, 2, 10).
No model weights / GPU needed.
"""
import os
import random
import sys
from pathlib import Path

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

run_ma_cap = pytest.importorskip("capture.run_ma_cap")


def _sam_order(names):
    """Mirror of the sort in predictor_factory's patched frame loader."""
    names = list(names)
    try:
        names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        names.sort()
    return names


def _gather(tmp_path, names):
    for n in names:
        (tmp_path / n).write_bytes(b"")
    return [os.path.basename(p) for p in run_ma_cap._gather_images(Path(tmp_path))]


@pytest.mark.parametrize("names, expected", [
    (["10.jpg", "2.jpg", "1.jpg"], ["1.jpg", "2.jpg", "10.jpg"]),
    (["0010.jpg", "0002.jpg", "0001.jpg"], ["0001.jpg", "0002.jpg", "0010.jpg"]),
    (["frame_10.jpg", "frame_2.jpg", "frame_1.jpg"], ["frame_1.jpg", "frame_10.jpg", "frame_2.jpg"]),
    (["10.jpg", "2.jpg", "a.jpg"], ["10.jpg", "2.jpg", "a.jpg"]),
])
def test_order_matches_sam_loader(tmp_path, names, expected):
    got = _gather(tmp_path, names)
    assert got == expected
    assert got == _sam_order(names)


def test_zero_padded_order_unchanged(tmp_path):
    names = [f"{i:06d}.png" for i in random.Random(0).sample(range(100000), 300)]
    assert _gather(tmp_path, names) == sorted(names)
