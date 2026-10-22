"""visualization.pipeline._vertex_bounds matches the float64 concat-then-reduce
bounds it replaced, bit for bit. No model weights / GPU needed."""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

pipeline = pytest.importorskip("visualization.pipeline")
from visualization.motion import PersonMotion  # noqa: E402


def _reference(motions):
    allv = np.concatenate(
        [np.asarray(m.vertices, dtype=np.float64).reshape(-1, 3) for m in motions], axis=0)
    return allv.min(0), allv.max(0)


def test_bounds_match_concat_reference():
    rng = np.random.default_rng(0)
    motions = [
        PersonMotion(body_id=1, vertices=(rng.normal(size=(7, 50, 3)) * 3).astype(np.float32)),
        PersonMotion(body_id=2, vertices=rng.normal(loc=5.0, size=(7, 50, 3))),  # float64
        PersonMotion(body_id=3, vertices=np.zeros((0, 50, 3), np.float32)),       # empty
    ]
    lo, hi = pipeline._vertex_bounds(motions)
    ref_lo, ref_hi = _reference(motions)
    assert lo.dtype == hi.dtype == np.float64
    np.testing.assert_array_equal(lo, ref_lo)
    np.testing.assert_array_equal(hi, ref_hi)
