"""Unit tests for the 'render all views' overlay controls.

Covers two layers:
* ``visualization.pipeline._resolve_overlay_request`` -- the ``all``/``*``
  sentinel expansion used by the CLI.
* ``inference.steps.ma_vis.MaVisBuilder`` -- forwarding the ``cam_names_overlay``
  (and ``max_preview_cams``) task-config knobs into the visualization argv.

No model weights / GPU needed.
"""
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class _Cam:
    """Minimal stand-in for visualization.cameras.Camera (only ``.name`` is used)."""

    def __init__(self, name):
        self.name = name


# --------------------------------------------------------------------------- #
# Part A -- CLI sentinel expansion
# --------------------------------------------------------------------------- #
try:
    from visualization.pipeline import _resolve_overlay_request
except Exception as exc:  # pragma: no cover - bare env without vis deps
    _resolve_overlay_request = None
    _VIS_IMPORT_ERR = exc


@pytest.mark.skipif(
    _resolve_overlay_request is None,
    reason="visualization runtime deps unavailable",
)
class TestResolveOverlayRequest:
    CAMS = [_Cam("B003"), _Cam("B001"), _Cam("B002")]

    def test_none_means_default(self):
        assert _resolve_overlay_request(None, self.CAMS) is None

    def test_empty_means_default(self):
        assert _resolve_overlay_request([], self.CAMS) is None

    @pytest.mark.parametrize("token", ["all", "ALL", "All", "*", " all "])
    def test_sentinel_expands_to_all_sorted(self, token):
        assert _resolve_overlay_request([token], self.CAMS) == ["B001", "B002", "B003"]

    def test_explicit_list_passthrough_preserves_order(self):
        assert _resolve_overlay_request(["B002", "B001"], self.CAMS) == ["B002", "B001"]

    def test_multi_token_with_all_is_not_a_sentinel(self):
        # More than one token -> treated as an explicit (possibly bogus) name list.
        assert _resolve_overlay_request(["all", "B001"], self.CAMS) == ["all", "B001"]

    def test_camera_literally_named_all_is_kept(self):
        cams = [_Cam("all"), _Cam("B001")]
        assert _resolve_overlay_request(["all"], cams) == ["all"]


# --------------------------------------------------------------------------- #
# Part B -- MaVisBuilder argv forwarding
# --------------------------------------------------------------------------- #
try:
    from inference.steps.ma_vis import MaVisBuilder
except Exception as exc:  # pragma: no cover
    MaVisBuilder = None
    _INF_IMPORT_ERR = exc


def _vals_after(argv, flag):
    """Return the contiguous run of values following ``flag`` in ``argv``."""
    i = argv.index(flag)
    out = []
    for tok in argv[i + 1:]:
        if tok.startswith("-"):
            break
        out.append(tok)
    return out


def _build(step_cfg):
    b = MaVisBuilder({"script": "run_ma_vis.py", **step_cfg}, {"dataset_name": "ds"}, "tag")
    return b.build_argv("seq0")


@pytest.mark.skipif(MaVisBuilder is None, reason="inference package unavailable")
class TestMaVisBuilderOverlay:
    def test_string_all_forwarded(self):
        argv = _build({"cam_names_overlay": "all"})
        assert _vals_after(argv, "--cam-names-overlay") == ["all"]

    def test_list_forwarded(self):
        argv = _build({"cam_names_overlay": ["B001", "B002"]})
        assert _vals_after(argv, "--cam-names-overlay") == ["B001", "B002"]

    def test_absent_knob_emits_no_flag(self):
        argv = _build({})
        assert "--cam-names-overlay" not in argv

    def test_max_preview_cams_forwarded(self):
        argv = _build({"max_preview_cams": 0})
        assert _vals_after(argv, "--max-preview-cams") == ["0"]

    def test_max_preview_cams_absent_emits_no_flag(self):
        argv = _build({})
        assert "--max-preview-cams" not in argv
