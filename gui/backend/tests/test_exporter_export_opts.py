"""Regression guard for option defaults on ``POST /api/exporter/export``.

The route collects its shared options into a dict and hands them to
``_export_one``, which reads each one as ``spec.get(key, default)``. If the
route puts a key in the dict with a ``None`` value when the caller omitted
it, every one of those defaults is silently defeated:

  * ``--unit`` and ``--blender-format`` receive ``None``, and subprocess
    rejects a ``None`` in argv with a ``TypeError`` that names no field;
  * ``ground`` inverts, because ``not None`` is True, so an omitted field
    appends ``--no-ground`` and the export comes out ungrounded with no
    error and a zero exit code.

The third is the dangerous one: wrong geometry, no signal. ``--ground``
defaults to True in ``optimization/export_blender.py``, so this is a real
behaviour change, not a no-op.

The GUI always sends every field (ExportPanel.tsx), so none of this is
reachable from the browser. It only bites API and script callers, which is
why it can regress unnoticed. These tests drive the real Flask route and
assert on the argv that reaches the subprocess.

Runs in any env, including one without the pipeline installed::

    pytest gui/backend/tests/test_exporter_export_opts.py
"""
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]  # gui/backend
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402
from flask import Flask  # noqa: E402

import exporter  # noqa: E402

#: Minimal valid body. Everything the route validates as required, and
#: nothing else, so each test adds back exactly the field it is about.
BASE = {"tag": "t", "capture": "c", "seq": "s", "ma_3d_dir": "/d", "formats": ["npz"]}

#: What ExportPanel.tsx sends today.
UI_FIELDS = {"unit": "m", "ground": True, "blender_format": "auto"}


class _InlineThread:
    """Stand-in for threading.Thread that runs the target synchronously.

    The route exports on a background thread; running it inline keeps the
    assertions deterministic without sleeping or joining.
    """

    def __init__(self, target=None, args=(), **_kwargs):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)


@pytest.fixture
def export_argv(monkeypatch):
    """Return `post(**body) -> argv`, capturing the command the route builds."""
    captured: list[list[str]] = []

    def _fake_run_stream(_jid, cmd, env=None):
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(exporter, "_run_stream", _fake_run_stream)
    monkeypatch.setattr(exporter.threading, "Thread", _InlineThread)

    app = Flask(__name__)
    exporter.register_routes(app)
    client = app.test_client()

    def post(**overrides):
        resp = client.post("/api/exporter/export", json=dict(BASE, **overrides))
        assert resp.status_code == 201, resp.get_json()
        assert captured, "export never reached the subprocess"
        return captured[-1]

    return post


def _flag_value(argv: list[str], flag: str) -> str | None:
    """Value following `flag` in argv, or None if the flag is absent."""
    return argv[argv.index(flag) + 1] if flag in argv else None


# --- the silent one -------------------------------------------------------

def test_omitted_ground_keeps_grounding_on(export_argv):
    """Omitting `ground` must NOT disable grounding.

    This is the regression that produces wrong geometry with a zero exit
    code, so it gets its own test rather than riding along with the argv
    sanity check below.
    """
    argv = export_argv(unit="m", blender_format="auto")  # ground omitted
    assert "--no-ground" not in argv, (
        "omitting `ground` disabled grounding; the documented default is on"
    )


def test_explicit_ground_false_still_disables(export_argv):
    """The opt-out must survive the fix. A truthiness filter would eat False."""
    argv = export_argv(unit="m", blender_format="auto", ground=False)
    assert "--no-ground" in argv


def test_explicit_ground_true_keeps_grounding(export_argv):
    argv = export_argv(**UI_FIELDS)
    assert "--no-ground" not in argv


# --- the two crashes ------------------------------------------------------

@pytest.mark.parametrize("omitted", ["unit", "blender_format", "fps", "output_dir", "ground"])
def test_no_none_reaches_argv(export_argv, omitted):
    """A None in argv makes subprocess raise a TypeError naming no field."""
    body = {**UI_FIELDS, "fps": 30, "output_dir": "/tmp/x"}
    body.pop(omitted)
    argv = export_argv(**body)
    assert all(a is not None for a in argv), f"None in argv when {omitted!r} omitted: {argv}"


def test_omitted_unit_falls_back_to_metres(export_argv):
    assert _flag_value(export_argv(blender_format="auto", ground=True), "--unit") == "m"


def test_omitted_blender_format_falls_back_to_auto(export_argv):
    assert _flag_value(export_argv(unit="m", ground=True), "--blender-format") == "auto"


# --- no behaviour change for the caller that already worked ---------------

def test_ui_payload_is_unchanged(export_argv):
    """The browser's payload must produce exactly what it produced before."""
    argv = export_argv(**UI_FIELDS)
    assert _flag_value(argv, "--unit") == "m"
    assert _flag_value(argv, "--blender-format") == "auto"
    assert _flag_value(argv, "--formats") == "npz"
    assert "--no-ground" not in argv


def test_explicit_values_still_pass_through(export_argv):
    argv = export_argv(unit="cm", blender_format="smplx", ground=True, fps=60)
    assert _flag_value(argv, "--unit") == "cm"
    assert _flag_value(argv, "--blender-format") == "smplx"
    assert _flag_value(argv, "--fps") == "60"


def test_missing_required_field_still_400s(export_argv, monkeypatch):
    """The fix must not weaken validation: formats is still required."""
    app = Flask(__name__)
    exporter.register_routes(app)
    body = dict(BASE)
    body.pop("formats")
    resp = app.test_client().post("/api/exporter/export", json=body)
    assert resp.status_code == 400
    assert "formats" in resp.get_json()["error"]
