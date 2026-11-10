"""The Exporter's results scan, streamed one capture folder at a time.

    pytest gui/backend/tests/test_exporter_sequences.py
"""
import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]  # gui/backend
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402
from flask import Flask  # noqa: E402

import exporter  # noqa: E402


def _npz(root: Path, tag: str, capture: str, seq: str, person: int) -> None:
    d = root / "ma_3d" / tag / capture / seq
    d.mkdir(parents=True, exist_ok=True)
    (d / f"smplx_params_body_id-{person}.npz").write_bytes(b"")


@pytest.fixture
def roots(tmp_path, monkeypatch):
    a, b = tmp_path / "local", tmp_path / "cluster"
    _npz(a, "t", "cap1", "s1", 0)
    _npz(a, "t", "cap1", "s1", 1)
    _npz(a, "t", "cap2", "s2", 0)
    _npz(b, "t", "cap1", "s1", 0)      # same sequence as the local root
    _npz(b, "u", "cap3", "s3", 0)
    monkeypatch.setattr(exporter, "_OUTPUT", a)
    monkeypatch.setattr(exporter, "_output_roots", lambda: [a, b])
    return a, b


def _expected(a: Path, b: Path) -> list:
    return [
        ("scanning", str(a / "ma_3d/t/cap1")), ("sequences", [("t", "cap1", "s1", 2)]),
        ("scanning", str(a / "ma_3d/t/cap2")), ("sequences", [("t", "cap2", "s2", 1)]),
        ("scanning", str(b / "ma_3d/t/cap1")),  # hidden by the earlier root: no sequences event
        ("scanning", str(b / "ma_3d/u/cap3")), ("sequences", [("u", "cap3", "s3", 1)]),
    ]


def _summary(events) -> list:
    out = []
    for ev in events:
        if "scanning" in ev:
            out.append(("scanning", ev["scanning"]))
        else:
            out.append(("sequences", [(s["tag"], s["capture"], s["seq"], s["people"]) for s in ev["sequences"]]))
    return out


def test_scan_yields_each_capture_in_order(roots):
    assert _summary(exporter._iter_exportable()) == _expected(*roots)


def test_flat_list_matches_the_stream(roots):
    flat = [(s["tag"], s["capture"], s["seq"], s["people"]) for s in exporter._exportable_sequences()]
    assert flat == [("t", "cap1", "s1", 2), ("t", "cap2", "s2", 1), ("u", "cap3", "s3", 1)]


def test_stream_route_sends_ndjson(roots):
    app = Flask(__name__)
    exporter.register_routes(app)
    r = app.test_client().get("/api/exporter/sequences/stream")
    assert r.status_code == 200 and r.mimetype == "application/x-ndjson"
    events = [json.loads(line) for line in r.get_data(as_text=True).splitlines()]
    assert _summary(events) == _expected(*roots)
