"""Cluster camera fan-out leaves per-cell DONE sentinels and logs; the GUI
must still see the step as done and open its log."""
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]  # gui/backend
for p in (_BACKEND, _BACKEND.parents[1]):  # sync, and the repo root for inference
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import sync  # noqa: E402


def test_done_scan_accepts_completed_fanout_cells(tmp_path):
    cfg = {"global": {"cam_names": ["A", "B"]}, "ma_2d": {"fanout_cameras": True}}
    for step, cells in (("ma_cap", ["DONE"]), ("ma_masks", ["DONE.init", "DONE.A", "DONE.finalize"]),
                        ("ma_2d", ["DONE.A", "DONE.B"]), ("ma_3d", ["DONE.A"])):
        d = tmp_path / step / "7" / "cap" / "s1"
        d.mkdir(parents=True)
        for c in cells:
            (d / c).touch()
    steps = ["ma_cap", "ma_masks", "ma_2d", "ma_3d"]
    done = sync.enumerate_done_sentinels(str(tmp_path), "7", "cap", ["s1"], steps, task_cfg=cfg)
    assert done == {"s1": ["ma_cap", "ma_masks", "ma_2d"]}
    # without the run config the ma_2d camera list is unknown
    assert sync.enumerate_done_sentinels(str(tmp_path), "7", "cap", ["s1"], steps) == \
        {"s1": ["ma_cap", "ma_masks"]}


def test_fanout_cell_logs_joins_cells_in_dag_order(tmp_path):
    for cell in ("finalize", "IOI_02", "init", "IOI_01"):
        (tmp_path / f"s1.{cell}.out").write_text("\n".join(f"{cell} line {i}" for i in range(100)))
    (tmp_path / "s1.IOI_01.err").write_text("other stream")
    body = sync.fanout_cell_logs(str(tmp_path / "s1.out"), tail_lines=2)
    order = [body.index(f"===== {c}:") for c in ("init", "IOI_01", "IOI_02", "finalize")]
    assert order == sorted(order)
    assert "init line 99" in body and "init line 97" not in body
    assert "other stream" not in body
    assert sync.fanout_cell_logs(str(tmp_path / "s2.out")) is None
