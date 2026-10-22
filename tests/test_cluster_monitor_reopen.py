"""ClusterMonitor: a non-terminal record after a terminal one reopens the cell.

DAGMan RETRY re-runs a failed node, and camera fan-out runs several nodes for
one (step, seq). Either way a later Running must make the cell eligible again
for the queue view and for finalize_missing, or a silently killed attempt
stays Running on the board forever.
"""
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from inference.cluster.monitor import ClusterMonitor  # noqa: E402
from inference.cluster.transport import CommandResult  # noqa: E402
from inference.status import StatusSink  # noqa: E402


class _Transport:
    def __init__(self):
        self.text, self.queue = "", "[]"

    def tail(self, _path, off):
        return self.text[off:], len(self.text)

    def run(self, _cmd):
        return CommandResult(0, self.queue)

    def emit(self, step, status, ts):
        self.text += json.dumps({"ts": ts, "step": step, "seq": "s", "status": status}) + "\n"


class _Sink(StatusSink):
    def __init__(self):
        self.last = {}

    def update(self, step, seq, status, pid=None):
        self.last[step] = status


def _monitor():
    tr, sink = _Transport(), _Sink()
    nodes = {"ma_2d_0": "s", "ma_masks_init_0": "s", "ma_masks_IOI_09_0": "s"}
    return tr, sink, ClusterMonitor(tr, "1", "/x/status.jsonl", nodes, sink)


def test_killed_retry_is_finalized():
    tr, sink, m = _monitor()
    for ts, status in enumerate(["Running", "Failed", "Running"]):
        tr.emit("ma_2d", status, ts)
    m.tick()
    assert m.finalize_missing() == 1
    assert sink.last["ma_2d"] == "Cancelled"


def test_held_retry_uses_queue_view():
    tr, sink, m = _monitor()
    for ts, status in enumerate(["Running", "Failed", "Running"]):
        tr.emit("ma_2d", status, ts)
    tr.queue = json.dumps([{"DAGNodeName": "ma_2d_0", "JobStatus": 5, "HoldReason": "oom"}])
    m.tick()
    assert sink.last["ma_2d"] == "Failed"


def test_fanout_cell_after_init_is_finalized():
    tr, sink, m = _monitor()
    for ts, status in enumerate(["Running", "Done", "Running"]):  # init cell, then a camera cell
        tr.emit("ma_masks", status, ts)
    m.tick()
    assert m.finalize_missing() == 1
    assert sink.last["ma_masks"] == "Cancelled"


def test_finished_cell_is_not_downgraded():
    tr, sink, m = _monitor()
    for ts, status in enumerate(["Running", "Failed", "Running", "Done"]):
        tr.emit("ma_2d", status, ts)
    m.tick()
    assert m.finalize_missing() == 0
    assert sink.last["ma_2d"] == "Done"
