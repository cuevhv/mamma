"""Backend for the GUI Exporter tab: SMPL-X export to npz / FBX / ABC / BVH / USD.

Self-contained Flask blueprint-style module (register via ``register_routes(app)``,
mirroring data_readiness.py). Provides:

  GET  /api/exporter/readiness          -> portable Blender + add-on presence
  POST /api/exporter/download-blender   -> run data/download_blender.sh (public)
  POST /api/exporter/download-addon     -> run data/download_smplx_blender_addon.sh (SMPL-X creds)
  GET  /api/exporter/sequences          -> completed ma_3d sequences available to export
  POST /api/exporter/export             -> run optimization/export_blender.py
  GET  /api/exporter/job/<job_id>       -> live job state (downloads + exports)

All long operations run in a daemon thread and report through an in-memory job
record (same shape as data_readiness jobs). Credentials are used once to set the
download script's env and never stored or echoed back.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import jsonify, request

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT = _REPO_ROOT / "output"
_DATA = _REPO_ROOT / "data"

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job(kind: str) -> str:
    jid = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[jid] = {"id": jid, "kind": kind, "state": "running",
                      "log_tail": [], "outputs": [], "error": None,
                      "started_at": time.time()}
    return jid


def _update_job(jid: str, **kw) -> None:
    with _jobs_lock:
        rec = _jobs.get(jid)
        if rec is None:
            return
        tail = kw.pop("log_line", None)
        if tail is not None:
            rec["log_tail"] = (rec["log_tail"] + [tail])[-40:]
        rec.update(kw)


# ---- readiness ----------------------------------------------------------

def _blender_bin() -> str | None:
    env = os.environ.get("MAMMA_BLENDER_BIN")
    if env and os.access(env, os.X_OK):
        return env
    hits = sorted(glob.glob(str(_DATA / "blender" / "blender-*" / "blender")))
    for h in hits:
        if os.access(h, os.X_OK):
            return h
    import shutil
    return shutil.which("blender")


def _addon_present() -> bool:
    return bool(glob.glob(str(_DATA / "blender_addon" / "smplx_blender_addon" / "data" / "*.blend")))


def _readiness() -> dict:
    bin_ = _blender_bin()
    return {
        "blender": {"present": bin_ is not None, "path": bin_ or ""},
        "addon": {"present": _addon_present(),
                  "path": str(_DATA / "blender_addon") if _addon_present() else ""},
    }


# ---- subprocess runners (threads) ---------------------------------------

def _run_stream(jid: str, cmd: list[str], env: dict | None = None) -> int:
    proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip("\n")
        if line.strip():
            _update_job(jid, log_line=line)
    return proc.wait()


def _run_script(jid: str, script: str, creds: dict | None = None) -> None:
    try:
        env = os.environ.copy()
        if creds:
            env["SMPLX_USERNAME"] = creds.get("username", "")
            env["SMPLX_PASSWORD"] = creds.get("password", "")
        rc = _run_stream(jid, ["bash", str(_REPO_ROOT / script)], env=env)
        _update_job(jid, state="ready" if rc == 0 else "error",
                    error=None if rc == 0 else f"{os.path.basename(script)} exited {rc}")
    except Exception as exc:  # noqa: BLE001
        _update_job(jid, state="error", error=str(exc))


def _run_export(jid: str, spec: dict) -> None:
    try:
        out_dir = _OUTPUT / "export" / spec["tag"] / spec["capture"] / spec["seq"]
        cmd = [sys.executable, str(_REPO_ROOT / "optimization" / "export_blender.py"),
               "--ma-3d-dir", spec["ma_3d_dir"], "--seq-name", spec["seq"],
               "--out-dir", str(out_dir), "--formats", ",".join(spec["formats"]),
               "--unit", spec.get("unit", "m"),
               "--blender-format", spec.get("blender_format", "auto")]
        if not spec.get("ground", True):
            cmd.append("--no-ground")
        if spec.get("ma_cap_dir"):
            cmd += ["--ma-cap-dir", spec["ma_cap_dir"]]
        if spec.get("fps"):
            cmd += ["--fps", str(int(spec["fps"]))]
        rc = _run_stream(jid, cmd)
        outs = sorted(str(p) for p in out_dir.glob(f"{spec['seq']}_*")) if out_dir.is_dir() else []
        _update_job(jid, state="ready" if rc == 0 else "error", outputs=outs,
                    error=None if rc == 0 else f"export exited {rc}")
    except Exception as exc:  # noqa: BLE001
        _update_job(jid, state="error", error=str(exc))


# ---- sequence discovery -------------------------------------------------

def _safe_mtime(path: str) -> float:
    """File mtime (epoch seconds) for a "when was this produced" hint; 0 on error."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _exportable_sequences() -> list[dict]:
    """Scan output/ma_3d/<tag>/<capture>/<seq>/ for smplx_params_body_id-*.npz."""
    seqs = []
    for params in sorted(glob.glob(str(_OUTPUT / "ma_3d" / "*" / "*" / "*" / "smplx_params_body_id-*.npz"))):
        seq_dir = Path(params).parent
        seq, capture, tag = seq_dir.name, seq_dir.parent.name, seq_dir.parent.parent.name
        key = (tag, capture, seq)
        existing = next((s for s in seqs if s["_key"] == key), None)
        if existing:
            existing["people"] += 1
            continue
        ma_cap = _OUTPUT / "ma_cap" / tag / capture
        seqs.append({
            "_key": key, "tag": tag, "capture": capture, "seq": seq, "people": 1,
            "ma_3d_dir": str(seq_dir.parent),
            "ma_cap_dir": str(ma_cap) if ma_cap.is_dir() else "",
            "already_exported": (_OUTPUT / "export" / tag / capture / seq).is_dir(),
            "mtime": _safe_mtime(params),  # result write time, for a "when" hint in the UI
        })
    for s in seqs:
        s.pop("_key", None)
    return seqs


def _scan_sequences(root: str):
    """Scan an arbitrary path for exportable sequences (results not under output/).
    Looks for smplx_params_body_id-*.npz at the path itself and up to 3 levels deep.
    Returns (sequences, error). ma_cap is inferred when the path mirrors output/."""
    p = Path(root).expanduser()
    if not p.exists():
        return [], f"path not found: {root}"
    if not p.is_dir():
        return [], f"not a directory: {root}"
    hits = []
    for depth in range(4):  # the path itself, or up to 3 levels deep (tag/capture/seq)
        hits += glob.glob(os.path.join(str(p), *(["*"] * depth), "smplx_params_body_id-*.npz"))
    seqs = []
    for params in sorted(set(hits)):
        seq_dir = Path(params).parent
        ma_3d_dir = str(seq_dir.parent)
        key = (ma_3d_dir, seq_dir.name)
        existing = next((s for s in seqs if s["_key"] == key), None)
        if existing:
            existing["people"] += 1
            continue
        ma_cap = ma_3d_dir.replace(f"{os.sep}ma_3d{os.sep}", f"{os.sep}ma_cap{os.sep}")
        seqs.append({
            "_key": key, "tag": seq_dir.parent.parent.name, "capture": seq_dir.parent.name,
            "seq": seq_dir.name, "people": 1, "ma_3d_dir": ma_3d_dir,
            "ma_cap_dir": ma_cap if (ma_cap != ma_3d_dir and Path(ma_cap).is_dir()) else "",
            "already_exported": False,
            "mtime": _safe_mtime(params),
        })
    for s in seqs:
        s.pop("_key", None)
    return seqs, None


# ---- routes -------------------------------------------------------------

def register_routes(app) -> None:
    @app.get("/api/exporter/readiness")
    def _exporter_readiness():
        return jsonify(_readiness())

    @app.post("/api/exporter/download-blender")
    def _exporter_dl_blender():
        jid = _new_job("blender")
        threading.Thread(target=_run_script, args=(jid, "data/download_blender.sh"),
                         daemon=True).start()
        return jsonify({"job_id": jid}), 201

    @app.post("/api/exporter/download-addon")
    def _exporter_dl_addon():
        body = request.get_json(silent=True) or {}
        creds = {"username": body.get("username", ""), "password": body.get("password", "")}
        jid = _new_job("addon")
        threading.Thread(target=_run_script,
                         args=(jid, "data/download_smplx_blender_addon.sh", creds),
                         daemon=True).start()
        return jsonify({"job_id": jid}), 201

    @app.get("/api/exporter/sequences")
    def _exporter_sequences():
        return jsonify({"sequences": _exportable_sequences()})

    @app.get("/api/exporter/scan")
    def _exporter_scan():
        path = request.args.get("path", "").strip()
        if not path:
            return jsonify({"sequences": [], "error": "no path given"})
        seqs, err = _scan_sequences(path)
        return jsonify({"sequences": seqs, "error": err})

    @app.post("/api/exporter/export")
    def _exporter_export():
        body = request.get_json(silent=True) or {}
        required = ("tag", "capture", "seq", "ma_3d_dir", "formats")
        missing = [k for k in required if not body.get(k)]
        if missing:
            return jsonify({"error": f"missing: {', '.join(missing)}"}), 400
        spec = {k: body.get(k) for k in
                ("tag", "capture", "seq", "ma_3d_dir", "ma_cap_dir",
                 "formats", "unit", "ground", "blender_format", "fps")}
        jid = _new_job("export")
        threading.Thread(target=_run_export, args=(jid, spec), daemon=True).start()
        return jsonify({"job_id": jid}), 201

    @app.get("/api/exporter/job/<job_id>")
    def _exporter_job(job_id):
        with _jobs_lock:
            rec = _jobs.get(job_id)
        if rec is None:
            return jsonify({"error": "unknown job"}), 404
        return jsonify(rec)
