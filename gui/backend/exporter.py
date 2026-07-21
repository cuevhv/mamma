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
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import jsonify, request, send_file

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
                      "progress": None, "results": [], "started_at": time.time()}
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


# The SMPL-X add-on requires Blender >= 4.5.0 (verified against the portable 4.5
# LTS); newer releases are fine. We surface the detected version so a fallback
# system Blender that's too old (< 4.5) — the only problematic case — is flagged.
_BLENDER_MIN = (4, 5, 0)
_VERSION_CACHE: dict[str, tuple | None] = {}


def _blender_version(bin_path: str | None) -> tuple | None:
    """(major, minor, patch) for a Blender binary, or None. The portable download
    encodes the version in its path (free); a system binary is queried once via
    --version and cached (snap can be slow to start)."""
    if not bin_path:
        return None
    if bin_path in _VERSION_CACHE:
        return _VERSION_CACHE[bin_path]
    ver = None
    m = re.search(r"blender-(\d+)\.(\d+)\.(\d+)", bin_path)
    if m:
        ver = tuple(int(x) for x in m.groups())
    else:
        try:
            out = subprocess.run([bin_path, "--version"], capture_output=True,
                                 text=True, timeout=20).stdout
            mm = re.search(r"Blender\s+(\d+)\.(\d+)\.(\d+)", out)
            if mm:
                ver = tuple(int(x) for x in mm.groups())
        except Exception:  # noqa: BLE001 — missing/slow/odd binary -> unknown
            ver = None
    _VERSION_CACHE[bin_path] = ver
    return ver


def _blender_compat(ver: tuple | None) -> str:
    """'ok' (>= 4.5.0, newer is fine) | 'too_old' (< 4.5) | 'unknown' (undetected)."""
    if not ver:
        return "unknown"
    if ver < _BLENDER_MIN:
        return "too_old"
    return "ok"


def _readiness() -> dict:
    bin_ = _blender_bin()
    ver = _blender_version(bin_)
    return {
        "blender": {
            "present": bin_ is not None, "path": bin_ or "",
            "version": ".".join(map(str, ver)) if ver else "",
            "compat": _blender_compat(ver),
        },
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


def _export_base(output_dir: str | None) -> Path:
    """Resolve the export root. Empty -> the default ``output/export``. A custom
    value is expanded (``~``) and, if relative, anchored to the repo root so it
    behaves the same regardless of the backend's cwd. The structured
    ``<tag>/<capture>/<seq>`` layout is always appended underneath by the caller."""
    raw = (output_dir or "").strip()
    if not raw:
        return _OUTPUT / "export"
    raw = os.path.expanduser(raw)
    return Path(raw) if os.path.isabs(raw) else (_REPO_ROOT / raw)


def _export_one(jid: str, spec: dict) -> tuple[bool, list[str], str | None]:
    """Run one sequence's Blender export. Returns (ok, output_files, error)."""
    out_dir = _export_base(spec.get("output_dir")) / spec["tag"] / spec["capture"] / spec["seq"]
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
    return (rc == 0, outs, None if rc == 0 else f"export exited {rc}")


def _run_export_batch(jid: str, specs: list[dict]) -> None:
    """Export sequences one after another in a single job, reporting aggregate
    progress and per-sequence results (a batch of one behaves like a single export)."""
    total = len(specs)
    all_outs: list[str] = []
    results: list[dict] = []
    failed = 0
    for i, spec in enumerate(specs, 1):
        _update_job(jid, progress={"done": i - 1, "total": total,
                                   "current": f"{spec['capture']}/{spec['seq']}"},
                    log_line=f"[{i}/{total}] {spec['capture']}/{spec['seq']}")
        try:
            ok, outs, err = _export_one(jid, spec)
        except Exception as exc:  # noqa: BLE001
            ok, outs, err = False, [], str(exc)
        if not ok:
            failed += 1
        all_outs += outs
        results.append({"seq": spec["seq"], "capture": spec["capture"], "tag": spec["tag"],
                        "ok": ok, "outputs": outs, "error": err})
    _update_job(jid, state="ready" if failed == 0 else "error", outputs=all_outs,
                results=results, progress={"done": total, "total": total, "current": None},
                error=None if failed == 0 else f"{failed}/{total} export(s) failed")


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


# ---- 3D mesh preview -----------------------------------------------------

# Serializes scene-blob generation so a double-clicked Preview button never
# runs two subprocesses racing to write the same cache file.
_preview_lock = threading.Lock()


def _anchor_repo(path: str) -> Path:
    """Expand ``~`` and anchor repo-relative paths (legacy DB rows carry
    "output/…") to the repo root so cwd never matters. Like _export_base."""
    p = Path(os.path.expanduser(path))
    return p if p.is_absolute() else _REPO_ROOT / p


def _blob_is_current(out: Path) -> bool:
    """True when the cached blob was written by the current schema version.
    The version field leads the JSON, so sniffing the head avoids parsing MBs."""
    try:
        with open(out, "r", encoding="utf-8") as f:
            return '"version": 2' in f.read(64)
    except OSError:
        return False


def _preview_scene_blob(tag: str, capture: str, seq: str, ma_3d_dir: str,
                        ma_cap_dir: str = "") -> tuple[Path | None, str | None]:
    """Return (path to a fresh scene3d.json, error). Generates via
    ``visualization.preview_scene`` (a ~1-2 s numpy subprocess) unless a
    cache newer than every source npz already exists."""
    root = _anchor_repo(ma_3d_dir)
    seq_dir = root / seq
    sources = sorted(glob.glob(str(seq_dir / "smplx_params_body_id-*.npz"))) + \
        sorted(glob.glob(str(seq_dir / "verts_joints_body_id-*.npz")))
    if not sources:
        return None, f"no ma_3d results (smplx_params_body_id-*.npz) under {seq_dir}"
    out = _OUTPUT / "preview" / tag / capture / seq / "scene3d.json"
    src_mtime = max(os.path.getmtime(p) for p in sources)
    with _preview_lock:
        if out.is_file() and out.stat().st_mtime >= src_mtime and _blob_is_current(out):
            return out, None
        cmd = [sys.executable, "-m", "visualization.preview_scene",
               "--ma-3d-dir", str(root), "--seq-name", seq, "--out", str(out)]
        if ma_cap_dir:
            cmd += ["--ma-cap-dir", str(_anchor_repo(ma_cap_dir))]
        try:
            r = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True,
                               text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return None, "preview generation timed out"
        if r.returncode != 0 or not out.is_file():
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
            return None, "preview generation failed: " + (" | ".join(tail) or f"exit {r.returncode}")
        return out, None


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
        opts = {k: body.get(k) for k in ("formats", "unit", "ground", "blender_format", "fps", "output_dir")}
        # `sequences` = batch (per-seq identity dicts sharing the top-level options);
        # otherwise the top-level body is a single sequence (back-compat).
        raw = body.get("sequences") or [body]
        specs = []
        for s in raw:
            spec = {k: s.get(k) for k in ("tag", "capture", "seq", "ma_3d_dir", "ma_cap_dir")}
            spec.update(opts)
            missing = [k for k in ("tag", "capture", "seq", "ma_3d_dir") if not spec.get(k)]
            if not spec.get("formats"):
                missing.append("formats")
            if missing:
                return jsonify({"error": f"missing: {', '.join(missing)}"}), 400
            specs.append(spec)
        if not specs:
            return jsonify({"error": "no sequences"}), 400
        jid = _new_job("export")
        threading.Thread(target=_run_export_batch, args=(jid, specs), daemon=True).start()
        return jsonify({"job_id": jid}), 201

    @app.get("/api/exporter/preview-scene")
    def _exporter_preview_scene():
        """Scene blob for the in-browser 3D mesh preview. Generation is fast
        (~1-2 s) and cached, so this is a single synchronous request — the
        frontend just shows a spinner until the JSON arrives."""
        q = {k: (request.args.get(k) or "").strip()
             for k in ("tag", "capture", "seq", "ma_3d_dir", "ma_cap_dir")}
        missing = [k for k in ("seq", "ma_3d_dir") if not q[k]]
        if missing:
            return jsonify({"error": f"missing: {', '.join(missing)}"}), 400
        seq_dir = Path(q["ma_3d_dir"])
        tag = q["tag"] or seq_dir.parent.name or "run"
        capture = q["capture"] or seq_dir.name or "capture"
        blob, err = _preview_scene_blob(tag, capture, q["seq"], q["ma_3d_dir"],
                                        ma_cap_dir=q["ma_cap_dir"])
        if err:
            return jsonify({"error": err}), 500 if "failed" in err or "timed" in err else 404
        resp = send_file(blob, mimetype="application/json")
        # Freshness is decided server-side (cache vs source npz mtimes);
        # the URL never changes, so the browser must not cache the body.
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/api/exporter/job/<job_id>")
    def _exporter_job(job_id):
        with _jobs_lock:
            rec = _jobs.get(job_id)
        if rec is None:
            return jsonify({"error": "unknown job"}), 404
        return jsonify(rec)

    @app.get("/api/exporter/jobs")
    def _exporter_jobs():
        # Recent jobs, newest first — lets the frontend re-attach to a running
        # export after a tab switch or page reload (the job registry is
        # server-side; component state is not). In-memory read, O(#jobs).
        with _jobs_lock:
            recs = sorted(_jobs.values(), key=lambda r: r["started_at"], reverse=True)[:10]
        return jsonify(recs)
