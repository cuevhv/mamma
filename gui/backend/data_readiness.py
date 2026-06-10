"""Data-readiness panel — inventory + per-asset acquisition.

Three acquisition modes, picked per asset by its source descriptor:

  PublicSource — single HTTPS GET, no credentials.
  MpiSource    — single HTTPS POST to download.is.tue.mpg.de with a
                 username+password form body, mirroring the wire format
                 used by data/download_mamma_*.sh. Credentials arrive in
                 ONE request body from the browser, are used exactly once
                 to compose the outbound POST, and are then dropped. They
                 are never logged, never echoed in any response or error
                 message, never written to disk, and never passed to a
                 subprocess (so they cannot leak via /proc/<pid>/cmdline).
  ManualSource — short documented steps the user has to follow themselves.

Both background-downloading modes stream into ``<dest>.part`` and
atomic-rename on success so a partial file never replaces the
destination. SMPL-X locked-head ships as a zip that is extracted in
place after download; the zip is deleted once extraction succeeds.

Routes (added to app.py via :func:`register_routes`):

  GET  /api/data/readiness/status     -> file-system probe, full inventory.
  POST /api/data/readiness/start      -> kick off a PublicSource download.
  POST /api/data/readiness/start-mpi  -> kick off an MpiSource download.
  GET  /api/data/readiness/job/<id>   -> per-job progress.
"""
from __future__ import annotations

import dataclasses
import http.cookiejar
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request

# Source-descriptor classes + the central asset registry are defined in
# inference.assets. We re-export the descriptor classes from this
# module so external imports keep working (some early GUI internals
# referenced ``from gui.backend.data_readiness import MpiSource``).
from inference.assets import (  # noqa: F401 — re-exported
    ASSETS as _REGISTRY_ASSETS,
    InstallationAsset as _InstallationAsset,
    PublicSource,
    MpiSource,
    ManualSource,
    GDriveSource,
    HFHubSource,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Outbound-download host allowlist. Adding a host is a code change and
# should be reviewed: the MPI flow ships user credentials in the request
# body, so the destination must be trustworthy.
_ALLOWED_HOSTS = frozenset({
    "download.is.tue.mpg.de",
    "mamma.is.tue.mpg.de",
    "smpl-x.is.tue.mpg.de",
    "dl.fbaipublicfiles.com",
    "github.com",
    "github-releases.githubusercontent.com",  # GH release redirect target
    "objects.githubusercontent.com",          # GH release redirect target
    # Google Drive — used by GDriveSource for files behind the >100 MB
    # anti-bot interstitial. The download itself never carries user
    # credentials; the only "auth" GDrive does is the confirm-token
    # dance, which we handle inline.
    "drive.google.com",
    "drive.usercontent.google.com",
})

_MPI_DOWNLOAD_URL = "https://download.is.tue.mpg.de/download.php"

# One friendly, consistent message for every "the download server won't
# give us the file right now" situation. The server signals this several
# ways (an HTML landing page (200), a 403, or a 429), but they all mean
# the same thing to a user, so they share one message. Kept deliberately
# vague on cause (we don't claim retrying makes it worse) and reassuring
# about credentials (sign-in is verified separately against login.php).
# Nudges the user to try the file directly on the project website: it's a
# quick sanity check that the problem is the server limiting their IP, and
# a browser download is a fallback if it happens to work there.
_DOWNLOAD_LIMIT_MSG = (
    "We couldn't fetch this file from the download server right now. It "
    "looks like the server is temporarily limiting downloads from your IP. "
    "This usually clears within about 24 hours, so please try again later. "
    "As a double-check, try signing in on the project website and "
    "downloading the file directly from the website. "
    "If downloading still fails, the limit is on the server, not your account."
)


# ─── Asset descriptors ───────────────────────────────────────────────────
#
# The five source-descriptor classes (PublicSource, MpiSource,
# ManualSource, GDriveSource, HFHubSource) are now defined in
# inference.assets and re-exported at the top of this module. The local
# `DataAsset` dataclass below is the GUI-facing shape that
# `_asset_record` serializes; it's built from the registry by
# `_to_data_asset` below.


@dataclasses.dataclass(frozen=True)
class DataAsset:
    id: str
    label: str
    rel_path: str              # relative to the repo root
    fs_kind: str               # "file" | "dir"
    purpose: str               # short caption shown next to the label
    source: object             # PublicSource | MpiSource | ManualSource
    size_hint_mb: int = 0      # used pre-download for the "~N MB" hint
    optional: bool = False     # ready-count denominator only includes
                               # required assets
    # Visual grouping in the readiness panel. Empty string is the
    # default (top) section, no header. Known groups (paired with their
    # display titles in DataReadinessPanel.tsx → SECTION_DEFS):
    #   "detectors"    → "Bounding box detectors"
    #   "segmenters"   → "Mask segmenters" (either-or; ma_masks needs one)
    #   "body_models"  → "Body models"
    #   "mamma_assets" → "MAMMA assets"  (the MAMMA-specific weights & data)
    #   "training"     → "Training only" (used by landmarks/train.py)
    group: str = ""


def _to_data_asset(a: _InstallationAsset) -> DataAsset:
    """Adapt a registry record to the GUI panel's DataAsset shape.

    For HFHubSource assets the displayed ``rel_path`` is the HF cache
    location (a hint for users — the probe still routes via
    ``_probe_hf_cache``). For other assets ``rel_path`` is the default
    repo-relative path from the registry.
    """
    if isinstance(a.source, HFHubSource):
        slug = a.source.model_id.replace("/", "--")
        rel_path = f"~/.cache/huggingface/hub/models--{slug}"
    else:
        rel_path = a.default or ""
    return DataAsset(
        id=a.id,
        label=a.label,
        rel_path=rel_path,
        fs_kind=a.fs_kind,
        purpose=a.purpose,
        source=a.source,
        size_hint_mb=a.size_hint_mb,
        optional=a.panel_optional,
        group=a.group,
    )


# Derived from inference.assets.ASSETS at import time. Adding a new
# asset is a one-place change in inference/assets.py.
ASSETS: tuple = tuple(_to_data_asset(a) for a in _REGISTRY_ASSETS)


# ─── Filesystem probe ────────────────────────────────────────────────────

def _probe(asset: DataAsset) -> dict:
    # HFHubSource assets live in the user's HF cache, not under the repo.
    # Resolve via $HF_HOME / $HUGGINGFACE_HUB_CACHE, falling back to the
    # documented default. Presence = at least one snapshot dir exists.
    if isinstance(asset.source, HFHubSource):
        return _probe_hf_cache(asset.source.model_id)

    abs_path = (_REPO_ROOT / asset.rel_path).resolve()
    if asset.fs_kind == "file":
        present = abs_path.is_file()
        size = abs_path.stat().st_size if present else 0
    else:
        present = abs_path.is_dir() and any(abs_path.iterdir())
        size = (
            sum(f.stat().st_size for f in abs_path.rglob("*") if f.is_file())
            if present else 0
        )
    return {"present": present, "size_bytes": size}


def _hf_hub_root() -> Path:
    """Return the HuggingFace Hub cache root, honouring the standard env
    vars. The HF library's resolution order is:
      1. HUGGINGFACE_HUB_CACHE (the hub root itself)
      2. HF_HOME/hub
      3. ~/.cache/huggingface/hub
    """
    cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache:
        return Path(cache)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _probe_hf_cache(model_id: str) -> dict:
    """Inspect the HF cache for the given model. Present iff there's at
    least one snapshot directory under ``models--<org>--<repo>/snapshots/``."""
    sanitized = model_id.replace("/", "--")
    model_dir = _hf_hub_root() / f"models--{sanitized}"
    if not model_dir.is_dir():
        return {"present": False, "size_bytes": 0}
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return {"present": False, "size_bytes": 0}
    try:
        has_snapshot = any(p.is_dir() for p in snapshots.iterdir())
    except OSError:
        return {"present": False, "size_bytes": 0}
    if not has_snapshot:
        return {"present": False, "size_bytes": 0}
    # Real files live under blobs/ as content-addressed blobs; snapshots/
    # are symlinks into blobs/. Walk blobs to count actual bytes.
    blobs = model_dir / "blobs"
    size = 0
    if blobs.is_dir():
        try:
            size = sum(
                p.stat().st_size for p in blobs.rglob("*") if p.is_file()
            )
        except OSError:
            size = 0
    return {"present": True, "size_bytes": size}


def _asset_record(asset: DataAsset) -> dict:
    probe = _probe(asset)
    src = asset.source
    if isinstance(src, PublicSource):
        source_dict = {"kind": "public"}
    elif isinstance(src, GDriveSource):
        source_dict = {"kind": "gdrive", "link": src.link}
    elif isinstance(src, HFHubSource):
        source_dict = {
            "kind": "hf_hub",
            "model_id": src.model_id,
            "account_label": src.account_label,
            "register_url": src.register_url,
            "gated": src.gated,
            "steps": list(src.steps),
        }
    elif isinstance(src, MpiSource):
        source_dict = {
            "kind": "mpi",
            "account_label": src.account_label,
            "register_url": src.register_url,
            "extract": src.extract,
        }
    elif isinstance(src, ManualSource):
        source_dict = {
            "kind": "manual",
            "steps": list(src.steps),
            "link": src.link,
            "link_label": src.link_label,
        }
    else:
        source_dict = {"kind": "unknown"}
    return {
        "id": asset.id,
        "label": asset.label,
        "rel_path": asset.rel_path,
        "fs_kind": asset.fs_kind,
        "purpose": asset.purpose,
        "size_hint_mb": asset.size_hint_mb,
        "optional": asset.optional,
        "group": asset.group,
        "present": probe["present"],
        "size_bytes": probe["size_bytes"],
        "source": source_dict,
    }


def _find_asset(asset_id: str) -> Optional[DataAsset]:
    return next((a for a in ASSETS if a.id == asset_id), None)


def _smallest_mpi_asset(domain: str) -> Optional[DataAsset]:
    """Return the smallest MpiSource asset whose source domain matches.

    Used by the credential-verify route: probing the smallest file in a
    domain keeps the check cheap (the worker only reads the first chunk
    anyway, but a small file also minimises wasted server work)."""
    candidates = [
        a for a in ASSETS
        if isinstance(a.source, MpiSource) and a.source.domain == domain
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda a: a.size_hint_mb or float("inf"))


# ─── Job table ───────────────────────────────────────────────────────────

# In-memory job table. Each entry holds progress + state for one download.
# Cleared on backend restart — these jobs are ephemeral by design.
_jobs_lock = threading.Lock()
_jobs: dict = {}


def _new_job(asset_id: str) -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "asset_id": asset_id,
            "state": "downloading",
            "bytes_downloaded": 0,
            "bytes_total": 0,
            "error": None,
            "started_at": time.time(),
        }
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        rec = _jobs.get(job_id)
        if rec is not None:
            rec.update(kwargs)


# ─── Outbound HTTP helpers ───────────────────────────────────────────────

def _check_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise RuntimeError("Only https:// download URLs are allowed.")
    if parsed.hostname not in _ALLOWED_HOSTS:
        raise RuntimeError(
            f"Download host '{parsed.hostname}' is not in the allowlist."
        )


def _is_html_error(prefix: bytes) -> bool:
    head = prefix[:128].lstrip().lower()
    return (
        head.startswith(b"<!doctype html")
        or head.startswith(b"<html")
        or b"error: file not found." in head
    )


def _stream_to_disk(resp, dest_tmp: Path, job_id: str) -> None:
    """Stream the HTTP response body into ``dest_tmp`` chunk by chunk.

    Updates the job's progress counters after each chunk. Detects the
    common 'HTML error page returned in place of a binary' failure mode
    of the MPI download endpoint by sniffing the first chunk."""
    total_hdr = resp.headers.get("Content-Length")
    total = int(total_hdr) if total_hdr and total_hdr.isdigit() else 0
    _update_job(job_id, bytes_total=total)
    first = True
    downloaded = 0
    with open(dest_tmp, "wb") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            if first:
                if _is_html_error(chunk):
                    # The MPI download endpoint serves an HTML landing page
                    # (200 OK + text/html) instead of the file when it's
                    # limiting downloads from this IP — the same situation a
                    # 403/429 signals on other requests. Surface the one
                    # shared, friendly message so every failure path reads
                    # the same. Sign-in credentials are verified separately
                    # (login.php), so this is not a "wrong password" case.
                    raise RuntimeError(_DOWNLOAD_LIMIT_MSG)
                first = False
            f.write(chunk)
            downloaded += len(chunk)
            _update_job(job_id, bytes_downloaded=downloaded)


def _atomic_install_file(tmp_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, dest_path)


def _flatten_single_wrapper_dir(dest_path: Path, expected_subdir: str) -> None:
    """Flatten an extra wrapper directory if a zip nested its content.

    No-op when ``expected_subdir`` already sits directly under ``dest_path``
    (extraction produced the correct layout). When ``dest_path`` instead
    contains exactly one directory whose ``expected_subdir`` is one level
    deeper, move the wrapper's contents up so consumers see the expected
    layout.

    The SMPL-X locked-head zip (``smplx_lockedhead_20230207.zip``) extracts
    to ``models_lockedhead/smplx/...``, but ``smplx.create()`` in ma_3d
    expects ``smplx/`` directly under the configured model path. Without
    this flatten, the GUI-downloaded model lands one level too deep.
    """
    if (dest_path / expected_subdir).exists():
        return
    entries = list(dest_path.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return
    wrapper = entries[0]
    if not (wrapper / expected_subdir).exists():
        return
    for child in list(wrapper.iterdir()):
        os.replace(child, dest_path / child.name)
    wrapper.rmdir()


def _safe_error(exc: BaseException) -> str:
    """Format a user-facing error string. Never includes credentials.

    HTTP code mapping is deliberately distinct: 401 and 403 mean very
    different things and conflating them ("Authentication failed.") sends
    users on a wild-goose chase through their password manager when the
    real cause is rate limiting or a file-specific ACL.
    """
    if isinstance(exc, urllib.error.HTTPError):
        # 401 is the only code that genuinely means "username or password
        # is wrong" — point users at their credentials in that case only.
        if exc.code == 401:
            return (
                "Those credentials weren't accepted. Please double-check the "
                "username (the email you registered with) and password."
            )
        # 403 and 429 are both the download server limiting this IP — NOT
        # an auth failure (a wrong password returns 401, above). Use the one
        # shared, friendly message so every limit path reads the same.
        if exc.code in (403, 429):
            return _DOWNLOAD_LIMIT_MSG
        # 5xx covers server-side faults; never the user's problem.
        if 500 <= exc.code < 600:
            return (
                "The download server hit a problem on its end. This isn't "
                "something on your side — please try again in a little while."
            )
        return (
            "The download server returned an unexpected response "
            f"(HTTP {exc.code}). Please try again later."
        )
    if isinstance(exc, urllib.error.URLError):
        return (
            "We couldn't reach the download server. Please check your "
            "internet connection and try again."
        )
    if isinstance(exc, RuntimeError):
        return str(exc)
    return "The download didn't complete. Please try again."


# ─── Workers ─────────────────────────────────────────────────────────────

def _run_gdrive(asset: DataAsset, job_id: str) -> None:
    """Download a Google Drive file, transparently handling the
    confirm-token interstitial that GDrive serves for files >100 MB.

    No credentials are involved. The cookie jar exists only to carry
    GDrive's own `download_warning_*` cookie between requests; it never
    touches user creds (those go through :func:`_run_mpi`)."""
    src: GDriveSource = asset.source  # type: ignore[assignment]
    dest = (_REPO_ROOT / asset.rel_path).resolve()
    tmp = dest.with_suffix(dest.suffix + ".part")

    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookie_jar)
    )

    base = "https://drive.usercontent.google.com/download"

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Step 1: ask for the file. Small files come straight back;
        # large files come back as an HTML form with a confirm token.
        params = {"id": src.file_id, "export": "download"}
        init_url = f"{base}?{urllib.parse.urlencode(params)}"
        _check_url(init_url)
        resp = opener.open(init_url, timeout=60)
        first_chunk = resp.read(64 * 1024)
        ctype = (resp.headers.get("Content-Type") or "").lower()

        if "text/html" in ctype or _is_html_error(first_chunk):
            # Confirm page. Read the rest, pull the uuid token out of
            # the hidden <input>, and re-request with it.
            rest = resp.read()
            resp.close()
            html = first_chunk + rest
            uuid_match = re.search(rb'name="uuid"\s+value="([^"]+)"', html)
            confirm_val = "t"
            for c in cookie_jar:
                if c.name.startswith("download_warning") and c.value:
                    confirm_val = c.value
                    break
            params2 = {
                "id": src.file_id,
                "export": "download",
                "confirm": confirm_val,
            }
            if uuid_match:
                params2["uuid"] = uuid_match.group(1).decode("ascii", errors="ignore")
            url2 = f"{base}?{urllib.parse.urlencode(params2)}"
            _check_url(url2)
            resp = opener.open(url2, timeout=60)
            _stream_to_disk(resp, tmp, job_id)
        else:
            # Streaming the file directly on the first try. Reuse the
            # 64 KB we already pulled instead of re-requesting.
            total_hdr = resp.headers.get("Content-Length")
            total = int(total_hdr) if total_hdr and total_hdr.isdigit() else 0
            _update_job(job_id, bytes_total=total)
            downloaded = len(first_chunk)
            with open(tmp, "wb") as f:
                f.write(first_chunk)
                _update_job(job_id, bytes_downloaded=downloaded)
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    _update_job(job_id, bytes_downloaded=downloaded)

        _atomic_install_file(tmp, dest)
        _update_job(job_id, state="ready")
        logger.info("GDrive download completed: asset=%s", asset.id)
    except Exception as exc:  # noqa: BLE001 — all failures surface to UI
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        _update_job(job_id, state="error", error=_safe_error(exc))
        logger.warning("GDrive download failed: asset=%s (%s)", asset.id, type(exc).__name__)


def _run_public(asset: DataAsset, job_id: str) -> None:
    src: PublicSource = asset.source  # type: ignore[assignment]
    dest = (_REPO_ROOT / asset.rel_path).resolve()
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        _check_url(src.url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(src.url, method="GET")
        with urllib.request.urlopen(req, timeout=60) as resp:
            _stream_to_disk(resp, tmp, job_id)
        _atomic_install_file(tmp, dest)
        _update_job(job_id, state="ready")
        logger.info("Public download completed: asset=%s", asset.id)
    except Exception as exc:  # noqa: BLE001 — all failures surface to UI
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        _update_job(job_id, state="error", error=_safe_error(exc))
        logger.warning("Public download failed: asset=%s (%s)", asset.id, type(exc).__name__)


def _run_mpi(asset: DataAsset, username: str, password: str, job_id: str) -> None:
    """One-shot MPI auth + download.

    Wire format (matches data/download_mamma_*.sh, which is the canonical
    reference; the server 500s on shapes that don't match):

      URL : https://download.is.tue.mpg.de/download.php
              ?domain=<domain>&resume=1&sfile=<sfile>
      POST: username=<u>&password=<p>      ← creds only

    Putting the *non-secret* request descriptors (sfile/domain/resume) in
    the URL keeps the wire shape identical to the shell downloaders.
    Credentials still go only in the POST body so they don't surface in
    URL logs, /proc/<pid>/cmdline, browser history, or any HTTP referer
    header that some downstream tool might emit.

    Security contract (unchanged):
      * `username` and `password` are local parameters only.
      * They are used to build a single URL-encoded form body, sent in
        one HTTPS POST to download.is.tue.mpg.de.
      * They are then deleted from this frame; nothing else in this
        module retains them.
      * They are never written to disk, logged, echoed in any response,
        or passed to a subprocess.
    """
    src: MpiSource = asset.source  # type: ignore[assignment]
    dest_path = (_REPO_ROOT / asset.rel_path).resolve()

    # URL: descriptor params in the query string. `safe='/'` keeps the
    # sfile path literal (matching the shell's unescaped slashes).
    query = urllib.parse.urlencode(
        {"domain": src.domain, "resume": "1", "sfile": src.sfile},
        safe="/",
    )
    request_url = f"{_MPI_DOWNLOAD_URL}?{query}"

    # POST body: credentials ONLY.
    form_body = urllib.parse.urlencode({
        "username": username,
        "password": password,
    }).encode("utf-8")
    del username, password

    # Asset id + domain are non-sensitive; credentials are not in the log line.
    logger.info(
        "MPI download starting: asset=%s domain=%s extract=%s",
        asset.id, src.domain, src.extract,
    )

    if src.extract:
        zip_path = (_REPO_ROOT / "data" / f"_{asset.id}_download.zip").resolve()
        tmp = zip_path.with_suffix(zip_path.suffix + ".part")
    else:
        zip_path = None
        tmp = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        _check_url(request_url)
        # Ensure the staging dir exists before streaming into `<dest>.part`.
        # For non-extract assets tmp.parent is the destination's parent
        # (e.g. data/weights/ma_2d/), which may not exist yet; for extract
        # assets it's data/. Mirrors the `mkdir -p` the shell downloaders do
        # and the dest.parent.mkdir() that _run_public/_run_gdrive already do.
        tmp.parent.mkdir(parents=True, exist_ok=True)
        # Match wget's request fingerprint exactly. The MPI download.php
        # endpoint serves the file body for wget but returns an HTML
        # landing page for "vanilla" urllib calls — observed difference
        # is the header set wget sends by default. Spoofing the
        # User-Agent alone is not enough; the full quad below is what
        # makes the server treat us as a download client.
        req = urllib.request.Request(
            request_url, data=form_body, method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "Wget/1.21.4",
                "Accept": "*/*",
                "Accept-Encoding": "identity",
                "Connection": "Keep-Alive",
            },
        )
        # The body is sent on the wire and the response is streamed.
        with urllib.request.urlopen(req, timeout=60) as resp:
            _stream_to_disk(resp, tmp, job_id)
        if src.extract and zip_path is not None:
            _atomic_install_file(tmp, zip_path)
            _update_job(job_id, state="extracting")
            dest_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest_path)
            if asset.id == "smplx_locked_head":
                _flatten_single_wrapper_dir(dest_path, expected_subdir="smplx")
            zip_path.unlink(missing_ok=True)
        else:
            _atomic_install_file(tmp, dest_path)
        _update_job(job_id, state="ready")
        logger.info("MPI download completed: asset=%s", asset.id)
    except Exception as exc:  # noqa: BLE001
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        if zip_path is not None:
            try:
                zip_path.unlink(missing_ok=True)
            except Exception:
                pass
        _update_job(job_id, state="error", error=_safe_error(exc))
        logger.warning(
            "MPI download failed: asset=%s (%s)", asset.id, type(exc).__name__,
        )
    finally:
        # Best-effort clear of the form body. CPython may still hold the
        # bytes in the garbage collector until collection; this is the
        # tightest we can reasonably get without an FFI dance.
        del form_body


def _login_url_for(src: MpiSource) -> str:
    """Derive a domain's website LOGIN url from its registration url.

    Every MPI project site (mamma.is.tue.mpg.de, smpl-x.is.tue.mpg.de, …)
    serves a ``login.php`` next to its ``register.php``. Deriving the login
    url from the registry's ``register_url`` keeps this data-driven — no
    second hard-coded host table to drift out of sync.
    """
    return src.register_url.replace("register.php", "login.php")


def _verify_mpi_credentials(asset: DataAsset, username: str, password: str) -> dict:
    """Verify MPI credentials against the project site's LOGIN endpoint.

    This deliberately does NOT touch ``download.php``. That endpoint is the
    rate-limited one: a wrong password there returns 401, but once an IP
    trips the (~24-hour) download block every request returns an opaque 403
    regardless of whether the credentials are valid — so it can neither
    verify reliably nor be probed safely (verifying could itself get the
    user blocked).

    ``login.php`` is a plain, non-rate-limited auth check. Observed wire
    behaviour (confirmed live 2026-06-10):

      * correct credentials  -> 302 redirect away from the login page
      * wrong credentials     -> 200 that re-renders the login form

    So a 3xx is the unambiguous "valid" signal and a 200 is "wrong username
    or password" — no rate-limit hedging needed, because logging in is not
    rate-limited. Returns ``{"valid": bool, "detail": str}``. Credentials
    live only in this frame's POST body.
    """
    src: MpiSource = asset.source  # type: ignore[assignment]
    login_url = _login_url_for(src)
    form_body = urllib.parse.urlencode(
        {"username": username, "password": password, "commit": "Log in"}
    ).encode("utf-8")
    del username, password

    # Don't follow the redirect — the 3xx itself is the success signal.
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
        _NoRedirect,
    )
    try:
        _check_url(login_url)
        # Prime a PHP session cookie (the login form sets PHPSESSID on GET;
        # the POST is accepted within that session).
        try:
            opener.open(
                urllib.request.Request(login_url, headers={"User-Agent": "Mozilla/5.0"}),
                timeout=30,
            ).read()
        except Exception:  # noqa: BLE001 — priming is best-effort
            pass
        req = urllib.request.Request(
            login_url, data=form_body, method="POST",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            resp = opener.open(req, timeout=30)
            code = resp.getcode()
        except urllib.error.HTTPError as exc:
            code = exc.code
        if 300 <= code < 400:
            return {"valid": True, "detail": "Credentials accepted."}
        site = src.register_url.replace("register.php", "")
        return {"valid": False, "detail": (
            "Those credentials weren't accepted. Please double-check the "
            f"username (the email you registered with at {site}) and password."
        )}
    except Exception as exc:  # noqa: BLE001 — surface a safe message to the UI
        return {"valid": False, "detail": _safe_error(exc)}
    finally:
        del form_body


# ─── Flask wiring ────────────────────────────────────────────────────────

def register_routes(app: Flask) -> None:
    """Install the data-readiness routes on the given Flask app."""

    @app.get("/api/data/readiness/status")
    def _status():  # noqa: D401
        items = [_asset_record(a) for a in ASSETS]
        required = [r for r in items if not r["optional"]]
        ready_required = sum(1 for r in required if r["present"])
        return jsonify({
            "items": items,
            "ready": ready_required,
            "total": len(required),
        })

    @app.post("/api/data/readiness/start")
    def _start():
        body = request.get_json(silent=True) or {}
        asset = _find_asset(str(body.get("id") or ""))
        if asset is None:
            return jsonify({"error": "unknown asset"}), 404
        if isinstance(asset.source, PublicSource):
            worker = _run_public
        elif isinstance(asset.source, GDriveSource):
            worker = _run_gdrive
        else:
            return jsonify({"error": "asset is not a credential-less download"}), 400
        job_id = _new_job(asset.id)
        threading.Thread(
            target=worker, args=(asset, job_id), daemon=True,
        ).start()
        return jsonify({"job_id": job_id})

    @app.post("/api/data/readiness/start-mpi")
    def _start_mpi():
        # Pull credentials from the request body. They live only inside
        # this function's frame and the worker thread it spawns.
        body = request.get_json(silent=True) or {}
        asset = _find_asset(str(body.get("id") or ""))
        username = str(body.get("username") or "")
        password = str(body.get("password") or "")
        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400
        if asset is None:
            return jsonify({"error": "unknown asset"}), 404
        if not isinstance(asset.source, MpiSource):
            return jsonify({"error": "asset is not an MPI-account download"}), 400
        job_id = _new_job(asset.id)
        # Pass creds positionally so they don't appear in a kwargs repr if
        # something deep in the threading machinery ever printed one.
        threading.Thread(
            target=_run_mpi, args=(asset, username, password, job_id),
            daemon=True,
        ).start()
        # Don't echo the username back. The frontend already has it.
        return jsonify({"job_id": job_id})

    @app.post("/api/data/readiness/verify-mpi")
    def _verify_mpi():
        # Credential check for a sign-in domain (mamma/smplx): authenticate
        # against the project site's login.php (NOT the rate-limited
        # download.php). Any asset in the domain supplies the site url, so
        # we reuse _smallest_mpi_asset. Creds live only in this frame.
        body = request.get_json(silent=True) or {}
        domain = str(body.get("domain") or "")
        username = str(body.get("username") or "")
        password = str(body.get("password") or "")
        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400
        asset = _smallest_mpi_asset(domain)
        if asset is None:
            return jsonify({"error": f"no MPI asset for domain '{domain}'"}), 404
        result = _verify_mpi_credentials(asset, username, password)
        return jsonify(result)

    @app.get("/api/data/readiness/job/<job_id>")
    def _job(job_id):
        with _jobs_lock:
            rec = _jobs.get(job_id)
        if rec is None:
            return jsonify({"error": "unknown job"}), 404
        return jsonify(rec)
