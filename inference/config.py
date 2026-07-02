"""Run-config loading, validation, and preset→run materialization.

The pipeline operates on a **run config** — a capture-bound execution
snapshot stored on disk (or built in memory at CLI/GUI submit time).
Run configs are recognized by suffix: ``.json`` (stdlib :mod:`json`)
and ``.yaml`` / ``.yml`` (PyYAML's ``safe_load``). The schema is
identical across both formats — YAML is friendlier for the long
multi-line argument lists that ``ma_masks`` and ``ma_3d`` carry.

A run config has this shape::

    {
        "global": {
            "dataset_name": "...",
            "capture_json": "/path/to/capture.json",
            "seq_ids": [0, 1, 2],
            "out_dir": "/path/to/output",
            "cam_names": ["cam01", "cam02", ...],
            "conda_env": "mamma",                  # optional
            "jobs_log_dir": "/path/to/logs",       # optional
            "username": "alice",                   # optional
            "bind": ["/data:/data"]                # optional, apptainer/docker
        },
        "<step_id>": {
            "enabled": true|false,
            "dependencies": ["<step_id>", ...],
            "engine": "conda" | "apptainer" | "docker",
            "script": "run_*.py",
            "repo_path": "${MAMMA_THIRD_PARTY}/<repo>",
            ...                                    # step-specific keys
        },
        ...
    }

A **preset** is the same shape minus ``global.capture_json`` (and
typically minus ``global.seq_ids`` / ``global.cam_names``). Presets
are reusable templates; binding one to a specific capture produces
a run config — see :func:`materialize_run_config`.

Validation here is intentionally lightweight (no ``jsonschema`` dep): we
check the small set of fields the runner actually relies on and produce
pointer-style error messages so users can fix their config quickly.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import List

from .runner import ALL_STEPS

VALID_ENGINES = ("conda", "apptainer", "docker")

_YAML_SUFFIXES = (".yaml", ".yml")
_JSON_SUFFIXES = (".json",)


class TaskConfigError(ValueError):
    """Raised when a run config (or preset) fails validation."""


def _parse(path: str) -> dict:
    suffix = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if suffix in _YAML_SUFFIXES:
            import yaml  # local import: pyyaml is a runtime dep but only for yaml run-config files
            data = yaml.safe_load(f)
        elif suffix in _JSON_SUFFIXES:
            data = json.load(f)
        else:
            raise TaskConfigError(
                f"unrecognized config-file suffix {suffix!r}; "
                f"expected one of {_JSON_SUFFIXES + _YAML_SUFFIXES}"
            )
    if not isinstance(data, dict):
        raise TaskConfigError(f"config file root must be a mapping, got {type(data).__name__}")
    return data


# Canonical videos subdirectory names (probed in order). Released
# captures use ``videos_crf24``; user-imported footage typically uses
# ``videos``. The ``_light`` / ``_crf16`` variants are optional
# alternate encodings that some pipelines ship alongside.
_VIDEOS_SUBDIRS = ("videos", "videos_light", "videos_crf24", "videos_crf16")


def synthesize_capture(
    footage_dir: str,
    calib_path: str,
    seq_name: str,
) -> dict:
    """Build an in-memory capture-config dict for one sequence.

    Lets callers run the pipeline without authoring a capture.json:
    given a dataset directory, a sequence subdirectory name within
    it, and a calibration file, this helper produces the same
    capture-dict shape the runner consumes when loading a real
    capture.json. The dict can then be written to a temp file and
    passed to :func:`materialize_run_config` like any other capture.

    Cameras and the videos-subdir layout are auto-detected from the
    actual on-disk contents of ``<footage_dir>/<seq_name>/`` — videos
    mode (``<videos_subdir>/<cam>.mp4``) or images mode
    (``<cam>/<frames>``). This is the same detection the GUI's
    `/api/captures/generate-json` endpoint uses.

    Args:
        footage_dir: Dataset root — a directory containing one or
            more sequence subdirectories. Becomes ``capture_root``.
        calib_path: Calibration file (`.yaml` / `.xcp` / OpenCV
            `.json`). Validated via :func:`capture.load_calibration`.
        seq_name: Name of the sequence subdirectory under
            ``footage_dir`` to run. The subdir must exist and contain
            either a videos subdir or per-camera image directories.

    Returns:
        Capture-config dict. Keys: ``capture_root``, ``calib``,
        ``cam_fps`` (default 30), ``cams``, ``sequences``,
        ``videos_subdir`` (only set in videos mode when not the
        default ``videos_crf24``), and an optional ``up_axis`` (signed
        world up-axis ``x|y|z|-x|-y|-z`` or ``auto``; auto-detected from
        the calibration when absent/``auto``).

    Raises:
        FileNotFoundError: If the footage_dir, the seq_name subdir,
            or the calibration file is missing.
        ValueError: If the calibration is malformed, or if neither a
            videos subdir nor per-camera image directories are found
            under ``<footage_dir>/<seq_name>/``.
    """
    import os
    from pathlib import Path

    footage = Path(footage_dir).resolve()
    if not footage.is_dir():
        raise FileNotFoundError(f"footage directory not found: {footage_dir}")

    seq_dir = footage / seq_name
    if not seq_dir.is_dir():
        raise FileNotFoundError(
            f"sequence directory not found: {seq_dir} "
            f"(expected '{seq_name}' to be a subdirectory of {footage_dir})"
        )

    # Validate the calibration up front — better to fail here than
    # midway through ma_3d.
    calib_abs = Path(calib_path).resolve()
    from capture.calibration import load_calibration, CalibrationError  # noqa: PLC0415
    try:
        load_calibration(str(calib_abs))
    except (FileNotFoundError, CalibrationError) as exc:
        raise ValueError(f"invalid calibration: {exc}") from exc

    # Auto-detect cameras + layout from the sequence directory.
    from capture.discovery import find_video_files, find_image_cam_dirs  # noqa: PLC0415
    layout: str | None = None
    cams: list[str] = []
    videos_subdir: str | None = None
    for sub in _VIDEOS_SUBDIRS:
        cand = seq_dir / sub
        if not cand.is_dir():
            continue
        files = find_video_files(str(cand))
        if files:
            cams = sorted({Path(v).stem for v in files})
            layout, videos_subdir = "videos", sub
            break
    if layout is None:
        cam_dirs = find_image_cam_dirs(str(seq_dir))
        if cam_dirs:
            cams = sorted(os.path.basename(d) for d in cam_dirs)
            layout = "images"

    if not cams:
        raise ValueError(
            f"could not detect cameras in {seq_dir}: expected either a "
            f"videos subdir ({'/'.join(_VIDEOS_SUBDIRS)}) of .mp4 files "
            f"or per-camera image directories"
        )

    capture: dict = {
        "capture_root": str(footage),
        "calib": str(calib_abs),
        "cam_fps": 30,
        "cams": cams,
        "sequences": {"000": {"name": seq_name}},
    }
    if layout == "videos" and videos_subdir and videos_subdir != "videos_crf24":
        capture["videos_subdir"] = videos_subdir
    return capture


def load_run_config(path: str) -> dict:
    """Load and validate a capture-bound run config. Returns the parsed dict.

    Dispatches on file suffix: ``.json`` (stdlib json), ``.yaml`` /
    ``.yml`` (PyYAML's ``safe_load``). Same schema either way.
    """
    cfg = _parse(path)
    validate(cfg)
    return cfg


def load_task(path: str) -> dict:
    """Deprecated alias for :func:`load_run_config`.

    Kept for one release so external callers (e.g. user scripts pinned
    to an earlier API) continue to work. New code should use
    ``load_run_config``.
    """
    warnings.warn(
        "inference.config.load_task is deprecated; use load_run_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_run_config(path)


def load_preset(path: str) -> dict:
    """Load a preset file (a capture-independent template). No validation.

    Presets intentionally lack ``global.capture_json`` and may also omit
    ``seq_ids`` / ``cam_names`` / ``out_dir``. Validation runs on the
    *materialized* run config, not on the preset itself.
    """
    return _parse(path)


def deep_merge(base: dict, override: dict | None) -> dict:
    """Recursively merge ``override`` into ``base`` in place.

    Nested dicts are recursed into; lists and scalars in ``override``
    replace whatever was in ``base`` (so users can fully redefine a
    flags array or bind path list, not just append). Returns the
    mutated ``base``. ``override=None`` is a no-op.
    """
    if override is None:
        return base
    if not isinstance(override, dict) or not isinstance(base, dict):
        return override if override is not None else base
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def materialize_run_config(
    preset_path: str,
    capture_path: str,
    *,
    seq_ids: List[int] | None = None,
    seq_names: List[str] | None = None,
    cam_names: List[str] | None = None,
    out_dir: str | None = None,
    dataset_name: str | None = None,
    username: str | None = None,
    enabled_steps: List[str] | None = None,
    overrides: dict | None = None,
    sequence_major: bool | None = None,
) -> dict:
    """Bind a capture-independent preset to a capture, producing a run config.

    The preset declares steps + hyperparameters; the capture declares
    sequences + calibration. Both the CLI (``run --preset … --capture …``)
    and the GUI's submit flow go through this single code path, so the
    materialization rules stay consistent across surfaces.

    Args:
        preset_path: Path to the preset file (``.yaml`` / ``.yml`` / ``.json``).
        capture_path: Path to the capture JSON. Recorded as ``global.capture_json``
            so the runner can resolve seq_ids later.
        seq_ids: Restrict to these zero-indexed sequence ids. Wins over ``seq_names``
            when both are given.
        seq_names: Sequence names to include; resolved against the capture's
            ``sequences`` map. Accepts both ``{"name": ...}`` and legacy
            ``{"ioi": ...}`` entry shapes.
        cam_names: Override ``global.cam_names``.
        out_dir: Override ``global.out_dir``.
        dataset_name: Override ``global.dataset_name``. If not supplied and the
            preset's value is empty or placeholder-like (``<...>``), a name is
            derived from the capture filename.
        username: Override ``global.username``.
        enabled_steps: Step ids to enable; all others are disabled. When ``None``
            the preset's per-step ``enabled`` flag is preserved.
        overrides: Deep-partial dict applied to the preset *before* the
            form-level field assignments above land — so explicit kwargs
            always win against overrides for their own keys.
        sequence_major: Toggle the runner's dispatch order. ``False``
            (default) = step-major, finishing each step across all
            sequences before advancing. ``True`` = sequence-major,
            finishing each sequence end-to-end before the next.
            Stored on ``global.sequence_major``; read by
            :func:`inference.runner.run_dag`.

    Returns:
        A fully-bound run config dict. Validate with :func:`validate` before
        persisting or running.
    """
    cfg = load_preset(preset_path)
    if overrides:
        deep_merge(cfg, overrides)

    # Capture data is consulted by several derivation rules below; load
    # once here so we don't re-parse it for each rule.
    with open(capture_path, "r") as f:
        capture_data = json.load(f)

    cfg.setdefault("global", {})
    g = cfg["global"]
    g["capture_json"] = capture_path

    if seq_ids is None and seq_names is not None:
        seq_ids = _seq_ids_from_names_in_data(capture_data, seq_names)
    if seq_ids is not None:
        g["seq_ids"] = seq_ids

    # Truthy check (not "is not None") so an empty list from the GUI
    # form — "user didn't pick any cameras in the multi-select" — falls
    # through to the derive-from-capture fallback rather than locking
    # in cam_names=[] and crashing argparse at ma_cap submit time.
    if cam_names:
        g["cam_names"] = list(cam_names)
    elif not g.get("cam_names"):
        # Fall back to the capture JSON's `cams` field when the preset
        # is silent. Lets a single shared preset target captures with
        # different camera rigs (the rig declaration lives with the
        # capture, not the preset).
        cap_cams = capture_data.get("cams") or []
        if cap_cams:
            g["cam_names"] = list(cap_cams)

    # World up-axis: an explicit capture.json `up_axis` wins; otherwise
    # auto-detect from the calibration geometry. Stored concretely (signed
    # x|y|z|-x|-y|-z) so the step builders emit a valid `--up-axis`. Best-effort:
    # any failure leaves it unset and steps fall back to their `z` default.
    if not g.get("up_axis"):
        try:
            from capture.calibration import (
                load_calibration, detect_up_axis, normalize_up_axis,
            )
            raw = str(capture_data.get("up_axis") or "").strip().lower()
            if raw and raw != "auto":
                g["up_axis"] = normalize_up_axis(raw)
            else:
                calib = capture_data.get("calib")
                if calib:
                    calib_path = calib if os.path.isabs(calib) else os.path.normpath(
                        os.path.join(os.path.dirname(capture_path), calib))
                    axis, conf = detect_up_axis(load_calibration(calib_path))
                    # Only trust a confident detection. An ambiguous rig
                    # (overhead-looking / heavily rolled cameras) can produce a
                    # low-confidence *wrong signed axis* that would silently
                    # reorient ma_3d/ma_vis output; below the threshold, leave
                    # up_axis unset so the steps keep their 'z' default.
                    if conf >= 0.8:
                        g["up_axis"] = axis
                    else:
                        import logging
                        logging.getLogger(__name__).warning(
                            "up-axis auto-detect is ambiguous (%s at confidence "
                            "%.2f < 0.8); keeping the default 'z'. Set up_axis "
                            "explicitly in capture.json to override.", axis, conf)
        except Exception:
            pass

    if out_dir is not None:
        g["out_dir"] = out_dir
    if username is not None:
        g["username"] = username
    if sequence_major is not None:
        g["sequence_major"] = bool(sequence_major)

    if dataset_name is not None:
        g["dataset_name"] = dataset_name
    else:
        existing = (g.get("dataset_name") or "").strip()
        if not existing or existing.startswith("<"):
            g["dataset_name"] = _derive_dataset_name(capture_path)

    # Per-step derivations: fill ma_cap.videos_dir and ma_cap.calibration
    # from the capture JSON when the preset doesn't pin them. Same
    # pattern as the global derivations — preset value (if non-empty)
    # always wins.
    ma_cap = cfg.setdefault("ma_cap", {})
    if not ma_cap.get("videos_dir") and not ma_cap.get("images_root_dir"):
        derived = _derive_videos_dir(capture_path, capture_data)
        if derived:
            ma_cap["videos_dir"] = derived
    if not ma_cap.get("calibration"):
        derived = _derive_calibration(capture_path, capture_data)
        if derived:
            ma_cap["calibration"] = derived

    if enabled_steps is not None:
        wanted = set(enabled_steps)
        for step in ALL_STEPS:
            if step in cfg:
                cfg[step]["enabled"] = step in wanted

    return cfg


def _seq_ids_from_names(capture_path: str, seq_names: List[str]) -> List[int]:
    """Convenience wrapper that loads the capture JSON then delegates.

    Most callers already have ``capture_data`` in hand and should use
    :func:`_seq_ids_from_names_in_data` to avoid re-parsing the JSON.
    """
    with open(capture_path, "r") as f:
        return _seq_ids_from_names_in_data(json.load(f), seq_names)


def _seq_ids_from_names_in_data(capture_data: dict, seq_names: List[str]) -> List[int]:
    """Look up the integer seq_ids matching the given sequence names.

    Accepts both ``{"name": ...}`` (current) and ``{"ioi": ...}`` (legacy)
    capture entry shapes. Returns ids in numeric order. Unknown names are
    silently dropped — the caller decides whether that's an error.
    """
    wanted = set(seq_names)
    matched: List[int] = []
    for seq_id, entry in (capture_data.get("sequences") or {}).items():
        if not isinstance(entry, dict):
            continue
        # Prefer ``name`` (current schema), fall back to ``ioi`` (legacy) —
        # mirrors inference/capture_loader.py._seq_name.
        name = entry.get("name") or entry.get("ioi")
        if name and name in wanted:
            try:
                matched.append(int(seq_id))
            except (TypeError, ValueError):
                continue
    matched.sort()
    return matched


def _derive_videos_dir(capture_path: str, capture_data: dict) -> str | None:
    """Build ``ma_cap.videos_dir`` from the capture's ``capture_root``.

    Convention: ``<capture_root>/{seq_name}/<videos_subdir>`` where
    ``videos_subdir`` defaults to ``"videos_crf24"``. A capture may
    override the subdir by setting ``videos_subdir`` at the top level
    (iphones captures do this — they ship raw ``videos/`` rather than
    the compressed ``videos_crf24/`` tier).

    ``capture_root`` is resolved against the capture file's parent dir
    (it's stored relative there so the JSONs are portable). The
    returned path is repo-root-relative when possible, otherwise
    absolute. ``{seq_name}`` is a template placeholder the step
    builder expands at runtime — kept literal here.
    """
    # Accept both the current ``capture_root`` key and the legacy/GUI
    # ``ioi_root`` key. The "New Capture" backend writes ``ioi_root``
    # (gui/backend/app.py), and run_ma_cap.py's JSON ingest already
    # falls back the same way — without this, videos_dir is never
    # derived for GUI-generated captures, so ma_cap silently drops to
    # image-discovery mode and fails on per-camera video footage.
    has_capture_root = bool(capture_data.get("capture_root"))
    capture_root = capture_data.get("capture_root") or capture_data.get("ioi_root") or ""
    if not capture_root:
        return None
    # Released captures (``capture_root``) have a known default video
    # tier, so a missing ``videos_subdir`` still means "videos" (default
    # ``videos_crf24``). GUI captures (``ioi_root`` only) must declare
    # their layout: the New Capture backend writes ``videos_subdir`` for
    # video footage and omits it for per-camera image dirs. So when only
    # ``ioi_root`` is present and ``videos_subdir`` is absent, this is an
    # image-layout capture — return None and let ma_cap take its
    # JSON/image-discovery path instead of inventing a videos dir.
    if not has_capture_root and not capture_data.get("videos_subdir"):
        return None
    if os.path.isabs(capture_root):
        anchored = capture_root
    else:
        # ``os.path.normpath`` (not ``Path.resolve``) — we want the
        # symlink-preserving form so the resulting path stays
        # repo-root-relative even when ``data/<dataset>`` is a symlink
        # into the cluster v1 release tree.
        anchored = os.path.normpath(os.path.join(os.path.dirname(capture_path), capture_root))
    anchored = _make_repo_relative_when_possible(anchored)
    subdir = capture_data.get("videos_subdir") or "videos_crf24"
    return os.path.join(anchored, "{seq_name}", subdir)


def _derive_calibration(capture_path: str, capture_data: dict) -> str | None:
    """Build ``ma_cap.calibration`` from the capture's ``calib`` field.

    The capture JSON stores ``calib`` as a path relative to the capture
    file's parent dir (portable across machines). Resolve it the same
    way :func:`_derive_videos_dir` does, then make it repo-root-relative
    when possible. Returns ``None`` if the capture doesn't declare a
    calibration path.
    """
    calib = capture_data.get("calib") or ""
    if not calib:
        return None
    if os.path.isabs(calib):
        anchored = calib
    else:
        anchored = os.path.normpath(os.path.join(os.path.dirname(capture_path), calib))
    return _make_repo_relative_when_possible(anchored)


def _make_repo_relative_when_possible(path: str) -> str:
    """Rewrite ``path`` as repo-root-relative if it lives under the repo.

    Otherwise return it unchanged. The repo root is inferred from this
    module's location (``inference/`` is a top-level package).
    """
    from pathlib import Path
    repo_root = str(Path(__file__).resolve().parent.parent)
    try:
        rel = os.path.relpath(path, repo_root)
        if not rel.startswith(".."):
            return rel
    except ValueError:
        pass  # different drive on Windows, etc.
    return path


def _derive_dataset_name(capture_path: str) -> str:
    """Pick a sensible dataset_name when the preset doesn't specify one.

    Uses the capture file's basename (without ``.json``). For generically
    named files (``capture.json``, ``default.json``), falls back to the
    parent directory's name.
    """
    fname = os.path.basename(capture_path).replace(".json", "")
    if fname.lower() in ("capture", "default", ""):
        return os.path.basename(os.path.dirname(capture_path)) or fname
    return fname


def validate(cfg: dict) -> None:
    """Field-by-field check. Raises :class:`TaskConfigError` on the first issue."""
    errors: List[str] = []

    g = cfg.get("global")
    if not isinstance(g, dict):
        raise TaskConfigError("global: missing or not an object")

    capture_json = os.path.expanduser(os.path.expandvars(g.get("capture_json", "")))
    if not capture_json:
        errors.append("global.capture_json: required")
    elif not os.path.exists(capture_json):
        errors.append(f"global.capture_json: file not found: {capture_json}")

    if not g.get("out_dir"):
        errors.append("global.out_dir: required")

    if not g.get("dataset_name"):
        errors.append("global.dataset_name: required")

    seq_ids = g.get("seq_ids", [])
    if seq_ids and not all(isinstance(s, int) for s in seq_ids):
        errors.append("global.seq_ids: must be a list of integers")

    enabled_count = 0
    for step in ALL_STEPS:
        s = cfg.get(step)
        if not isinstance(s, dict):
            continue
        if not s.get("enabled"):
            continue
        enabled_count += 1

        engine = (s.get("engine") or "conda").lower()
        if engine not in VALID_ENGINES:
            errors.append(
                f"{step}.engine: {engine!r} not in {VALID_ENGINES}"
            )

        if not s.get("script"):
            errors.append(f"{step}.script: required when enabled")
        if not s.get("repo_path"):
            errors.append(f"{step}.repo_path: required when enabled")

        if engine == "apptainer" and not s.get("sif_path"):
            errors.append(f"{step}.sif_path: required for engine=apptainer")
        if engine == "docker" and not s.get("docker_image"):
            errors.append(f"{step}.docker_image: required for engine=docker")

        deps = s.get("dependencies", []) or []
        if not isinstance(deps, list):
            errors.append(f"{step}.dependencies: must be a list")
        else:
            for d in deps:
                if d not in ALL_STEPS:
                    errors.append(f"{step}.dependencies: unknown step {d!r}")

    if enabled_count == 0:
        errors.append("no steps are enabled (set <step>.enabled = true)")

    if errors:
        raise TaskConfigError(
            "run-config validation failed:\n  - " + "\n  - ".join(errors)
        )
