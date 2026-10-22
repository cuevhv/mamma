#!/usr/bin/env python
"""MAMMA V1 smoke test.

A standardised set of fast checks that the V1 configuration plumbing
(``inference.env.bootstrap_env``, engine env-stripping, runner
env-to-argv translation, optimization submodule refactor) did not
break the inference pipeline. Each check is short, self-contained,
and prints a single pass/fail line with timing.

Every invocation creates a timestamped directory under
``output/smoke_test/`` containing the full transcript, per-check
captured stdout/stderr, a machine-readable ``results.json``, and (for
the DAG-walk check) an isolated runner output tree at ``pipeline/``
that you can browse to verify the pipeline produced the expected
files. A ``latest`` symlink at ``output/smoke_test/latest`` always
points at the most recent run.

Usage::

    python scripts/smoke_test.py             # quick checks only (~14s)
    python scripts/smoke_test.py --full      # also walk the Breakdance DAG (~5min)
    python scripts/smoke_test.py --no-color
    python scripts/smoke_test.py -v          # verbose: print failure detail inline
    python scripts/smoke_test.py --list      # list checks without running
    python scripts/smoke_test.py --no-write  # do not create output/smoke_test/<ts>/

Exit code is 0 if every check passes, 1 otherwise. Checks that depend
on data the user has not yet populated are reported as ``SKIP`` (not
counted as failures). A ``KNOWN`` status marks a pre-existing failure
that is documented and outside this test's scope (e.g. a missing
weights file unrelated to V1).

The script is invoked from the repo root and expects the ``mamma``
conda env to be the active interpreter (it doesn't activate one
itself — see the docstring for docs/INSTALL.md guidance).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]
# Allow `import inference, ...` when invoked from anywhere.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Canonical fixture used by every builder-shape check and by the
# end-to-end DAG-walk. Breakdance quick preset: 4 cameras × 30 frames,
# exercises the videos workflow (ma_cap → ma_masks → ma_2d → ma_3d
# → ma_vis) with the new ma_cap-centric frame range plumbing.
#
# Two files now: a capture-independent preset and the capture it binds to.
# Run as: ``python -m inference run --preset <preset> --capture <capture>``.
_FIXTURE_PRESET = _REPO_ROOT / "configs" / "examples" / "presets" / "quick.yaml"
_FIXTURE_CAPTURE = _REPO_ROOT / "configs" / "examples" / "captures" / "140725_Breakdance.json"
_FIXTURE_SEQ = "140725_Breakdance_Improv_1_03684_03686_1"

# Backwards-compat alias for any check still grepping for _FIXTURE_TASK —
# the materialized (preset+capture) run-config now lives at a temp path
# computed on demand (see `_materialized_fixture_path`).
_FIXTURE_TASK = _FIXTURE_PRESET

_MATERIALIZED_PATH: Optional[Path] = None  # set lazily by _materialized_fixture_path()


def _load_fixture():
    """Load the canonical fixture as a bound run config.

    Materializes the preset against the capture in memory so the
    returned dict has ``global.capture_json`` and validates cleanly.

    Raises :class:`_Skip` if either file is absent (e.g. someone
    deleted ``configs/examples/`` or the ``data/`` symlinks aren't in
    place).
    """
    if not _FIXTURE_PRESET.exists():
        raise _Skip(f"preset fixture not present: {_FIXTURE_PRESET}")
    if not _FIXTURE_CAPTURE.exists():
        raise _Skip(f"capture fixture not present: {_FIXTURE_CAPTURE}")
    from inference import config
    return config.materialize_run_config(
        str(_FIXTURE_PRESET), str(_FIXTURE_CAPTURE),
    )


def _materialized_fixture_path() -> Path:
    """Materialize the fixture (preset+capture) to a temp file and return its path.

    Cached for the duration of the smoke run so subprocess invocations
    that expect a single fully-bound run config (e.g. ``doctor --task``)
    can be pointed at one file. Removed in :func:`main` cleanup.
    """
    global _MATERIALIZED_PATH
    if _MATERIALIZED_PATH and _MATERIALIZED_PATH.exists():
        return _MATERIALIZED_PATH
    if not _FIXTURE_PRESET.exists():
        raise _Skip(f"preset fixture not present: {_FIXTURE_PRESET}")
    if not _FIXTURE_CAPTURE.exists():
        raise _Skip(f"capture fixture not present: {_FIXTURE_CAPTURE}")
    from inference import config
    cfg = config.materialize_run_config(
        str(_FIXTURE_PRESET), str(_FIXTURE_CAPTURE),
    )
    # The shared quick.yaml is silent on cam_names AND seq_ids — those
    # are capture-coupled and live in the capture / runner-default
    # respectively. Smoke needs both pinned for predictable, fast
    # runs. Without these caps the DAG walk would run ALL sequences
    # (Breakdance has 30+) on ALL 32 cameras, ballooning past the
    # 900s timeout. Local to the smoke fixture; doesn't affect the
    # shipped preset.
    cap_cams = cfg.get("global", {}).get("cam_names") or []
    if len(cap_cams) > 4:
        cfg["global"]["cam_names"] = cap_cams[:4]
    cfg.setdefault("global", {}).setdefault("seq_ids", [0])
    import tempfile
    fd, path = tempfile.mkstemp(prefix="mamma_smoke_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=2)
    _MATERIALIZED_PATH = Path(path)
    return _MATERIALIZED_PATH


# Run-dir state. Set once in main(). Each check can opt in to writing
# captured subprocess logs and pipeline artifacts under it.
_RUN_DIR: Optional[Path] = None


def _set_run_dir(p: Optional[Path]) -> None:
    global _RUN_DIR
    _RUN_DIR = p


def _run_dir() -> Optional[Path]:
    return _RUN_DIR


def _slugify(name: str) -> str:
    """Filesystem-safe filename derived from a check name."""
    safe = [c if c.isalnum() else "_" for c in name]
    return "".join(safe).strip("_").lower()[:80] or "check"


def _capture_subprocess(check_name: str, cmd: list, *, cwd: Optional[Path] = None,
                        timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """Run a subprocess and persist stdout/stderr under the run dir.

    Returns the CompletedProcess so callers can branch on returncode.
    """
    rc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, timeout=timeout,
    )
    rd = _run_dir()
    if rd is not None:
        slug = _slugify(check_name)
        try:
            (rd / "subprocess").mkdir(parents=True, exist_ok=True)
            (rd / "subprocess" / f"{slug}.cmd.txt").write_text(
                " ".join(map(str, cmd)) + ("\n" if cwd else "")
                + (f"# cwd: {cwd}\n" if cwd else "")
                + f"# rc: {rc.returncode}\n"
            )
            (rd / "subprocess" / f"{slug}.stdout.log").write_text(rc.stdout or "")
            (rd / "subprocess" / f"{slug}.stderr.log").write_text(rc.stderr or "")
        except OSError:
            # Don't let logging failures break the check.
            pass
    return rc


# ─── Output formatting ────────────────────────────────────────────────────

class _Ansi:
    enabled = True
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    GREEN = "\x1b[32m"
    RED = "\x1b[31m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    CYAN = "\x1b[36m"

    @classmethod
    def disable(cls) -> None:
        cls.enabled = False
        for attr in ("RESET", "BOLD", "DIM", "GREEN", "RED", "YELLOW", "BLUE", "CYAN"):
            setattr(cls, attr, "")


def _color(s: str, code: str) -> str:
    return f"{code}{s}{_Ansi.RESET}" if _Ansi.enabled else s


# ─── Check infrastructure ─────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    fn: Callable[[], None]
    requires_full: bool = False
    description: str = ""


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "FAIL" | "SKIP" | "KNOWN"
    duration_s: float
    detail: Optional[str] = None


CHECKS: List[Check] = []


def check(name: str, *, requires_full: bool = False, description: str = ""):
    """Register a check. Used as a decorator."""
    def wrap(fn: Callable[[], None]) -> Callable[[], None]:
        CHECKS.append(Check(name=name, fn=fn, requires_full=requires_full, description=description))
        return fn
    return wrap


class _Skip(Exception):
    """Raise from a check body to mark it skipped (e.g. data not present)."""


class _Known(Exception):
    """Raise from a check body to mark it a documented pre-existing failure."""


def _format_status(status: str) -> str:
    if status == "PASS":
        return _color("PASS", _Ansi.GREEN)
    if status == "FAIL":
        return _color("FAIL", _Ansi.RED + _Ansi.BOLD)
    if status == "SKIP":
        return _color("SKIP", _Ansi.YELLOW)
    if status == "KNOWN":
        return _color("KNOWN", _Ansi.YELLOW)
    return status


def _run_one(c: Check, verbose: bool) -> CheckResult:
    start = time.monotonic()
    try:
        c.fn()
        dur = time.monotonic() - start
        return CheckResult(c.name, "PASS", dur)
    except _Skip as e:
        return CheckResult(c.name, "SKIP", time.monotonic() - start, detail=str(e))
    except _Known as e:
        return CheckResult(c.name, "KNOWN", time.monotonic() - start, detail=str(e))
    except AssertionError as e:
        msg = str(e) if str(e) else "assertion failed"
        if verbose:
            msg = msg + "\n" + traceback.format_exc()
        return CheckResult(c.name, "FAIL", time.monotonic() - start, detail=msg)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if verbose:
            msg = msg + "\n" + traceback.format_exc()
        return CheckResult(c.name, "FAIL", time.monotonic() - start, detail=msg)


# ─── Helpers ──────────────────────────────────────────────────────────────

# MAMMA_* env keys the test harness clears between checks. Must stay
# in sync with inference/assets.py:ASSETS — anything a check sets via
# os.environ.update() must be listed here, otherwise it leaks across
# checks (e.g. into the DAG walk, which would then run with a stale
# override).
_MAMMA_ENV_KEYS = (
    "MAMMA_SMPLX_LOCKHEAD_MODELS",
    "MAMMA_DOWNSAMPLED_VERTS_PKL",
    "MAMMA_MA2D_CHECKPOINT",
    "MAMMA_SAM2_CHECKPOINT",
    "MAMMA_BUN_MODELS",
    "MAMMA_PART_MESH_PATH",
)


@contextmanager
def _clean_mamma_env():
    """Temporarily clear MAMMA_* env vars; restore on exit."""
    saved = {k: os.environ[k] for k in _MAMMA_ENV_KEYS if k in os.environ}
    for k in _MAMMA_ENV_KEYS:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k in _MAMMA_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(saved)


def _have_dotenv_local() -> bool:
    return (_REPO_ROOT / ".env.local").exists()


# ─── Checks ───────────────────────────────────────────────────────────────

@check("import inference modules",
       description="The runner, engines, and steps subpackages import cleanly.")
def _c_import_runner():
    import inference  # noqa: F401
    from inference import config, runner, engines  # noqa: F401
    from inference.cli import run, run_step  # noqa: F401
    from inference.steps import ma_cap, ma_masks, ma_2d, ma_3d, ma_vis  # noqa: F401


@check("import env module",
       description="inference.env exposes bootstrap_env, DEFAULTS, repo_root.")
def _c_import_env():
    from inference.env import bootstrap_env, DEFAULTS, repo_root  # noqa: F401
    assert callable(bootstrap_env)
    assert isinstance(DEFAULTS, dict)
    assert callable(repo_root)


@check("DEFAULTS keys cover required MAMMA_* paths",
       description="The DEFAULTS dict must include every MAMMA_* the pipeline reads. "
                   "MAMMA_SAM2_CHECKPOINT is included for the sam2 branch; SAM 3 "
                   "self-resolves through HuggingFace Hub so there's no env var.")
def _c_defaults_shape():
    from inference.env import DEFAULTS
    required = {
        "MAMMA_SMPLX_LOCKHEAD_MODELS",
        "MAMMA_DOWNSAMPLED_VERTS_PKL",
        "MAMMA_MA2D_CHECKPOINT",
        "MAMMA_SAM2_CHECKPOINT",
    }
    missing = required - set(DEFAULTS)
    assert not missing, f"DEFAULTS missing required keys: {sorted(missing)}"


@check("bootstrap_env populates DEFAULTS into env",
       description="With no MAMMA_* set and no .env.local, bootstrap_env must populate DEFAULTS.")
def _c_bootstrap_no_env():
    from inference.env import bootstrap_env, DEFAULTS, repo_root, _looks_like_hf_model_id
    if _have_dotenv_local():
        raise _Skip(".env.local present — would mask DEFAULTS-only behavior")
    with _clean_mamma_env():
        bootstrap_env()
        for key, default in DEFAULTS.items():
            if default is None:
                # Optional; should not be set.
                assert key not in os.environ or not os.environ[key], (
                    f"{key} should remain unset for None default, "
                    f"got {os.environ.get(key)!r}"
                )
            else:
                val = os.environ.get(key)
                assert val, f"{key} not populated"
                if _looks_like_hf_model_id(val):
                    # HF model id (e.g. "facebook/sam3") — intentionally left
                    # unanchored so consumers pass it to from_pretrained().
                    continue
                # Anchored to repo root (absolute path).
                assert Path(val).is_absolute(), f"{key} not anchored: {val}"
                assert val.startswith(str(repo_root())), (
                    f"{key} anchored elsewhere: {val}"
                )


@check("bootstrap_env is idempotent",
       description="Calling bootstrap_env twice must not corrupt env state.")
def _c_bootstrap_idempotent():
    from inference.env import bootstrap_env
    if _have_dotenv_local():
        raise _Skip(".env.local present — would mask the test")
    with _clean_mamma_env():
        bootstrap_env()
        snapshot = {k: os.environ.get(k) for k in _MAMMA_ENV_KEYS}
        bootstrap_env()
        for k, v in snapshot.items():
            assert os.environ.get(k) == v, f"{k} changed on second bootstrap"


@check("bootstrap_env preserves pre-set env",
       description="setdefault semantics: an existing env var beats the in-code DEFAULT.")
def _c_bootstrap_preserves_user_env():
    from inference.env import bootstrap_env
    if _have_dotenv_local():
        raise _Skip(".env.local present — would override the user-set value")
    with _clean_mamma_env():
        os.environ["MAMMA_SMPLX_LOCKHEAD_MODELS"] = "/some/user/override"
        bootstrap_env()
        assert os.environ["MAMMA_SMPLX_LOCKHEAD_MODELS"] == "/some/user/override", (
            "bootstrap_env clobbered a pre-set MAMMA_*"
        )


@check("engine _child_env strips MAMMA_* but keeps PATH",
       description="Subprocesses spawned by the runner must not inherit MAMMA_* keys.")
def _c_child_env_strips_mamma():
    from inference.env import bootstrap_env
    from inference.engines import _child_env
    with _clean_mamma_env():
        bootstrap_env()
        child = _child_env()
        leaked = [k for k in child if k.startswith("MAMMA_")]
        assert leaked == [], f"MAMMA_* keys leaked to child env: {leaked}"
        assert "PATH" in child, "PATH unexpectedly stripped"


@check("fixture task config loads and validates",
       description="The canonical Breakdance quick task config loads via the runner without errors.")
def _c_fixture_config_loads():
    from inference import config
    cfg = _load_fixture()
    config.validate(cfg)


@check("preset.yaml + capture.json dispatch — both formats load",
       description="Phase 3 dispatch: load_run_config + materialize_run_config handle YAML presets and JSON captures.")
def _c_yaml_json_dispatch():
    from inference import config
    cfg = _load_fixture()  # materialized from YAML preset + JSON capture
    if _FIXTURE_CAPTURE.exists():
        cfg_json = config._parse(str(_FIXTURE_CAPTURE))
        assert "sequences" in cfg_json, f"capture JSON missing 'sequences': {_FIXTURE_CAPTURE}"
    # Sanity: the materialized config has the expected shape AND a
    # bound capture_json (proof the materializer wired the two files).
    assert "global" in cfg and "ma_cap" in cfg, (
        f"materialized run config missing required sections: keys={sorted(cfg.keys())}"
    )
    assert cfg["global"].get("capture_json"), (
        "materialize_run_config did not record capture_json under global."
    )


@check("materializer derives capture-coupled fields from the capture JSON",
       description="The shared quick.yaml preset is silent on cam_names, videos_dir, "
                   "and calibration; the materializer must derive them from the bound capture. "
                   "This guards the 1-preset / many-captures collapse.")
def _c_materializer_derivation():
    from inference import config
    cfg = _load_fixture()  # full.yaml or quick.yaml + Breakdance capture

    # 1. cam_names: derived from capture.cams (Breakdance has 32; the
    # capture JSON now carries them after the migration enriched it).
    cams = cfg["global"].get("cam_names") or []
    assert cams, "materializer did not populate global.cam_names from the capture"
    assert len(cams) >= 4, (
        f"expected at least 4 cameras for Breakdance, got {len(cams)}: {cams}"
    )

    # 2. videos_dir: derived from capture_root + videos_subdir.
    # ``{seq_name}`` is a template placeholder; we just verify the
    # surrounding shape since the actual data path is dataset-specific.
    videos_dir = cfg["ma_cap"].get("videos_dir") or ""
    assert "{seq_name}" in videos_dir, (
        f"ma_cap.videos_dir missing {{seq_name}} template: {videos_dir!r}"
    )
    assert "videos_crf24" in videos_dir, (
        f"ma_cap.videos_dir missing expected videos_subdir: {videos_dir!r}"
    )

    # 3. calibration: derived from capture.calib. Must resolve to a
    # file that actually exists on disk.
    calib = cfg["ma_cap"].get("calibration") or ""
    assert calib, "materializer did not populate ma_cap.calibration"
    calib_abs = calib if os.path.isabs(calib) else os.path.join(str(_REPO_ROOT), calib)
    assert os.path.isfile(calib_abs), (
        f"derived calibration path does not exist on disk: {calib_abs}"
    )


@check("ma_3d builder produces argv with required path flags",
       description="With DEFAULTS-only env, the builder must emit --smplx-models and --downsampled-verts.")
def _c_ma_3d_argv_has_paths():
    from inference.env import bootstrap_env
    from inference.steps.ma_3d import Ma3dBuilder
    # uses the canonical fixture via _load_fixture()
    if _have_dotenv_local():
        raise _Skip(".env.local present — would influence DEFAULTS test")
    with _clean_mamma_env():
        bootstrap_env()
        cfg = _load_fixture()
        b = Ma3dBuilder(cfg["ma_3d"], cfg["global"], "local")
        argv = b.python_argv(_FIXTURE_SEQ)
    for flag in ("--smplx-models", "--downsampled-verts"):
        assert flag in argv, f"argv missing required flag {flag!r}: {argv}"
    # Paths should be absolute (anchored at bootstrap).
    for flag in ("--smplx-models", "--downsampled-verts"):
        idx = argv.index(flag)
        val = argv[idx + 1]
        assert val.startswith("/"), f"{flag} not absolute: {val}"
    # Optional flags must NOT appear because the corresponding DEFAULTS are None.
    for opt_flag in ("--bun-models", "--part-mesh"):
        assert opt_flag not in argv, (
            f"{opt_flag} unexpectedly present without explicit env: {argv}"
        )


@check("ma_3d builder raises actionable error when required path missing",
       description="Without MAMMA_SMPLX_LOCKHEAD_MODELS, the builder must surface a clear error (not a deep KeyError).")
def _c_ma_3d_actionable_error():
    from inference.steps.ma_3d import Ma3dBuilder
    cfg = _load_fixture()
    with _clean_mamma_env():
        # Intentionally do NOT call bootstrap_env. All MAMMA_* unset.
        b = Ma3dBuilder(cfg["ma_3d"], cfg["global"], "local")
        try:
            b.python_argv(_FIXTURE_SEQ)
        except RuntimeError as e:
            msg = str(e)
            assert "MAMMA_SMPLX_LOCKHEAD_MODELS" in msg, (
                f"error message missing var name: {msg!r}"
            )
            assert ".env.local" in msg or "DEFAULTS" in msg, (
                f"error message missing remediation hint: {msg!r}"
            )
        else:
            raise AssertionError("expected RuntimeError, none raised")


def _scan_env_reads(root: Path, skip_files: set) -> List[str]:
    """Return file:line:source for any bare MAMMA_* env reads outside comments."""
    hits: List[str] = []
    for py in root.rglob("*.py"):
        if py.name in skip_files:
            continue
        rel = py.relative_to(root)
        try:
            text = py.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if 'os.environ["MAMMA_' in line or "os.environ.get(\"MAMMA_" in line:
                hits.append(f"{rel}:{i}: {line.strip()}")
    return hits


@check("optimization/ has no bare MAMMA_* env reads (excluding scope-out files)",
       description="Phase 2 acceptance: the pipeline path through optimization/ is env-free.")
def _c_no_env_reads_optimization():
    opt = _REPO_ROOT / "optimization"
    if not opt.exists():
        raise _Skip("optimization/ not present")
    # export_blender.py is the optional Blender export tool, off the pipeline path;
    # its MAMMA_BLENDER_BIN fallback is documented user-facing configuration.
    hits = _scan_env_reads(opt, skip_files={"paths_config.py", "export_blender.py"})
    assert not hits, "Bare MAMMA_* env reads remain:\n  " + "\n  ".join(hits)


@check("segmentation/ has no bare MAMMA_* env reads",
       description="Phase-4-ish acceptance: ma_masks paths flow via argv from the runner, not via env.")
def _c_no_env_reads_segmentation():
    seg = _REPO_ROOT / "segmentation"
    if not seg.exists():
        raise _Skip("segmentation/ not present")
    hits = _scan_env_reads(seg, skip_files=set())
    assert not hits, "Bare MAMMA_* env reads remain:\n  " + "\n  ".join(hits)


@check("ma_masks builder: no detector flag injected; --sam_checkpoint is version-aware",
       description="Five cases. (A) fixture default (sam2): no detector flag (the "
                   "default detector self-resolves through the HF cache); inject "
                   "--sam_checkpoint <MAMMA_SAM2_CHECKPOINT default, anchored>. "
                   "(B) override to sam3_prompt: no --sam_checkpoint (SAM 3 "
                   "self-resolves through HF cache). (C) sam2 + env override: "
                   "inject override value. (D) sam2 + preset supplies "
                   "--sam_checkpoint: no extra runner injection (preset wins). "
                   "(E) sam3_prompt + preset supplies --sam_checkpoint: same.")
def _c_ma_masks_argv_has_paths():
    import os
    from inference.env import bootstrap_env
    from inference.steps.ma_masks import MaMasksBuilder
    if _have_dotenv_local():
        raise _Skip(".env.local present — would influence DEFAULTS test")

    def _build_argv(extra_flags=None, extra_env=None):
        with _clean_mamma_env():
            bootstrap_env()
            if extra_env:
                os.environ.update(extra_env)
            cfg = _load_fixture()
            ma_cfg = dict(cfg["ma_masks"])
            if extra_flags is not None:
                # Append test-specific flags to the fixture's existing
                # `flags` list. The version resolver returns the LAST
                # --sam_version, so the appended value wins.
                ma_cfg["flags"] = list(cfg["ma_masks"].get("flags", [])) + list(extra_flags)
            b = MaMasksBuilder(ma_cfg, cfg["global"], "local")
            return b.python_argv(_FIXTURE_SEQ)

    # (A) Fixture defaults — quick.yaml uses --sam_version sam2. Runner
    # injects --sam_checkpoint from MAMMA_SAM2_CHECKPOINT (anchored).
    argv = _build_argv()
    assert not {"--yolo-checkpoint", "--detector-checkpoint"} & set(argv), (
        f"(A) runner must not inject a detector checkpoint: {argv}"
    )
    assert "--sam_checkpoint" in argv, (
        f"(A) sam2 fixture should trigger runner --sam_checkpoint injection: {argv}"
    )
    idx = argv.index("--sam_checkpoint")
    sam_val = argv[idx + 1]
    assert sam_val.startswith("/") and sam_val.endswith("sam2.1_hiera_large.pt"), (
        f"(A) sam2 default value wrong: {sam_val}"
    )

    # (B) Override the fixture's sam_version to sam3_prompt (last-wins).
    # SAM 3 self-resolves; runner must NOT inject --sam_checkpoint.
    argv = _build_argv(extra_flags=["--sam_version", "sam3_prompt"])
    assert "--sam_checkpoint" not in argv, (
        f"(B) sam3_prompt should NOT trigger runner --sam_checkpoint injection: {argv}"
    )

    # (C) sam2 + MAMMA_SAM2_CHECKPOINT override.
    argv = _build_argv(
        extra_env={"MAMMA_SAM2_CHECKPOINT": "/tmp/custom_sam2.pt"},
    )
    idx = argv.index("--sam_checkpoint")
    assert argv[idx + 1] == "/tmp/custom_sam2.pt", (
        f"(C) override not forwarded: argv[{idx+1}]={argv[idx+1]!r}"
    )

    # (D) sam2 (fixture default) + preset supplies --sam_checkpoint →
    # preset wins, runner doesn't add another.
    argv = _build_argv(
        extra_flags=["--sam_checkpoint", "/preset/sam2.pt"],
    )
    sam_count = sum(1 for a in argv if a == "--sam_checkpoint")
    assert sam_count == 1, f"(D) expected exactly 1 --sam_checkpoint, got {sam_count}: {argv}"
    idx = argv.index("--sam_checkpoint")
    assert argv[idx + 1] == "/preset/sam2.pt", (
        f"(D) preset value not preserved: argv[{idx+1}]={argv[idx+1]!r}"
    )

    # (E) Override to sam3_prompt + preset supplies --sam_checkpoint →
    # preset wins.
    argv = _build_argv(
        extra_flags=["--sam_version", "sam3_prompt", "--sam_checkpoint", "/preset/sam3.pt"],
    )
    sam_count = sum(1 for a in argv if a == "--sam_checkpoint")
    assert sam_count == 1, f"(E) expected exactly 1 --sam_checkpoint, got {sam_count}: {argv}"


@check("ma_masks builder needs no installation path",
       description="With every MAMMA_* unset the builder still builds: the person "
                   "detector self-resolves through the HF cache and the SAM "
                   "checkpoint has a per-version default.")
def _c_ma_masks_no_required_paths():
    from inference.steps.ma_masks import MaMasksBuilder
    cfg = _load_fixture()
    with _clean_mamma_env():
        # No bootstrap_env: every MAMMA_* unset.
        b = MaMasksBuilder(cfg["ma_masks"], cfg["global"], "local")
        b.python_argv(_FIXTURE_SEQ)


@check("builders emit --undistort when global.undistort is set",
       description="Phase 5 wiring: global.undistort propagates to ma_masks/ma_2d/ma_vis argv, never to ma_3d (no distortion math).")
def _c_undistort_flag_propagation():
    from inference.env import bootstrap_env
    from inference.steps.ma_masks import MaMasksBuilder
    from inference.steps.ma_2d import Ma2dBuilder
    from inference.steps.ma_3d import Ma3dBuilder
    from inference.steps.ma_vis import MaVisBuilder
    # uses the canonical fixture via _load_fixture()
    if _have_dotenv_local():
        raise _Skip(".env.local present — would influence DEFAULTS test")
    with _clean_mamma_env():
        bootstrap_env()
        cfg = _load_fixture()
        cfg["global"]["undistort"] = True
        cfg["global"]["calibration"] = "/tmp/calib.yaml"
        # Frame-reading steps should emit --undistort. Strip any per-step
        # `undistort: False` from the fixture so global.undistort wins.
        for step_name in ("ma_masks", "ma_2d", "ma_vis"):
            cfg[step_name].pop("undistort", None)
        for name, cls in [("ma_masks", MaMasksBuilder), ("ma_2d", Ma2dBuilder), ("ma_vis", MaVisBuilder)]:
            b = cls(cfg[name], cfg["global"], "local")
            argv = b.python_argv(_FIXTURE_SEQ)
            assert "--undistort" in argv, f"{name}: --undistort missing: {argv}"
        # ma_3d should NOT emit --undistort (no distortion math; flag would be a no-op):
        b = Ma3dBuilder(cfg["ma_3d"], cfg["global"], "local")
        argv = b.python_argv(_FIXTURE_SEQ)
        assert "--undistort" not in argv, f"ma_3d: --undistort should not be emitted: {argv}"


@check("step.undistort overrides global.undistort=false",
       description="Per-step override: setting ma_2d.undistort=true while global.undistort=false should emit --undistort on ma_2d only.")
def _c_undistort_per_step_override():
    from inference.env import bootstrap_env
    from inference.steps.ma_masks import MaMasksBuilder
    from inference.steps.ma_2d import Ma2dBuilder
    # uses the canonical fixture via _load_fixture()
    if _have_dotenv_local():
        raise _Skip(".env.local present — would influence DEFAULTS test")
    with _clean_mamma_env():
        bootstrap_env()
        cfg = _load_fixture()
        cfg["global"]["undistort"] = False
        cfg["ma_2d"]["undistort"] = True
        b2 = Ma2dBuilder(cfg["ma_2d"], cfg["global"], "local")
        bm = MaMasksBuilder(cfg["ma_masks"], cfg["global"], "local")
        argv_2d = b2.python_argv(_FIXTURE_SEQ)
        argv_masks = bm.python_argv(_FIXTURE_SEQ)
        assert "--undistort" in argv_2d, f"ma_2d should have step-level --undistort: {argv_2d}"
        assert "--undistort" not in argv_masks, f"ma_masks should not have --undistort: {argv_masks}"


@check("ma_masks builder translates global.videos_dir to --videos_dir argv",
       description="Phase 1: setting global.videos_dir routes through MaMasksBuilder._frame_source_flags into argv.")
def _c_ma_masks_videos_dir_translates():
    from inference.env import bootstrap_env
    from inference.steps.ma_masks import MaMasksBuilder
    # uses the canonical fixture via _load_fixture()
    if _have_dotenv_local():
        raise _Skip(".env.local present — would influence DEFAULTS test")
    with _clean_mamma_env():
        bootstrap_env()
        cfg = _load_fixture()
        cfg["global"]["videos_dir"] = "/tmp/fixture_videos"
        b = MaMasksBuilder(cfg["ma_masks"], cfg["global"], "local")
        argv = b.python_argv(_FIXTURE_SEQ)
    assert "--videos_dir" in argv, f"--videos_dir missing: {argv}"
    idx = argv.index("--videos_dir")
    assert argv[idx + 1] == "/tmp/fixture_videos", f"unexpected value: {argv[idx+1]!r}"
    assert "--images_root_dir" not in argv, f"both flags emitted: {argv}"


@check("ma_masks builder raises on videos_dir + images_root_dir mutex violation",
       description="Setting both videos_dir and images_root_dir must raise an actionable RuntimeError.")
def _c_ma_masks_mutex_error():
    from inference.steps.ma_masks import MaMasksBuilder
    cfg = _load_fixture()
    cfg["global"]["videos_dir"] = "/tmp/v"
    cfg["global"]["images_root_dir"] = "/tmp/i"
    with _clean_mamma_env():
        b = MaMasksBuilder(cfg["ma_masks"], cfg["global"], "local")
        try:
            b.python_argv(_FIXTURE_SEQ)
        except RuntimeError as e:
            msg = str(e)
            assert "mutually exclusive" in msg, f"unexpected message: {msg!r}"
        else:
            raise AssertionError("expected RuntimeError, none raised")


@check("ma_2d builder resolves --weights with task.json > env precedence",
       description="task.json `weights:` wins; if absent, MAMMA_MA2D_CHECKPOINT is the fallback; if neither, RuntimeError.")
def _c_ma_2d_weights_precedence():
    from inference.env import bootstrap_env
    from inference.steps.ma_2d import Ma2dBuilder
    # uses the canonical fixture via _load_fixture()
    if _have_dotenv_local():
        raise _Skip(".env.local present — would influence env-fallback test")
    cfg = _load_fixture()
    pinned = cfg["ma_2d"].get("weights")
    assert pinned, "fixture task should pin ma_2d.weights for this test"

    # 1. task.json wins when both are set. The fixture pins `weights`
    # as a relative path; the builder anchors it to repo root, so we
    # compare against the anchored value.
    expected_anchored = pinned if os.path.isabs(pinned) else str(_REPO_ROOT / pinned)
    with _clean_mamma_env():
        bootstrap_env()  # sets MAMMA_* from DEFAULTS
        os.environ["MAMMA_MA2D_CHECKPOINT"] = "/tmp/should-not-win.ckpt"
        b = Ma2dBuilder(cfg["ma_2d"], cfg["global"], "local")
        argv = b.python_argv(_FIXTURE_SEQ)
    idx = argv.index("--weights")
    assert argv[idx + 1] == expected_anchored, (
        f"task.json `weights:` should win, got {argv[idx + 1]!r}, expected {expected_anchored!r}"
    )

    # 2. env is the fallback when task.json omits `weights:`.
    cfg_no_weights = json.loads(json.dumps(cfg))
    cfg_no_weights["ma_2d"].pop("weights", None)
    with _clean_mamma_env():
        # Builder also requires MAMMA_DOWNSAMPLED_VERTS_PKL (asset path);
        # supply it so the test focuses on the weights-resolution branch.
        os.environ["MAMMA_DOWNSAMPLED_VERTS_PKL"] = "/tmp/verts_512.pkl"
        os.environ["MAMMA_MA2D_CHECKPOINT"] = "/tmp/from-env.ckpt"
        b = Ma2dBuilder(cfg_no_weights["ma_2d"], cfg_no_weights["global"], "local")
        argv = b.python_argv(_FIXTURE_SEQ)
    idx = argv.index("--weights")
    assert argv[idx + 1] == "/tmp/from-env.ckpt", (
        f"env fallback should resolve, got {argv[idx + 1]!r}"
    )

    # 3. Neither weights source set: actionable RuntimeError that names both.
    with _clean_mamma_env():
        # Supply the unrelated required path so the weights error surfaces first.
        os.environ["MAMMA_DOWNSAMPLED_VERTS_PKL"] = "/tmp/verts_512.pkl"
        b = Ma2dBuilder(cfg_no_weights["ma_2d"], cfg_no_weights["global"], "local")
        try:
            b.python_argv(_FIXTURE_SEQ)
        except RuntimeError as e:
            msg = str(e)
            assert "MAMMA_MA2D_CHECKPOINT" in msg and "weights" in msg, (
                f"error message should mention both sources: {msg!r}"
            )
        else:
            raise AssertionError("expected RuntimeError when neither source is set")


@check("ma_cap builder translates global.videos_dir to --videos_dir argv",
       description="Phase 3: setting global.videos_dir + global.calibration routes ma_cap into video-ingest mode.")
def _c_ma_cap_videos_dir_translates():
    from inference.steps.ma_cap import MaCapBuilder
    cfg = _load_fixture()
    cfg["global"]["videos_dir"] = "/tmp/fixture_videos"
    cfg["global"]["calibration"] = "/tmp/fixture_calib.yaml"
    b = MaCapBuilder(cfg["ma_cap"], cfg["global"], "local")
    argv = b.python_argv(_FIXTURE_SEQ)
    assert "--videos_dir" in argv, f"--videos_dir missing: {argv}"
    assert "--calibration" in argv, f"--calibration missing: {argv}"
    assert "--json" not in argv, f"--json should be skipped in standalone mode: {argv}"


@check("ma_3d builder translates global.videos_dir + calibration to argv",
       description="Phase 4: standalone ma_3d emits --videos_dir + --calibration and skips --ma_cap_dir.")
def _c_ma_3d_videos_dir_translates():
    from inference.env import bootstrap_env
    from inference.steps.ma_3d import Ma3dBuilder
    # uses the canonical fixture via _load_fixture()
    if _have_dotenv_local():
        raise _Skip(".env.local present — would influence DEFAULTS test")
    with _clean_mamma_env():
        bootstrap_env()
        cfg = _load_fixture()
        cfg["global"]["videos_dir"] = "/tmp/fixture_videos"
        cfg["global"]["calibration"] = "/tmp/fixture_calib.yaml"
        b = Ma3dBuilder(cfg["ma_3d"], cfg["global"], "local")
        argv = b.python_argv(_FIXTURE_SEQ)
    assert "--videos_dir" in argv, f"--videos_dir missing: {argv}"
    assert "--calibration" in argv, f"--calibration missing: {argv}"
    assert "--ma_cap_dir" not in argv, f"--ma_cap_dir should be skipped: {argv}"


@check("ma_vis builder translates global.videos_dir + calibration to argv",
       description="Phase 5: standalone ma_vis emits --videos_dir + --calibration + --cam_names and skips --ma_cap_dir.")
def _c_ma_vis_videos_dir_translates():
    from inference.steps.ma_vis import MaVisBuilder
    cfg = _load_fixture()
    cfg["global"]["videos_dir"] = "/tmp/fixture_videos"
    cfg["global"]["calibration"] = "/tmp/fixture_calib.yaml"
    b = MaVisBuilder(cfg["ma_vis"], cfg["global"], "local")
    argv = b.python_argv(_FIXTURE_SEQ)
    assert "--videos_dir" in argv, f"--videos_dir missing: {argv}"
    assert "--calibration" in argv, f"--calibration missing: {argv}"
    assert "--ma_cap_dir" not in argv, f"--ma_cap_dir should be skipped: {argv}"
    assert "--cam_names" in argv, f"--cam_names required in ma_vis standalone: {argv}"


@check("ma_2d builder translates global.videos_dir to --videos_dir argv",
       description="Phase 2: setting global.videos_dir routes through Ma2dBuilder._frame_source_flags into argv, skipping --img_folder.")
def _c_ma_2d_videos_dir_translates():
    from inference.env import bootstrap_env
    from inference.steps.ma_2d import Ma2dBuilder
    # uses the canonical fixture via _load_fixture()
    if _have_dotenv_local():
        raise _Skip(".env.local present — would influence DEFAULTS test")
    with _clean_mamma_env():
        bootstrap_env()
        cfg = _load_fixture()
        cfg["global"]["videos_dir"] = "/tmp/fixture_videos"
        b = Ma2dBuilder(cfg["ma_2d"], cfg["global"], "local")
        argv = b.python_argv(_FIXTURE_SEQ)
    assert "--videos_dir" in argv, f"--videos_dir missing: {argv}"
    idx = argv.index("--videos_dir")
    assert argv[idx + 1] == "/tmp/fixture_videos", f"unexpected value: {argv[idx+1]!r}"
    assert "--img_folder" not in argv, f"--img_folder should be skipped in standalone mode: {argv}"


@check("compileall inference/",
       description="Byte-compile every module under inference/.")
def _c_compileall_inference():
    rc = _capture_subprocess(
        "compileall inference",
        [sys.executable, "-m", "compileall", "-q", "inference/"],
        cwd=_REPO_ROOT,
    )
    assert rc.returncode == 0, f"compileall failed:\n{rc.stdout}\n{rc.stderr}"


@check("compileall optimization/",
       description="Byte-compile every module under optimization/.")
def _c_compileall_optimization():
    rc = _capture_subprocess(
        "compileall optimization",
        [sys.executable, "-m", "compileall", "-q", "optimization/"],
        cwd=_REPO_ROOT,
    )
    assert rc.returncode == 0, f"compileall failed:\n{rc.stdout}\n{rc.stderr}"


@check("python -m inference doctor (no task)",
       description="Phase 5: doctor returns 0 with no task argument when the env table is healthy.")
def _c_doctor_no_task():
    rc = _capture_subprocess(
        "inference doctor",
        [sys.executable, "-m", "inference", "doctor"],
        cwd=_REPO_ROOT, timeout=20,
    )
    assert rc.returncode == 0, f"doctor exited {rc.returncode}:\n{rc.stdout}\n{rc.stderr}"
    assert "PASS" in rc.stdout, f"doctor stdout missing PASS marker:\n{rc.stdout!r}"


@check("python -m inference doctor --task <materialized fixture>",
       description="Phase 5: doctor walks the materialized run config and reports ma_2d.weights source.")
def _c_doctor_with_task():
    bound = _materialized_fixture_path()
    rc = _capture_subprocess(
        "inference doctor --task",
        [sys.executable, "-m", "inference", "doctor", "--task", str(bound)],
        cwd=_REPO_ROOT, timeout=20,
    )
    assert rc.returncode == 0, f"doctor --task exited {rc.returncode}:\n{rc.stdout}\n{rc.stderr}"
    assert "ma_2d.weights" in rc.stdout, f"doctor stdout missing ma_2d.weights row:\n{rc.stdout!r}"
    assert "source=task.json" in rc.stdout, (
        f"doctor should report task.json as source for ma_2d.weights:\n{rc.stdout!r}"
    )


@check("python -m inference --help",
       description="The runner CLI responds to --help without crashing.")
def _c_inference_help():
    rc = _capture_subprocess(
        "inference --help",
        [sys.executable, "-m", "inference", "--help"],
        cwd=_REPO_ROOT, timeout=20,
    )
    assert rc.returncode == 0, f"--help exited {rc.returncode}:\n{rc.stderr}"
    assert "run" in rc.stdout and "run-step" in rc.stdout, f"unexpected help: {rc.stdout!r}"


@check("optimization/run_ma_3d.py --help lists new path flags",
       description="The argparse refactor must expose the path flags in --help output.")
def _c_ma_3d_help_flags():
    rc = _capture_subprocess(
        "run_ma_3d --help",
        [sys.executable, "run_ma_3d.py", "--help"],
        cwd=_REPO_ROOT / "optimization", timeout=30,
    )
    assert rc.returncode == 0, f"--help exited {rc.returncode}:\n{rc.stderr}"
    expected = ["--smplx-models", "--downsampled-verts",
                "--bun-models", "--part-mesh"]
    missing = [f for f in expected if f not in rc.stdout]
    assert not missing, f"--help missing flags: {missing}"


@check("PathsConfig field set agrees with registry's ma_3d CLI flags",
       description="Phase 6 drift-guard: build an argparse.Namespace from "
                   "the central registry's ma_3d flag list and pass it to "
                   "optimization.utils.paths_config.PathsConfig.from_args. "
                   "Any rename of a CLI flag on the inference side that "
                   "isn't mirrored in PathsConfig.from_args surfaces here "
                   "as an AttributeError instead of in production.")
def _c_pathsconfig_drift_guard():
    import argparse
    import sys as _sys
    # PathsConfig lives in optimization/utils — add to path explicitly
    # rather than depending on the submodule's own sys.path setup.
    opt_dir = _REPO_ROOT / "optimization"
    if not (opt_dir / "utils" / "paths_config.py").exists():
        raise _Skip("optimization/utils/paths_config.py not present")
    if str(opt_dir) not in _sys.path:
        _sys.path.insert(0, str(opt_dir))
    from utils.paths_config import PathsConfig  # type: ignore

    from inference.assets import (
        step_argv_translation, step_optional_translation,
    )
    required = step_argv_translation("ma_3d")
    optional = step_optional_translation("ma_3d")

    # Build a fake Namespace with each CLI flag's argparse dest set to
    # a sentinel. argparse converts ``--smplx-models`` → ``smplx_models``.
    ns_kwargs = {}
    for env_key, cli_flag in (*required, *optional):
        dest = cli_flag.lstrip("-").replace("-", "_")
        ns_kwargs[dest] = f"<sentinel:{dest}>"
    ns = argparse.Namespace(**ns_kwargs)

    # If a flag was renamed in the registry but not in PathsConfig.from_args,
    # the next call raises AttributeError naming the missing attribute.
    cfg = PathsConfig.from_args(ns)

    # Every dataclass field must be populated. None or empty string
    # would indicate from_args read the wrong attribute or skipped a
    # flag the registry now requires.
    for f_name in ("smplx_lockhead_models", "downsampled_verts_pkl"):
        v = getattr(cfg, f_name)
        assert v, (
            f"PathsConfig.{f_name} is empty after from_args; registry "
            f"declares ma_3d requires it but PathsConfig may not be "
            f"reading the right argparse dest."
        )


@check("committed .env.example matches dump_env_example() from registry",
       description="Phase 8 drift-guard: the .env.example file at the repo "
                   "root is generated from inference/assets.py:ASSETS. If "
                   "anyone hand-edits .env.example or renames an env key "
                   "without regenerating, this check fails with a clear "
                   "diff. Regenerate with: "
                   "`python -m inference dump-env-example -o .env.example`.")
def _c_env_example_drift_guard():
    from inference.assets import dump_env_example
    expected = dump_env_example()
    committed_path = _REPO_ROOT / ".env.example"
    if not committed_path.exists():
        raise AssertionError(
            ".env.example is missing from the repo root. Generate it with "
            "`python -m inference dump-env-example -o .env.example`."
        )
    actual = committed_path.read_text(encoding="utf-8")
    if actual != expected:
        # Show a brief diff hint, not the full content (which can be long).
        diff_lines = []
        a_lines, e_lines = actual.splitlines(), expected.splitlines()
        n = max(len(a_lines), len(e_lines))
        for i in range(n):
            al = a_lines[i] if i < len(a_lines) else "<EOF>"
            el = e_lines[i] if i < len(e_lines) else "<EOF>"
            if al != el:
                diff_lines.append(f"  line {i+1}: committed={al!r}")
                diff_lines.append(f"           expected={el!r}")
                if len(diff_lines) >= 6:
                    diff_lines.append("  …")
                    break
        raise AssertionError(
            "Committed .env.example is out of date relative to "
            "inference/assets.py:ASSETS. Regenerate it:\n"
            "  python -m inference dump-env-example -o .env.example\n"
            "Differences:\n" + "\n".join(diff_lines)
        )


@check("Breakdance quick DAG walk (isolated output dir)",
       description="Runs the canonical Breakdance quick preset end-to-end "
                   "(4 cams × 30 frames, videos workflow). Materializes a "
                   "smoke-specific run config (preset + capture + cam_names "
                   "subset to the first 4 cams) and feeds it via --task, so "
                   "the smoke completes in ~5 min even though quick.yaml "
                   "itself is capture-independent and would otherwise pick "
                   "up all 32 cams from the capture's `cams` field.",
       requires_full=False)
def _c_dag_walk_dry():
    if not _FIXTURE_PRESET.exists():
        raise _Skip(f"{_FIXTURE_PRESET} not present")
    if not _FIXTURE_CAPTURE.exists():
        raise _Skip(f"{_FIXTURE_CAPTURE} not present")
    bound = _materialized_fixture_path()  # already cam-subsetted to 4
    rd = _run_dir()
    cmd = [sys.executable, "-m", "inference", "run",
           "--task", str(bound),
           "--out-tag", "smoke", "-v"]
    if rd is not None:
        # Force runner output to land under this smoke run, not under
        # the global ./output. Lets the user inspect exactly what this
        # invocation produced.
        cmd += ["--out-dir", str(rd / "pipeline")]
    rc = _capture_subprocess("dag walk", cmd, cwd=_REPO_ROOT, timeout=900)
    if rc.returncode != 0:
        if "DAG completed with failures" in rc.stdout or "DAG completed with failures" in rc.stderr:
            raise _Known(
                "DAG completed with downstream failures (likely data gaps); "
                "the runner itself walked cleanly. Inspect "
                f"{rd / 'subprocess' / 'dag_walk.stderr.log' if rd else 'output/logs/jobs/'}"
            )
        raise AssertionError(f"runner crashed with rc={rc.returncode}:\n{rc.stderr[-2000:]}")


@check("ma_cap artifacts exist after DAG walk",
       description="Following the DAG walk, the ma_cap step should have written gt/*.npz files into the isolated pipeline dir.")
def _c_ma_cap_artifacts():
    rd = _run_dir()
    if rd is None:
        raise _Skip("--no-write: no run dir to inspect")
    pipeline = rd / "pipeline"
    if not pipeline.exists():
        raise _Skip("DAG walk did not run (or used a different out dir)")
    # ma_cap writes <out>/ma_cap/<tag>/<dataset>/<seq>/gt/{global,<cam>}.npz
    matches = list(pipeline.rglob("ma_cap/**/gt/*.npz"))
    assert matches, (
        f"No ma_cap NPZ artifacts under {pipeline}. "
        "Either ma_cap did not run or wrote elsewhere."
    )
    # Sanity: must have a global.npz and at least one per-cam.
    has_global = any(p.name == "global.npz" for p in matches)
    has_cam = any(p.name != "global.npz" for p in matches)
    assert has_global, f"ma_cap did not write global.npz under {pipeline}"
    assert has_cam, f"ma_cap did not write any per-camera NPZ under {pipeline}"


@check("Breakdance quick pipeline --force run",
       requires_full=True,
       description="(--full only) Re-drive the canonical Breakdance quick "
                   "preset end-to-end with --force, ignoring DONE sentinels.")
def _c_dag_walk_full():
    if not _FIXTURE_PRESET.exists():
        raise _Skip(f"{_FIXTURE_PRESET} not present")
    if not _FIXTURE_CAPTURE.exists():
        raise _Skip(f"{_FIXTURE_CAPTURE} not present")
    bound = _materialized_fixture_path()  # cam-subsetted to 4
    rd = _run_dir()
    cmd = [sys.executable, "-m", "inference", "run",
           "--task", str(bound),
           "--out-tag", "smoke", "--force", "-v"]
    if rd is not None:
        cmd += ["--out-dir", str(rd / "pipeline")]
    rc = _capture_subprocess("dag walk full", cmd, cwd=_REPO_ROOT, timeout=1800)
    if rc.returncode != 0:
        raise AssertionError(f"runner rc={rc.returncode}:\n{rc.stderr[-3000:]}")


# ─── Main ─────────────────────────────────────────────────────────────────

def _parse_args(argv: List[str]) -> "argparse.Namespace":
    import argparse
    p = argparse.ArgumentParser(
        prog="scripts/smoke_test.py",
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    p.add_argument("--full", action="store_true",
                   help="Run end-to-end pipeline checks (slower).")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI color in output.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print traceback on failure.")
    p.add_argument("--list", action="store_true",
                   help="List checks without running anything.")
    p.add_argument("--only", default=None,
                   help="Run only checks whose name contains this substring.")
    p.add_argument("--no-write", action="store_true",
                   help="Do not create output/smoke_test/<timestamp>/ "
                        "(useful for fast in-place re-runs).")
    p.add_argument("--out-base", default=None,
                   help="Override the base directory for per-run output "
                        "(default: <repo>/output/smoke_test).")
    return p.parse_args(argv)


def _create_run_dir(out_base: Path) -> Path:
    """Create a fresh timestamped run dir and update the `latest` symlink."""
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run = out_base / ts
    suffix = 0
    while run.exists():
        suffix += 1
        run = out_base / f"{ts}_{suffix:02d}"
    run.mkdir(parents=True)
    # Maintain a `latest` symlink for convenience.
    latest = out_base / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run.name, target_is_directory=True)
    except (OSError, NotImplementedError):
        pass  # filesystem doesn't support symlinks; the run dir still exists
    return run


def _print_header(args, run_dir: Optional[Path]) -> None:
    when = time.strftime("%Y-%m-%d %H:%M:%S")
    title = _color("MAMMA V1 smoke test", _Ansi.BOLD)
    print(f"{title}  {_color(when, _Ansi.DIM)}")
    venv = os.environ.get("CONDA_DEFAULT_ENV") or "?"
    print(_color(f"  conda env: {venv}   python: {sys.executable}", _Ansi.DIM))
    print(_color(f"  repo:      {_REPO_ROOT}", _Ansi.DIM))
    mode = "full" if args.full else "quick"
    print(_color(f"  mode:      {mode}", _Ansi.DIM))
    if run_dir is not None:
        print(_color(f"  run dir:   {run_dir}", _Ansi.CYAN))
    else:
        print(_color("  run dir:   (--no-write)", _Ansi.DIM))
    print()


def _print_result(idx: int, total: int, result: CheckResult, verbose: bool) -> None:
    name_w = 56
    name = result.name
    if len(name) > name_w:
        name = name[: name_w - 1] + "…"
    dots = "." * max(2, name_w - len(name))
    tag = _format_status(result.status)
    dur = _color(f"({result.duration_s:5.2f}s)", _Ansi.DIM)
    print(f"  [{idx:>2}/{total}] {name} {_color(dots, _Ansi.DIM)} {tag} {dur}")
    if result.detail and (verbose or result.status == "FAIL"):
        for line in result.detail.splitlines():
            print(f"          {_color(line, _Ansi.DIM)}")


def _strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences for plain-text persistence."""
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _write_artifacts(run_dir: Path, args, results: List[CheckResult], total_s: float,
                     transcript: List[str]) -> None:
    """Persist summary.txt + results.json after the run."""
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    known = sum(1 for r in results if r.status == "KNOWN")

    summary_path = run_dir / "summary.txt"
    summary_path.write_text("\n".join(_strip_ansi(line) for line in transcript) + "\n")

    results_payload = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "full" if args.full else "quick",
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "repo_root": str(_REPO_ROOT),
        "total_seconds": round(total_s, 3),
        "counts": {"passed": passed, "failed": failed, "skipped": skipped, "known": known},
        "checks": [
            {
                "name": r.name,
                "status": r.status,
                "duration_seconds": round(r.duration_s, 3),
                "detail": r.detail,
            }
            for r in results
        ],
    }
    (run_dir / "results.json").write_text(json.dumps(results_payload, indent=2) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.no_color or not sys.stdout.isatty():
        _Ansi.disable()

    if args.list:
        print(_color("MAMMA V1 smoke test — registered checks", _Ansi.BOLD))
        for i, c in enumerate(CHECKS, 1):
            tag = "[full] " if c.requires_full else ""
            print(f"  {i:>2}. {tag}{c.name}")
            if c.description:
                print(_color(f"      {c.description}", _Ansi.DIM))
        return 0

    runnable = [c for c in CHECKS if args.full or not c.requires_full]
    if args.only:
        runnable = [c for c in runnable if args.only in c.name]
        if not runnable:
            print(_color(f"No checks match --only {args.only!r}", _Ansi.RED))
            return 1

    # Set up per-run output dir unless suppressed.
    run_dir: Optional[Path] = None
    if not args.no_write:
        out_base = Path(args.out_base) if args.out_base else (_REPO_ROOT / "output" / "smoke_test")
        out_base.mkdir(parents=True, exist_ok=True)
        run_dir = _create_run_dir(out_base)
    _set_run_dir(run_dir)

    transcript: List[str] = []

    def emit(line: str = "") -> None:
        transcript.append(line)
        print(line)

    # Re-implement _print_header/_print_result to capture for transcript.
    when = time.strftime("%Y-%m-%d %H:%M:%S")
    emit(f"{_color('MAMMA V1 smoke test', _Ansi.BOLD)}  {_color(when, _Ansi.DIM)}")
    venv = os.environ.get("CONDA_DEFAULT_ENV") or "?"
    emit(_color(f"  conda env: {venv}   python: {sys.executable}", _Ansi.DIM))
    emit(_color(f"  repo:      {_REPO_ROOT}", _Ansi.DIM))
    emit(_color(f"  mode:      {'full' if args.full else 'quick'}", _Ansi.DIM))
    if run_dir is not None:
        emit(_color(f"  run dir:   {run_dir}", _Ansi.CYAN))
    else:
        emit(_color("  run dir:   (--no-write)", _Ansi.DIM))
    emit()

    started = time.monotonic()
    results: List[CheckResult] = []
    for i, c in enumerate(runnable, 1):
        r = _run_one(c, verbose=args.verbose)
        # Build the result line, also capturing it.
        name_w = 56
        name = r.name if len(r.name) <= name_w else r.name[: name_w - 1] + "…"
        dots = "." * max(2, name_w - len(name))
        line = (
            f"  [{i:>2}/{len(runnable)}] {name} "
            f"{_color(dots, _Ansi.DIM)} {_format_status(r.status)} "
            f"{_color(f'({r.duration_s:5.2f}s)', _Ansi.DIM)}"
        )
        emit(line)
        if r.detail and (args.verbose or r.status == "FAIL"):
            for dl in r.detail.splitlines():
                emit(f"          {_color(dl, _Ansi.DIM)}")
        results.append(r)
    total = time.monotonic() - started

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    known = sum(1 for r in results if r.status == "KNOWN")

    emit()
    summary = (
        f"{_color(str(passed), _Ansi.GREEN)} passed · "
        f"{_color(str(failed), _Ansi.RED if failed else _Ansi.DIM)} failed · "
        f"{_color(str(skipped), _Ansi.YELLOW if skipped else _Ansi.DIM)} skipped · "
        f"{_color(str(known), _Ansi.YELLOW if known else _Ansi.DIM)} known · "
        f"total {total:.1f}s"
    )
    emit("  " + summary)

    if failed:
        emit()
        emit(_color("FAIL", _Ansi.RED + _Ansi.BOLD) + " — V1 smoke detected breakage.")
        if not args.verbose:
            emit(_color("  Re-run with -v for traceback detail.", _Ansi.DIM))
    elif known:
        emit(_color(
            "  PASS — V1 plumbing intact. Documented pre-existing failures noted above.",
            _Ansi.GREEN,
        ))
    else:
        emit(_color("  PASS — every check green.", _Ansi.GREEN))

    if run_dir is not None:
        _write_artifacts(run_dir, args, results, total, transcript)
        emit("")
        emit(_color(f"  artifacts: {run_dir}", _Ansi.CYAN))
        emit(_color(f"  latest:    {_REPO_ROOT / 'output' / 'smoke_test' / 'latest'}", _Ansi.DIM))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
