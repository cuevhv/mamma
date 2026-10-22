"""MAMMA environment bootstrap.

Single source of truth for installation paths (body models, weights,
priors) and any other env-driven config the pipeline consumes.

Resolution order, applied by :func:`bootstrap_env`:

1. :data:`DEFAULTS` — sensible relative paths under ``data/`` that work
   out-of-the-box for users following the documented data layout.
   Applied via ``os.environ.setdefault`` so any value already in the
   process environment (exported by the shell, set by a SLURM job, etc.)
   takes precedence over the in-code default.
2. ``.env.local`` (gitignored, optional) at the repo root — when present,
   loaded via ``python-dotenv`` with ``override=True``. Use this file
   for per-installation overrides and secrets (e.g. ``WANDB_API_KEY``).
3. Relative path values for the known ``MAMMA_*`` (plus ``BEDLAM_*``)
   keys are anchored to the repo root, so callers do not have to be
   run from a specific working directory.

The function is idempotent and cheap; call it at the top of every
entry point. Subprocesses spawned by the runner do **not** see the
``MAMMA_*`` keys (see :mod:`inference.engines`) — the runner translates
them into explicit ``--flag`` arguments at the boundary.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    # python-dotenv is optional: it is only used to load an (also optional)
    # .env.local. On a minimal python (e.g. a cluster login node with just
    # the stdlib) the package may be absent; the pipeline still runs, it just
    # skips the .env.local convenience. Every real MAMMA_* value has a default
    # or is passed explicitly, so nothing depends on dotenv being present.
    load_dotenv = None

from .assets import defaults_dict as _registry_defaults
from .assets import path_keys as _registry_path_keys


_REPO_ROOT = Path(__file__).resolve().parents[1]


# Keys that hold filesystem paths and should have relative values
# anchored to the repo root. Derived from :data:`inference.assets.ASSETS`
# so adding a new env var is a one-place change. The per-value HF-id
# skip is still done by ``_looks_like_hf_model_id`` below.
_PATH_KEYS = _registry_path_keys()


# Defaults that match the documented ``data/`` layout. Derived from
# :data:`inference.assets.ASSETS`. Users who follow the layout need no
# ``.env.local`` at all. ``None`` entries are env vars without a
# sensible default — they remain unset unless the user opts in.
#
# To change a default path, edit ``inference/assets.py:ASSETS``; this
# dict and the GUI panel's asset list both pick the change up.
DEFAULTS: Dict[str, Optional[str]] = _registry_defaults()


def repo_root() -> Path:
    """Return the absolute path to the repo root."""
    return _REPO_ROOT


def bootstrap_env() -> None:
    """Populate ``os.environ`` from defaults + optional ``.env.local``.

    Idempotent. Safe to call from any entry point.
    """
    # 1. In-code defaults. setdefault means an already-set env var wins.
    for key, value in DEFAULTS.items():
        if value is None:
            continue
        os.environ.setdefault(key, value)

    # 2. Optional user overrides from .env.local (skipped if dotenv is absent).
    local = _REPO_ROOT / ".env.local"
    if load_dotenv is not None and local.exists():
        load_dotenv(local, override=True)

    # 3. Anchor relative path values to the repo root.
    for key in _PATH_KEYS:
        value = os.environ.get(key)
        if not value:
            continue
        if _looks_like_hf_model_id(value):
            # e.g. "facebook/sam3" — a HuggingFace model id, not a filesystem
            # path. Leave it alone so consumers (segmentation/process_sequence.py)
            # can pass it straight to from_pretrained().
            continue
        p = Path(value)
        if p.is_absolute():
            continue
        os.environ[key] = str((_REPO_ROOT / p).resolve())

    # 4. v1.0.x set the YOLO weights here; the runner no longer passes them.
    if os.environ.get("MAMMA_YOLO_CHECKPOINT"):
        warnings.warn("MAMMA_YOLO_CHECKPOINT is ignored: ma_masks detects people with D-FINE-X. "
                      "To use YOLO, add --detector yolo --detector-checkpoint <file>.pt to the "
                      "ma_masks flags.", stacklevel=2)


def _looks_like_hf_model_id(value: str) -> bool:
    """Heuristic — is ``value`` a HuggingFace `<org>/<repo>` model id rather
    than a filesystem path?

    HF ids: exactly one slash, no leading dot/slash/tilde, no file extension
    on the final segment. Filesystem paths almost always violate at least
    one of these (extra slashes, a `.ext` suffix, an absolute prefix, etc.).
    Cheap-and-good-enough; the alternative is forcing users to spell
    ``hf:facebook/sam3`` which is more friction for the common case.
    """
    if not value or value.startswith(("/", "./", "../", "~")):
        return False
    if value.count("/") != 1:
        return False
    last = value.rsplit("/", 1)[-1]
    if "." in last:
        return False
    return True
