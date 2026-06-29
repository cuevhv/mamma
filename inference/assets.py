"""Central registry of MAMMA installation assets.

Single source of truth for the per-installation paths the inference
pipeline consumes. Every downstream module that needs to know the env
keys, default locations, or per-step CLI flags derives its data
structures from :data:`ASSETS` at module-load time rather than
restating them:

* :mod:`inference.env`'s ``DEFAULTS`` and ``_PATH_KEYS``
* :mod:`inference.steps.*`'s ``_REQUIRED_ENV_FLAGS`` / ``_OPTIONAL_ENV_FLAGS``
* :mod:`inference.cli.doctor`'s per-step requirement table
* :mod:`gui.backend.data_readiness`'s ``ASSETS`` (the GUI panel)

The registry is a *leaf* module — it imports nothing from elsewhere in
``inference/`` or ``gui/``. This is deliberate so the rest of the
codebase can depend on it without circular-import risk.

Adding a new installation asset is a one-place change: append a record
to :data:`ASSETS` and every consumer above picks it up automatically.

Each record bundles three concerns:

* **Identity**: ``id``, ``label``, ``purpose``, ``group`` — descriptive
  metadata used by the doctor and the GUI panel.
* **Installation contract**: ``env_key``, ``default``, ``fs_kind``,
  ``consumers`` — what env var holds the path, where the default
  expects it on disk, and which step subprocesses receive it as a CLI
  flag.
* **Provenance**: ``source`` — how the asset is obtained (public URL,
  MPI gated download, HF cache, GDrive interstitial, manual). The
  source descriptor classes are defined here so a future CLI
  downloader can consume them without depending on the GUI module.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple


# ─── Source descriptors ──────────────────────────────────────────────────
# Moved from gui/backend/data_readiness.py. The descriptors only carry
# provenance metadata; the actual download workers stay in the GUI
# module (data_readiness.py) since they pull in Flask + threading
# concerns that don't belong in inference/.

@dataclasses.dataclass(frozen=True)
class PublicSource:
    """A single HTTPS GET, no credentials."""
    url: str


@dataclasses.dataclass(frozen=True)
class MpiSource:
    """A POST to download.is.tue.mpg.de with username+password in the body."""
    domain: str                # e.g. "smplx" or "mamma"
    sfile: str                 # remote path under that domain
    account_label: str         # shown in the sign-in form ("SMPL-X", "MAMMA")
    register_url: str          # registration page for new users
    extract: bool = False      # treat the downloaded file as a zip & expand


@dataclasses.dataclass(frozen=True)
class ManualSource:
    """Short documented steps the user has to follow themselves."""
    steps: tuple
    link: Optional[str] = None
    link_label: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class GDriveSource:
    """Single-click download for a Google Drive file.

    The plain HTTP GET path can't handle GDrive files >100 MB because
    Google interposes an anti-bot HTML page that requires a confirm
    token to bypass; the GUI worker walks that two-step dance.
    """
    file_id: str               # the GDrive file id (the 33-char slug)
    link: str                  # human-readable URL (opened in error UI)


@dataclasses.dataclass(frozen=True)
class HFHubSource:
    """Asset is loaded via huggingface_hub (e.g. ``from_pretrained("facebook/sam3")``).

    The readiness probe checks for a populated HF cache directory at
    ``$HF_HOME/hub/models--<org>--<repo>/snapshots/<commit>/`` rather
    than a repo-relative file. The first ``ma_masks`` run downloads
    the weights lazily.
    """
    model_id: str              # e.g. "facebook/sam3"
    account_label: str         # shown in the UI ("Hugging Face")
    register_url: str          # model card / access request page
    gated: bool = False        # true → user must request access first
    steps: tuple = ()          # short manual prep steps (login etc.)


# ─── Installation-contract records ───────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class StepConsumer:
    """One step subprocess's consumption of one asset.

    Attributes
    ----------
    step:
        Step name as used by the runner (``"ma_2d"``, ``"ma_3d"``,
        ``"ma_masks"``).
    cli_flag:
        CLI flag passed to the subprocess (e.g. ``"--smplx-models"``).
    required:
        ``False`` when the step has an alternative source for the
        value (e.g. ``ma_2d`` can take ``--weights`` from
        ``task.json:weights:`` instead of the env-var default). The
        builder is expected to handle the precedence chain; the env
        var alone may be unset without failing the build.
    mechanical:
        ``True`` if the step's builder auto-translates the env-var to
        argv via the simple loop in :func:`step_argv_translation`.
        ``False`` means the builder handles this argument specially
        (today: ``ma_2d``'s ``--weights`` precedence chain). Special
        cases stay in the builder code; the registry just declares
        the env_key and CLI flag so the doctor can still surface them.
    """
    step: str
    cli_flag: str
    required: bool = True
    mechanical: bool = True


@dataclasses.dataclass(frozen=True)
class InstallationAsset:
    """One installation asset: where it lives, who reads it, how it's obtained."""

    # Identity / display
    id: str
    label: str
    purpose: str
    group: str = ""                          # GUI panel grouping
    size_hint_mb: int = 0
    panel_optional: bool = False             # GUI ready-count denominator only

    # Installation contract
    env_key: Optional[str] = None            # MAMMA_* key (None = no env contract)
    default: Optional[str] = None            # default rel path OR HF model id
    fs_kind: str = "file"                    # "file" | "dir" — for probe + display
    consumers: Tuple[StepConsumer, ...] = ()  # step subprocesses that consume it

    # Provenance
    source: object = None                    # PublicSource | MpiSource | ...

    # Optional one-line note rendered above this asset's entry in the
    # generated `.env.example`. Use for values that aren't obvious
    # filesystem paths (e.g. HF model ids) so users don't get confused.
    note: Optional[str] = None


# ─── ASSETS registry ─────────────────────────────────────────────────────
# Each record is the single declaration site for one installation asset.
# Reordering is safe; the registry is consumed via id-keyed accessors.

ASSETS: Tuple[InstallationAsset, ...] = (
    InstallationAsset(
        id="yolo",
        label="YOLO12x detector",
        purpose="ma_masks · person detection",
        group="detectors",
        size_hint_mb=119,
        env_key="MAMMA_YOLO_CHECKPOINT",
        default="data/weights/yolo/yolo12x.pt",
        fs_kind="file",
        consumers=(
            StepConsumer(step="ma_masks", cli_flag="--yolo-checkpoint"),
        ),
        source=PublicSource(
            url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12x.pt",
        ),
    ),
    InstallationAsset(
        id="sam3",
        label="SAM 3",
        purpose="ma_masks · default segmenter (HF-gated)",
        group="segmenters",
        size_hint_mb=3300,
        # No env_key: SAM 3 self-resolves through HuggingFace Hub —
        # the subprocess calls from_pretrained("facebook/sam3") which
        # downloads to ~/.cache/huggingface/hub on first use (gated,
        # needs `hf auth login`). The runner never needs to
        # inject --sam_checkpoint for sam3*. Users wanting a pinned
        # revision or local .pt file pass --sam_checkpoint in the
        # preset's `flags` list directly.
        env_key=None,
        default=None,
        fs_kind="dir",
        consumers=(),
        source=HFHubSource(
            model_id="facebook/sam3",
            account_label="Hugging Face",
            register_url="https://huggingface.co/facebook/sam3",
            gated=True,
            steps=(
                "Request access on the SAM 3 Hugging Face page; wait for approval.",
                "Activate the env: micromamba activate mamma (or conda activate mamma).",
                "Authenticate once: hf auth login.",
                "Weights download lazily on the first ma_masks run — no action needed here.",
            ),
        ),
    ),
    InstallationAsset(
        id="sam2",
        label="SAM 2.1 (large)",
        purpose="ma_masks · fallback (sam2 engine)",
        group="segmenters",
        size_hint_mb=856,
        panel_optional=True,
        env_key="MAMMA_SAM2_CHECKPOINT",
        default="data/weights/sam2/sam2.1_hiera_large.pt",
        fs_kind="file",
        # mechanical=False: the runner handles SAM injection in code
        # because the choice between MAMMA_SAM2_CHECKPOINT and
        # MAMMA_SAM3_CHECKPOINT depends on the preset's --sam_version.
        # See inference/steps/ma_masks.py.
        consumers=(
            StepConsumer(
                step="ma_masks", cli_flag="--sam_checkpoint",
                required=False, mechanical=False,
            ),
        ),
        note=(
            "Used only when --sam_version sam2. Path to the SAM2 hiera-large\n"
            "checkpoint. Default points to the documented data/ layout."
        ),
        source=PublicSource(
            url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        ),
    ),
    InstallationAsset(
        id="smplx_locked_head",
        label="SMPL-X locked head",
        purpose="ma_3d · body model",
        group="body_models",
        size_hint_mb=95,
        env_key="MAMMA_SMPLX_LOCKHEAD_MODELS",
        default="data/body_models/smplx_locked_head",
        fs_kind="dir",
        consumers=(
            StepConsumer(step="ma_3d", cli_flag="--smplx-models"),
        ),
        source=MpiSource(
            domain="smplx",
            sfile="smplx_lockedhead_20230207.zip",
            account_label="SMPL-X",
            register_url="https://smpl-x.is.tue.mpg.de/register.php",
            extract=True,
        ),
    ),
    InstallationAsset(
        id="downsampled_verts",
        label="Downsampled SMPL-X vertices",
        purpose="ma_3d · vertex downsampling",
        group="mamma_assets",
        size_hint_mb=20,
        env_key="MAMMA_DOWNSAMPLED_VERTS_PKL",
        default="data/body_models/downsampled_verts/verts_512.pkl",
        fs_kind="file",
        consumers=(
            StepConsumer(step="ma_2d", cli_flag="--downsampled-verts"),
            StepConsumer(step="ma_3d", cli_flag="--downsampled-verts"),
        ),
        source=MpiSource(
            domain="mamma",
            # The download.php endpoint maps this to FTP protected/assets/
            # — confirmed against the live server on 2026-05-27.
            sfile="mamma_assets/verts_512.pkl",
            account_label="MAMMA",
            register_url="https://mamma.is.tue.mpg.de/register.php",
            extract=False,
        ),
    ),
    InstallationAsset(
        id="ma_2d_checkpoint",
        label="MammaNet landmark .ckpt model file",
        purpose="ma_2d · trained landmark detector",
        group="mamma_assets",
        size_hint_mb=1600,
        env_key="MAMMA_MA2D_CHECKPOINT",
        default="data/weights/ma_2d/mamma_mask_full_cvpr.ckpt",
        fs_kind="file",
        # ma_2d's --weights has a precedence chain (task.json > env >
        # error). The builder handles this in code; the registry just
        # declares the env_key and flag so the doctor can show them.
        # required=False because the task.json source can satisfy the
        # step without the env-var being set. mechanical=False so the
        # generic argv-translation loop skips it (the builder handles
        # this flag specially).
        consumers=(
            StepConsumer(
                step="ma_2d", cli_flag="--weights",
                required=False, mechanical=False,
            ),
        ),
        source=MpiSource(
            domain="mamma",
            sfile="weights/mamma_mask_full_cvpr.ckpt",
            account_label="MAMMA",
            register_url="https://mamma.is.tue.mpg.de/register.php",
            extract=False,
        ),
    ),
    InstallationAsset(
        id="vitpose",
        label="ViTPose-B backbone",
        purpose="landmarks/train.py · pretrained ViT backbone",
        group="training",
        size_hint_mb=330,
        panel_optional=True,
        # Training-only; the path is hardcoded inside the landmarks
        # submodule (see landmarks/configs/constants.py:
        # PATHS.PRETRAINED_VITPOSE_CKPT_PTH). There is no MAMMA_* env
        # var for it — the env.py defaults intentionally do not include
        # MAMMA_VITPOSE_CHECKPOINT. The asset stays in the registry so
        # the GUI panel still surfaces it for users who want to train.
        env_key=None,
        default="data/weights/vitpose/vitpose-b-multi-coco.pth",
        fs_kind="file",
        consumers=(),
        source=GDriveSource(
            file_id="1sCkVDSSqyzltPyGDaBKsTwY-Adag2Vgr",
            link="https://drive.google.com/file/d/1sCkVDSSqyzltPyGDaBKsTwY-Adag2Vgr/view?usp=sharing",
        ),
    ),
)


# ─── Accessors ───────────────────────────────────────────────────────────
# Downstream consumers use these instead of touching ASSETS directly.

def get_asset(asset_id: str) -> InstallationAsset:
    """Return the asset with this id, or raise ``KeyError``."""
    for a in ASSETS:
        if a.id == asset_id:
            return a
    raise KeyError(f"No asset with id={asset_id!r} in inference.assets.ASSETS")


def defaults_dict() -> dict:
    """Return ``{env_key: default}`` for env-key-bearing assets.

    Feeds ``inference.env.DEFAULTS``. Optional assets without a sensible
    default (the ``BUN_MODELS`` / ``PART_MESH_PATH`` slots) are included
    with value ``None`` so the env-var key is still known.
    """
    out: dict = {}
    for a in ASSETS:
        if a.env_key:
            out[a.env_key] = a.default
    # Optional env vars that have no default but should still be
    # tracked by the env machinery. Today only the two ma_3d optionals
    # — they're declared via empty consumers below, but they need an
    # env_key entry so the runner can pick up `.env.local` overrides.
    for env_key in _OPTIONAL_KEYS_WITHOUT_DEFAULT:
        out.setdefault(env_key, None)
    return out


def path_keys() -> Tuple[str, ...]:
    """Return env-keys that participate in the anchoring loop.

    Feeds ``inference.env._PATH_KEYS``. Every env-bearing asset is
    included — even HF-id assets like ``MAMMA_SAM_CHECKPOINT`` —
    because the user may override the env var with a relative file
    path that legitimately needs anchoring to the repo root. The
    per-value HF-id-vs-path decision is made by env.py's
    ``_looks_like_hf_model_id`` heuristic inside the anchoring loop.
    """
    keys = []
    for a in ASSETS:
        if not a.env_key:
            continue
        keys.append(a.env_key)
    for env_key in _OPTIONAL_KEYS_WITHOUT_DEFAULT:
        if env_key not in keys:
            keys.append(env_key)
    return tuple(keys)


def step_argv_translation(step: str) -> Tuple[Tuple[str, str], ...]:
    """Return ``((env_key, cli_flag), ...)`` for **mechanical** required consumers of ``step``.

    Feeds each step builder's ``_REQUIRED_ENV_FLAGS`` tuple. The
    builder iterates this and translates env → argv in a single loop.
    Non-mechanical consumers (precedence chains like ma_2d
    ``--weights``) are *not* returned; the builder handles those in
    code, reading the env_key directly via :func:`get_asset`.
    """
    out = []
    for a in ASSETS:
        if not a.env_key:
            continue
        for c in a.consumers:
            if c.step != step:
                continue
            if not c.required or not c.mechanical:
                continue
            out.append((a.env_key, c.cli_flag))
    return tuple(out)


def step_optional_translation(step: str) -> Tuple[Tuple[str, str], ...]:
    """Return ``((env_key, cli_flag), ...)`` for **mechanical** optional consumers of ``step``.

    Feeds each step builder's ``_OPTIONAL_ENV_FLAGS`` tuple. The
    builder iterates this and only appends ``flag value`` when the env
    var is set; an unset value means "let the subprocess use its own
    default."

    Two sources contribute:

    1. Registry consumers with ``required=False`` and ``mechanical=True``
       (e.g. ``MAMMA_SAM_CHECKPOINT`` on ma_masks — an opt-in override).
    2. The :data:`_OPTIONAL_KEYS_WITH_FLAGS` mapping for env keys that
       don't have an :class:`InstallationAsset` record (ma_3d's
       ``--bun-models`` / ``--part-mesh``).
    """
    out = []
    # 1. Registry consumers.
    for a in ASSETS:
        if not a.env_key:
            continue
        for c in a.consumers:
            if c.step != step:
                continue
            if c.required or not c.mechanical:
                continue
            out.append((a.env_key, c.cli_flag))
    # 2. Non-registry optional keys.
    for env_key in _OPTIONAL_KEYS_WITH_FLAGS:
        flag_for_step = _OPTIONAL_KEYS_WITH_FLAGS[env_key].get(step)
        if flag_for_step:
            out.append((env_key, flag_for_step))
    return tuple(out)


def step_consumers(step: str) -> Tuple[Tuple[InstallationAsset, StepConsumer], ...]:
    """Return ``((asset, consumer), ...)`` for *all* consumers of ``step``.

    Includes non-mechanical ones (so the doctor can display the full
    per-step requirement table). Order follows :data:`ASSETS`.
    """
    out = []
    for a in ASSETS:
        for c in a.consumers:
            if c.step == step:
                out.append((a, c))
    return tuple(out)


# Optional env vars that aren't tied to a specific InstallationAsset
# but participate in the inference contract (ma_3d's ``--bun-models``
# and ``--part-mesh``). They're documented inline rather than as full
# registry records because:
#   - they have no canonical default location;
#   - they have no provenance (no public download); and
#   - they're only required when the optim config selects them.
# The mapping is env_key → {step: cli_flag, ...}.
_OPTIONAL_KEYS_WITH_FLAGS = {
    "MAMMA_BUN_MODELS":     {"ma_3d": "--bun-models"},
    "MAMMA_PART_MESH_PATH": {"ma_3d": "--part-mesh"},
}

# Env keys to surface in DEFAULTS / _PATH_KEYS with value None, so
# bootstrap_env recognises them and `.env.local` overrides flow.
_OPTIONAL_KEYS_WITHOUT_DEFAULT = tuple(_OPTIONAL_KEYS_WITH_FLAGS.keys())


# ─── .env.example generator ──────────────────────────────────────────────


_ENV_EXAMPLE_HEADER = """\
# MAMMA path overrides. Copy to .env.local and uncomment lines to change defaults.
# Regenerate from registry: python -m inference dump-env-example -o .env.example
#
"""


def dump_env_example() -> str:
    """Render a `.env.example` body from the registry.

    Each env-bearing asset becomes a commented line of the form
    ``# KEY=value``. Loading the file as-is is a no-op
    (every line is a comment); users uncomment the keys they
    want to override.
    """
    lines = [_ENV_EXAMPLE_HEADER.rstrip("\n")]
    # Registry-backed assets first, in registry order. Assets with a
    # `note` get a blank-line separator above so the comment doesn't
    # visually attach to the preceding entry. Multi-line notes are
    # rendered as consecutive `# <line>` comments — author the note
    # with explicit ``\n`` line breaks to control wrapping.
    for a in ASSETS:
        if not a.env_key:
            continue
        value = a.default if a.default is not None else ""
        if a.note:
            lines.append("")
            for note_line in a.note.splitlines():
                lines.append(f"# {note_line}")
        lines.append(f"# {a.env_key}={value}")
    # Optional keys without defaults (ma_3d's bun-models / part-mesh).
    for env_key in _OPTIONAL_KEYS_WITHOUT_DEFAULT:
        lines.append(f"# {env_key}=")
    lines.append("")  # trailing newline
    return "\n".join(lines)
