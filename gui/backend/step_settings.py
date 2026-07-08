"""Curated "common settings" schema per pipeline step.

Backs the friendly widgets in the New Task preset editor (Step 2). Each
:class:`Setting` declares ONE important knob — a label, a widget type, and how
it maps onto the preset:

* ``flag``      — a token in the step's ``flags`` list (``--x`` for a
                  store_true toggle, ``--x VALUE`` otherwise).
* ``flag_pair`` — an ``argparse.BooleanOptionalAction`` (``--x`` / ``--no-x``).
* ``extra``     — a dedicated step key (``config_file`` / ``config_path`` /
                  ``undistort``), surfaced by the digest as ``extras``.

The frontend renders these generically and round-trips them through the SAME
``flags``/``extras`` the raw editor uses, so the "Common settings" widgets and
the "Advanced flags" list stay in sync (single source of truth).

This is intentionally a CURATED subset — the args users actually tune, grounded
in the four shipped presets (configs/examples/presets/*.yaml) plus the per-step
argparse — NOT the full surface; "View available flags" still exposes the rest.
``tests/test_step_settings.py`` asserts every flag name here still exists in the
step's argparse, so the manifest can't silently drift from the scripts.

The deepest quality knobs (SAM thresholds, ma_3d iterations/LR/loss weights) are
not CLI flags — they live inside the YAML recipes chosen by the
``config_file`` / ``config_path`` selects here, so we expose the recipe, not its
internals.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]  # gui/backend -> repo root

# step -> repo subdir (mirrors help_cache.STEP_SCRIPTS); used to resolve the
# recipe paths the dropdowns store, relative to each step's own tree.
_STEP_REPO = {
    "ma_cap": "capture",
    "ma_masks": "segmentation",
    "ma_2d": "landmarks",
    "ma_3d": "optimization",
    "ma_vis": "visualization",
}


# --------------------------------------------------------------------------- #
# Target constructors — how a setting maps onto the preset.
# --------------------------------------------------------------------------- #
def _flag(name: str, valued: bool = True, invert: bool = False) -> dict:
    """A token in the step's ``flags`` list. ``valued=False`` = store_true.

    ``invert=True`` makes a store_true toggle read "on when the flag is ABSENT"
    (e.g. a "Save visualizations" toggle backed by ``--disable-visualizations``).
    """
    t = {"kind": "flag", "flag": name, "valued": valued}
    if invert:
        t["invert"] = True
    return t


def _flag_pair(on: str, off: str, default_on: bool = True) -> dict:
    """An argparse BooleanOptionalAction (``--x`` / ``--no-x``)."""
    return {"kind": "flag_pair", "on": on, "off": off, "defaultOn": default_on}


def _extra(key: str) -> dict:
    """A dedicated step key (config_file / config_path / undistort)."""
    return {"kind": "extra", "key": key}


def _flags(names: list) -> dict:
    """One toggle backing several store_true flags — all emitted/removed together
    (e.g. a "Low-memory mode" switch that sets both --lazy-frames and
    --prune-memory). Reads as ON when ANY of the flags is present.
    """
    return {"kind": "flags", "flags": list(names)}


@dataclasses.dataclass(frozen=True)
class Setting:
    id: str
    label: str
    widget: str                       # toggle | select | slider | number | text
    target: dict
    help: str = ""
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    unit: Optional[str] = None
    choices: Optional[list] = None         # static [{value,label}, ...]
    choices_from: Optional[str] = None     # dynamic key resolved at build time
    omit_when_default: bool = False        # selecting the default drops the token
    depends_on: Optional[dict] = None      # {"id": ..., "equals": ...}
    peek: bool = False                     # value is a recipe file viewable via /recipe
    note: Optional[str] = None             # small help line rendered above the control
    placeholder: Optional[str] = None      # in-field hint for text / number inputs

    def to_json(self, choices: Optional[list] = None) -> dict:
        out: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "widget": self.widget,
            "target": self.target,
            "help": self.help,
            "default": self.default,
        }
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.step is not None:
            out["step"] = self.step
        if self.unit is not None:
            out["unit"] = self.unit
        resolved = choices if choices is not None else self.choices
        if resolved is not None:
            out["choices"] = resolved
        if self.omit_when_default:
            out["omitWhenDefault"] = True
        if self.depends_on is not None:
            out["dependsOn"] = self.depends_on
        if self.peek:
            out["peek"] = True
        if self.note:
            out["note"] = self.note
        if self.placeholder:
            out["placeholder"] = self.placeholder
        return out


# --------------------------------------------------------------------------- #
# The manifest.
# --------------------------------------------------------------------------- #
_STEP_SETTINGS: dict[str, list[Setting]] = {
    # ma_cap has no per-step quality knobs — frame range is the global control.
    "ma_cap": [],

    "ma_masks": [
        Setting(
            id="sam_version", label="Segmenter", widget="select",
            target=_flag("--sam_version"), default="sam2",
            choices=[
                {"value": "sam2", "label": "SAM 2 + YOLO",
                 "description": "Fastest, lowest VRAM. Frames stream from host RAM to the GPU "
                                "(CPU offload), so VRAM stays bounded — handles the longest clips."},
                {"value": "sam3", "label": "SAM 3 + YOLO",
                 "description": "Better tracking (SAM 3 joint consolidation). Same CPU offload, "
                                "so still good on long clips; more VRAM than SAM 2."},
                {"value": "sam3_prompt", "label": "SAM 3 text-prompt",
                 "description": "No YOLO — finds people from the text prompt \"person\". Uses the most "
                                "GPU memory, and it grows with clip length, so it may run out of memory "
                                "and crash on long clips."},
                {"value": "sam3_prompt_light", "label": "SAM 3 text-prompt, light (recommended)",
                 "description": "No YOLO — same text-prompt detection, but a leaner tracker keeps GPU "
                                "memory bounded, to handle longer clips."},
            ],
            help="How people are detected (YOLO boxes vs SAM 3 text-prompt) and tracked. "
                 "Pick by VRAM / clip length — each option explains its trade-off.",
        ),
        Setting(
            id="assignment_cfg", label="Segmenter config (.yaml)", widget="select",
            target=_flag("--cfg"), default="", choices_from="ma_masks_configs",
            peek=True,
            note="Manage the config YAML files in segmentation/configs/.",
            help="Tuning for the segmenter (CPU offload, memory, tracking thresholds). "
                 "Leave on Auto to use the file matching the segmenter; override only if "
                 "needed — the config must match the chosen segmenter.",
        ),
        Setting(
            id="cam_init", label="Subject reference camera", widget="text",
            target=_flag("--cam_init"), default=None, omit_when_default=True,
            placeholder="e.g. IOI_09",
            note="Camera whose view seeds the people's IDs — every person you want "
                 "tracked must be visible in it. Leave blank to auto-pick the view "
                 "with the most people detected.",
            help="Camera used to initialize person identities.",
        ),
        Setting(
            id="save_masked_outputs", label="Save visualizations", widget="toggle",
            target=_flag("--skip_masked_outputs", valued=False, invert=True), default=True,
            help="Write the mask overlay images/MP4 (viz-only, not consumed downstream). "
                 "On by default; turning it off passes --skip_masked_outputs (saves time and disk).",
        ),
        Setting(
            id="low_memory", label="Low-memory mode", widget="toggle",
            target=_flags(["--lazy-frames", "--prune-memory"]), default=False,
            help="For long clips that would otherwise run out of memory. Streams frames on "
                 "demand and prunes the tracker's accumulated per-frame memory, so memory stays "
                 "roughly flat instead of growing with the clip. Near-lossless; costs ~+24% wall "
                 "on mp4 (frame re-decode). Off by default — normal-length clips don't need it.",
        ),
        Setting(
            id="debug_crop_summary", label="Debug masks", widget="toggle",
            target=_flag("--debug_crop_summary", valued=False), default=False,
            help="Render per-person crop-summary + t-SNE + cross-camera debug PNGs "
                 "(extra wall time; off by default).",
        ),
    ],

    "ma_2d": [
        Setting(
            id="config_path", label="Model config (.yaml)", widget="select",
            target=_extra("config_path"), default=None, choices_from="ma2d_configs",
            peek=True,
            note="Manage the config YAML files in landmarks/configs/train/models_2d/.",
            help="2D landmark model config (architecture / input size).",
        ),
        Setting(
            id="tensorrt", label="TensorRT", widget="toggle",
            target=_flag("--tensorrt", valued=False), default=False,
            help="Compile the landmark net to TensorRT: ~5x faster on the network itself, "
                 "biggest on long / many-camera captures. NVIDIA-only and needs a one-time "
                 "extra install (requirements-tensorrt.txt); the first run compiles for "
                 "~1 minute, then it's cached. Accuracy is unchanged (verified against "
                 "ground truth); without the install the run safely falls back to normal "
                 "speed, so it never breaks anything.",
        ),
        Setting(
            id="save_visualizations", label="Save visualizations", widget="toggle",
            target=_flag("--disable-visualizations", valued=False, invert=True), default=True,
            help="Write per-body 2D-landmark viz frames + a preview video for inspection. "
                 "Viz-only — nothing downstream reads them. On by default; turn off for "
                 "faster runs and less disk when you don't need to eyeball the landmarks.",
        ),
    ],

    "ma_3d": [
        Setting(
            id="config_file", label="Optimization config (.yaml)", widget="select",
            target=_extra("config_file"), default=None, choices_from="ma3d_recipes",
            peek=True,
            note="Manage the config YAML files in optimization/config_files/contact_configs/.",
            help="The optimization config (betas, contact, iterations, loss weights). "
                 "The main ma_3d quality lever.",
        ),
        Setting(
            id="tf32", label="Fast fits (TF32)", widget="toggle",
            target=_flag("--tf32", valued=False), default=False,
            help="Faster optimization on modern NVIDIA GPUs; however it might give worse results. "
                 "Keep it off for final, best-quality fits.",
        ),
        Setting(
            id="occlusion_aware_weights", label="Heavy occlusion-aware", widget="toggle",
            target=_flag("--occlusion-aware-weights", valued=False), default=False,
            help="Enable to handle extreme occlusion, i.e. when body parts stay hidden "
                 "across the majority of camera views.",
        ),
        Setting(
            id="use_vposer", label="Enable Pose Prior", widget="toggle",
            target=_flag("--use-vposer", valued=False), default=False,
            help="A learned prior of realistic human poses; a fix for heavy occlusion, not a "
                 "general quality boost. "
                 # ""
                 # "Turn it on only when too few 2D landmarks are visible to "
                 # "constrain a joint and limbs settle into implausible poses (e.g. a knee or elbow "
                 # "rotating the wrong way); it nudges those under-constrained joints back onto "
                 # "plausible ones. Leave it off otherwise — when the 2D evidence is good it can bias "
                 # "the fit toward generic poses and reduce accuracy. Requires the VPoser weights in "
                 # "data/body_models/vposer."
            ,
        ),
    ],

    "ma_vis": [
        Setting(
            id="overlay_num_workers", label="Overlay workers", widget="slider",
            target=_flag("--overlay-num-workers"), default=1, min=1, max=8, step=1,
            help="Parallel overlay-rendering workers (~2x on multi-cam; raise on big machines).",
        ),
        Setting(
            id="skip_overlay", label="Skip overlay videos", widget="toggle",
            target=_flag("--skip-overlay", valued=False), default=False,
            help="Skip the per-camera mesh-over-footage overlay videos — the slowest part "
                 "of visualization. The interactive 3D scene (.rrd) is still produced; you "
                 "only lose the ready-made overlay MP4s.",
        ),
        Setting(
            id="rerun_light", label="Light .rrd", widget="toggle",
            target=_flag("--rerun-light", valued=False), default=False,
            help="Skip 2D landmark logging for a lighter, faster Rerun scene.",
        ),
        Setting(
            id="rerun_video", label="H.264 backdrop", widget="toggle",
            target=_flag_pair("--rerun-video", "--no-rerun-video", default_on=True), default=True,
            help="Log each camera backdrop as H.264 (~10-15x smaller .rrd) instead of "
                 "per-frame JPEG.",
        ),
        Setting(
            id="rerun_video_crf", label="Backdrop quality (CRF)", widget="slider",
            target=_flag("--rerun-video-crf"), default=20, min=18, max=28, step=1,
            help="H.264 quality for the .rrd backdrop. Lower = better but larger.",
        ),
        Setting(
            id="overlay_resolution", label="Overlay resolution", widget="number",
            target=_flag("--overlay-resolution"), default=1280,
            min=0, step=1, unit="px", omit_when_default=True,
            help="Long-side resolution for overlay videos. 0 keeps the source resolution.",
        ),
        Setting(
            id="fps", label="FPS", widget="number",
            target=_flag("--fps"), default=30, min=1, step=1, omit_when_default=True,
            help="Frame rate for the Rerun timeline and overlay videos.",
        ),
    ],
}


# --------------------------------------------------------------------------- #
# Dynamic choices — config recipes enumerated from disk.
# --------------------------------------------------------------------------- #
def _recipe_choices(subdir: str, rel_glob: str) -> list[dict]:
    """List config YAMLs under ``<repo>/<subdir>`` as ``{value, label}``.

    ``value`` is the path relative to the step's repo subdir (exactly how the
    preset stores ``config_file`` / ``config_path``); ``label`` is the stem.
    """
    base = _REPO_ROOT / subdir
    out: list[dict] = []
    for p in sorted(base.glob(rel_glob)):
        out.append({"value": p.relative_to(base).as_posix(), "label": p.stem})
    return out


# ma_2d model configs: only config_mammanet_mask_512 matches the shipped ma_2d
# weights (data/weights/ma_2d/mamma_mask_full_cvpr.ckpt). The rest are other
# architectures / training variants — shown behind a "show other architectures"
# expander in the GUI (advanced=True), each with a one-line hint. Unknown/new
# files fall back to advanced with a generic hint.
_MA2D_RECOMMENDED = "config_mammanet_mask_512"
_MA2D_CONFIG_META = {
    "config_mammanet_mask_512":            "MammaNet + mask + contact — matches the shipped ma_2d weights.",
    "config_mammanet_mask_512_no_contact": "MammaNet + mask, no contact head.",
    "config_mammanet_no-mask_512":         "MammaNet, no mask input.",
    "config_camerahmr_mask_512":           "CameraHMR backbone + mask.",
    "config_camerahmr_no-mask_512":        "CameraHMR backbone, no mask.",
    "config_hrnet_no-mask_512":            "HRNet backbone.",
    "config":                              "Base template config.",
}


def _ma2d_config_choices() -> list[dict]:
    """ma_2d model-config choices, tagged for the GUI: the recommended one
    (matches the shipped weights) first and shown by default; every other
    architecture/variant marked ``advanced`` so the UI tucks it behind a
    "show other architectures" expander.
    """
    base = _REPO_ROOT / "landmarks"
    out: list[dict] = []
    for p in sorted((base / "configs/train/models_2d").glob("config*.yaml")):
        stem = p.stem
        recommended = stem == _MA2D_RECOMMENDED
        out.append({
            "value": p.relative_to(base).as_posix(),
            "label": stem + (" — recommended" if recommended else ""),
            "description": _MA2D_CONFIG_META.get(stem, "Other config."),
            "advanced": not recommended,
        })
    out.sort(key=lambda c: (c["advanced"], c["label"]))  # recommended (non-advanced) first
    return out


# ma_3d optimization recipes. All are valid variants of the same pipeline (not
# tied to specific weights), so none are hidden — just described. The shipped
# presets use the "no_vtemplate" one; it's marked the default and sorted first.
_MA3D_RECOMMENDED = "config_real_gmf_small_vals_detr_exp_no_vtemplate"
_MA3D_CONFIG_META = {
    "config_real_gmf_small_vals_detr_exp_no_vtemplate":
        "16 shape betas, contact on — the shipped default.",
    "config_real_gmf_small_vals_detr_exp_no_vtemplate_no_contact":
        "16 shape betas, contact off.",
    "config_real_gmf_small_vals_detr_exp_no_vtemplate_occlusion_vposer":
        "16 betas, contact on, plus occlusion-aware weights and a VPoser prior (for heavy occlusion).",
    "config_real_gmf_small_vals_detr_exp_betas16":
        "16 shape betas, contact on.",
    "config_real_gmf_small_vals_detr_exp_betas16_no_contact":
        "16 shape betas, contact off.",
    "config_real_gmf_small_vals_detr_exp_betas10":
        "10 shape betas, contact on.",
    "config_real_gmf_small_vals_detr_exp_betas10_no_contact":
        "10 shape betas, contact off.",
    "config_real_gmf_small_vals_detr_exp_betas10_bun":
        "10 shape betas, contact on, BUN body model.",
}


def _ma3d_config_choices() -> list[dict]:
    """ma_3d optimization-config choices with one-line descriptions; the shipped
    default sorted first and labelled. All stay visible (no expander)."""
    base = _REPO_ROOT / "optimization"
    out: list[dict] = []
    for p in sorted((base / "config_files/contact_configs").glob("*.yaml")):
        stem = p.stem
        is_default = stem == _MA3D_RECOMMENDED
        out.append({
            "value": p.relative_to(base).as_posix(),
            "label": stem + (" — default" if is_default else ""),
            "description": _MA3D_CONFIG_META.get(stem, "Optimization config."),
            "_default": is_default,
        })
    out.sort(key=lambda c: (not c.pop("_default"), c["label"]))  # default first
    return out


def _ma_masks_config_choices() -> list[dict]:
    """ma_masks segmenter configs (the ``--cfg`` YAML). The first option leaves
    --cfg unset so the pipeline auto-picks the file matching --sam_version;
    server_mounts.yaml etc. are excluded by the sam*.yaml glob."""
    base = _REPO_ROOT / "segmentation"
    meta = {
        "sam2": "SAM 2 — recommended for most cases; robust on dense multi-person scenes.",
        "sam3": "SAM 3 — tuned to reduce tracklet fragmentation (30 fps, 3–6 people).",
        "sam3_default": "SAM 3 unmodified defaults (experimental).",
    }
    out: list[dict] = [{
        "value": "",
        "label": "Auto (match segmenter)",
        "description": "Pick the config matching the segmenter (sam2 → configs/sam2.yaml, sam3* → configs/sam3.yaml).",
    }]
    for p in sorted((base / "configs").glob("sam*.yaml")):
        out.append({
            "value": p.relative_to(base).as_posix(),
            "label": p.stem,
            "description": meta.get(p.stem, "Segmenter config."),
        })
    return out


_CHOICE_RESOLVERS: dict[str, Callable[[], list[dict]]] = {
    "ma3d_recipes": _ma3d_config_choices,
    "ma2d_configs": _ma2d_config_choices,
    "ma_masks_configs": _ma_masks_config_choices,
}


def build_settings() -> dict[str, list[dict]]:
    """``{step: [setting-json, ...]}`` with dynamic choices resolved.

    Cheap (a couple of directory globs); the route can call it per request.
    """
    out: dict[str, list[dict]] = {}
    for step, settings in _STEP_SETTINGS.items():
        items = []
        for s in settings:
            choices = None
            if s.choices_from:
                resolver = _CHOICE_RESOLVERS.get(s.choices_from)
                choices = resolver() if resolver else []
            items.append(s.to_json(choices=choices))
        out[step] = items
    return out


def read_recipe(step: str, rel_path: str) -> tuple[str, str]:
    """Return ``(rel_path, file_text)`` for a recipe YAML shown in a select.

    Path-safe: only paths that are *actual dropdown choices* for that step are
    readable, so this can never be used to read arbitrary files. ``rel_path`` is
    the value stored in the preset (relative to the step's repo subdir).
    """
    if step not in _STEP_REPO:
        raise ValueError(f"unknown step: {step!r}")
    allowed: set[str] = set()
    for s in _STEP_SETTINGS.get(step, []):
        if s.choices_from:
            resolver = _CHOICE_RESOLVERS.get(s.choices_from)
            if resolver:
                allowed.update(c["value"] for c in resolver())
    if rel_path not in allowed:
        raise ValueError(f"{rel_path!r} is not a known recipe for {step}")
    base = (_REPO_ROOT / _STEP_REPO[step]).resolve()
    abs_path = (base / rel_path).resolve()
    if not str(abs_path).startswith(str(base)):  # defence in depth
        raise ValueError("recipe path escapes its directory")
    return rel_path, abs_path.read_text()


def iter_flag_names():
    """Yield ``(step, flag_name)`` for every flag-backed setting.

    For ``flag_pair`` only the canonical ``on`` form is yielded — that's the
    name argparse prints in ``--help`` (the drift test matches against it).
    """
    for step, settings in _STEP_SETTINGS.items():
        for s in settings:
            t = s.target
            if t["kind"] == "flag":
                yield step, t["flag"]
            elif t["kind"] == "flag_pair":
                yield step, t["on"]
            elif t["kind"] == "flags":
                for f in t["flags"]:
                    yield step, f
