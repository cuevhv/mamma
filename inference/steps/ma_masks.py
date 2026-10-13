"""ma_masks (segmentation) — per-camera person masks for a sequence."""
from __future__ import annotations

import os
from typing import List

from ..assets import get_asset, step_argv_translation, step_optional_translation
from .base import StepBuilder


# Mapping: required MAMMA_* env var -> ma_masks CLI flag. Derived from
# the central registry (inference/assets.py) so adding a new flag for
# ma_masks is a one-place change. Translated to argv at builder time
# and validated up-front so users see an actionable error here instead
# of a deep crash inside the segmentation submodule's module-import or
# model-init.
_REQUIRED_ENV_FLAGS = step_argv_translation("ma_masks")

# Optional env vars; passed only when the user has set them. The
# generic loop only picks up mechanical consumers; the SAM checkpoint
# is non-mechanical (see _inject_sam_checkpoint below).
_OPTIONAL_ENV_FLAGS = step_optional_translation("ma_masks")

# SAM2 env-key looked up once at module load. SAM3 self-resolves
# through the HF cache; there's no MAMMA_SAM3_CHECKPOINT.
_MA_MASKS_SAM2_ENV_KEY = get_asset("sam2").env_key


class MaMasksBuilder(StepBuilder):
    step_name = "ma_masks"

    def python_argv(self, seq_name: str) -> List[str]:
        frame_source_flags = self._frame_source_flags()

        # ma_cap_dir: when a frame-source override is set, it is allowed
        # as a calibration source only (segmentation/process_sequence.py
        # picks up calibration from NPZs when present). When neither
        # frame-source override is set, fall back to the default ma_cap
        # output path so existing pipeline-chained tasks keep working.
        ma_cap_dir = self._resolve("ma_cap_dir")
        if not ma_cap_dir and not frame_source_flags:
            ma_cap_dir = os.path.join(
                self.out_root, "ma_cap", self.tag, self.dataset_name
            )

        # ma_masks --out is the step root WITHOUT dataset; the script
        # composes <out>/<dataset>/<seq>/ itself.
        out = os.path.join(self.out_root, self.step_name, self.tag)
        argv: List[str] = [self.script]
        if ma_cap_dir:
            argv += ["--ma_cap_dir", ma_cap_dir]
        argv += frame_source_flags
        argv += self._calibration_flag()
        argv += self._undistort_flag()
        argv += ["--seq_name", seq_name]
        argv += ["--out", out]
        if self.dataset_name:
            argv += ["--dataset_name", self.dataset_name]
        # Global camera list goes BEFORE the step flags so that flags win
        # under argparse last-occurrence semantics: presets and camera
        # fan-out cells narrow the camera set via their own --cam_names.
        if self.cam_names:
            argv += ["--cam_names", *self.cam_names]
        argv += self.flags

        # Translate per-installation paths from env -> argv.
        missing: List[str] = []
        for env_key, flag in _REQUIRED_ENV_FLAGS:
            value = os.environ.get(env_key)
            if not value:
                missing.append(env_key)
                continue
            argv += [flag, value]
        if missing:
            raise RuntimeError(
                "ma_masks cannot be built: the following installation paths "
                "are not set. Set them in .env.local at the repo root or rely "
                "on the in-code defaults from inference.env.DEFAULTS. "
                f"Missing: {', '.join(missing)}"
            )

        for env_key, flag in _OPTIONAL_ENV_FLAGS:
            value = os.environ.get(env_key)
            if value:
                argv += [flag, value]

        # SAM checkpoint: version-aware. Three rules:
        #   1. If the preset's `flags` already supplies --sam_checkpoint,
        #      that value wins. We don't inject anything.
        #   2. Otherwise, when --sam_version sam2, inject the SAM2
        #      checkpoint path from MAMMA_SAM2_CHECKPOINT (defaulted in
        #      inference/assets.py to data/weights/sam2/...).
        #   3. Otherwise (sam3 / sam3_prompt / unspecified), inject
        #      nothing. SAM 3 self-resolves through HuggingFace Hub
        #      (~/.cache/huggingface/hub/), so the subprocess doesn't
        #      need a path from the runner.
        if not _preset_supplies_sam_checkpoint(self.flags):
            sam_version = _resolve_sam_version(self.flags)
            if sam_version == "sam2":
                sam2_value = os.environ.get(_MA_MASKS_SAM2_ENV_KEY)
                if sam2_value:
                    argv += ["--sam_checkpoint", sam2_value]

        return argv


def _preset_supplies_sam_checkpoint(flags) -> bool:
    """Return True iff the preset's flags include --sam_checkpoint in any form."""
    return any(
        f == "--sam_checkpoint" or f.startswith("--sam_checkpoint=")
        for f in flags
    )


def _resolve_sam_version(flags) -> str:
    """Read --sam_version from the preset's flags. Returns the LAST
    occurrence to match argparse semantics (a later override wins over
    an earlier one in the same flags list).

    Defaults to ``sam2`` because that is what the segmentation
    subprocess's own argparse defaults to when the flag is absent: the
    injection decision must mirror what will actually run. (The old
    ``sam3_prompt`` fallback made a bare-flags config run sam2 WITHOUT a
    checkpoint injection: fine locally where cwd-relative weights exist,
    a guaranteed FileNotFoundError on cluster nodes.)"""
    flag_list = list(flags)
    found: str = "sam2"
    for i, f in enumerate(flag_list):
        if f == "--sam_version" and i + 1 < len(flag_list):
            found = flag_list[i + 1]
        elif f.startswith("--sam_version="):
            found = f.split("=", 1)[1]
    return found
