"""ma_3d (optimization) — multi-view SMPL-X fitting from 2D landmarks."""
from __future__ import annotations

import os
from typing import List

from ..assets import step_argv_translation, step_optional_translation
from .base import StepBuilder


# Mapping: required MAMMA_* env var -> ma_3d CLI flag. Derived from
# the central registry (inference/assets.py) so adding a new flag for
# ma_3d is a one-place change. Translated to argv at builder time and
# validated up-front so users see an actionable error here instead of
# a bare KeyError deep inside the optimization submodule's
# module-import.
_REQUIRED_ENV_FLAGS = step_argv_translation("ma_3d")

# Optional env vars; passed only when the user has set them.
# Whether they are *required* depends on the algorithm config_file
# selected for the run (use_bun_model: True, SDF loss). Validating
# that conditional requirement is the job of
# `python -m inference doctor`, not this builder.
_OPTIONAL_ENV_FLAGS = step_optional_translation("ma_3d")


class Ma3dBuilder(StepBuilder):
    step_name = "ma_3d"

    def python_argv(self, seq_name: str) -> List[str]:
        config_file = self._expand(self.step_cfg.get("config_file", ""))
        ma_2d_dir = self._resolve("ma_2d_dir") or os.path.join(
            self.out_root, "ma_2d", self.tag, self.dataset_name
        )
        # Calibration source priority:
        #   1. Explicit ma_cap_dir (step or global override)   -> chained mode
        #   2. videos_dir / images_root_dir + calibration file -> standalone synthesis
        #   3. Default ma_cap output path                      -> chained mode
        # ma_3d does NOT consume frames itself, so videos_dir is only
        # relevant as a calibration-synthesis source.
        ma_cap_dir_explicit = self._resolve("ma_cap_dir")
        frame_source_flags = self._frame_source_flags()
        calibration_flag = self._calibration_flag()

        out = self.step_out_dir(with_dataset=True)
        argv: List[str] = [self.script]
        argv += ["--config_file", config_file]
        argv += ["--seq_name", seq_name]
        argv += ["--ma_2d_dir", ma_2d_dir]
        argv += ["--out_path", out]
        if ma_cap_dir_explicit:
            argv += ["--ma_cap_dir", ma_cap_dir_explicit]
        elif frame_source_flags:
            argv += frame_source_flags
            argv += calibration_flag
        else:
            argv += [
                "--ma_cap_dir",
                os.path.join(self.out_root, "ma_cap", self.tag, self.dataset_name),
            ]
            argv += calibration_flag
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
                "ma_3d cannot be built: the following installation paths are "
                "not set. Set them in .env.local at the repo root or rely on "
                "the in-code defaults from inference.env.DEFAULTS. "
                f"Missing: {', '.join(missing)}"
            )

        for env_key, flag in _OPTIONAL_ENV_FLAGS:
            value = os.environ.get(env_key)
            if value:
                argv += [flag, value]

        if self.cam_names:
            argv += ["--cam_names", *self.cam_names]
        up_axis = self.global_cfg.get("up_axis")
        if up_axis and str(up_axis) != "auto":
            # '=' form: a signed value like '-y' would otherwise be parsed as a flag.
            argv += [f"--up-axis={up_axis}"]
        return argv
