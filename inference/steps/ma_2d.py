"""ma_2d (landmarks) — 2D pose / keypoint estimation per camera."""
from __future__ import annotations

import logging
import os
from typing import List

from ..assets import get_asset, step_argv_translation
from .base import StepBuilder

log = logging.getLogger(__name__)


# Per-installation paths translated from env to argv. Derived from the
# central registry (inference/assets.py) so adding/renaming a flag for
# ma_2d is a one-place change.
_REQUIRED_ENV_FLAGS = step_argv_translation("ma_2d")

# ma_2d's --weights env-key, looked up from the registry once at module
# load. The precedence logic (task.json > env > error) stays in
# python_argv below — the registry just declares the env_key so we
# don't repeat the string literal here.
_MA2D_WEIGHTS_ENV_KEY = get_asset("ma_2d_checkpoint").env_key


class Ma2dBuilder(StepBuilder):
    step_name = "ma_2d"

    def python_argv(self, seq_name: str) -> List[str]:
        config_path = self._expand(self.step_cfg.get("config_path", ""))
        # Resolve --weights with precedence: explicit --weights among the
        # task's raw flags (argparse's last-occurrence would pick it anyway;
        # honoring it here keeps the command to ONE --weights and the log
        # truthful) > task.json `weights:` (pinned for this experiment) >
        # MAMMA_MA2D_CHECKPOINT (per-installation default from
        # inference/env.py::DEFAULTS or .env.local). _resolve anchors
        # relative paths to the repo root so subprocess (cwd=landmarks/)
        # can find data/weights/... reliably.
        flag_weights = ""
        user_flags = self.flags
        for i, tok in enumerate(user_flags):
            if tok == "--weights" and i + 1 < len(user_flags):
                flag_weights = user_flags[i + 1]
            elif tok.startswith("--weights="):
                flag_weights = tok.split("=", 1)[1]
        task_weights = self._resolve("weights")
        env_weights = os.environ.get(_MA2D_WEIGHTS_ENV_KEY) or ""
        if flag_weights:
            weights = flag_weights
            source = "explicit --weights flag"
        elif task_weights:
            weights = task_weights
            source = "task.json"
        elif env_weights:
            weights = env_weights
            source = "env"
        else:
            raise RuntimeError(
                "ma_2d cannot be built: --weights is unresolved. Either "
                "set `weights:` for the ma_2d step in the task file, or "
                f"set {_MA2D_WEIGHTS_ENV_KEY} in .env.local (or rely on the "
                "in-code default from inference.env.DEFAULTS)."
            )
        log.info("ma_2d weights resolved from %s: %s", source, weights)

        out = self.step_out_dir(with_dataset=True)
        frame_source_flags = self._frame_source_flags()

        # Frame source: ``ma_cap_dir`` (-> --img_folder, NPZ mode) is the
        # default when no override is set; videos_dir / images_root_dir
        # override and skip --img_folder entirely.
        argv: List[str] = [self.script]
        argv += ["--config_path", config_path]
        if not flag_weights:
            # The explicit flag arrives via `self.flags` below; injecting
            # here too would put --weights on the command twice.
            argv += ["--weights", weights]
        argv += ["--out_folder", out]
        argv += ["--seq_name", seq_name]
        if frame_source_flags:
            argv += frame_source_flags
        else:
            img_folder = self._resolve("ma_cap_dir") or os.path.join(
                self.out_root, "ma_cap", self.tag, self.dataset_name
            )
            argv += ["--img_folder", img_folder]

        mask_path = self._resolve("ma_masks_dir") or os.path.join(
            self.out_root, "ma_masks", self.tag, self.dataset_name
        )
        argv += ["--mask_path", mask_path]
        argv += self._calibration_flag()
        argv += self._undistort_flag()
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
                "ma_2d cannot be built: the following installation paths "
                "are not set. Set them in .env.local at the repo root or "
                "rely on the in-code defaults from inference.env.DEFAULTS. "
                f"Missing: {', '.join(missing)}"
            )

        if self.cam_names:
            argv += ["--cam_names", *self.cam_names]
        return argv
