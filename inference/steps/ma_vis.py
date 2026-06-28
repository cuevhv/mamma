"""ma_vis (visualization) — multi-view SMPL-X result viewer / renderer."""
from __future__ import annotations

import os
from typing import List

from .base import StepBuilder


class MaVisBuilder(StepBuilder):
    step_name = "ma_vis"

    def python_argv(self, seq_name: str) -> List[str]:
        ma_2d_dir = self._resolve("ma_2d_dir") or os.path.join(
            self.out_root, "ma_2d", self.tag, self.dataset_name
        )
        ma_3d_dir = self._resolve("ma_3d_dir") or os.path.join(
            self.out_root, "ma_3d", self.tag, self.dataset_name
        )
        out = self.step_out_dir(with_dataset=True)
        frame_source_flags = self._frame_source_flags()
        calibration_flag = self._calibration_flag()

        # Calibration source priority (same rule as ma_3d):
        #   1. Explicit ma_cap_dir (step or global override)
        #   2. videos_dir / images_root_dir + calibration file
        #   3. Default ma_cap output path
        ma_cap_dir_explicit = self._resolve("ma_cap_dir")

        argv: List[str] = [self.script]
        argv += ["--ma_2d_dir", ma_2d_dir]
        argv += ["--ma_3d_dir", ma_3d_dir]
        argv += ["--seq_name", seq_name]
        argv += ["--out_path", out]
        if ma_cap_dir_explicit:
            argv += ["--ma_cap_dir", ma_cap_dir_explicit]
        elif frame_source_flags:
            argv += frame_source_flags
            argv += calibration_flag
            if self.cam_names:
                argv += ["--cam_names", *self.cam_names]
            # Standalone mode: thread global.start_frame/end_frame through so
            # overlay backgrounds align with ma_3d's meshes. Chained mode
            # (ma_cap_dir_explicit) reads the range from the per-camera NPZ.
            argv += self._frame_range_flags()
        else:
            argv += [
                "--ma_cap_dir",
                os.path.join(self.out_root, "ma_cap", self.tag, self.dataset_name),
            ]
            argv += calibration_flag
        argv += self._undistort_flag()
        # Overlay camera selection (applies in both chained and standalone modes).
        # Accept "all" (-> render every view) or an explicit list; the CLI expands
        # the 'all' sentinel against the cameras it loads. Placed before
        # ``self.flags`` so an explicit flags entry still wins (argparse keeps the
        # last occurrence of a nargs="+" option).
        overlay = self.step_cfg.get("cam_names_overlay")
        if overlay:
            if isinstance(overlay, str):
                overlay = [overlay]
            argv += ["--cam-names-overlay", *[str(c) for c in overlay]]
        mpc = self.step_cfg.get("max_preview_cams")
        if mpc is not None:
            argv += ["--max-preview-cams", str(int(mpc))]
        up_axis = self.global_cfg.get("up_axis")
        if up_axis and str(up_axis) != "auto":
            # '=' form: a signed value like '-y' would otherwise be parsed as a flag.
            argv += [f"--up-axis={up_axis}"]
        argv += self.flags
        return argv
