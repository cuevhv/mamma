"""StepBuilder base class.

A :class:`StepBuilder` knows how to invoke one pipeline step for a single
sequence. The runner asks the builder for:

* the python argv (after the engine prefix),
  e.g. ``["run_ma_2d.py", "--seq_name", ...]``;
* the output dir under which the DONE sentinel lives;
* the repo_path to ``cd`` into before invoking python (conda engine);
* the bind list (apptainer/docker engines).

Subclasses encode per-step argument shapes that mirror the cluster
runner at ``mamma_apptainer/steps/<step>/run/step.py``.
"""
from __future__ import annotations

import os
import shlex
from typing import List


class StepBuilder:
    """Translate one step config into an executable invocation."""

    step_name: str = ""

    def __init__(self, step_cfg: dict, global_cfg: dict, tag: str) -> None:
        self.step_cfg = step_cfg or {}
        self.global_cfg = global_cfg or {}
        self.tag = str(tag)

    # --- common helpers ---------------------------------------------------

    @staticmethod
    def _expand(p: str) -> str:
        """Expand ``~`` and environment variables (e.g. ``${MAMMA_THIRD_PARTY}``).

        Does NOT anchor relative paths — some fields are intentionally
        submodule-relative (``config_path`` for ma_2d, ``config_file``
        for ma_3d) since the subprocess ``cd``s into the submodule dir
        before invoking python. Fields that need repo-root anchoring
        for public-release portability should go through :meth:`_resolve`.
        """
        return os.path.expanduser(os.path.expandvars(p)) if p else p

    @staticmethod
    def _anchor(p: str) -> str:
        """Anchor a non-empty relative path to the repo root."""
        if not p or os.path.isabs(p):
            return p
        from ..env import repo_root
        return str(repo_root() / p)

    @property
    def repo_path(self) -> str:
        # Anchored: subprocess uses this as cwd, so it must be absolute.
        return self._anchor(self._expand(self.step_cfg.get("repo_path", "")))

    @property
    def script(self) -> str:
        return self.step_cfg.get("script", "")

    @property
    def sif_path(self) -> str:
        # Anchored like repo_path: a relative sif_path in a preset must
        # resolve against the repo root, not whatever cwd the runner (or the
        # GUI backend) happens to have.
        val = self._expand(self.step_cfg.get("sif_path", "") or "")
        return self._anchor(val) if val else ""

    @property
    def docker_image(self) -> str:
        return self.step_cfg.get("docker_image", "") or ""

    @property
    def engine(self) -> str:
        # Per-step override; default conda.
        return (self.step_cfg.get("engine") or "conda").lower()

    @property
    def conda_env(self) -> str:
        # Per-step override → global → built-in default.
        return (
            self.step_cfg.get("conda_env")
            or self.global_cfg.get("conda_env")
            or "mamma"
        )

    @property
    def out_root(self) -> str:
        # Anchored: every step pastes this into argv path positions
        # that subprocesses (with cwd=submodule) must be able to open.
        return self._anchor(self._expand(self.global_cfg.get("out_dir", "")))

    @property
    def dataset_name(self) -> str:
        return self.global_cfg.get("dataset_name", "")

    @property
    def cam_names(self) -> List[str]:
        return list(self.global_cfg.get("cam_names", []))

    @property
    def flags(self) -> List[str]:
        """Extra argv flags, parsed shell-style.

        ``task.json`` stores flags as a list of strings; entries may contain
        spaces (e.g. ``"--sam_version sam3_prompt"``) or quoted args. We use
        :func:`shlex.split` so quoted spaces inside a single value survive.
        """
        out: List[str] = []
        for f in self.step_cfg.get("flags", []) or []:
            out.extend(shlex.split(str(f)))
        return out

    def step_out_dir(self, with_dataset: bool = True) -> str:
        parts = [self.out_root, self.step_name, self.tag]
        if with_dataset and self.dataset_name:
            parts.append(self.dataset_name)
        return os.path.join(*parts)

    def done_file(self, seq_name: str) -> str:
        """Per-(step, seq) DONE sentinel path."""
        return os.path.join(
            self.out_root,
            self.step_name,
            self.tag,
            self.dataset_name,
            seq_name,
            "DONE",
        )

    def binds(self) -> List[str]:
        """Bind list for apptainer/docker engines.

        The **repo root** is bound at ``/repo`` and the step runs with
        ``container_cwd()`` = ``/repo/<step subdir>`` — the same shape as the
        conda engine (cwd = ``<root>/<step subdir>``), so step scripts that
        import their own package through the repo root (``run_ma_cap.py``
        does ``from capture import …``) work identically in containers.
        Extra entries come from ``step_cfg["bind"]`` and
        ``global_cfg["bind"]``.
        """
        binds: List[str] = []
        if self.repo_path:
            from ..env import repo_root
            binds.append(f"{repo_root()}:/repo")
        for b in self.step_cfg.get("bind", []) or []:
            binds.append(self._expand(b))
        for b in self.global_cfg.get("bind", []) or []:
            binds.append(self._expand(b))
        return binds

    # --- tri-mode input helpers ------------------------------------------
    #
    # Every step accepts the same three input modes:
    #   1. ``ma_cap_dir <dir>``    — NPZ manifest from a prior ma_cap run
    #   2. ``videos_dir <dir>``    — <dir>/<cam_name>.mp4
    #   3. ``images_root_dir <dir>`` — <dir>/<cam_name>/*.{jpg,png}
    #
    # ``videos_dir`` and ``images_root_dir`` are mutually exclusive
    # frame-source flags; ``ma_cap_dir`` may still be set alongside one of
    # them to supply calibration. The fields live on ``global_cfg`` so a
    # single ``global.videos_dir`` propagates to every step that needs
    # frames; ``step_cfg`` may override per step.

    def _resolve(self, field: str) -> str:
        """Resolve a path-valued config field with step→global precedence.

        Step config wins if set; otherwise falls back to global. The
        returned value is ``_expand``-ed and anchored to the repo root
        if relative — so callers always get an absolute path safe to
        pass to subprocesses (which ``cd`` into the submodule dir).
        """
        val = self.step_cfg.get(field) or self.global_cfg.get(field) or ""
        if not val:
            return ""
        return self._anchor(self._expand(str(val)))

    def _frame_source_flags(self) -> List[str]:
        """Translate ``videos_dir`` / ``images_root_dir`` to argv flags.

        Raises ``RuntimeError`` if both are set (mutually exclusive). Returns
        an empty list if neither is set (caller falls back to ``ma_cap_dir``).
        """
        videos_dir = self._resolve("videos_dir")
        images_root_dir = self._resolve("images_root_dir")
        if videos_dir and images_root_dir:
            raise RuntimeError(
                f"{self.step_name}: 'videos_dir' and 'images_root_dir' are "
                "mutually exclusive; set at most one (on global or step "
                "config). Got: "
                f"videos_dir={videos_dir!r}, images_root_dir={images_root_dir!r}"
            )
        if videos_dir:
            return ["--videos_dir", videos_dir]
        if images_root_dir:
            return ["--images_root_dir", images_root_dir]
        return []

    def _calibration_flag(self) -> List[str]:
        """Translate ``calibration`` to ``["--calibration", <path>]`` or []."""
        path = self._resolve("calibration")
        return ["--calibration", path] if path else []

    def _undistort_flag(self) -> List[str]:
        """Translate the ``undistort`` task-config field to ``["--undistort"]`` or [].

        ``step.undistort`` (per-step override) wins over ``global.undistort``;
        both default ``False``. Only meaningful on frame-reading steps
        (ma_masks, ma_2d, ma_vis). ma_3d has no distortion math, so the
        flag would be a no-op there and is intentionally omitted from
        :class:`Ma3dBuilder`.
        """
        val = self.step_cfg.get("undistort")
        if val is None:
            val = self.global_cfg.get("undistort", False)
        return ["--undistort"] if bool(val) else []

    def _frame_range_flags(self) -> List[str]:
        """Translate ``global.start_frame`` / ``global.end_frame`` to argv.

        Returns ``["--start", "<n>", "--end", "<n>"]`` (or a subset)
        depending on which fields are set. **Only emitted by
        :class:`MaCapBuilder`** — ma_cap is the canonical owner of the
        frame range for a run; downstream steps inherit it via the
        per-camera NPZ (``frame_start`` / ``frame_end`` fields read by
        :func:`capture.frame_source_from_cam_data`).
        """
        flags: List[str] = []
        start = self.global_cfg.get("start_frame")
        end = self.global_cfg.get("end_frame")
        if start is not None:
            flags += ["--start", str(int(start))]
        if end is not None:
            flags += ["--end", str(int(end))]
        return flags

    # --- subclass interface ----------------------------------------------

    def python_argv(self, seq_name: str) -> List[str]:
        """Return ``[script, *args]`` — what to invoke under python."""
        raise NotImplementedError

    def build_argv(self, seq_name: str) -> List[str]:
        """Call :meth:`python_argv` and substitute ``{seq_name}`` placeholders.

        Task configs may reference per-sequence paths via the literal
        token ``{seq_name}`` (e.g. ``videos_dir: data/<dataset>/{seq_name}/videos_crf24``)
        so a single task config can iterate over many sequences within a
        capture. The substitution is applied after ``python_argv`` runs,
        so subclasses don't need to know about templating.
        """
        argv = self.python_argv(seq_name)
        return [a.replace("{seq_name}", seq_name) if isinstance(a, str) else a for a in argv]

    def container_cwd(self) -> str:
        """Working dir inside the container for apptainer/docker engines.

        ``/repo`` is the repo root (see :meth:`binds`); the step runs from
        its subdir under it, mirroring the conda engine's
        ``cwd = <root>/<step subdir>``.
        """
        if self.repo_path:
            from ..env import repo_root
            try:
                rel = os.path.relpath(self.repo_path, str(repo_root()))
            except ValueError:
                rel = ""
            if rel and rel != "." and not rel.startswith(".."):
                return f"/repo/{rel}"
        return "/repo"

    def host_cwd(self) -> str:
        """Working dir on host for the conda engine."""
        return self.repo_path
