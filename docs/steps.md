# Pipeline steps

The MAMMA local runner orchestrates five pipeline steps. Each step's code is
tracked directly under its step directory in this repo; the runner provides a
thin builder per step that knows how to invoke the step's entry script with
the right argv shape.

| Step id    | Human name      | Step directory     | Builder                       |
|------------|-----------------|--------------------|-------------------------------|
| `ma_cap`   | capture (input) | `capture/`         | `inference/steps/ma_cap.py`   |
| `ma_masks` | segmentation    | `segmentation/`    | `inference/steps/ma_masks.py` |
| `ma_2d`    | landmarks       | `landmarks/`       | `inference/steps/ma_2d.py`    |
| `ma_3d`    | optimization    | `optimization/`    | `inference/steps/ma_3d.py`    |
| `ma_vis`   | visualization   | `visualization/`   | `inference/steps/ma_vis.py`   |

`visualization/` is a library-quality Python module. Importable as
``from visualization import run_visualization`` and runnable as
``python -m visualization``. The runner subprocesses
``visualization/run_ma_vis.py`` (a thin shim around the same CLI).

## Default pipeline

```
ma_cap ──► ma_masks ──► ma_2d ──► ma_3d ──► ma_vis
                          ▲                    ▲
                          └─── ma_cap ─────────┘
```

`ma_2d` consumes both `ma_cap` (frames) and `ma_masks` (person masks).
`ma_3d` consumes `ma_cap` (cameras + frames) and `ma_2d` (2D landmarks).
`ma_vis` consumes everything upstream and renders the final SMPL-X result
overlays.

To run a subset, set `enabled: false` on the steps you want to skip, the
runner drops them and reorders the remaining steps so each one's inputs
are produced before it runs.

## Adding a step

1. Drop a new builder under `inference/steps/<step_id>.py` that subclasses
   `StepBuilder` and implements `python_argv(seq_name)`.
2. Register it in `inference/steps/__init__.py::BUILDERS`.
3. Add `<step_id>` to `ALL_STEPS` in `inference/runner.py`.
