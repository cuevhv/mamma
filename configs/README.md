# `configs/`

Configuration files used by the runner. The full schema reference (preset vs run config, per-step fields, output layout, re-run semantics) lives at [`docs/CONFIGS.md`](../docs/CONFIGS.md).

Layout:

- `examples/presets/`: shipped pipeline presets (`full.yaml`, `quick.yaml`).
- `examples/captures/`: shipped capture descriptors (one JSON per capture).
- `examples/calib/`: shipped calibration files (one YAML per capture).
- `task_local_template.json`: template for the GUI's local runs.
