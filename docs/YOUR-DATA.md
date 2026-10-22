# Run MAMMA on your own data

To run MAMMA on a new multi-view capture you bring two pieces:

1. **Multi-view footage**, one video file (or image directory) per camera, organized into sequences.
2. **Calibration**, per-camera intrinsics + extrinsics.

Then either let the GUI mint a capture descriptor for you, or hand-author one. Both paths end with the same pipeline-run command.

---

## 1. Lay out your footage

Pick a directory under `data/` (gitignored) and use one of two layouts.

**Videos** (one `.mp4` per camera per sequence):

```
data/<your-dataset>/
├── <sequence_1>/
│   └── videos/
│       ├── cam_01.mp4
│       ├── cam_02.mp4
│       └── …
├── <sequence_2>/
│   └── videos/
│       └── …
```

**Image directories** (one directory of frames per camera per sequence):

```
data/<your-dataset>/
├── <sequence_1>/
│   └── images/
│       ├── cam_01/
│       │   ├── 00000.jpg
│       │   └── …
│       ├── cam_02/
│       └── …
```

Camera names (`cam_01`, etc.) are yours to choose; they just have to match the names you list in the calibration and capture descriptor below.

---

## 2. Author the calibration file

Save it alongside the shipped examples (`configs/examples/calib/<your-dataset>.yaml`) or anywhere on disk you like. Three formats are accepted:

| Extension       | Format                                                            |
|-----------------|-------------------------------------------------------------------|
| `.yaml` / `.yml`| Pinhole + radtan distortion, Hamilton `[w, x, y, z]` quaternion.  |
| `.xcp`          | Vicon export (5-parameter Vicon radial distortion).               |
| `.json`         | OpenCV-style: 3×3 intrinsics + 3×4 extrinsics + 5-param distortion. |

Minimal YAML (one block per camera):

```yaml
cameras:
  cam_01:
    camera_model: pinhole
    distortion_model: radtan         # or vicon_radial_2 for Vicon exports
    intrinsics: [fx, fy, cx, cy]
    distortion_coeffs: [k1, k2, p1, p2, k3]
    resolution: [width, height]
    translation: [tx, ty, tz]                  # in metres
    rotation_quaternion: [w, x, y, z]          # Hamilton convention
  cam_02:
    …
```

The keys under `cameras:` must match the camera names you use in your footage filenames (or image-dir names) and in the capture descriptor below.

---

## 3. Mint a capture descriptor

A capture JSON tells the runner where your footage lives and which cameras + sequences belong to it. Two ways:

**GUI (recommended).** Open the **Captures** tab in the GUI (`bash gui/scripts/dev.sh`), click **+ New capture**, point it at your footage root + calibration file. The backend auto-detects sequences (each subfolder) and cameras (from the first sequence's subfolders) and writes the JSON for you.

**Manual.** Create the file at e.g. `configs/examples/captures/<your-dataset>.json`:

```json
{
  "capture_root":  "../../../data/<your-dataset>",
  "calib":         "../calib/<your-dataset>.yaml",
  "cam_fps":       30,
  "videos_subdir": "videos",
  "cams":      ["cam_01", "cam_02", "cam_03", "cam_04"],
  "sequences": {
    "000": { "name": "<sequence_1>" },
    "001": { "name": "<sequence_2>" }
  }
}
```

Notes:

- Paths in `capture_root` and `calib` are resolved **relative to the capture JSON file's parent directory**. Use absolute paths if your JSON lives outside `configs/examples/captures/`.
- For the image-directory layout, set `"videos_subdir": "images"` (or whatever your per-camera subdirectory is called).
- `cam_fps` is read by `ma_cap` and stamped into the per-camera NPZs the rest of the pipeline consumes.

---

## 4. Run the pipeline

```bash
python -m inference run \
  --cfg     configs/examples/presets/quick.yaml \
  --capture configs/examples/captures/<your-dataset>.json \
  --out-tag run01 -v
```

`quick.yaml` runs a 30-frame slice (`start_frame: 60`, `end_frame: 90`) for a fast end-to-end check. Switch to `presets/full.yaml` for full-frame runs.

Or use the GUI's **Tasks** board: pick the preset + your capture, click **Run**.

Outputs land under `output/ma_*/run01/<your-dataset>/<sequence>/…`, see [CONFIGS.md → Output path layout](CONFIGS.md#output-path-layout) for the per-segment meaning.

### Even quicker: skip the capture JSON

If you only want to run *one* sequence and don't want to author the capture descriptor, you can pass the footage + calibration + sequence name directly. The runner synthesizes the capture in memory:

```bash
python -m inference run \
  --cfg      configs/examples/presets/quick.yaml \
  --footage  data/<your-dataset> \
  --seq_name <sequence_1> \
  --calib    configs/examples/calib/<your-dataset>.yaml \
  --out-tag  run01 -v
```

Cameras and the videos-subdir layout are auto-detected from `<footage>/<seq_name>/`. Equivalent to building a capture JSON for one sequence and running with `--capture`, but no file written to disk. One sequence per invocation.

---

## Reference

- Capture JSON + preset schema: [`docs/CONFIGS.md`](CONFIGS.md)
- Pipeline step → builder mapping: [`docs/steps.md`](steps.md)
- What each step produces: [README → Pipeline](../README.md#pipeline)
