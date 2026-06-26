# `capture/loaders` — calibration file loaders

Format-specific parsers for multi-camera calibration. Each loader reads one file
format and returns `{cam_name: Camera}`; the public entry point
[`capture.calibration.load_calibration`](../calibration.py) picks the right loader
and wraps the result in a `Calibration`.

```python
from capture.calibration import load_calibration

calib = load_calibration("path/to/calib.yaml")   # or a folder, .xcp, .json, …
for name, cam in calib.cameras.items():
    cam.intrinsics    # 3x3 K (fx, fy, cx, cy)
    cam.T_cam_world   # 4x4 world -> camera  (projection-ready)
    cam.T_world_cam   # 4x4 camera pose in the world (inverse)
```

## The one thing to get right: extrinsics convention

This is the historical source of user confusion. Two ways to express extrinsics:

- **world→camera (`world2cam`)**: `X_cam = R · X_world + t`. The matrix `[R|t]`
  is `T_cam_world`. (OpenCV / EasyMocap / COLMAP use this.)
- **camera→world / camera pose (`cam2world`)**: `R`,`t` are the camera's
  orientation/position *in the world*. The matrix is `T_world_cam`; its inverse
  is `T_cam_world`. The camera centre is `t` directly (vs `-Rᵀt` for world2cam).

Every loader normalizes to the same in-memory `Camera`, which stores **both**
`T_cam_world` (what the projection pipeline uses) and `T_world_cam` (the natural
pose). What differs is the *input* convention each format assumes — see the table.

A wrong convention does **not** error: the pipeline runs and produces garbage 3D.
Use the GUI's **"Preview camera rig"** (`gui/backend/calib_preview.py`) to check —
a correct rig shows cameras around the scene, looking inward.

## Supported formats

| Loader | Extension / shape | Input convention | Distortion | Quaternion | Notes |
|---|---|---|---|---|---|
| `yaml_loader.py` | single `.yaml`/`.yml`, MAMMA schema | **declared** via `extrinsics_convention` (default `cam2world`) | `radtan` (4) / `vicon_radial_2` (5) | `[w,x,y,z]` Hamilton (`quaternion_order: xyzw` to flip) | The recommended format for new rigs. |
| `opencv_loader.py` | OpenCV FileStorage `.yml` (single file, or a **folder** of per-camera files) | **world2cam** (`K`/`D`/`R`/`T`) | `opencv_brown` (5) | — (matrices) | Tolerant parser (handles tabs + the non-standard `%YAML:1.0` directive that crash `cv2.FileStorage`). Camera name = file stem; resolution inferred from the principal point if absent. |
| `easymocap_loader.py` | a **folder** with `intri.yml` + `extri.yml` | **world2cam** | `opencv_brown` (5) | — (`Rot_*` matrix or `R_*` Rodrigues) | Camera names from the `names` list. |
| `xcp_loader.py` | Vicon `.xcp` (XML) | camera pose in world | `vicon_radial_2` (5) | `[x,y,z,w]` JPL/scipy | Millimetres → metres at parse time. |
| `json_loader.py` | OpenCV-style `.json` (nested / flat / legacy) | OpenCV layout: **world2cam** `extrinsics_matrix`; legacy: pose-in-world | `opencv_brown` (5) / `vicon_radial_2` | — (matrices) | Auto-detects the sub-layout. |

The MAMMA YAML schema (with the new explicit fields):

```yaml
extrinsics_convention: cam2world   # optional; or world2cam. Default = cam2world.
quaternion_order: wxyz             # optional; or xyzw. Default = wxyz (Hamilton).
cameras:
  <cam_name>:
    camera_model: pinhole
    distortion_model: radtan          # or vicon_radial_2
    intrinsics: [fx, fy, cx, cy]
    distortion_coeffs: [k1, k2, p1, p2]
    resolution: [W, H]
    translation: [tx, ty, tz]         # metres
    rotation_quaternion: [w, x, y, z] # unit norm
```

## Dispatch rules (`load_calibration`)

1. **Directory** → EasyMocap if it has `intri.yml` + `extri.yml`, otherwise a
   folder of OpenCV FileStorage files (`opencv_loader`).
2. **`.xcp`** → `xcp_loader`; **`.json`** → `json_loader`.
3. **`.yaml` / `.yml`** → content-sniffed: an OpenCV FileStorage header
   (`%YAML:1.0` / `!!opencv-matrix`) routes to `opencv_loader`, otherwise the
   MAMMA `yaml_loader`.

## The shared data model

All loaders return [`capture.calibration.Camera`](../calibration.py):
`name`, `width`, `height`, `intrinsics` (3×3 float64), `distortion_model`
(one of `VALID_DISTORTION_MODELS`: `radtan`, `opencv_brown`, `vicon_radial_2`),
`distortion_coeffs`, `T_cam_world` (4×4), `T_world_cam` (4×4). `ma_cap` bakes
`T_cam_world` into each per-camera NPZ as `cam_ext`; everything downstream
projects with `K · T_cam_world · X`.

## Adding a new loader

1. Create `myformat_loader.py` exposing `load(path) -> Dict[str, Camera]`. Build
   each `Camera` with **both** transforms — set whichever the format gives you and
   compute the inverse (`T_world_cam = inv(T_cam_world)` etc.). Raise
   `CalibrationError` (from `..calibration`) on malformed input.
2. Wire it into `load_calibration` in [`../calibration.py`](../calibration.py)
   (extension or directory/content detection).
3. Add a row to the table above and, ideally, a fixture + a round-trip test that
   projects a known world point.

Reuse helpers: `opencv_loader.parse_fs` (tolerant FileStorage reader),
`opencv_loader._camera_from_KDRT` (world2cam K/D/R/T → `Camera`),
`.._quat.hamilton_quat_to_rotmat`.
