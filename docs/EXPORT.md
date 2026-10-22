# Exporting SMPL-X results (npz / FBX / Alembic / BVH / USD)

MAMMA's `ma_3d` fit can be exported into formats that drop straight into the
**SMPL-X Blender add-on** and on into game/render engines:

| format | what it is | needs Blender? |
|---|---|---|
| **`.npz`** | the SMPL-X Blender add-on's native animation file (AMASS-style) | **No** |
| **`.abc`** (Alembic) | animated **vertex/geometry cache**, render engines (Houdini, Maya, UE geometry cache) | yes |
| **`.fbx`** | **rigged** skeleton + skinned mesh + animation, game engines (Unity, Unreal) | yes |
| **`.bvh`** | skeleton-only motion (mocap retargeting) | yes |
| **`.usd`** | USD scene (UE5/Unity/Houdini import natively) | yes |

One person → one set of files (`<seq>_body-NN_smplx.npz`, `<seq>_body-NN.fbx`, …).

## Quick start: npz only (no setup)

The npz is pure-Python; it needs nothing beyond the `mamma` env:

```bash
python optimization/export_blender.py \
  --ma-3d-dir output/ma_3d/<tag>/<capture> \
  --seq-name  <sequence> \
  --ma-cap-dir output/ma_cap/<tag>/<capture> \
  --out-dir   output/export/<sequence> \
  --formats   npz
```

It **validates correctness**: it reconstructs MAMMA's saved vertices from the
exported parameters and refuses to write if they don't match (so a subtly wrong
file can't slip through). Import the `.npz` via the add-on's *Add Animation*.

## FBX / Alembic / BVH / USD: one-time setup

These build a rig in Blender, so they need a Blender runtime + the add-on. Both
are fetched with the two scripts below (or via the GUI **Exporter** tab):

```bash
# 1) Portable Blender 4.5 LTS (no install/root; isolated from the mamma env)
bash data/download_blender.sh

# 2) SMPL-X Blender add-on (gated; SMPL-X account at https://smpl-x.is.tue.mpg.de/)
#    Self-contained: add-on code + the locked-head .blend model.
bash data/download_smplx_blender_addon.sh
```

`download_blender.sh` lands a portable Blender in `data/blender/`; the add-on
extracts to `data/blender_addon/`. The exporter auto-detects both. (You can point
at a different Blender with `MAMMA_BLENDER_BIN=/path/to/blender`.)

## Export all formats

```bash
python optimization/export_blender.py \
  --ma-3d-dir output/ma_3d/<tag>/<capture> \
  --seq-name  <sequence> \
  --ma-cap-dir output/ma_cap/<tag>/<capture> \
  --out-dir   output/export/<sequence> \
  --formats   npz,fbx,abc,bvh,usd \
  --unit      m             # or cm (Unreal); --no-ground to keep the exact fit translation
```

If Blender or the add-on isn't present, the exporter **falls back to npz-only**
with a warning; it never fails the run for a missing optional dependency.

## Options

| flag | meaning |
|---|---|
| `--formats` | comma list of `npz,fbx,abc,bvh,usd` (default `npz`) |
| `--blender-format` | which add-on import **Format** the npz is prepared for so it imports **upright**: `auto` (default, keeps your data's axes, reports the matching Format), `amass` (orients npz Z-up → import as *AMASS*), `smplx` (orients npz Y-up → import as *SMPL-X*) |
| `--unit` | units for FBX/ABC/USD/BVH: `m` (meters, default, Blender/Unity/Maya) or `cm` (centimeters, Unreal). The npz is always meters (SMPL-X convention) |
| `--ground` / `--no-ground` | drop the feet to the floor (0 along the auto-detected up-axis). On by default; `--no-ground` keeps the fit's exact translation. **Never changes the axes** |
| `--up-axis` | source up-axis for grounding + geometry normalization. `auto` (default) detects it (foot-plane + body-vertical); `x`/`y`/`z` force it |
| `--fps` | override `mocap_frame_rate` (else read from ma_cap's `global.npz`, else 30) |
| `--smplx-models` | SMPL-X model folder (default `data/body_models/smplx_locked_head`) |
| `--blender-bin` | explicit Blender binary (else `MAMMA_BLENDER_BIN` / `data/blender/` / `PATH`) |
| `--addon-dir` | add-on folder (default `data/blender_addon`) |
| `--no-validate` | skip the round-trip vertex check (not recommended) |

## How the conventions are handled (the tricky bits)

- **Hand pose / `flat_hand_mean`**: the fit's hand convention is **auto-detected**
  from the saved vertices (normal inference is `flat_hand_mean=False`); the relaxed
  mean is baked into absolute hand angles so the add-on imports them as **FLAT**.
- **Coordinate system**: the npz keeps the fit's **own axes**, untouched
  (auto-detected up-axis; works for any capture, not just Z-up). The only optional
  change is **floor grounding** (a pure translation along the up-axis, `--ground`,
  on by default). FBX/ABC/BVH/USD are all built from that single npz via the add-on,
  so they inherit its orientation (FBX additionally gets the add-on's engine adapter).
- **Units**: `--unit m`/`cm` scales the rigged formats only; the npz stays meters.
- **FPS**: taken from ma_cap so animation timing is right.

## Viewing in Blender

Open Blender (the add-on installed normally), use **Add Animation** on the `.npz`
with the **Format** the export reports (`--blender-format auto` logs it; `amass`/
`smplx` make it explicit), the body imports upright + grounded. Or import the
`.fbx`/`.abc`/`.usd` directly. `--unit cm` pre-scales the rigged formats (×100) for
Unreal; `--unit m` keeps meters (Blender/Unity/Maya).

> **Tip:** the `.npz` is the most faithful (it's the add-on's own format and is
> round-trip-validated). FBX/ABC/etc. are derived from it through Blender.
