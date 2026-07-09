"""Blender headless SMPL-X exporter — a CLIENT of the SMPL-X Blender add-on.

Run via the portable Blender:
  blender --background --python scripts/blender_smplx_export.py -- \
      --addon-dir data/blender_addon --npz body.npz --out-prefix out/body \
      --formats fbx,abc,bvh,usd --fps 30 --target blender

We do NOT install or duplicate the add-on: we import it as a library, call
register(), and drive its operators (scene.smplx_add_gender,
object.smplx_add_animation, object.smplx_export_fbx/_alembic). BVH/USD aren't
add-on features, so those use Blender's native exporters on the add-on-built rig.

The incoming npz is Z-up-normalized upstream with the relaxed-hand mean baked in,
so we import it as anim_format=AMASS + hand_reference=FLAT (which reproduces the
Z-up body faithfully). --target sets each format's destination up-axis + units.
"""
import sys
import os

import bpy


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--addon-dir", required=True,
                   help="Dir containing the 'smplx_blender_addon' package (with data/*.blend).")
    p.add_argument("--npz", required=True, help="Add-on-format animation npz to import.")
    p.add_argument("--out-prefix", required=True, help="Output path prefix (no extension).")
    p.add_argument("--formats", default="fbx,abc", help="Comma list: fbx,abc,bvh,usd.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--unit", default="m", choices=["m", "cm"],
                   help="Units for the geometry: m (meters) or cm (centimeters).")
    return p.parse_args(argv)


def _enable_addon(addon_dir):
    addon_dir = os.path.abspath(addon_dir)
    if addon_dir not in sys.path:
        sys.path.insert(0, addon_dir)
    import smplx_blender_addon
    try:
        smplx_blender_addon.register()
    except Exception as e:
        # Already-registered is fine; anything else re-raises.
        print(f"[blender_export] register note: {e}")
    return smplx_blender_addon


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _select(objs, active):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        if o:
            o.select_set(True)
    bpy.context.view_layer.objects.active = active


def _bake_shape_betas(mesh):
    """Bake the static Shape### (beta) shape keys into the mesh points, keeping
    the animated Pose*/Exp* keys intact as deltas. The betas are weights outside
    [0,1] and Blender's USD *importer* clamps blendshape weights to [0,1], which
    reverts the body toward the template (feet float by a per-subject amount);
    with the shape baked there is nothing left to clamp. Mutates the rig, so the
    USD export must stay last in the format order."""
    keys = mesh.data.shape_keys
    if not keys:
        return
    kb = keys.key_blocks
    shape = [k for k in kb if k.name.startswith("Shape")]
    if not shape:
        return
    import numpy as np
    n = len(mesh.data.vertices)
    basis = np.empty(n * 3)
    kb[0].data.foreach_get("co", basis)
    offset = np.zeros(n * 3)
    for k in shape:
        if k.value == 0.0:
            continue
        co = np.empty(n * 3)
        k.data.foreach_get("co", co)
        offset += k.value * (co - basis)
    for k in shape:
        mesh.shape_key_remove(k)
    # Shift Basis + all remaining keys by the same offset: every key's delta to
    # its relative key is preserved, so the Pose correctives keep working.
    for k in kb:
        co = np.empty(n * 3)
        k.data.foreach_get("co", co)
        k.data.foreach_set("co", co + offset)
    co = np.empty(n * 3)
    mesh.data.vertices.foreach_get("co", co)
    mesh.data.vertices.foreach_set("co", co + offset)
    mesh.data.update()
    print(f"[blender_export] baked {len(shape)} Shape keys into the mesh for USD")


def main():
    a = _args()
    formats = [f.strip().lower() for f in a.formats.split(",") if f.strip()]
    _reset_scene()
    _enable_addon(a.addon_dir)

    # smplx_add_animation ADDS the model AND animates it (it calls
    # scene.smplx_add_gender internally). Do NOT add a body first or you get a
    # duplicate and export the wrong, un-animated one. Set the model variant +
    # UV first (the add-model reads them), like the reference does.
    wm = bpy.context.window_manager
    wm.smplx_tool.smplx_version = "locked_head"
    wm.smplx_tool.smplx_uv = "UV_2023"
    bpy.ops.object.smplx_add_animation(
        filepath=os.path.abspath(a.npz),
        anim_format="AMASS",            # our npz is AMASS Y-up
        hand_reference="FLAT",          # relaxed-hand mean already baked in
        target_framerate=a.fps,
        keyframe_corrective_pose_weights=True,  # keyframe pose-correctives per frame
    )
    mesh = bpy.context.view_layer.objects.active
    arm = mesh.parent if mesh else None
    sc = bpy.context.scene
    print(f"[blender_export] animated body: mesh={mesh.name if mesh else None} "
          f"arm={arm.name if arm else None} frames={sc.frame_start}-{sc.frame_end}")

    out = os.path.abspath(a.out_prefix)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # All formats derive from the single npz source — no per-format reorientation.
    # `unit` is the only knob: m (scale 1) or cm (scale 100). The add-on FBX UNREAL
    # preset bakes the x100, so cm -> UNREAL preset, m -> UNITY preset.
    cm = (a.unit == "cm")
    scale = 100.0 if cm else 1.0
    fbx_preset = "UNREAL" if cm else "UNITY"
    print(f"[blender_export] unit={a.unit} (scale {scale:g}) -> fbx preset={fbx_preset}")

    # Export each requested format in isolation: one format's failure (e.g. a
    # Blender-version quirk in the add-on's cm/UNREAL FBX path) must NOT abort
    # the others. Collect per-format outcomes and only hard-fail if nothing wrote.
    written, failed = [], []

    def _do(fmt, path, fn):
        if fmt not in formats:
            return
        try:
            fn()
            if os.path.exists(path) and os.path.getsize(path) > 0:
                written.append(path)
            else:
                failed.append(fmt)
                print(f"[blender_export] {fmt.upper()} produced no file")
        except Exception as e:
            failed.append(fmt)
            print(f"[blender_export] {fmt.upper()} export FAILED: {type(e).__name__}: {e}")

    def _fbx():
        _select([mesh, arm], mesh)
        bpy.ops.object.smplx_export_fbx(filepath=out + ".fbx", target_format=fbx_preset)

    def _abc():  # native (the add-on's ABC op is this call without a scale arg)
        _select([mesh, arm], mesh)
        bpy.ops.wm.alembic_export(filepath=out + ".abc", selected=True, packuv=False,
                                  face_sets=True, global_scale=scale)

    def _bvh():  # native exporter (no add-on BVH op)
        _select([arm], arm)
        bpy.ops.export_anim.bvh(filepath=out + ".bvh", frame_start=bpy.context.scene.frame_start,
                                frame_end=bpy.context.scene.frame_end, root_transform_only=False,
                                global_scale=scale)

    def _usd():
        _bake_shape_betas(mesh)
        _select([mesh, arm], mesh)
        usd_kw = dict(filepath=out + ".usd", selected_objects_only=True, export_animation=True)
        if cm:
            usd_kw.update(meters_per_unit=0.01)  # centimeters
        bpy.ops.wm.usd_export(**usd_kw)

    _do("fbx", out + ".fbx", _fbx)
    _do("abc", out + ".abc", _abc)
    _do("bvh", out + ".bvh", _bvh)
    _do("usd", out + ".usd", _usd)

    for w in written:
        print(f"[blender_export] OK   {w} ({os.path.getsize(w)} B)")
    requested = [f for f in ("fbx", "abc", "bvh", "usd") if f in formats]
    if failed:
        print(f"[blender_export] {len(failed)}/{len(requested)} format(s) failed: {','.join(failed)}")
    if requested and not written:
        raise SystemExit("[blender_export] all requested formats failed")


if __name__ == "__main__":
    main()
