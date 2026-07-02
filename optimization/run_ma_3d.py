import os, glob, sys
import shutil
import numpy as np
import torch
import cv2
from utils_smplx import get_smplx_models, get_smplx_forward

from scene_debug.vis import build_scene
from scene_debug.renderer import Renderer
from utils.fitting import fit_smplx
from utils.utils_camera import w2c
from utils.paths_config import PathsConfig

import argparse
from typing import List, Optional
import yaml
from utils.mesh_post_proc import GeometryModifierCutHands
import pandas as pd
from utils_draw import save_images
from utils.get_metrics import procrustes_analysis_batch

if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_id)
    print(f"Using GPU: {gpu_name} (ID: {gpu_id})")
else:
    print("No GPU available, using CPU.")


ROTATED_CAMS = [10, 11, 12, 13, 14, 15, 16]


def _to_path_string(path_value):
    if isinstance(path_value, bytes):
        return path_value.decode("utf-8")
    return str(path_value)


def _resolve_cam_img_path(cam_metadata, frame_idx: int, imgs_pth: str):
    if "img_abs_path" in cam_metadata.files:
        path_value = _to_path_string(cam_metadata["img_abs_path"][frame_idx])
        return os.path.normpath(os.path.join(imgs_pth, path_value))
    if "img_rel_path" in cam_metadata.files:
        path_value = _to_path_string(cam_metadata["img_rel_path"][frame_idx])
        return os.path.normpath(os.path.join(imgs_pth, path_value))
    return None


def _write_run_settings(save_prediction_fn: str, optim_cfg_fn: Optional[str] = None, cli_args=None):
    import json
    os.makedirs(save_prediction_fn, exist_ok=True)
    if cli_args is not None:
        args_fn = os.path.join(save_prediction_fn, "run_args.json")
        with open(args_fn, "w", encoding="utf-8") as f:
            json.dump(vars(cli_args), f, indent=2)
        print(f"Saved CLI args to: {args_fn}")
    if optim_cfg_fn is not None:
        cfg_src = os.path.abspath(optim_cfg_fn)
        cfg_dst = os.path.join(save_prediction_fn, "run_config.yaml")
        if os.path.exists(cfg_src):
            shutil.copy2(cfg_src, cfg_dst)
            print(f"Saved run config copy to: {cfg_dst}")
        else:
            print(f"Could not copy run config: source does not exist: {cfg_src}")


def _run_detection_analysis(pred_pth: str, out_path: str, out_fn: str, cam_names: List[str],
                            cam_name_prefix: str, top_k_frames: int = 30):
    if pred_pth is None or not os.path.exists(pred_pth):
        print("Skipping detection analysis: prediction path is missing.")
        return None
    try:
        from analyze_ma_2d_detections import run_analysis
        analysis_out_dir = os.path.join(out_path, out_fn, "detection_analysis")
        run_analysis(
            pred_dir=pred_pth,
            out_dir=analysis_out_dir,
            cam_names=cam_names or [],
            cam_name_prefix=cam_name_prefix or "IOI_",
            top_k_frames=max(1, int(top_k_frames)),
        )
        return analysis_out_dir
    except Exception as e:
        print(f"Detection analysis failed (non-fatal): {e}")
        return None


def _reencode_to_h264(path: str) -> None:
    """Re-encode an mp4v file in place to browser-friendly H.264/yuv420p.

    OpenCV's pip wheel can only encode mp4v (MPEG-4 Part 2), which browsers
    won't play inline, so we transcode the finished file with ffmpeg (same
    recipe ma_vis/ma_masks use). Best-effort: no-ops if ffmpeg is unavailable
    or the encode fails, leaving the original mp4v in place.
    """
    import shutil
    import subprocess
    path = str(path)
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not os.path.exists(path):
        return
    tmp = path + ".tmp.mp4"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-i", path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", tmp],
            check=True,
        )
        os.replace(tmp, path)
    except (subprocess.CalledProcessError, OSError):
        if os.path.exists(tmp):
            os.remove(tmp)


def render_scene_videos(smplx_preds_by_body: dict, cameras_metadata_fns: List[str], imgs_pth: str, out_dir: str,
                        faces, fps: int = 30):
    if len(smplx_preds_by_body) == 0:
        print("No SMPL-X predictions available for rendering scene videos.")
        return []

    os.makedirs(out_dir, exist_ok=True)
    body_ids = sorted(list(smplx_preds_by_body.keys()))
    pred_vertices_world = {
        body_id: smplx_preds_by_body[body_id].vertices.detach().cpu().numpy()
        for body_id in body_ids
    }
    max_frames = min(pred_vertices_world[body_id].shape[0] for body_id in body_ids)
    video_paths = []

    renderer_init_failed = False
    for camera_metadata_fn in cameras_metadata_fns:
        cam_metadata = np.load(camera_metadata_fn, allow_pickle=True)
        cam_id = os.path.splitext(os.path.basename(camera_metadata_fn))[0]
        print(f"Rendering scene video for camera: {cam_id}")

        if "cam_int" not in cam_metadata.files or "cam_ext" not in cam_metadata.files:
            print(f"Skipping camera {cam_id} because cam_int/cam_ext are missing.")
            continue

        cam_int = cam_metadata["cam_int"]
        cam_ext = cam_metadata["cam_ext"].copy()
        # Ensure extrinsics translation is in meters (auto-detect mm by magnitude).
        if np.abs(cam_ext[:3, -1]).max() > 200:
            cam_ext[:3, -1] = cam_ext[:3, -1] / 1000.0

        frames_to_render = max_frames
        if "img_abs_path" in cam_metadata.files:
            frames_to_render = min(frames_to_render, len(cam_metadata["img_abs_path"]))
        elif "img_rel_path" in cam_metadata.files:
            frames_to_render = min(frames_to_render, len(cam_metadata["img_rel_path"]))

        if frames_to_render <= 0:
            print(f"Skipping camera {cam_id}: no frames available.")
            continue

        first_img = None
        first_img_fn = _resolve_cam_img_path(cam_metadata, 0, imgs_pth)
        if first_img_fn is not None and os.path.exists(first_img_fn):
            first_img = cv2.imread(first_img_fn)

        if first_img is not None:
            img_h, img_w = first_img.shape[:2]
        else:
            img_h = int(cam_metadata["cam_img_h"]) if "cam_img_h" in cam_metadata.files else 3008
            img_w = int(cam_metadata["cam_img_w"]) if "cam_img_w" in cam_metadata.files else 4112
            print(f"Could not load camera images for {cam_id}. Using black canvas {img_w}x{img_h}.")

        try:
            renderer = Renderer(
                focal_length_px=float(cam_int[0, 0]),
                img_w=img_w,
                img_h=img_h,
                faces=faces,
                principal_p_x=float(cam_int[0, 2]),
                principal_p_y=float(cam_int[1, 2]),
            )
        except Exception as e:
            if not renderer_init_failed:
                print(f"Skipping scene videos: renderer init failed ({e})")
                print(
                    "Hint: use --skip_scene_videos for non-render runs, "
                    "or configure EGL/OSMesa on the cluster "
                    "(PYOPENGL_PLATFORM and EGL_DEVICE_ID)."
                )
                renderer_init_failed = True
            break

        output_video_path = os.path.join(out_dir, f"{cam_id}_smplx_scene.mp4")
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (img_w, img_h),
        )
        if not writer.isOpened():
            print(f"Failed to open video writer for {output_video_path}. Skipping camera.")
            renderer.delete()
            continue

        for frame_idx in range(frames_to_render):
            frame_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            frame_img_fn = _resolve_cam_img_path(cam_metadata, frame_idx, imgs_pth)
            if frame_img_fn is not None and os.path.exists(frame_img_fn):
                loaded_img = cv2.imread(frame_img_fn)
                if loaded_img is not None:
                    if loaded_img.shape[0] != img_h or loaded_img.shape[1] != img_w:
                        loaded_img = cv2.resize(loaded_img, (img_w, img_h))
                    frame_img = loaded_img

            vertices_cam = []
            for body_id in body_ids:
                verts_world = pred_vertices_world[body_id][frame_idx]
                vertices_cam.append(w2c(verts_world, cam_ext))

            rendered = renderer.render_front_view(verts=vertices_cam, bg_img_bgr=frame_img)
            writer.write(rendered)

        writer.release()
        _reencode_to_h264(output_video_path)
        renderer.delete()
        video_paths.append(output_video_path)
        print(f"Saved scene video for {cam_id}: {output_video_path}")

    return video_paths


def read_gt(metadata_world: dict, smplx_model, batch_size: int, paths: PathsConfig,
            body_id: int = 0, device: str = "cuda",
            v_template=None, use_bun_model=False, flat_hand=True):

    if metadata_world["pose_world"].shape[-2:] == (3,3):
    # from rotmat to axis angle
        from scipy.spatial.transform import Rotation as R
        rot_mats = metadata_world["pose_world"].reshape(-1, 3, 3)
        rots = R.from_matrix(rot_mats)
        axis_angles = rots.as_rotvec()  # shape (N, 3)

        # Reshape back to original leading dimensions + (3,)
        axis_angles = axis_angles.reshape(*metadata_world["pose_world"].shape[:-2], 3)
        axis_angles = axis_angles.squeeze(2)
        # flatten last dimension
        axis_angles = axis_angles.reshape(*axis_angles.shape[:-2], -1)
    else:
        axis_angles = metadata_world["pose_world"]
    if "ioi_dir" in metadata_world and "CHI3D" in str(metadata_world["ioi_dir"]):
        print("changing jaw pose to zero for CHI3D")
        axis_angles[:, body_id, 66:69] = 0.0  #

    smplx_pose = torch.tensor(axis_angles[:, body_id]).float().to(device)

    smplx_trans = torch.tensor(metadata_world["pose_trans_world"][:, body_id]).float().to(device)


    if len(metadata_world["shape"].shape) == 3:
        smplx_betas = torch.tensor(metadata_world["shape"][:, body_id]).float().to(device)
    else:
        smplx_betas = torch.tensor(metadata_world["shape"][body_id])[None,].repeat((batch_size, 1)).float().to(device)

    gender = "neutral"
    smplx_model_name = paths.smplx_lockhead_models
    if use_bun_model:
        if not paths.bun_models:
            raise ValueError(
                "use_bun_model: True requires --bun-models / MAMMA_BUN_MODELS to be set."
            )
        smplx_model_name = paths.bun_models
        print("Using bun model")
    else:
        print("Using lockhead model")
    n_betas = smplx_model["neutral"].betas.shape[-1]
    print("using flat hand: ", flat_hand)
    if v_template is not None:
        import smplx
        gender = metadata_world["gender"][0] if v_template is not None else "neutral"
        print("Using custom v_template")

        smplx_model_gt = {gender: smplx.create(smplx_model_name, model_type='smplx',
                                        gender=gender,
                                        ext='npz',
                                        flat_hand_mean=flat_hand,
                                        num_pca_comps=45,
                                        num_betas=n_betas, v_template=v_template[body_id],
                                        use_pca=False).to(device)}
    elif metadata_world["gender"][0] != "neutral":
        import smplx
        gender = metadata_world["gender"][0]
        print("Using 12 pca components")

        smplx_model_gt = {gender: smplx.create(smplx_model_name, model_type='smplx',
                                        gender=gender,
                                        ext='npz',
                                        flat_hand_mean=flat_hand,
                                        num_pca_comps=12,
                                        num_betas=n_betas, v_template=None,
                                        use_pca=True).to(device)}

    else:
        smplx_model_gt = smplx_model

    smplx_out = get_smplx_forward(smplx_pose,
                                    smplx_betas[:, :n_betas],
                                    smplx_trans,
                                    gender=gender,
                                    smplx_models=smplx_model_gt)


    return smplx_out


def process_results(smplx_out_gt, smplx_out_pred, body_id, out_fn, cameras_metadata_fns,
                    imgs_pth, batch_size, pred_fns, faces, downsampled_verts_mat=None, save_imgs=False, out_folder="tmp", start_frame=5,
                    save_one_cam=True):
    verts_gt = smplx_out_gt.vertices.detach().cpu() # [5:,]
    joints_gt = smplx_out_gt.joints.detach().cpu() # [5:,]
    verts_pred = smplx_out_pred.vertices.detach().cpu() # [5:,]
    joints_pred = smplx_out_pred.joints.detach().cpu() # [5:,]

    pred_joints_pa, scale, R, t  = procrustes_analysis_batch(joints_pred.numpy(), joints_gt.numpy(), list(range(22)))
    pred_verts_pa, scale, R, t = procrustes_analysis_batch(verts_pred.numpy(), verts_gt.numpy())

    geo_mod = GeometryModifierCutHands('./utils/hand_removal.npz')
    vertices_idx = geo_mod.vertices_to_keep
    gt_verts_no_hands = verts_gt[:, vertices_idx]
    pred_verts_no_hands = verts_pred[:, vertices_idx]
    pred_verts_pa_no_hands = pred_verts_pa[:, vertices_idx]

    metrics = {
        'pve': torch.linalg.norm(verts_gt[start_frame:] - verts_pred[start_frame:], axis=-1),
        'mpjpe': torch.linalg.norm(joints_gt[start_frame:] - joints_pred[start_frame:], axis=-1),
        'pve_pa': torch.linalg.norm(verts_gt[start_frame:] - pred_verts_pa[start_frame:], axis=-1),
        'mpjpe_pa': torch.linalg.norm(joints_gt[start_frame:] - pred_joints_pa[start_frame:], axis=-1),
        'pve_no_hands': torch.linalg.norm(gt_verts_no_hands[start_frame:] - pred_verts_no_hands[start_frame:], axis=-1),
        'pve_pa_no_hands': torch.linalg.norm(gt_verts_no_hands[start_frame:] - pred_verts_pa_no_hands[start_frame:], axis=-1),
    }

    for key, value in metrics.items():
        print(f"{key}: {value.mean().item()} median: {value.median().item()}")

    metrics_out = {key: value.mean(axis=-1) for key, value in metrics.items()}

    df = pd.DataFrame(metrics_out)

    # save diffs per sequence in csv
    csv_path = os.path.join(out_folder, out_fn)
    os.makedirs(csv_path, exist_ok=True)

    csv_path = os.path.join(csv_path, f"body_id-{body_id:02d}.csv")
    df.to_csv(csv_path, index=False)

    if save_imgs:
        save_images(body_id, cameras_metadata_fns, smplx_out_gt, smplx_out_pred, faces, batch_size, imgs_pth, pred_fns, out_folder, out_fn+"_"+f"body_id-{body_id:02d}",
                    save_one_cam=save_one_cam)

    print("done processing results")


def process_metadata(optim_cfg_fn, metadata_data_pth, pred_pth, downsampled_verts_mat_path, smplx_model_pth, cam_names,
                     hand_joints_pred_pth=None, use_gt=True, device="cuda", cam_name_prefix=None):
    with open(optim_cfg_fn, 'r') as f:
        optim_cfg = yaml.load(f, Loader=yaml.FullLoader)
    use_v_template = optim_cfg["use_v_template"]
    num_betas = optim_cfg["n_betas"]

    use_cam_names = cam_names is not None and len(cam_names) > 0
    print("Camera names provided: ", cam_names)
    cam_name_prefix = cam_name_prefix or "IOI_"

    print(metadata_data_pth, cam_names if use_cam_names else cam_name_prefix)
    print(os.path.exists(metadata_data_pth))

    if use_cam_names:
        # Get all available camera metadata files and filter by cam_names
        all_cameras_metadata_fns = glob.glob(os.path.join(metadata_data_pth, '*.npz'))
        all_cameras_metadata_fns = sorted(all_cameras_metadata_fns)

        # Filter by camera names
        cameras_metadata_fns = []
        for cam_name in cam_names:
            matching_files = [f for f in all_cameras_metadata_fns if cam_name in os.path.basename(f)]
            cameras_metadata_fns.extend(matching_files)
    else:
        cameras_metadata_fns = glob.glob(os.path.join(metadata_data_pth, f'{cam_name_prefix}*.npz'))
        cameras_metadata_fns = sorted(cameras_metadata_fns)

    print("cameras_metadata_fns: ", cameras_metadata_fns)
    selected_cameras_id = [i for i in range(len(cameras_metadata_fns))]

    cameras_metadata_fns = [cameras_metadata_fns[i] for i in selected_cameras_id]
    print("selected cameras: ", cameras_metadata_fns)
    print("prediction path: ", pred_pth)

    if pred_pth is not None:
        if use_cam_names:
            # Get all prediction files and filter by camera names
            all_pred_fns = [
                f for f in glob.glob(os.path.join(pred_pth, '*.npz'))
                if os.path.basename(f).rsplit("_",1)[-1] != 'diff.npz'
            ]

            # Filter by camera names
            pred_fns = []
            for cam_name in cam_names:
                matching_files = [f for f in all_pred_fns if cam_name in os.path.basename(f)]
                pred_fns.extend(matching_files)
        else:
            pred_fns = [
                f for f in glob.glob(os.path.join(pred_pth, f'{cam_name_prefix}*.npz'))
                if os.path.basename(f).rsplit("_",1)[-1] != 'diff.npz'
            ]

        if pred_fns == []:
            # Fallback to IOI_ prefix if no matches found
            pred_fns = [
                f for f in glob.glob(os.path.join(pred_pth, f'IOI_*.npz'))
                if os.path.basename(f).rsplit("_",1)[-1] != 'diff.npz'
            ]

        pred_fns = sorted(pred_fns)
        print(pred_pth, cam_names if use_cam_names else cam_name_prefix)

        if len(selected_cameras_id) <= len(pred_fns):
            pred_fns = [pred_fns[i] for i in selected_cameras_id]
        else:
            pred_fns = [pred_fns[i] for i in range(len(pred_fns))]
            selected_cameras_id = [i for i in range(len(pred_fns))]
            cameras_metadata_fns = [cameras_metadata_fns[i] for i in selected_cameras_id]
    else:
        pred_fns = None
        downsampled_verts_mat_path = None

    if hand_joints_pred_pth is not None:
        if use_cam_names:
            all_hand_joints_pred_fns = glob.glob(os.path.join(hand_joints_pred_pth, '*.npz'))

            # Filter by camera names
            hand_joints_pred_fns = []
            for cam_name in cam_names:
                matching_files = [f for f in all_hand_joints_pred_fns if cam_name in os.path.basename(f)]
                hand_joints_pred_fns.extend(matching_files)
        else:
            hand_joints_pred_fns = glob.glob(os.path.join(hand_joints_pred_pth, f'{cam_name_prefix}*.npz'))

        hand_joints_pred_fns = sorted(hand_joints_pred_fns)
        hand_joints_pred_fns = [hand_joints_pred_fns[i] for i in selected_cameras_id]
    else:
        hand_joints_pred_fns = None

    # Load camera parameters
    if not os.path.exists(os.path.join(metadata_data_pth, 'global.npz')) or 'MPI_Dance' in metadata_data_pth:
        pred_data = np.load(pred_fns[0], allow_pickle=True)

        metadata_world = None
        batch_size = pred_data["landmarks"].shape[0] # 400
        n_people = pred_data["landmarks"].shape[1] # 1
        use_gt = False

    else:
        metadata_world = np.load(os.path.join(metadata_data_pth, 'global.npz'))
        batch_size = metadata_world["frames_len"].item()
        pred_data = np.load(pred_fns[0], allow_pickle=True)
        if 'people_len' in metadata_world:
            n_people = metadata_world["people_len"].item()
        elif 'landmarks' in pred_data:
            batch_size = pred_data["landmarks"].shape[0]
            n_people = pred_data["landmarks"].shape[1]
        else:
            n_people = 1

    print("sequence length: ", batch_size)

    v_template_pred = None
    if use_v_template:
        v_template_pred = metadata_world["v_template"] if "v_template" in metadata_world else None
    smplx_model = get_smplx_models(smplx_model_pth, num_betas, v_template=v_template_pred, n_people=n_people, device=device)

    per_frame_count = np.max((1, np.round(batch_size/600).astype(int)))
    frames_to_use_idx = list(range(0,batch_size,np.round(per_frame_count).astype(int)))
    if optim_cfg["optim"]['first_run'].get("only_ten_first_frames", False):
        print("Ignoring legacy only_ten_first_frames=True in config; full-sequence mode is enforced.")


    if downsampled_verts_mat_path is not None:
        downsampled_verts_mat = np.load(downsampled_verts_mat_path, allow_pickle=True)
        downsampled_verts_mat = {"downsampled_verts_mat": downsampled_verts_mat}
        base_dir = os.path.dirname(downsampled_verts_mat_path)
        npz_fn = os.path.join(base_dir, "*.npz")
        if npz_fn:
            part_idx = np.load(glob.glob(npz_fn)[0], allow_pickle=True)
            for key in part_idx.files:
                downsampled_verts_mat[key] = part_idx[key]
    else:
        downsampled_verts_mat = None
    return (optim_cfg, metadata_world, smplx_model, batch_size, n_people, cameras_metadata_fns, pred_fns,
            hand_joints_pred_fns, frames_to_use_idx, downsampled_verts_mat, v_template_pred, use_gt)


def save_mesh_seq(smplx_out_pred, frames_to_use_idx, cam_imgs_seq, cam_names, body_id, smplx_model, out_path, out_fn, smplx_out_gt=None):
    mesh_results_list = []
    if smplx_out_gt is not None:
        offsets = [("gt", np.array([[1, 0, 0]])), ("gt_overlapped", np.array([[3, 0, 0]]))]
        for name, offset in offsets:
            mesh_results = dict()
            mesh_results["name"] = name
            mesh_results["vertices"] = smplx_out_gt.vertices.detach().cpu().numpy()[frames_to_use_idx] + offset
            mesh_results["faces"] = smplx_model["neutral"].faces
            mesh_results["color"] = np.array((255, 200, 220))/255.
            mesh_results_list.append(mesh_results)


    for name, offset, color in [("pred", np.array([[0, 0, 0]]), np.array((67, 133, 255))/255.),
                        ("pred_overlapped", np.array([[3, 0, 0]]), np.array((67, 133, 255))/255.)]:
        mesh_results = dict()
        mesh_results["name"] = name
        verts = smplx_out_pred.vertices.detach().cpu().numpy()[frames_to_use_idx]
        mesh_results["vertices"] = verts + offset
        mesh_results["faces"] = smplx_model["neutral"].faces
        mesh_results["color"] = color
        mesh_results_list.append(mesh_results)

    # save mesh
    out_meshes_fn = os.path.join(out_path, "meshes", out_fn)
    os.makedirs(out_meshes_fn, exist_ok=True)
    cam_imgs_seq = None
    build_scene(mesh_results_list, cam_imgs_seq, cam_names, file_name=os.path.join(out_meshes_fn, f"body_id-{body_id:02d}"))


def save_scene_meshes(smplx_out_preds, frames_to_use_idx, cam_imgs_seq, cam_names, smplx_model, out_path, out_fn, smplx_out_gts=None):
    mesh_results_list = []
    color = np.array((67, 133, 255))/255.
    for body_id, smplx_out_pred in enumerate(smplx_out_preds):
        body_id = f"{body_id:02d}"
        mesh_results = dict()
        mesh_results["name"] = "pred_"+body_id
        verts = smplx_out_pred.vertices.detach().cpu().numpy()[frames_to_use_idx]
        mesh_results["vertices"] = verts
        mesh_results["faces"] = smplx_model["neutral"].faces
        # permute color
        mesh_results["color"] = color
        color = color[[2, 0, 1]]
        mesh_results_list.append(mesh_results)

    color = np.array((67, 133, 255))/255.
    if smplx_out_gts is not None:
        for body_id, smplx_out_gt in enumerate(smplx_out_gts):
            body_id = f"{body_id:02d}"
            mesh_results = dict()
            mesh_results["name"] = "gt_"+body_id
            verts = smplx_out_gt.vertices.detach().cpu().numpy()[frames_to_use_idx]
            mesh_results["vertices"] = verts + np.array([[3, 0, 0]])
            mesh_results["faces"] = smplx_model["neutral"].faces
            mesh_results["color"] = color
            color = color[[2, 0, 1]]
            mesh_results_list.append(mesh_results)

    # save mesh
    out_meshes_fn = os.path.join(out_path, "meshes", out_fn)
    os.makedirs(out_meshes_fn, exist_ok=True)
    build_scene(mesh_results_list, cam_imgs_seq, cam_names, file_name=os.path.join(out_meshes_fn, f"optim_seq"))


def _apply_occlusion_gating(optim_cfg, enable_via_cli=False):
    """
    Resolve the opt-in ``occlusion_gating`` config so each optimization stage
    carries its own block (read in ``optimization.fit``).

    Precedence: a per-stage ``occlusion_gating`` wins; otherwise a top-level
    ``occlusion_gating`` is propagated into every stage; the ``--occlusion-aware-weights``
    CLI flag synthesizes a sensible top-level block (hybrid, tau=0.30, keep_top_m=1)
    when none is present. The gate applies to all landmarks. When gating is off
    everywhere this is a no-op and the optimizer behaves exactly as before.
    """
    top = optim_cfg.get("occlusion_gating", None)

    if enable_via_cli:
        top = dict(top) if isinstance(top, dict) else {}
        top.setdefault("mode", "hybrid")
        top.setdefault("tau", 0.30)
        top.setdefault("beta", 0.05)
        top.setdefault("keep_top_m", 1)
        top.setdefault("temporal", True)
        top["enabled"] = True
        optim_cfg["occlusion_gating"] = top

    stages = optim_cfg.get("optim", {}) or {}
    if isinstance(top, dict):
        for run_cfg in stages.values():
            if isinstance(run_cfg, dict) and "occlusion_gating" not in run_cfg:
                run_cfg["occlusion_gating"] = dict(top)

    enabled_stages = [n for n, c in stages.items()
                      if isinstance(c, dict) and isinstance(c.get("occlusion_gating"), dict)
                      and c["occlusion_gating"].get("enabled")]
    if enabled_stages:
        sample = stages[enabled_stages[0]]["occlusion_gating"]
        print(f"[occlusion_gating] occlusion-aware gating ENABLED for stages {enabled_stages} "
              f"(mode={sample.get('mode')}, tau={sample.get('tau')}, "
              f"keep_top_m={sample.get('keep_top_m')}, temporal={sample.get('temporal')}).")
    return optim_cfg


_VPOSER_DEFAULT_WEIGHT = 0.25


def _apply_vposer(optim_cfg, enable_via_cli=False):
    """Opt-in VPoser pose prior (the ``--use-vposer`` CLI flag).

    Adds ``vposer_recon_loss`` (default weight 0.25) to every optimization
    stage's ``losses`` that doesn't already define it, so the VPoser manifold
    prior can be enabled on ANY recipe without editing YAML — the same loss the
    ``*_occlusion_vposer`` recipe configures by hand. No-op when off or where a
    stage already sets ``vposer_recon_loss``. Requires the VPoser weights
    (``MAMMA_VPOSER_DIR``, default ``data/body_models/vposer/V02_05``).
    """
    if not enable_via_cli:
        return optim_cfg
    stages = optim_cfg.get("optim", {}) or {}
    added = []
    for name, run_cfg in stages.items():
        if not isinstance(run_cfg, dict):
            continue
        losses = run_cfg.setdefault("losses", {})
        if "vposer_recon_loss" not in losses:
            losses["vposer_recon_loss"] = {"weight": _VPOSER_DEFAULT_WEIGHT}
            added.append(name)
    if added:
        print(f"[vposer] VPoser pose prior ENABLED (weight={_VPOSER_DEFAULT_WEIGHT}) "
              f"for stages {added} via --use-vposer.")
    return optim_cfg


def main(optim_cfg_fn, cam_names, metadata_data_pth:str, imgs_pth:str, paths: PathsConfig,
         pred_pth:str = None, hand_joints_pred_pth:str = None,
         out_fn:str = None, downsampled_verts_mat_path:str = None,
         smplx_model_pth:Optional[str] = None, device:str = "cuda",
         out_path:str = "out", use_gt = True, cam_name_prefix: str = None, save_scene_videos: bool = True,
         start_frame: int = 0, end_frame: int = None, ignore_start_frames: int = 0,
         save_detection_analysis: bool = True,
         detection_analysis_top_k: int = 30, up_axis: str = "z", cli_args=None):

    if smplx_model_pth is None:
        smplx_model_pth = paths.smplx_lockhead_models

    flat_hand = True
    if "rich" in pred_pth or "chi3d" in pred_pth:
        flat_hand = False

    (optim_cfg, metadata_world, smplx_model, batch_size, n_people, cameras_metadata_fns, pred_fns,
            hand_joints_pred_fns, frames_to_use_idx, downsampled_verts_mat, v_template_pred, use_gt) = process_metadata(optim_cfg_fn,
                                                                                                                        metadata_data_pth,
                                                                                                                        pred_pth,
                                                                                                                        downsampled_verts_mat_path,
                                                                                                                        smplx_model_pth,
                                                                                                                        cam_names,
                                                                                                                        hand_joints_pred_pth,
                                                                                                                        use_gt,
                                                                                                                        device,
                                                                                                                        cam_name_prefix)

    # Opt-in occlusion-aware gating: resolve config (CLI flag +
    # top-level/per-stage blocks) so each stage carries its own settings.
    _apply_occlusion_gating(
        optim_cfg,
        enable_via_cli=bool(getattr(cli_args, "occlusion_aware_weights", False)),
    )
    # Opt-in VPoser pose prior: inject vposer_recon_loss into the stages.
    _apply_vposer(
        optim_cfg,
        enable_via_cli=bool(getattr(cli_args, "use_vposer", False)),
    )
    # Preflight: if any stage (via --use-vposer OR a yaml recipe like
    # *_occlusion_vposer) uses the VPoser prior, verify the optional package
    # and weights NOW — failing seconds in beats a ModuleNotFoundError after
    # re-ID and triangulation have already burned minutes.
    if any("vposer_recon_loss" in (rc.get("losses") or {})
           for rc in (optim_cfg.get("optim", {}) or {}).values()
           if isinstance(rc, dict)):
        from utils.vposer import check_available as _vposer_check
        _vposer_check()

    body_ids = [i for i in range(n_people)]

    total_bodies = len(body_ids)
    processed = 0
    save_prediction_fn = os.path.join(out_path, out_fn)
    _write_run_settings(save_prediction_fn, optim_cfg_fn=optim_cfg_fn, cli_args=cli_args)
    for body_id in body_ids:
        npz_result = os.path.join(save_prediction_fn, f"smplx_params_body_id-{body_id:02d}.npz")
        # Require BOTH the smplx params (the optimization output) AND the
        # verts/joints export (consumed by ma_vis). Either missing means
        # we fall through and re-run the full body loop so the missing
        # artifact gets produced.
        verts_result = os.path.join(save_prediction_fn, f"verts_joints_body_id-{body_id:02d}.npz")
        if os.path.exists(npz_result) and os.path.exists(verts_result):
            print(f"Already processed body_id {body_id} in {npz_result}")
            processed += 1
    if processed == total_bodies:
        print("All bodies already processed")

        if save_scene_videos:
            print(
                "Scene video overlay rendering has moved to mv-rerun/run_ma_vis.py "
                "(smplx_overlay.py). Skipping scene videos in run_ma_3d.py."
            )
        analysis_out_dir = None
        if save_detection_analysis:
            analysis_out_dir = _run_detection_analysis(
                pred_pth=pred_pth,
                out_path=out_path,
                out_fn=out_fn,
                cam_names=cam_names,
                cam_name_prefix=cam_name_prefix,
                top_k_frames=detection_analysis_top_k,
            )
        return
    # GET PRED result

    smplx_out_gts = []
    for body_id in body_ids:
        smplx_out_gt = None
        if use_gt:
            v_template = metadata_world["v_template"] if "v_template" in metadata_world else None
            smplx_out_gt = read_gt(metadata_world, smplx_model[body_id], batch_size, paths,
                                   body_id, device,
                                   v_template=v_template, use_bun_model=optim_cfg['use_bun_model'], flat_hand=flat_hand)
        smplx_out_gts.append(smplx_out_gt)
    smplx_poses, smplxs_betas, smplxs_trans, triangulated_3d_pts, smplx_contact, smplx_floor_contact, cam_imgs_seq, cam_names = fit_smplx(body_ids, pred_fns, cameras_metadata_fns, imgs_pth, smplx_model,
                                                                                batch_size, downsampled_verts_mat, smplx_out_gts, device=device,
                                                                                optim_cfg=optim_cfg, valid_frames_idx=frames_to_use_idx,
                                                                                hand_joints_pred_fns=hand_joints_pred_fns, rotate_cams=ROTATED_CAMS,
                                                                                save_prediction_fn=save_prediction_fn,
                                                                                start_frame=start_frame, end_frame=end_frame,
                                                                                ignore_start_frames=ignore_start_frames,
                                                                                paths=paths,
                                                                                up_axis=up_axis,
                                                                                parallel=True)

    # Fill ignored start frames by copying from the first optimized frame
    if ignore_start_frames > 0:
        with torch.no_grad():
            for body_id in body_ids:
                smplx_poses[body_id].data[:ignore_start_frames] = smplx_poses[body_id].data[ignore_start_frames]
                smplxs_trans[body_id].data[:ignore_start_frames] = smplxs_trans[body_id].data[ignore_start_frames]
        print(f"Filled frames 0:{ignore_start_frames} by copying from frame {ignore_start_frames}")

    smplx_preds_by_body = {}
    for body_id in body_ids:
        smplx_pose = smplx_poses[body_id]
        smplx_betas = smplxs_betas[body_id]
        smplx_trans = smplxs_trans[body_id]
        smplx_preds_by_body[body_id] = get_smplx_forward(
            smplx_pose,
            smplx_betas,
            smplx_trans,
            gender="neutral",
            smplx_models=smplx_model[body_id],
        )

    smplx_preds = []
    smplx_gts = []
    for body_id in body_ids:
        save_prediction_fn = os.path.join(out_path, out_fn)
        npz_result = os.path.join(save_prediction_fn, f"smplx_params_body_id-{body_id:02d}.npz")
        # Match the outer guard: a body is "done" only when BOTH the smplx
        # params and the verts/joints export exist. If verts_joints is
        # missing, fall through so the unconditional savez below produces it.
        verts_result = os.path.join(save_prediction_fn, f"verts_joints_body_id-{body_id:02d}.npz")
        if os.path.exists(npz_result) and os.path.exists(verts_result):
            print(f"Skipping body_id {body_id} as it's already processed")
            continue

        smplx_pose = smplx_poses[body_id]
        smplx_betas = smplxs_betas[body_id]
        smplx_trans = smplxs_trans[body_id]
        smplx_out_pred = smplx_preds_by_body[body_id]
        smplx_preds.append(smplx_out_pred)

        if metadata_world is not None:
            v_template = metadata_world["v_template"] if "v_template" in metadata_world else None
        if use_gt:
            smplx_out_gt = read_gt(metadata_world, smplx_model[body_id], batch_size, paths,
                                   body_id, device,
                                   v_template=v_template, use_bun_model=optim_cfg['use_bun_model'],
                                   flat_hand=flat_hand)
            smplx_gts.append(smplx_out_gt)
        else:
            smplx_out_gt = smplx_out_pred
            smplx_gts.append(smplx_out_gt)

        save_prediction_fn = os.path.join(out_path, out_fn)
        os.makedirs(save_prediction_fn, exist_ok=True)
        # Read the PREDICTION model's actual flat_hand_mean (the neutral model
        # built by get_smplx_models, which uses the SMPL-X default for normal
        # inference — NOT the rich/chi3d eval-only `flat_hand` above). Defensive:
        # never let metadata stamping break a run for existing users.
        try:
            _export_flat_hand_mean = bool(smplx_model[body_id]["neutral"].flat_hand_mean)
        except Exception:
            _export_flat_hand_mean = False

        np.savez(os.path.join(save_prediction_fn, f"smplx_params_body_id-{body_id:02d}.npz"),
                smplx_pose=smplx_pose.detach().cpu().numpy(),
                smplx_betas=smplx_betas.detach().cpu().numpy(),
                smplx_translation=smplx_trans.detach().cpu().numpy(),
                triangulated_3d_pts=triangulated_3d_pts[body_id].cpu().numpy() if triangulated_3d_pts[body_id] is not None else None,
                smplx_contact=smplx_contact[body_id].cpu().numpy() if smplx_contact[body_id] is not None else None,
                smplx_floor_contact=smplx_floor_contact[body_id].cpu().numpy() if smplx_floor_contact[body_id] is not None else None,
                v_template_pred=v_template_pred,
                # Self-describing metadata for the SMPL-X/Blender exporter
                # (export_blender.py). Additive + namespaced: existing readers
                # ignore these keys, and the exporter falls back to defaults (and
                # auto-detects flat_hand_mean from the verts) for older files.
                smplx_export_model_type="smplx",
                smplx_export_gender="neutral",
                smplx_export_flat_hand_mean=_export_flat_hand_mean,
                smplx_export_num_betas=int(smplx_betas.shape[-1]))

        # ma_vis (and regression_check) consume pred_vertices/pred_joints from
        # this file, so always write those. The gt_* arrays are only meaningful
        # for ground-truth eval runs (use_gt=True); in normal inference there's
        # no GT — smplx_out_gt falls back to the prediction — so writing them
        # would just duplicate pred_*. Omit them unless a real GT was loaded.
        verts_payload = dict(
            pred_joints=smplx_out_pred.joints.detach().cpu().numpy(),
            pred_vertices=smplx_out_pred.vertices.detach().cpu().numpy(),
        )
        if use_gt:
            verts_payload["gt_joints"] = smplx_out_gt.joints.detach().cpu().numpy()
            verts_payload["gt_vertices"] = smplx_out_gt.vertices.detach().cpu().numpy()
        np.savez(os.path.join(save_prediction_fn, f"verts_joints_body_id-{body_id:02d}.npz"),
                 **verts_payload)

        # Only emit per-frame error metrics (body_id-NN.csv) when a real GT
        # exists. With use_gt=False the "GT" tensor was just a copy of the
        # prediction, so the CSV would be noise.
        if use_gt:
            process_results(smplx_out_gt, smplx_out_pred, body_id, out_fn, cameras_metadata_fns,
                            imgs_pth, batch_size, pred_fns, smplx_model[body_id]["neutral"].faces, downsampled_verts_mat,
                            save_imgs=False, out_folder=out_path, start_frame=0)

    if save_scene_videos:
        print(
            "Scene video overlay rendering has moved to mv-rerun/run_ma_vis.py "
            "(smplx_overlay.py). Skipping scene videos in run_ma_3d.py."
        )

    analysis_out_dir = None
    if save_detection_analysis:
        analysis_out_dir = _run_detection_analysis(
            pred_pth=pred_pth,
            out_path=out_path,
            out_fn=out_fn,
            cam_names=cam_names,
            cam_name_prefix=cam_name_prefix,
            top_k_frames=detection_analysis_top_k,
        )


def parser():
    args = argparse.ArgumentParser(
        description="Predict 3D pose from 2D keypoints. Accepts either "
                    "--ma_cap_dir (chained mode) or --calibration combined "
                    "with --videos_dir / --images_root_dir (standalone mode).",
    )
    args.add_argument("--seq_name", type=str)
    args.add_argument("--ma_2d_dir", type=str, help="Path to the 2D predictions.")
    args.add_argument("--ma_cap_dir", type=str, default=None,
                      help="Path to the ma_cap NPZ output (chained mode). "
                           "Omit and use --calibration for standalone mode.")
    args.add_argument("--videos_dir", type=str, default=None,
                      help="Standalone mode: directory of <cam_name>.mp4 files. "
                           "Requires --calibration.")
    args.add_argument("--images_root_dir", type=str, default=None,
                      help="Standalone mode: directory of <cam_name>/*.jpg "
                           "subdirectories. Requires --calibration.")
    args.add_argument("--calibration", type=str, default=None,
                      help="Calibration file (yaml/xcp/json). Required when "
                           "--ma_cap_dir is not set.")
    args.add_argument('--config_file', type=str, default="config_files/config.yaml", help="Path to the config file")
    args.add_argument('--cam_names', type=str, nargs='+', default=[], help="List of camera names (e.g., IOI_01 IOI_02 IOI_03)")
    args.add_argument('--cam_name_prefix', type=str, default="IOI_", help="Camera name prefix when --cam_names is empty")
    args.add_argument('--up-axis', '--up_axis', dest='up_axis', type=str, default="z",
                      choices=["x", "y", "z", "-x", "-y", "-z"],
                      help="Signed world up axis for the triangulated-points .rrd (default: z).")
    args.add_argument('--out_path', type=str, required=True, help="Files to process")
    args.add_argument('--skip_scene_videos', action='store_true',
                      help="Deprecated: scene videos are rendered in mv-rerun/run_ma_vis.py.")
    args.add_argument('--start_frame', type=int, default=0,
                      help="First frame index to use (all earlier frames are excluded from the data).")
    args.add_argument('--end_frame', type=int, default=None,
                      help="End frame index exclusive (all later frames are excluded from the data). Omit to use all remaining frames.")
    args.add_argument('--ignore_start_frames', type=int, default=0,
                      help="Number of initial frames to exclude from optimization losses (but keep in the output). "
                           "Unlike --start_frame which removes frames entirely, --ignore_start_frames keeps the frames "
                           "but does not optimize for them; after fitting, they are filled by copying from the first "
                           "optimized frame. Applied after --start_frame/--end_frame slicing.")
    args.add_argument('--skip_detection_analysis', action='store_true',
                      help="Skip post-run 2D detection analysis plots/CSVs.")
    args.add_argument('--use-gt', dest='use_gt', action='store_true',
                      help="Evaluate against ground truth: also compute MPJPE/PVE vs the "
                           "SMPL-X poses in <ma_cap_dir>/<seq>/gt/global.npz (e.g. the "
                           "mamma_eval_dance sequences). Off by default — normal captures have "
                           "no GT, so the fit's prediction is the only output.")
    args.add_argument('--detection_analysis_top_k', type=int, default=30,
                      help="Top-K least confident frames to report in detection analysis.")
    # ── Per-installation paths (previously read from MAMMA_* env vars at
    # module-import time). Required at the CLI; the pipeline runner
    # constructs them from the .env file via inference.env.
    args.add_argument('--smplx-models', dest='smplx_models', required=True,
                      help="Directory containing SMPL-X lockhead models. "
                           "Previously MAMMA_SMPLX_LOCKHEAD_MODELS.")
    args.add_argument('--downsampled-verts', dest='downsampled_verts', required=True,
                      help="Path to verts_512.pkl. "
                           "Previously MAMMA_DOWNSAMPLED_VERTS_PKL.")
    args.add_argument('--bun-models', dest='bun_models', default=None,
                      help="Directory containing BUN SMPL-X models. "
                           "Required only when the algorithm config_file "
                           "has use_bun_model: True. Previously MAMMA_BUN_MODELS.")
    args.add_argument('--part-mesh', dest='part_mesh', default=None,
                      help="Directory containing per-part SMPL-X mesh files. "
                           "Required only when SDF-based loss is enabled. "
                           "Previously MAMMA_PART_MESH_PATH.")
    args.add_argument('--occlusion-aware-weights', '--stricter-weights',
                      dest='occlusion_aware_weights', action='store_true',
                      help="Opt-in occlusion-aware gating of 2D landmarks in the "
                           "reprojection loss. Replaces the visibility clamp (vis_clip_value) "
                           "with a gate that "
                           "drives the weight of consistently low-visibility (occluded) landmarks "
                           "toward zero, so the views that actually see a body part dominate. "
                           "Enables a hybrid soft-gate (tau=0.30) with keep_top_m=1 safety in "
                           "every optimization stage unless the config sets occlusion_gating "
                           "explicitly. Default off (behaviour unchanged).")
    args.add_argument('--use-vposer', '--use_vposer', dest='use_vposer', action='store_true',
                      help="Opt-in VPoser pose prior. Adds a gentle vposer_recon_loss "
                           "(weight 0.25) to every optimization stage unless the config already "
                           "sets it, so the VPoser manifold prior can be enabled on any recipe "
                           "without editing YAML. Pulls the body pose toward VPoser's learned "
                           "pose manifold (helps where 2D evidence is weak/occluded; can bias "
                           "toward typical poses). Requires the VPoser weights "
                           "(MAMMA_VPOSER_DIR, default data/body_models/vposer/V02_05). Default off.")
    return args.parse_args()


if __name__ == "__main__":
    args = parser()
    paths = PathsConfig.from_args(args)
    seq_name = args.seq_name

    # Input mode dispatch: chained (--ma_cap_dir) wins; standalone
    # synthesizes the same NPZ scaffolding from --calibration + a frame
    # source (videos_dir / images_root_dir), writing under <out_path>/
    # _synth_ma_cap/<seq>/gt/ so downstream code is unchanged.
    if args.ma_cap_dir:
        npz_gt_path = os.path.join(args.ma_cap_dir, seq_name, "gt")
    else:
        if not args.calibration:
            sys.stderr.write(
                "error: --calibration is required when --ma_cap_dir is omitted.\n"
            )
            sys.exit(2)
        if not (args.videos_dir or args.images_root_dir):
            sys.stderr.write(
                "error: --videos_dir or --images_root_dir is required when "
                "--ma_cap_dir is omitted (otherwise frames_len is undefined).\n"
            )
            sys.exit(2)
        # Local sys.path bump so capture/ is importable when this script
        # is invoked with cwd=optimization/.
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from capture.run_ma_cap import synthesize_ma_cap_npzs  # noqa: E402
        synth_gt = synthesize_ma_cap_npzs(
            out_dir=os.path.join(args.out_path, "_synth_ma_cap"),
            seq_name=seq_name,
            cam_names=args.cam_names,
            calibration_path=args.calibration,
            videos_dir=args.videos_dir,
            images_root_dir=args.images_root_dir,
        )
        npz_gt_path = str(synth_gt)
        print(f"Synthesized ma_cap scaffolding at: {npz_gt_path}")

    ldmks_pred_path = os.path.join(args.ma_2d_dir, seq_name)
    print("Processing sequence: ", seq_name)
    img_pth = ""
    hand_joints_pred_pth = None

    print("NPZ GT PATH: ", npz_gt_path)
    print("2D LDMKS PRED PATH: ", ldmks_pred_path)
    print("OUT PATH: ", args.out_path)

    print("Given cam names: ", args.cam_names)

    main(
        args.config_file,
        args.cam_names,
        npz_gt_path,
        img_pth,
        paths,
        pred_pth=ldmks_pred_path,
        hand_joints_pred_pth=hand_joints_pred_pth,
        out_fn=seq_name,
        downsampled_verts_mat_path=paths.downsampled_verts_pkl,
        out_path=args.out_path,
        use_gt=args.use_gt,
        cam_name_prefix=args.cam_name_prefix,
        save_scene_videos=not args.skip_scene_videos,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        ignore_start_frames=args.ignore_start_frames,
        save_detection_analysis=not args.skip_detection_analysis,
        detection_analysis_top_k=args.detection_analysis_top_k,
        up_axis=args.up_axis,
        cli_args=args,
        )
