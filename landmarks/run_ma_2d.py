"""Run 2D dense landmark prediction on multi-camera `.npz` sequences.

Example (run from the repo root):
    python landmarks/run_ma_2d.py \\
        --config_path landmarks/configs/train/models_2d/config_mammanet_mask_512.yaml \\
        --weights <path/to/model.ckpt> \\
        --downsampled-verts data/body_models/downsampled_verts/verts_512.pkl \\
        --img_folder <path/to/sequence_root> \\
        --mask_path <path/to/masks> \\
        --out_folder out
"""
import os
import sys

# Make ``capture`` importable regardless of the launch dir. The repo
# root is the parent of ``landmarks/``; both are derived from __file__
# so the script runs from anywhere (repo root or cwd=landmarks/).
_LANDMARKS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_LANDMARKS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv

load_dotenv()

import torch
from loguru import logger

from capture import (  # noqa: E402  (after sys.path bump)
    FrameSource,
    ImageFileSource,
    cam_data_from_image_dir,
    cam_data_from_video,
    find_image_cam_dirs,
    find_video_files,
    frame_source_from_cam_data,
)


from utils.util import init_random_seed, set_random_seed
from utils.img_utils import DrawUV
from lib.models import build_model

import cv2
import tqdm
import glob
import numpy as np
from pathlib import Path
from utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
from lib.datasets.vitdet_dataset import ViTDetDataset
from utils.video_utils import create_video_from_images
from utils.post_video_from_imgs import process_sequence
from typing import Any
import argparse
from collections.abc import Mapping
from omegaconf import OmegaConf, DictConfig, ListConfig

def recursive_to(x: Any, target: torch.device):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x


def process_data(frame_source, detector, device, model, cfg, out_folder, save_cam_output, masks_path=None,
                 downsampled_verts_pth='assets/verts_512.pkl'):
    """Run dense 2D landmarks for a single camera, given a :class:`FrameSource`.

    The source may wrap an NPZ image-path list (chained mode), an MP4
    video, or a directory of image frames. Output is always one NPZ per
    camera at ``<out_folder>/<cam>.npz`` — the runner composes any
    higher-level nesting via ``--out_folder``.
    """
    camera_id = str(frame_source.cam_name)
    n_frames = len(frame_source)
    all_verts = []
    all_vis = []
    all_contact = []
    all_floor_contact = []

    if masks_path is not None:
        pred_mask_dir = os.path.join(masks_path, camera_id, "masks")
        pred_masks = sorted(glob.glob(os.path.join(pred_mask_dir, "*.png")))
        # Keep contiguous IDs up to the max observed ID so missing bodies can be zero-filled.
        observed_people_ids = []
        for pred_mask in pred_masks:
            mask_id = int(os.path.basename(pred_mask).split("_")[2].replace(".png", ""))
            if mask_id not in observed_people_ids:
                observed_people_ids.append(mask_id)

        if len(observed_people_ids) == 0:
            raise ValueError(f"No mask files found in {pred_mask_dir}")

        max_person_id = max(observed_people_ids)
        people_ids = list(range(1, max_person_id + 1))
        logger.info(
            f"Found mask IDs {sorted(observed_people_ids)} in {pred_mask_dir}. "
            f"Will export bodies: {people_ids}"
        )
    else:
        people_ids = [1]
        mask = None

    draw_uv = DrawUV(downsampled_verts_mat_path=downsampled_verts_pth)

    if os.path.exists(f"{out_folder}/{camera_id}.npz"):
        logger.info(f"skipping {camera_id}: output file already exists")
        return
    # Decode each frame once per camera and run every body on it (frame-major),
    # rather than re-decoding the whole clip once per person. Per-body results are
    # collected in dicts and stacked afterwards, so the .npz layout is unchanged.
    verts_per_body = {pid: [] for pid in people_ids}
    vis_per_body = {pid: [] for pid in people_ids}
    contact_per_body = {pid: [] for pid in people_ids}
    floor_contact_per_body = {pid: [] for pid in people_ids}
    body_dirs = {pid: f"{out_folder}/{camera_id}/body_{pid:02d}" for pid in people_ids}

    for frame_n in tqdm.tqdm(range(n_frames)):
        # FrameSource gives RGB; cv2/downstream expects BGR. Single decode/frame.
        frame_bgr = cv2.cvtColor(frame_source.read_rgb(frame_n), cv2.COLOR_RGB2BGR)
        for body_id in people_ids:
            folder_path = body_dirs[body_id]
            body_verts = verts_per_body[body_id]
            body_vis = vis_per_body[body_id]
            body_contact = contact_per_body[body_id]
            body_floor_contact = floor_contact_per_body[body_id]
            # frame_bgr is read-only downstream (detector reads it; ViTDetDataset
            # copies internally), so the bodies can share it without a copy.
            img = frame_bgr
            if masks_path is None:
                det_out = detector(img)
                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
                valid_scores = det_instances.scores[valid_idx].cpu().numpy()
                boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                boxes = boxes[np.argsort(-valid_scores)]  # shape (n, 4)
                valid_scores = valid_scores[np.argsort(-valid_scores)]

                if len(boxes) > 0:
                    # NOTE: SIMPLEST ASSUMPTION THAT THE BODY IS AT THE CENTER OF THE IMAGE
                    boxes_center = (boxes[:, :2] + boxes[:, 2:])/2
                    img_center = np.array(img.shape[:2][::-1])/2
                    box2center_dist = np.linalg.norm(boxes_center - img_center, axis=1)
                    # pick the closest box to the center
                    boxes = boxes[np.argmin(box2center_dist)][None]
                    valid_scores = valid_scores[np.argmin(box2center_dist)][None]
            else:
                masks_path_person = os.path.join(masks_path, camera_id, "masks", f"mask_{frame_n:04d}_{body_id:02d}.png")
                if not os.path.exists(masks_path_person):
                    body_verts.append(np.zeros((1, cfg.num_joints, 3)))
                    body_vis.append(np.zeros((1, cfg.num_joints)))
                    body_contact.append(np.zeros((1, cfg.num_joints)))
                    body_floor_contact.append(np.zeros((1, cfg.num_joints)))
                    continue

                mask = cv2.imread(str(masks_path_person), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    body_verts.append(np.zeros((1, cfg.num_joints, 3)))
                    body_vis.append(np.zeros((1, cfg.num_joints)))
                    body_contact.append(np.zeros((1, cfg.num_joints)))
                    body_floor_contact.append(np.zeros((1, cfg.num_joints)))
                    continue

                total_pixels = mask.shape[0] * mask.shape[1]
                mask_sum_ratio = np.sum(mask) / total_pixels
                threshold_ratio = 0.01
                if mask_sum_ratio < threshold_ratio:
                    logger.info(f"skipping because mask sum ratio {mask_sum_ratio} < {threshold_ratio}: {masks_path_person}")
                    body_verts.append(np.zeros((1, cfg.num_joints, 3)))
                    body_vis.append(np.zeros((1, cfg.num_joints)))
                    body_contact.append(np.zeros((1, cfg.num_joints)))
                    body_floor_contact.append(np.zeros((1, cfg.num_joints)))
                    continue
                else:
                    ys, xs = np.where(mask)
                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                    boxes = np.array([[x1, y1, x2, y2]])
                    valid_scores = np.array([1.0])

            dataset = ViTDetDataset(cfg, img, mask, boxes=boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            if len(dataset) == 0:
                logger.warning(f"skipping because dataloader length: {len(dataset)}")
                body_verts.append(np.zeros((1, cfg.num_joints, 3)))
                body_vis.append(np.zeros((1, cfg.num_joints)))
                body_contact.append(np.zeros((1, cfg.num_joints)))
                body_floor_contact.append(np.zeros((1, cfg.num_joints)))
                continue

            # one iteration
            batch = next(iter(dataloader))
            batch = recursive_to(batch, device)
            img_crop = batch["img"].cpu() * dataset.std[None,:, None, None] + dataset.mean[None,:, None, None]
            img_crop = (img_crop[0].numpy().transpose(1,2,0))[:,:,::-1].astype(np.uint8).copy()
            mask_crop = batch["mask"][0].cpu().numpy().transpose(1,2,0).copy() if batch["mask"] is not None else None

            with torch.no_grad():
                out = model(batch["img"], batch["mask"])
                box_center = batch["box_center"].float().cpu().numpy()
                box_size = batch["box_size"].float().cpu().numpy()
                joints2d = out["joints2d"].cpu().numpy()
                if cfg.data["train"]["normalize_plus_min_one"]:
                    normalize_scale = 2
                    joints2d[:,:,:2] = (joints2d[:,:,:2] + 1) / normalize_scale

                h, w, _ = img_crop.shape

                visibilities = torch.sigmoid(out["visibility"].squeeze(-1)).cpu().numpy()
                contact = torch.sigmoid(out["contact"].squeeze(-1)).cpu().numpy()
                floor_contact = torch.sigmoid(out["floor_contact"].squeeze(-1)).cpu().numpy()

                pred_joints2d = joints2d.copy()
                pred_joints2d[:, :, 0] = joints2d[:, :, 0] * w
                pred_joints2d[:, :, 1] = joints2d[:, :, 1] * h

                # This scales the points back to the original image space
                joints2d[:,:,0] = joints2d[:,:,0] * box_size[0, 0] - box_size[0, 0]/2 + box_center[0,0]
                joints2d[:,:,1] = joints2d[:,:,1] * box_size[0, 1] - box_size[0, 1]/2 + box_center[0,1]

                body_verts.append(joints2d)
                body_vis.append(visibilities)
                body_contact.append(contact)
                body_floor_contact.append(floor_contact)

            if save_cam_output  and (frame_n % 20 == 0):
                count = 0
                pred_img = img_crop.copy()
                pred_vis_img = img_crop.copy()
                pred_not_vis_img = img_crop.copy()
                pred_uncertainty_img = img_crop.copy()
                pred_contact_img = img_crop.copy()
                pred_floor_contact_img = img_crop.copy()
                uv_img = draw_uv.new_uv_img()
                for joint, vis, cont, floor_cont in zip(pred_joints2d[0], visibilities[0], contact[0], floor_contact[0]):
                    if joint.shape[-1] == 3:
                        x_pred, y_pred, sigma = joint
                        sigma = np.sqrt(np.exp(sigma)) / normalize_scale * max(img_crop.shape[:2])
                        sigma = sigma/50 #px
                        sigma = np.clip(sigma, 0, 1)

                    else:
                        x_pred, y_pred = joint
                        sigma = 0
                    circle_size = 2
                    if vis > 0.5:
                        cv2.circle(pred_vis_img, (int(x_pred), int(y_pred)), circle_size, (0, int(255*vis), int(255*(1-vis))), -1)
                    else:
                        cv2.circle(pred_not_vis_img, (int(x_pred), int(y_pred)), circle_size, (0, int(255*vis), int(255*(1-vis))), -1)
                    cont = cont/0.6
                    cont = np.clip(cont, 0, 1)

                    floor_cont = floor_cont/1.
                    floor_cont = np.clip(floor_cont, 0, 1)

                    cv2.circle(pred_contact_img, (int(x_pred), int(y_pred)), circle_size, (0, int(255*(cont)), int(255*(1-cont))), -1)
                    cv2.circle(pred_floor_contact_img, (int(x_pred), int(y_pred)), circle_size, (0, int(255*(floor_cont)), int(255*(1-floor_cont))), -1)

                    draw_uv.draw_visibility_img(uv_img, count, color=(0, int(255*vis), int(255*(1-vis)), 255))
                    cv2.circle(pred_uncertainty_img, (int(x_pred), int(y_pred)), 4, (0, int(255*(1-sigma)), int(255*sigma)), -1)
                    count += 1

                cv2.putText(pred_img, f"{frame_n:05d}", (10, 20), fontScale=1, color=(0, 0, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=3)
                cv2.putText(pred_vis_img, f"Pred_vis", (10, 20), fontScale=1, color=(0, 0, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=3)
                cv2.putText(pred_not_vis_img, f"Pred_not_vis", (10, 20), fontScale=1, color=(0, 0, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=3)
                cv2.putText(pred_uncertainty_img, f"Pred_uncertainty", (10, 20), fontScale=1, color=(0, 0, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=3)

                uv_img = cv2.resize(uv_img, (pred_img.shape[0], pred_img.shape[0]))[:, :, :3]

                pred_img = np.concatenate([pred_img, uv_img, pred_vis_img, pred_not_vis_img, pred_uncertainty_img], axis=1)
                if mask_crop is not None:
                    img_mask = (mask_crop.repeat(3, axis=-1) * 255).astype(np.uint8)
                    pred_img = np.concatenate([pred_img, img_mask], axis=1)
                    pred_img = np.concatenate([pred_img, pred_contact_img, pred_floor_contact_img], axis=1)
                pred_img = cv2.resize(pred_img, (pred_img.shape[1]//2, pred_img.shape[0]//2))

                os.makedirs(folder_path, exist_ok=True)
                cv2.imwrite(f"{folder_path}/img_{frame_n:04d}.jpg", pred_img)

    # Stack per body in people_ids order so the output layout is unchanged.
    for body_id in people_ids:
        all_verts.append(np.array(verts_per_body[body_id]).squeeze(1))
        all_vis.append(np.array(vis_per_body[body_id]).squeeze(1))
        all_contact.append(np.array(contact_per_body[body_id]).squeeze(1))
        all_floor_contact.append(np.array(floor_contact_per_body[body_id]).squeeze(1))

        # Preview MP4 is optional and only meaningful when frames were
        # actually written above (``save_cam_output=True``, gated by
        # ``frame_n % 20 == 0`` so e.g. quick smoke runs may produce
        # zero preview frames). The helper now no-ops when the
        # ``folder_path`` is empty, so this call is safe either way;
        # the explicit guard avoids spamming "creating video" prints
        # on common no-preview runs. Format matches the .jpg writes
        # above (was .png — silent ffmpeg failure on every call).
        if save_cam_output:
            folder_path = body_dirs[body_id]
            create_video_from_images(folder_path, f"{folder_path}/{camera_id}.mp4", img_format="img_%04d.jpg")

    landmarks = np.stack(all_verts, axis=1)
    visibilities = np.stack(all_vis, axis=1)
    contacts = np.stack(all_contact, axis=1)
    floor_contacts = np.stack(all_floor_contact, axis=1)

    np.savez(f"{out_folder}/{camera_id}.npz", landmarks=landmarks,
             visibilities=visibilities, contacts=contacts, floor_contacts=floor_contacts,
             contacts_gt=None, floor_contacts_gt=None)


def parser():
    args = argparse.ArgumentParser(
        description="Run 2D dense landmarks. Accepts three input modes "
                    "(mutually exclusive): --img_folder (NPZ manifest from "
                    "ma_cap), --videos_dir (one MP4 per camera), or "
                    "--images_root_dir (one directory per camera).",
    )
    # Tri-mode input flags (exactly one required).
    args.add_argument('--img_folder', type=str, default=None,
                      help='Chained mode: <ma_cap_out>/<seq>/gt/ — NPZ manifest with img_abs_path per camera.')
    args.add_argument('--videos_dir', type=str, default=None,
                      help='Standalone mode: directory of <cam_name>.mp4 files.')
    args.add_argument('--images_root_dir', type=str, default=None,
                      help='Standalone mode: directory of <cam_name>/*.{jpg,png} subdirectories.')
    args.add_argument('--calibration', type=str, default=None,
                      help='Optional calibration file (yaml/xcp/json). Required when '
                           '--undistort is set; otherwise accepted for CLI parity.')
    args.add_argument('--undistort', action='store_true',
                      help='Undistort frames via Vicon-radial-2 coefficients '
                           '(from --calibration) before running the landmark '
                           'network. Default off.')
    args.add_argument('--start', type=int, default=None,
                      help='First frame index to process (0-based, inclusive). '
                           'Default: 0 (process from the beginning).')
    args.add_argument('--end', type=int, default=None,
                      help='Last frame index to process (0-based, exclusive). '
                           'Default: process all frames.')
    # Existing args.
    args.add_argument('--config_path', type=str, default='configs/train/models_2d/config_mammanet_mask_512.yaml', help='train config file path')
    args.add_argument('--task', type=str, default='landmarks_2d_dense_512', help='train task')
    args.add_argument('--weights', type=str, help='checkpoint weights')
    args.add_argument('--out_folder', type=str, default='out', help='out folder name')
    args.add_argument('--seq_name', type=str, default='', help='sequence name')
    args.add_argument('--dataset_name', type=str, default='', help='dataset name')
    args.add_argument('--camera_id', type=str, default='', help='camera id number')
    args.add_argument('--mask_path', type=str, default=None, help='path to detectron2 mask model')
    args.add_argument('--video_fps', type=float, default=5.0, help='FPS for generated videos')
    args.add_argument('--cam_names', nargs='*', default=None, help='space-separated camera names (e.g., IOI_01 IOI_02)')
    args.add_argument('--save_cam_output', action=argparse.BooleanOptionalAction, default=True, help='save per-camera viz frames + video (use --no-save_cam_output to disable)')
    args.add_argument('--downsampled-verts', dest='downsampled_verts',
                      default='assets/verts_512.pkl',
                      help='Path to verts_512.pkl. Previously hard-coded to '
                           'assets/verts_512.pkl; the inference runner injects '
                           'this from MAMMA_DOWNSAMPLED_VERTS_PKL.')
    parsed = args.parse_args()

    # Post-parse mutex: exactly one input mode.
    input_flags = [
        ("--img_folder", parsed.img_folder),
        ("--videos_dir", parsed.videos_dir),
        ("--images_root_dir", parsed.images_root_dir),
    ]
    set_flags = [name for name, val in input_flags if val]
    if len(set_flags) == 0:
        raise SystemExit(
            "error: one of --img_folder / --videos_dir / --images_root_dir is required."
        )
    if len(set_flags) > 1:
        raise SystemExit(
            f"error: {' and '.join(set_flags)} are mutually exclusive; set exactly one."
        )

    return parsed


def sanitize_omegaconf_inplace(cfg):
    if isinstance(cfg, DictConfig):
        for k in list(cfg.keys()):
            v = cfg[k]
            if isinstance(v, (DictConfig, ListConfig)):
                sanitize_omegaconf_inplace(v)
            elif isinstance(v, np.ndarray):
                cfg[k] = v.tolist()
            # Add support for other types here if needed
    elif isinstance(cfg, ListConfig):
        for i in range(len(cfg)):
            v = cfg[i]
            if isinstance(v, (DictConfig, ListConfig)):
                sanitize_omegaconf_inplace(v)
            elif isinstance(v, np.ndarray):
                cfg[i] = v.tolist()


def load_hydra_style_config(cfg_file="conf/config.yaml"):
    cfg = OmegaConf.load(cfg_file)

    base = OmegaConf.create()
    # Process defaults
    if "defaults" in cfg:
        for entry in cfg.defaults:
            if entry == '_self_':
                continue
            if isinstance(entry, Mapping):
                for group, name in entry.items():
                    path = os.path.join(os.path.dirname(cfg_file), group, f"{name}.yaml")
                    subcfg = OmegaConf.load(path)
                    base[group] = subcfg
            elif isinstance(entry, str):
                path = os.path.join(os.path.dirname(cfg_file), f"{entry}.yaml")
                subcfg = OmegaConf.load(path)
                base = OmegaConf.merge(base, subcfg)

    # Merge the base with the rest of config.yaml (excluding 'defaults')
    del cfg["defaults"]
    merged = OmegaConf.merge(base, cfg)
    sanitize_omegaconf_inplace(merged)

    resolved = OmegaConf.create(OmegaConf.to_container(merged, resolve=True, throw_on_missing=True))
    return resolved

def _build_cam_sources(args, img_folder=None):
    """Build a list of :class:`FrameSource` for whichever input mode is active.

    Exactly one of ``args.img_folder`` / ``args.videos_dir`` /
    ``args.images_root_dir`` must be set (the parser enforces this).
    Returns sources sorted by camera name (stable ordering).

    When ``args.undistort`` is set, each source is configured to apply
    Vicon-radial-2 undistortion on every frame read; the per-camera
    :class:`Camera` is taken from ``args.calibration``.
    """
    sources = []

    calib_cams = None
    if args.undistort:
        if not args.calibration:
            raise SystemExit("error: --undistort requires --calibration")
        from capture import load_calibration
        calib_cams = load_calibration(args.calibration).cameras
        logger.info(f"undistort: loaded calibration with {len(calib_cams)} cameras")

    def _cam_for(name):
        if calib_cams is None:
            return None
        cam = calib_cams.get(name)
        if cam is None:
            logger.warning(f"--undistort: no calibration entry for camera {name!r}; skipping undistort for it")
        return cam

    start, end = args.start, args.end

    if args.videos_dir:
        video_paths = find_video_files(args.videos_dir, cam_names=args.cam_names)
        if not video_paths:
            logger.warning(f"No MP4 files found under {args.videos_dir}")
        for vp in video_paths:
            cam_data = cam_data_from_video(vp, start=start, end=end)
            cam = _cam_for(str(cam_data['cam_name']))
            sources.append(frame_source_from_cam_data(
                cam_data, camera=cam, undistort=args.undistort and cam is not None,
            ))
        return sources

    if args.images_root_dir:
        cam_dirs = find_image_cam_dirs(args.images_root_dir, cam_names=args.cam_names)
        if not cam_dirs:
            logger.warning(f"No camera image dirs found under {args.images_root_dir}")
        for cd in cam_dirs:
            cam_data = cam_data_from_image_dir(cd, start=start, end=end)
            cam = _cam_for(str(cam_data['cam_name']))
            sources.append(frame_source_from_cam_data(
                cam_data, camera=cam, undistort=args.undistort and cam is not None,
            ))
        return sources

    # NPZ (chained) mode: img_folder is the resolved
    # <ma_cap_out>/<seq>/gt/ path.
    if args.cam_names:
        cams_data = []
        for cam_name in args.cam_names:
            if any(ch in cam_name for ch in "*?[]"):
                pattern = (os.path.join(img_folder, f"{cam_name}.npz")
                           if not cam_name.endswith(".npz")
                           else os.path.join(img_folder, cam_name))
                cams_data.extend(glob.glob(pattern))
            else:
                cam_file = cam_name if cam_name.endswith(".npz") else f"{cam_name}.npz"
                cams_data.extend(glob.glob(os.path.join(img_folder, cam_file)))
        cams_data = sorted(set(cams_data))
        if not cams_data:
            logger.warning(f"No camera files found for cam_names={args.cam_names} in {img_folder}")
    else:
        cams_data = sorted(glob.glob(os.path.join(img_folder, "IOI_*.npz")))

    for cam_data_path in cams_data:
        data = np.load(cam_data_path, allow_pickle=True)
        # Convert NpzFile to a plain dict so frame_source_from_cam_data
        # can pick up video_path / frame_start / frame_end (videos
        # workflow) — or fall through to img_abs_path (chained NPZ).
        cam_data = {k: data[k] for k in data.files}
        cam_name = str(cam_data['cam_name'])
        cam = _cam_for(cam_name)
        # CLI --start/--end aren't honoured in NPZ mode: the NPZ's
        # frame_start/frame_end is the canonical range (set by ma_cap).
        # For ad-hoc users who want a different slice, use --videos_dir
        # or --images_root_dir directly with --start/--end.
        sources.append(frame_source_from_cam_data(
            cam_data, camera=cam, undistort=args.undistort and cam is not None,
        ))
    return sources


def main(args, out_folder, masks_folder, img_folder=None):
    OmegaConf.register_new_resolver("mult", lambda x,y: x*y)
    OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
    OmegaConf.register_new_resolver("div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("concat", lambda x: np.concatenate(x))
    OmegaConf.register_new_resolver("sorted", lambda x: np.argsort(x))

    cfg_file = args.config_path
    cfg = load_hydra_style_config(cfg_file)

    # set cudnn_benchmark
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    seed = init_random_seed(cfg.seed)
    set_random_seed(seed, deterministic=cfg.deterministic)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(torch.cuda.get_device_properties(device))

    model = build_model(cfg).to(device)

    logger.info(f"loading weights from: {args.weights}")
    # Trusted local checkpoint: PyTorch >=2.6 defaults torch.load(...)
    # to weights_only=True, which rejects the OmegaConf metadata stored in
    # this release checkpoint. Opt into the full checkpoint payload here.
    try:
        checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Resolve relative to this script (landmarks/) so the detector config
    # is found whether launched from the repo root or from cwd=landmarks/.
    cfg_path = os.path.join(_LANDMARKS_DIR, "configs/cascade_mask_rcnn_vitdet_h_75ep.py")
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    os.makedirs(out_folder, exist_ok=True)

    sources = _build_cam_sources(args, img_folder=img_folder)
    logger.info(f"processing {len(sources)} cameras: {[s.cam_name for s in sources]}")
    for source in sources:
        process_data(source, detector, device, model, cfg, out_folder,
                     args.save_cam_output, masks_folder,
                     downsampled_verts_pth=args.downsampled_verts)


def make_videos_from_args(args):
    dataset_dir = Path(os.path.join(args.out_folder, args.seq_name)).expanduser()
    if not dataset_dir.exists():
        logger.warning(f"Video dataset directory does not exist: {dataset_dir}")
        return

    # Only process the single sequence specified by seq_name
    seq_dir = dataset_dir
    process_sequence(seq_dir, args.video_fps, True, cleanup_frames=True)


if __name__ == '__main__':
    args = parser()

    # Resolve the chained-mode NPZ dir; left None for standalone modes.
    img_folder = (
        os.path.join(args.img_folder, args.seq_name, "gt")
        if args.img_folder else None
    )
    # Out + masks dirs are seq-scoped only when seq_name is provided
    # (true for chained mode; usually empty in standalone use).
    out_folder = (
        str(os.path.join(args.out_folder, args.seq_name))
        if args.seq_name else str(args.out_folder)
    )
    masks_folder = (
        os.path.join(args.mask_path, args.seq_name)
        if (args.mask_path and args.seq_name) else args.mask_path
    )

    logger.info(f"img_folder: {img_folder}")
    logger.info(f"videos_dir: {args.videos_dir}")
    logger.info(f"images_root_dir: {args.images_root_dir}")
    logger.info(f"out_folder: {out_folder}")

    main(args, out_folder=out_folder, masks_folder=masks_folder, img_folder=img_folder)

    # Post-pipeline video stitching reads per-camera viz dirs at
    # <out>/<seq>/<cam>/<body>/ — only meaningful when seq_name is set.
    if args.seq_name:
        make_videos_from_args(args)
