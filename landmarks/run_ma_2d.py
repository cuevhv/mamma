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
from lib.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD
from lib.datasets.utils_eval import gen_trans_from_patch_cv, expand_to_aspect_ratio
from kornia.geometry.transform import warp_affine
from kornia.filters import gaussian_blur2d
from utils.video_utils import create_video_from_images
from utils.post_video_from_imgs import process_sequence
import argparse
from collections.abc import Mapping
from omegaconf import OmegaConf, DictConfig, ListConfig


def _gpu_make_batch(frame_t, mask, boxes, cfg, device, mean_t, std_t):
    """Build the model-input batch on the GPU, replacing the per-crop CPU
    ``ViTDetDataset`` path.

    ``frame_t`` is the RGB frame already uploaded to the device once per frame
    ((1, 3, H, W) float); ``mask`` is the raw 0/255 mask (mask mode) or None.
    The math mirrors ``ViTDetDataset.__getitem__``: a single affine warp from the
    bbox to the fixed network patch, with an anti-alias gaussian blur before the
    downsampling. Warp/blur/normalize run as kornia/torch ops on-device, so the
    frame stays resident and no per-crop CPU work or host->device copy is needed.
    Returns the batch dict, or None when there is no valid box.
    """
    if len(boxes) == 0:
        return None
    bbox_shape = cfg.data_cfg["image_size"]
    patch_w, patch_h = int(bbox_shape[0]), int(bbox_shape[1])
    boxes = np.asarray(boxes, np.float32)
    center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
    scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
    bbox_size = expand_to_aspect_ratio(scale[0] * 200 * 1.2, target_aspect_ratio=bbox_shape)
    trans = gen_trans_from_patch_cv(float(center[0, 0]), float(center[0, 1]),
                                    bbox_size[0], bbox_size[1], patch_w, patch_h, 1.0, 0)
    M = torch.from_numpy(np.asarray(trans, np.float32)).to(device)[None]

    src = frame_t
    downsampling_factor = (float(bbox_size.max()) / patch_w) / 2.0
    if downsampling_factor > 1.1:
        sigma = (downsampling_factor - 1) / 2
        k = 2 * int(np.ceil(3 * sigma)) + 1
        src = gaussian_blur2d(frame_t, (k, k), (sigma, sigma))
    img = warp_affine(src, M, (patch_h, patch_w), mode="bilinear", padding_mode="zeros")
    img = (img - mean_t) / std_t

    mask_patch = None
    if mask is not None:
        mask_t = torch.from_numpy(mask).to(device).float()[None, None] / 255.0
        mask_patch = warp_affine(mask_t, M, (patch_h, patch_w), mode="nearest", padding_mode="zeros")

    return {
        "img": img,
        "mask": mask_patch,
        "box_center": torch.from_numpy(np.ascontiguousarray(center[:1])).float(),
        "box_size": torch.from_numpy(np.asarray(bbox_size, np.float32)[None]).float(),
    }


def process_data(frame_source, detector, device, model, cfg, out_folder, save_cam_output, masks_path=None,
                 downsampled_verts_pth='assets/verts_512.pkl',
                 contacts_gt=None, floor_contacts_gt=None):
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
    mean_t = torch.tensor(DEFAULT_MEAN, device=device).view(1, 3, 1, 1).float()
    std_t = torch.tensor(DEFAULT_STD, device=device).view(1, 3, 1, 1).float()

    from concurrent.futures import ThreadPoolExecutor

    def _load_frame_inputs(frame_n):
        # Runs on the single prefetch worker: decode the frame and read this
        # frame's mask PNGs while the MAIN thread is still computing the
        # previous frame. One worker + in-order submission keeps the video
        # decoder strictly sequential; values and their consumption order are
        # unchanged, so outputs stay bit-identical.
        frame_rgb = frame_source.read_rgb(frame_n)
        frame_masks = None
        if masks_path is not None:
            frame_masks = {}
            for pid in people_ids:
                p = os.path.join(masks_path, camera_id, "masks", f"mask_{frame_n:04d}_{pid:02d}.png")
                frame_masks[pid] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
        return frame_rgb, frame_masks

    _prefetcher = ThreadPoolExecutor(max_workers=1)
    _next_inputs = _prefetcher.submit(_load_frame_inputs, 0) if n_frames > 0 else None
    for frame_n in tqdm.tqdm(range(n_frames)):
        # Decode the frame once and upload it to the GPU once; every body warps
        # its crop from this resident RGB tensor (the network's input order).
        frame_rgb, frame_masks = _next_inputs.result()
        _next_inputs = (_prefetcher.submit(_load_frame_inputs, frame_n + 1)
                        if frame_n + 1 < n_frames else None)
        frame_t = torch.from_numpy(np.ascontiguousarray(frame_rgb)).to(device).permute(2, 0, 1).float()[None]
        for body_id in people_ids:
            folder_path = body_dirs[body_id]
            body_verts = verts_per_body[body_id]
            body_vis = vis_per_body[body_id]
            body_contact = contact_per_body[body_id]
            body_floor_contact = floor_contact_per_body[body_id]
            if masks_path is None:
                # Standalone (no-mask) mode: the detector finds the person box on
                # a BGR frame; the crop itself is still warped on the GPU below.
                img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
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
                # Read on the prefetch worker; None covers both missing-file
                # and failed-read (the two zero-fill branches were identical).
                mask = frame_masks.get(body_id)
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
                    # cv2.boundingRect is ~100x cheaper than np.where + min/max
                    # over a 4K mask and returns the same tight box:
                    # x2 = x + w - 1 == xs.max(), y2 = y + h - 1 == ys.max().
                    x, y, bw, bh = cv2.boundingRect(mask)
                    x1, y1, x2, y2 = x, y, x + bw - 1, y + bh - 1
                    boxes = np.array([[x1, y1, x2, y2]])
                    valid_scores = np.array([1.0])

            batch = _gpu_make_batch(frame_t, mask, boxes, cfg, device, mean_t, std_t)
            if batch is None:
                logger.warning(f"skipping body {body_id} at frame {frame_n}: no valid box")
                body_verts.append(np.zeros((1, cfg.num_joints, 3)))
                body_vis.append(np.zeros((1, cfg.num_joints)))
                body_contact.append(np.zeros((1, cfg.num_joints)))
                body_floor_contact.append(np.zeros((1, cfg.num_joints)))
                continue

            img_crop = batch["img"].cpu() * DEFAULT_STD[None, :, None, None] + DEFAULT_MEAN[None, :, None, None]
            img_crop = (img_crop[0].numpy().transpose(1, 2, 0))[:, :, ::-1].astype(np.uint8).copy()
            mask_crop = batch["mask"][0].cpu().numpy().transpose(1, 2, 0).copy() if batch["mask"] is not None else None

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

    # Pin the output dtype due to the zero-fill placeholders for missing detections
    landmarks = np.stack(all_verts, axis=1).astype(np.float32, copy=False)
    visibilities = np.stack(all_vis, axis=1).astype(np.float32, copy=False)
    contacts = np.stack(all_contact, axis=1).astype(np.float32, copy=False)
    floor_contacts = np.stack(all_floor_contact, axis=1).astype(np.float32, copy=False)

    # GT contacts are written only when provided (evaluation runs); omitted during
    # normal inference instead of being stored as empty pickled None placeholders.
    gt = {k: v for k, v in {"contacts_gt": contacts_gt,
                            "floor_contacts_gt": floor_contacts_gt}.items() if v is not None}
    np.savez(f"{out_folder}/{camera_id}.npz", landmarks=landmarks,
             visibilities=visibilities, contacts=contacts, floor_contacts=floor_contacts, **gt)


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
                      help='Optional calibration file (yaml/xcp/json). Overrides the '
                           'distortion source for --undistort; if omitted, --undistort '
                           'falls back to the per-camera distortion in the ma_cap NPZ '
                           '(chained --ma_cap_dir mode).')
    args.add_argument('--undistort', action='store_true',
                      help='Undistort frames (any supported lens model) before running '
                           'the landmark network. Coefficients come from --calibration '
                           'when given, else from the per-camera NPZ. No-op for cameras '
                           'with no distortion data. Default off. CAUTION: the network '
                           'was trained on distorted frames, so undistorted input is '
                           'out-of-distribution and measurably worsens the 3D fit — '
                           'prefer undistorting on the ma_vis (overlay) side.')
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
    args.add_argument('--disable-visualizations', '--disable_visualizations',
                      dest='disable_visualizations', action='store_true',
                      help='Skip the per-body 2D-landmark viz frames + preview video. '
                           'Visualizations are written by default; pass this to turn '
                           'them off (they are not consumed by downstream steps).')
    # Deprecated alias, kept so older presets / run-configs keep working:
    # --save_cam_output (now the default) / --no-save_cam_output still toggle
    # the same visualizations. Hidden from --help in favour of
    # --disable-visualizations; default None = "not passed".
    args.add_argument('--save_cam_output', action=argparse.BooleanOptionalAction,
                      default=None, help=argparse.SUPPRESS)
    args.add_argument('--downsampled-verts', dest='downsampled_verts',
                      default='assets/verts_512.pkl',
                      help='Path to verts_512.pkl. Previously hard-coded to '
                           'assets/verts_512.pkl; the inference runner injects '
                           'this from MAMMA_DOWNSAMPLED_VERTS_PKL.')
    args.add_argument('--tensorrt', action='store_true',
                      help='Compile the landmark network to a TensorRT engine for a faster '
                           'forward (~5x FP16). NVIDIA-only, and the FP16 speedup needs a '
                           'Volta-or-newer GPU (tensor cores); falls back to PyTorch when '
                           'torch-tensorrt is unavailable. Best for long / many-camera runs '
                           '(the one-time engine build amortizes over all frames).')
    args.add_argument('--tensorrt-fp32', dest='tensorrt_fp32', action='store_true',
                      help='With --tensorrt, use FP32 (~2x, near-exact) instead of FP16 (~5x).')
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

    When ``args.undistort`` is set, each source is configured to undistort
    every frame read (any supported lens model). The per-camera distortion
    comes from ``args.calibration`` when given; otherwise it falls back to the
    distortion carried in the per-camera ma_cap NPZ (``frame_source`` builds it
    from the cam_data dict and no-ops where absent).
    """
    sources = []

    calib_cams = None
    if args.undistort and args.calibration:
        from capture import load_calibration
        calib_cams = load_calibration(args.calibration).cameras
        logger.info(f"undistort: loaded calibration with {len(calib_cams)} cameras")
    elif args.undistort:
        # No explicit calibration: fall back to the per-camera distortion the
        # ma_cap NPZ carries (works in chained --ma_cap_dir mode). frame_source
        # builds the Camera from the cam_data dict and no-ops where absent
        # (e.g. raw --videos_dir/--images_root_dir without an NPZ).
        logger.info("undistort: no --calibration; using per-camera distortion "
                    "from the NPZ where available (chained --ma_cap_dir mode)")

    def _cam_for(name):
        if calib_cams is None:
            return None
        cam = calib_cams.get(name)
        if cam is None:
            logger.warning(f"--undistort: no calibration entry for camera {name!r}; "
                           "falling back to NPZ distortion if present")
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
                cam_data, camera=cam, undistort=args.undistort,
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
                cam_data, camera=cam, undistort=args.undistort,
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
            cam_data, camera=cam, undistort=args.undistort,
        ))
    return sources


def _build_tensorrt_forward(model, cfg, device, fp16=True, weights_path=None):
    """Compile the landmark network to a TensorRT engine and return a callable
    with the same ``(img, mask) -> dict`` contract as the eager model.

    NVIDIA-only opt-in fast path. The compiled engine is cached to disk (next to
    the weights), keyed by the weight *contents* (hash) + input shape + precision
    + GPU + TRT version, so the first run pays the ~minute build and later runs
    load it in ~2 s; swapping in a different checkpoint -- even at the same path,
    size and mtime -- changes the content hash and rebuilds. The single forward
    call site is unchanged — this just swaps what ``model`` points at, so there is
    no duplicated inference code.
    """
    import torch_tensorrt
    import hashlib
    keys = ("joints2d", "visibility", "contact", "floor_contact")

    class _Tuple(torch.nn.Module):  # ONNX/TRT export needs tuple (not dict) outputs
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, mask):
            o = self.m(x, mask)
            return tuple(o[k] for k in keys)

    h, w = int(cfg.data_cfg["image_size"][1]), int(cfg.data_cfg["image_size"][0])
    example = (torch.randn(1, 3, h, w, device=device), torch.rand(1, 1, h, w, device=device))

    cache_path = None
    if weights_path:
        # Key on the weight *contents* (streamed hash), not path+size+mtime, so a
        # checkpoint swapped in at the same path -- even with size and mtime
        # preserved (rsync -a / cp -p / tar) -- yields a different key and rebuilds
        # instead of silently loading a stale engine. The full read is cheap next
        # to the ~minute engine build.
        wh = hashlib.md5()
        with open(weights_path, "rb") as _wf:
            for _chunk in iter(lambda: _wf.read(1 << 20), b""):
                wh.update(_chunk)
        gpu = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"
        key = hashlib.md5(
            f"{wh.hexdigest()}|{h}x{w}|"
            f"{'fp16' if fp16 else 'fp32'}|{gpu}|trt{torch_tensorrt.__version__}".encode()
        ).hexdigest()[:16]
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(weights_path)), ".trt_cache")
        cache_path = os.path.join(cache_dir, f"ma2d_trt_{key}.ep")

    engine = None
    if cache_path and os.path.exists(cache_path):
        try:
            engine = torch.export.load(cache_path).module()
            logger.info(f"ma_2d: loaded cached TensorRT engine {cache_path}")
        except Exception as e:
            logger.warning(f"ma_2d: cached engine load failed ({e}); rebuilding.")
            engine = None

    if engine is None:
        engine = torch_tensorrt.compile(
            _Tuple(model).eval().to(device), ir="dynamo", arg_inputs=example,
            enabled_precisions={torch.float16 if fp16 else torch.float32},
            truncate_double=True, min_block_size=1,
        )
        if cache_path:
            # Atomic publish: write to a private temp file then rename, so a
            # concurrent cold-cache run can never read a half-written .ep.
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch_tensorrt.save(engine, tmp_path, arg_inputs=example)
                os.replace(tmp_path, cache_path)
                logger.info(f"ma_2d: cached TensorRT engine to {cache_path}")
            except Exception as e:
                logger.warning(f"ma_2d: could not cache TensorRT engine ({e}).")
                try:
                    os.path.exists(tmp_path) and os.remove(tmp_path)
                except Exception:
                    pass

    def forward(img, mask):
        return dict(zip(keys, engine(img, mask)))

    return forward


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
    model.load_state_dict(torch.load(args.weights)['state_dict'])
    model.eval()

    # Forward backend: eager PyTorch by default; opt-in TensorRT fast path that
    # falls back to eager on any failure (non-NVIDIA host, missing torch-tensorrt,
    # unsupported op) so the flag never breaks a run.
    forward = model
    if getattr(args, "tensorrt", False) and masks_folder is None:
        # The engine is compiled with a mask *tensor* example input; standalone
        # (no-mask) mode calls forward(img, None), which the compiled module
        # rejects at runtime — and the try/except below only covers compile
        # failures. Refuse up front and run eager instead of crashing mid-run.
        logger.warning("ma_2d: --tensorrt is only supported in mask mode "
                       "(standalone mode passes mask=None); using the PyTorch forward.")
    elif getattr(args, "tensorrt", False):
        try:
            forward = _build_tensorrt_forward(model, cfg, device, fp16=not args.tensorrt_fp32,
                                              weights_path=args.weights)
            logger.info(f"ma_2d: using TensorRT {'FP32' if args.tensorrt_fp32 else 'FP16'} forward backend.")
        except Exception as e:
            logger.warning(f"ma_2d: TensorRT compile failed ({e}); using PyTorch forward.")
            forward = model

    # The detector only finds person boxes in standalone (no-mask) mode; in mask
    # mode the boxes come from the segmentation masks, so the detector is never
    # called. Skip building it then — loading cascade Mask-RCNN ViTDet-H is pure
    # wasted startup (weights download + load) on every masked ma_2d run.
    detector = None
    if masks_folder is None:
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
    # Visualizations write by default; --disable-visualizations turns them off.
    # The deprecated --save_cam_output/--no-save_cam_output still wins when passed.
    save_viz = not args.disable_visualizations
    if args.save_cam_output is not None:
        save_viz = args.save_cam_output
    for source in sources:
        process_data(source, detector, device, forward, cfg, out_folder,
                     save_viz, masks_folder,
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
