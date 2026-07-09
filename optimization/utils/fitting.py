import numpy as np
import torch
from typing import List, Optional
import time

import torch.multiprocessing as mp
from multiprocessing import Pool
import os
import numpy as np
from utils_draw import draw_2d_pts
from utils.utils_camera import get_projected_points
from utils.optimization import OptimizeSMPLX
from utils.paths_config import PathsConfig


def process_2d_pred(body_id, pred_fns, smplx_out, cam_metadata, batch_size, i):
    pts2d_np, pts2d_vis_np, contacts_np, floor_contacts_np = None, None, None, None
    if pred_fns is None:
        print("Using SMPLX projected 2D points for body id", body_id)
        pts2d_np = get_projected_points(smplx_out[body_id], cam_metadata, batch_size)
        pts2d_vis_np = np.ones(pts2d_np.shape[:2], dtype=np.float32)
    else:
        pts2d_data = np.load(pred_fns[i], allow_pickle=True)
        pts2d_np = pts2d_data["landmarks"][:, body_id]
        if pts2d_np.shape[-1] == 3:
            pts2d_np[..., -1] = np.exp(pts2d_np[..., -1])
            # var to std
            pts2d_np[..., -1] = np.sqrt(pts2d_np[..., -1])
        else:
            pts2d_np = pts2d_np[..., :2]
        if "visibilities" in pts2d_data:
            pts2d_vis_np = pts2d_data["visibilities"][:, body_id]
        else:
            pts2d_vis_np = None
        if "contacts" in pts2d_data and pts2d_data["contacts"][0] is not None:
            contacts_np = pts2d_data["contacts"][:, body_id]
        else:
            contacts_np = None

        if "floor_contacts" in pts2d_data and pts2d_data["floor_contacts"][0] is not None:
            floor_contacts_np = pts2d_data["floor_contacts"][:, body_id]
        else:
            floor_contacts_np = None

    return pts2d_np, pts2d_vis_np, contacts_np, floor_contacts_np


def load_frames_in_parallel(dat_args):
    camera_metadata_fn, pred_fns, hand_joints_pred_fns, smplx_out, imgs_pth, batch_size, valid_frames_idx, i, every_n_frames, scale, rotate_cams, body_id = dat_args
    cam_metadata = np.load(camera_metadata_fn, allow_pickle=True)
    cam_id = os.path.splitext(os.path.basename(camera_metadata_fn))[0]
    cam_imgs_dir = os.path.join(imgs_pth, cam_id)

    intrinsics_np = np.repeat(cam_metadata["cam_int"][None], batch_size, axis=0)
    extrinsics_np = np.repeat(cam_metadata["cam_ext"][None], batch_size, axis=0)
    # from cm to m
    # extrinsics_np[:, :3, -1] = extrinsics_np[:, :3, -1] / 100
    # Ensure extrinsics translation is in meters (auto-detect mm by magnitude).
    if extrinsics_np[:, :3, -1].max() > 200:
        print("Extrinsics are in mm, converting to m")
        extrinsics_np[:, :3, -1] = extrinsics_np[:, :3, -1] / 1000

    camera_width = np.repeat(cam_metadata["cam_img_w"][None], batch_size, axis=0)
    camera_height = np.repeat(cam_metadata["cam_img_h"][None], batch_size, axis=0)

    try:
        pts2d_np, pts2d_vis_np, contacts_np, floor_contacts_np = process_2d_pred(body_id, pred_fns, smplx_out, cam_metadata, batch_size, i)
    except Exception as e:
        print(f"ma_3d: {cam_id}: could not load 2D predictions for body {body_id} "
              f"({type(e).__name__}: {e}): zero-filling, this camera won't constrain body {body_id}. "
              f"An IndexError just means the 2D file has fewer body slots: subject not visible in this view.")
        # pts2d_np, pts2d_vis_np, contacts_np, floor_contacts_np = None, None, None, None
        pts2d_np = np.zeros((batch_size, 512, 3), np.float32)
        pts2d_vis_np = np.zeros((batch_size, 512), dtype=np.float32)
        contacts_np = np.zeros((batch_size, 512), dtype=np.float32)
        floor_contacts_np = np.zeros((batch_size, 512), dtype=np.float32)


    img_h = int(cam_metadata["cam_img_h"])
    img_w = int(cam_metadata["cam_img_w"])

    cam_imgs = []
    for frame_idx in range(0, batch_size, every_n_frames):
        if frame_idx in valid_frames_idx:
            cam_imgs.append(None)

    if hand_joints_pred_fns is not None:
        hand_joints2d_np = np.load(hand_joints_pred_fns[i], allow_pickle=True)["landmarks"][:, ] # TODO: add extra dimension for the person id
        within_bounds_mask = (
                (hand_joints2d_np[:, :, 0] >= 0) & (hand_joints2d_np[:, :, 0] < img_w) &  # x within bounds
                (hand_joints2d_np[:, :, 1] >= 0) & (hand_joints2d_np[:, :, 1] < img_h)    # y within bounds
            )
        hand_joints2d_np[~within_bounds_mask, :2] = 0
    else:
        hand_joints2d_np = None

    print("Loaded", cam_id, "with", len(cam_imgs), "frames")
    return cam_id, pts2d_np, hand_joints2d_np, intrinsics_np, extrinsics_np, cam_imgs, pts2d_vis_np, camera_height, camera_width, contacts_np, floor_contacts_np


def extract_data_from_metadata(body_id, pred_fns, hand_joints_pred_fns, smplx_out, imgs_pth, batch_size, cameras_metadata_fns, valid_frames_idx, rotate_cams, parallel=True):
    cam_intrinsics = []
    cam_extrinsics = []
    dense_lndmks2d = []
    hand_joints2d = []
    cam_imgs_seq = []
    cam_ids = []
    pts2d_vis_weight = []
    cam_height = []
    cam_width = []
    contacts = []
    floor_contacts = []
    time_start = time.time()
    if parallel:
        if not mp.get_start_method(allow_none=True):
            mp.set_start_method('spawn')
        with Pool(4) as pool:
            all_data = [(camera_metadata_fn, pred_fns, hand_joints_pred_fns, smplx_out, imgs_pth, batch_size, valid_frames_idx, i, 1, 1, rotate_cams, body_id) for i, camera_metadata_fn in enumerate(cameras_metadata_fns)]
            out = pool.map(load_frames_in_parallel, all_data)
            cam_ids, dense_lndmks2d, hand_joints2d, cam_intrinsics, cam_extrinsics, cam_imgs_seq, pts2d_vis_weight, cam_height, cam_width, contacts, floor_contacts = zip(*out)
    else:
        for i, camera_metadata_fn in enumerate(cameras_metadata_fns):
            dat_args = (camera_metadata_fn, pred_fns, hand_joints_pred_fns, smplx_out, imgs_pth, batch_size, valid_frames_idx, i, 1, 1, rotate_cams, body_id)
            cam_id, pts2d_np, hand_joints2d_np, intrinsics_np, extrinsics_np, cam_imgs, pts2d_vis_np, camera_height_np, camera_width_np, contacts_np, floor_contacts_np = load_frames_in_parallel(dat_args)

            cam_ids.append(cam_id)
            dense_lndmks2d.append(pts2d_np)
            hand_joints2d.append(hand_joints2d_np)
            cam_intrinsics.append(intrinsics_np)
            cam_extrinsics.append(extrinsics_np)
            cam_imgs_seq.append(cam_imgs)
            pts2d_vis_weight.append(pts2d_vis_np)
            cam_height.append(camera_height_np)
            cam_width.append(camera_width_np)
            contacts.append(contacts_np)
            floor_contacts.append(floor_contacts_np)
    print(f"Loading frames from {len(cameras_metadata_fns)} cameras took", time.time()-time_start, "seconds")
    return dense_lndmks2d, hand_joints2d, cam_intrinsics, cam_extrinsics, cam_height, cam_width, cam_imgs_seq, pts2d_vis_weight, cam_ids, contacts, floor_contacts


def fit_smplx(body_ids: str, pred_fns: List[str], cameras_metadata_fns: List[str], imgs_pth: str, smplx_model, batch_size: int,
                downsampled_verts_mat: Optional[torch.Tensor] = None, smplx_out=None, device="cuda",
                optim_cfg=None, valid_frames_idx=None,
                hand_joints_pred_fns=None, rotate_cams=None,  save_prediction_fn=None,
                start_frame=0, end_frame=None, ignore_start_frames: int = 0,
                paths: Optional[PathsConfig] = None, up_axis: str = "z", parallel=True):

    body_params = {body_id: None for body_id in body_ids}
    world_params = None
    for body_id in body_ids:
        dense_lndmks2d, hand_joints2d, cam_intrinsics, cam_extrinsics, cam_height, cam_width, cam_imgs_seq, pts2d_vis_weight, cam_ids, contacts, floor_contacts = extract_data_from_metadata(
                                                                                                                                            body_id,
                                                                                                                                            pred_fns,
                                                                                                                                            hand_joints_pred_fns,
                                                                                                                                            smplx_out,
                                                                                                                                            imgs_pth,
                                                                                                                                            batch_size,
                                                                                                                                            cameras_metadata_fns,
                                                                                                                                            valid_frames_idx,
                                                                                                                                            rotate_cams,
                                                                                                                                            parallel
                                                                                                                                            )

        # Slice frames according to start_frame / end_frame from config
        dense_lndmks2d = [d[start_frame:end_frame] for d in dense_lndmks2d]
        pts2d_vis_weight = [v[start_frame:end_frame] if v is not None else v for v in pts2d_vis_weight]
        contacts = [c[start_frame:end_frame] if c is not None else c for c in contacts]
        floor_contacts = [fc[start_frame:end_frame] if fc is not None else fc for fc in floor_contacts]
        hand_joints2d = [h[start_frame:end_frame] if h is not None else h for h in hand_joints2d]
        cam_intrinsics = [k[start_frame:end_frame] for k in cam_intrinsics]
        cam_extrinsics = [e[start_frame:end_frame] for e in cam_extrinsics]
        cam_height = [ch[start_frame:end_frame] for ch in cam_height]
        cam_width = [cw[start_frame:end_frame] for cw in cam_width]

        n_frames = dense_lndmks2d[0].shape[0]
        if body_id == body_ids[0]:
            print(f"Using frames [{start_frame}:{end_frame}] → {n_frames} frames")

        n_betas = smplx_model[body_id]["neutral"].betas.shape[-1]
        smplx_params_size = {"pose": (n_frames, int(55*3)), "trans": (n_frames, 3), "betas": (n_frames, n_betas)}

        body_params[body_id] = {"dense_lndmks2d": dense_lndmks2d, "hand_joints2d": hand_joints2d, "pts2d_vis_weight": pts2d_vis_weight, "smplx_params_size": smplx_params_size,
                                "contacts": contacts, "floor_contacts": floor_contacts}
        if world_params is None:
            world_params = {"cam_intrinsics": cam_intrinsics, "cam_extrinsics": cam_extrinsics,
                            "cam_height": cam_height, "cam_width": cam_width,
                            "cam_metadata_fns": cameras_metadata_fns,
                            "imgs_pth": imgs_pth}

    # print optim_cfg
    print("Optimization Configurations:")
    for key, value in optim_cfg.items():
        print(f"{key}: {value}")
    print("Optimization started")
    if paths is None:
        raise ValueError(
            "fit_smplx requires a PathsConfig (paths=). The caller (run_ma_3d.py) "
            "builds it from argparse flags; do not omit it when calling directly."
        )
    smplx_fitter = OptimizeSMPLX(body_params, world_params, smplx_model, downsampled_verts_mat, optim_cfg,
                                 paths,
                                 save_prediction_fn=save_prediction_fn, device=device,
                                 skip_start=ignore_start_frames, up_axis=up_axis)
    smplx_pose, smplx_betas, smplx_trans, triangulated_3d_pts, smplx_contact, smplx_floor_contact = smplx_fitter.fit(optim_verts=True,
                                                            optim_cfg=optim_cfg)

    print(smplx_betas)

    return smplx_pose, smplx_betas, smplx_trans, triangulated_3d_pts, smplx_contact, smplx_floor_contact, list(zip(*cam_imgs_seq)), cam_ids
