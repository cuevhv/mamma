import numpy as np
import torch
from torch.optim import LBFGS
from utils_smplx import get_smplx_models, get_smplx_forward, get_smplx_forward_per_parts
from losses.losses import (proj_pts_loss, trans_temp_loss, rotation_prior_loss, body_shape_prior_loss,
                           temp_pose_loss, SMPLifyAnglePrior, rot_consistency_loss, acceleration_loss,
                           angular_acceleration_loss, proj_pts_error,
                           visibility_loss)
from utils.vposer import vposer_recon_loss
from losses.sdf import MultiSDF
import utils.triangulation_functions as triang_funcs
from utils.paths_config import PathsConfig
from typing import List, Optional
import time
import os
import copy
from collections import Counter


LEFT_IDX=[20,37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70]
RIGHT_IDX=[21,52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75]
HANDS_IDX=LEFT_IDX+RIGHT_IDX
LIMBS_BODY_JOINTS_SMPLX = np.array([1, 4, 7, 2, 5, 8, 16, 18, 20, 17, 19, 21])  - 1  # ignore root joint
LIMBS_BODY_JOINTS_SMPLX = LIMBS_BODY_JOINTS_SMPLX.tolist()

def closest_point_rays(origins, directions):
    I = torch.eye(3, device=origins.device)
    A = torch.zeros((3, 3), device=origins.device)
    b = torch.zeros(3, device=origins.device)

    for o, d in zip(origins, directions):
        d = d / torch.norm(d)
        A += I - torch.outer(d, d)
        b += (I - torch.outer(d, d)) @ o

    point = torch.linalg.solve(A, b)
    return point


class OptimizeSMPLX:
    def __init__(self, body_params, world_params, smplx_model, downsampled_verts_mat, optim_cfg,
                 paths: PathsConfig,
                 save_prediction_fn=None, device="cuda", skip_start: int = 0, up_axis: str = "z"):
        self._paths = paths
        self.up_axis = up_axis           # signed world up-axis for the contact .rrd
        # Opt-in robust triangulation (--tri-irls): cross-view residual (default off)
        self.tri_irls = bool(optim_cfg.get("tri_irls", False)) if isinstance(optim_cfg, dict) else False
        # Opt-in temporal outlier gate on the triangulated 3D seed (--tri-outlier-gate):
        # bridges isolated "teleport" frames before optimizing (default off).
        self.tri_outlier_gate = bool(optim_cfg.get("tri_outlier_gate", False)) if isinstance(optim_cfg, dict) else False
        self.tri_outlier_gate_cfg = (optim_cfg.get("outlier_gate", {}) if isinstance(optim_cfg, dict) else {}) or {}
        intrinsics_np = world_params["cam_intrinsics"]
        extrinsics_np = world_params["cam_extrinsics"]
        self.cam_heights = world_params["cam_height"]
        self.cam_widths = world_params["cam_width"]
        self.decim_mesh = np.load("smplx_simplified_face_idx.npy").astype(np.int32)
        self.limb_vertices_idx = np.load("vertex_limbs.npy").astype(np.int32)

        self.save_prediction_fn = save_prediction_fn
        os.makedirs(self.save_prediction_fn, exist_ok=True)
        self.skip_start = max(0, int(skip_start))
        # Camera metadata file paths and image root dir — used for re-ID debug visualizations.
        self.cam_metadata_fns = world_params.get("cam_metadata_fns", None)
        self.imgs_pth = world_params.get("imgs_pth", "")
        self.downsampled_verts_mat = downsampled_verts_mat["downsampled_verts_mat"].to(device) if downsampled_verts_mat is not None else None

        self.body_idx = np.array([296, 40, 338, 82, 281, 25, 278, 22, 269, 13, 467, 211, 305, 79]) if downsampled_verts_mat is not None else None
        self.body_idx = np.concatenate([downsampled_verts_mat["body"], downsampled_verts_mat["right_head"], downsampled_verts_mat["left_head"]]) if downsampled_verts_mat is not None else None
        self.angle_prior = SMPLifyAnglePrior(device=device, dtype=torch.float64)
        self.sdf_loss = MultiSDF(mesh_faces=(smplx_model[-1]["neutral"].faces).astype(np.int32), max_batch_size=4,
                                 valid_vertices_idx=self.limb_vertices_idx,
                                 part_mesh_path=self._paths.part_mesh_path)
        self.faces = smplx_model[-1]["neutral"].faces

        self.K = [torch.from_numpy(intrinsic_np).float().to(device) for intrinsic_np in intrinsics_np]
        self.extrinsics = [torch.from_numpy(extrinsic).float().to(device) for extrinsic in extrinsics_np]

        self.smplx_model = smplx_model

        kwargs = {"dtype": torch.float, "device": device}
        self.pts2d = {body_id: [torch.from_numpy(pts_2d).float().to(device) for pts_2d in body_params[body_id]["dense_lndmks2d"]] for body_id in body_params.keys()}

        if body_params[0]["pts2d_vis_weight"][0] is not None:
            self.pts2d_vis_weight = {
                body_id: [
                    torch.from_numpy(pts2d).float().to(device) if pts2d.squeeze().ndim == 1
                    else torch.from_numpy(pts2d.squeeze()).float().to(device)
                    for pts2d in body_params[body_id]["pts2d_vis_weight"]
                ]
                for body_id in body_params.keys()
            }
        else:
            self.pts2d_vis_weight = {body_id: None for body_id in body_params.keys()}

        if body_params[0]["contacts"][0] is not None and body_params[0]["contacts"][0][0] is not None:
            self.contacts = {
                body_id: [
                    torch.from_numpy(contacts).float().to(device) if contacts.squeeze().ndim == 1
                    else torch.from_numpy(contacts.squeeze()).float().to(device)
                    for contacts in body_params[body_id]["contacts"]
                ]
                for body_id in body_params.keys()
            }
            self.floor_contacts = {
                body_id: [
                    torch.from_numpy(floor_contacts).float().to(device) if floor_contacts.squeeze().ndim == 1
                    else torch.from_numpy(floor_contacts.squeeze()).float().to(device)
                    for floor_contacts in body_params[body_id]["floor_contacts"]
                ]
                for body_id in body_params.keys()
            }
        else:
            self.contacts = {body_id: None for body_id in body_params.keys()}
            self.floor_contacts = {body_id: None for body_id in body_params.keys()}


        # A camera that essentially never sees a subject must not vote in that subject's fit:
        self._gate_unsupported_views()

        self.triangulated_contact = None
        self.triangulated_floor_contact = None
        self.triangulated_3d_points = None


        if body_params[0]["hand_joints2d"][0] is not None:
            self.hand_joints2d = {body_id: [torch.from_numpy(hand_joints2d).float().to(device) for hand_joints2d in body_params[body_id]["hand_joints2d"]] for body_id in body_params.keys()}
        else:
            self.hand_joints2d = {body_id: None for body_id in body_params.keys()}

        self.poses = {}
        self.smplx_betas = {}
        self.smplx_trans = {}
        for body_id in body_params.keys():
            smplx_params_size = body_params[body_id]["smplx_params_size"]
            poses, smplx_betas, smplx_trans = self.init_learnable_params(smplx_params_size, init_trans_w_extrinsics=True, device=device)
            self.poses[body_id] = poses
            self.smplx_betas[body_id] = smplx_betas
            self.smplx_trans[body_id] = smplx_trans


    # View-support gate thresholds: a (camera, body) pair counts as supported in
    # a frame when >= MIN_LANDMARKS of its 512 landmarks are inside the image AND
    # have visibility > VIS_THRESH; the pair is kept only if >= MIN_FRAME_FRAC of
    # frames are supported. Support is a property of the (view, person) pair over the
    # clip, not of frames.
    _VIEW_SUPPORT_MIN_LANDMARKS = 20
    _VIEW_SUPPORT_VIS_THRESH = 0.5
    _VIEW_SUPPORT_MIN_FRAME_FRAC = 0.5

    def _view_support_fractions(self, body_id):
        """Per-camera support fraction for one body: the fraction of frames with
        >= _VIEW_SUPPORT_MIN_LANDMARKS in-bounds landmarks at vis >
        _VIEW_SUPPORT_VIS_THRESH. Shared by the view-support gate (pre-re-ID)
        and the untriangulatable-body exclusion (post-re-ID)."""
        vis_list = self.pts2d_vis_weight.get(body_id)
        if vis_list is None:
            return None
        fractions = []
        for cam_i, (pts, vis) in enumerate(zip(self.pts2d[body_id], vis_list)):
            w = float(np.asarray(self.cam_widths[cam_i]).reshape(-1)[0])
            h = float(np.asarray(self.cam_heights[cam_i]).reshape(-1)[0])
            xy = pts[..., :2]
            in_bounds = (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
            strong = in_bounds & (vis > self._VIEW_SUPPORT_VIS_THRESH)
            fractions.append((strong.sum(dim=1) >= self._VIEW_SUPPORT_MIN_LANDMARKS).float().mean().item())
        return fractions

    def _gate_unsupported_views(self):
        """Zero the 2D landmarks, visibility and contacts of every unsupported
        (camera, body) pair, so the pair drops out of triangulation (its points
        fail the abs-sum>0 validity mask), the reprojection loss (pred_in_img
        mask), the epipolar re-ID (zero trajectory -> zero affinity), and the
        contact aggregation alike. Zeroing the points is what removes the view
        from the reprojection loss — zeroed visibility alone would be clipped
        back up to vis_clip_value there.

        Safeguard: if gating would leave a body with fewer than 2 supported
        views (possible only when no camera sees the person for the majority of
        the clip — e.g. moving cameras or mid-sequence entrances, never observed
        on rig-style captures), the gate abstains for that body rather than
        deleting its remaining 2D evidence.
        """
        for body_id, cams_pts in self.pts2d.items():
            supports = self._view_support_fractions(body_id)
            if supports is None:
                continue
            gated = [i for i, s in enumerate(supports) if s < self._VIEW_SUPPORT_MIN_FRAME_FRAC]
            if gated and len(cams_pts) - len(gated) < 2:
                print(f"ma_3d: view-support gate: body {body_id} would keep fewer than 2 supported "
                      f"views — leaving all its views ungated instead of removing its remaining "
                      f"2D evidence.")
                continue
            for cam_i in gated:
                cams_pts[cam_i].zero_()
                self.pts2d_vis_weight[body_id][cam_i].zero_()
                # Contacts too: contact_prob is an unweighted camera mean, and
                # zeroed visibility would otherwise disable the visible-based
                # suppression that normally mutes this view's contact junk.
                if self.contacts[body_id] is not None:
                    self.contacts[body_id][cam_i].zero_()
                    self.floor_contacts[body_id][cam_i].zero_()
                print(f"ma_3d: view-support gate: camera {cam_i} sees body {body_id} in only "
                      f"{supports[cam_i]:.0%} of frames (needs {self._VIEW_SUPPORT_MIN_FRAME_FRAC:.0%} with "
                      f">={self._VIEW_SUPPORT_MIN_LANDMARKS} in-bounds points at vis>"
                      f"{self._VIEW_SUPPORT_VIS_THRESH}) — excluding this view from body "
                      f"{body_id}'s fit.")

    def init_learnable_params(self, smplx_params_size: dict, init_trans_w_extrinsics: bool = False, device: str = "cuda"):
        kwargs = {"dtype": torch.float, "device": device}
        poses = {}
        poses["global"] = torch.tensor(np.zeros([smplx_params_size["pose"][0], 3]), **kwargs).requires_grad_()
        poses["body"] = torch.tensor(np.zeros([smplx_params_size["pose"][0], 63]), **kwargs)#.requires_grad_()
        poses["jaw"] = torch.tensor(np.zeros([smplx_params_size["pose"][0], 3]), **kwargs).requires_grad_()
        poses["leye"] = torch.tensor(np.zeros([smplx_params_size["pose"][0], 3]), **kwargs).requires_grad_()
        poses["reye"] = torch.tensor(np.zeros([smplx_params_size["pose"][0], 3]), **kwargs).requires_grad_()
        poses["left_hand"] = torch.tensor(np.zeros([smplx_params_size["pose"][0], 45]), **kwargs).requires_grad_()
        poses["right_hand"] = torch.tensor(np.zeros([smplx_params_size["pose"][0], 45]), **kwargs).requires_grad_()


        # make both hands be in front of the body with the elbows bent
        poses["body"][:, -3+17*3+2] = np.pi/4  # shoulder right z
        poses["body"][:, -3+16*3+2] = -np.pi/4  # shoulder left z

        poses["body"][:, -3+17*3+1] = np.pi/7  # shoulder right y
        poses["body"][:, -3+16*3+1] = -np.pi/7  # shoulder left y

        poses["body"][:, -3+19*3+1] = np.pi/1.7  # elbow right
        poses["body"][:, -3+18*3+1] = -np.pi/1.7  # elbow left
        poses["body"] = poses["body"].requires_grad_()

        smplx_betas = torch.tensor(np.zeros(smplx_params_size["betas"][1]), **kwargs)[None,].requires_grad_()
        # make z -1
        smplx_trans_np = np.zeros(smplx_params_size["trans"])
        if init_trans_w_extrinsics:
            pass

        extrinsics = torch.cat([ext[:1].detach().cpu() for ext in self.extrinsics]) # 20, 4, 4
        if extrinsics.shape[1] == 3:
            extrinsics = torch.cat([extrinsics, torch.tensor([[0, 0, 0, 1]], device=extrinsics.device).expand(extrinsics.shape[0], -1, -1)], dim=1)

        if torch.linalg.det(extrinsics)[0] < 0 or torch.linalg.det(extrinsics)[0] > 1:
            print("determinant of extrinsics is not 1, using svd to fix it")
            print(torch.linalg.det(extrinsics))
            U, S, V = torch.linalg.svd(extrinsics[:, :3, :3], full_matrices=False)
            extrinsics[:, :3, :3] = U @ V

        ray_z = extrinsics[:, 2, :3]/torch.linalg.norm(extrinsics[:, 2, :3])
        ray_o = -torch.einsum("bkj,bk->bj", extrinsics[:, :3, :3], extrinsics[:, :3, 3])
        smplx_trans_np = smplx_trans_np + closest_point_rays(ray_o, ray_z).numpy()

        smplx_trans = torch.tensor(smplx_trans_np, **kwargs).requires_grad_()
        return poses, smplx_betas, smplx_trans


    def register_limbs_grad_masks(self):
        """
        Register gradient hooks so that only upper-body limb DOFs of poses['body']
        (collars, shoulders, elbows, wrists) are updated during optimization.

        Assumes SMPL-X body_pose is 21x3 in the order:
        [left_hip, right_hip, spine1, left_knee, right_knee, spine2,
        left_ankle, right_ankle, spine3, left_foot, right_foot, neck,
        left_collar, right_collar, head,
        left_shoulder, right_shoulder, left_elbow, right_elbow,
        left_wrist, right_wrist]
        """

        if hasattr(self, "_limb_masks_registered") and self._limb_masks_registered:
            # Avoid registering hooks multiple times
            return

        # Body-pose joint indices (0-based in body_pose) for upper body limbs
        # (see mapping above)

        print("Registering limb gradient masks for body poses...")
        ARM_BODY_JOINTS = [0, 3, 6, 1, 4, 7, 15, 17, 19, 16, 18, 20]
        weights = [0.2, 0.4, 1.0,  # left leg
                   0.2, 0.4, 1.0,  # right leg
                   0.001, 0.001, 1.5,  # left arm
                   0.001, 0.001, 1.5,  # right arm
                   ]

        for body_id, poses in self.poses.items():
            body_pose = poses["body"]  # [T, 63]
            device = body_pose.device
            D = body_pose.shape[1]     # should be 63

            # Build mask [1, 63] where only arm DOFs are 1
            body_grad_mask = torch.zeros(1, D, device=device)

            dof_idx = []
            weights_idx = []
            for idx, j in enumerate(ARM_BODY_JOINTS):
                base = 3 * j
                dof_idx.extend([base + 0, base + 1, base + 2])
                weights_idx.extend([weights[idx]] * 3)

            body_grad_mask[0, dof_idx] = torch.tensor(weights_idx, device=device)
            body_grad_mask.requires_grad_(False)

            # Store mask if needed later
            if not hasattr(self, "body_grad_masks"):
                self.body_grad_masks = {}
            self.body_grad_masks[body_id] = body_grad_mask

            # Register hook for this body's body_pose
            def make_hook(mask):
                def hook(grad):
                    # grad: [T, 63]
                    return grad * mask  # mask broadcast along T dimension
                return hook

            body_pose.register_hook(make_hook(body_grad_mask))

        self._limb_masks_registered = True


    def loss_functions(self, pts2d: List[torch.Tensor], K: List[torch.Tensor], extrinsics: List[torch.Tensor],
                       pts3d: torch.Tensor, joints3d_pred: torch.Tensor, joint_angles: torch.Tensor,
                       body_shape: torch.Tensor, losses_names: List[str],
                       reprojection_loss_name: str = "mse",
                       hand_joints2d: Optional[List[torch.Tensor]] = None,
                       weights_cfg=None,
                       pts2d_vis_weight: Optional[torch.Tensor] = None,
                       verts3d_pred: Optional[torch.Tensor] = None,
                       poses: Optional[dict] = None,
                       triangulated_3d_points: Optional[torch.Tensor] = None,
                       vis_clip_value: float = 0.8,
                       occlusion_gating: Optional[dict] = None,
                       ):

        losses_functions = {
                "reproj_loss": (proj_pts_loss, (pts2d, pts3d, K, extrinsics, reprojection_loss_name, pts2d_vis_weight, vis_clip_value, occlusion_gating)),
                "pts3d_temp_loss": (trans_temp_loss, (joints3d_pred,)),
                "rotation_prior_loss": (rotation_prior_loss, (joint_angles,)),
                "vposer_recon_loss": (vposer_recon_loss, (joint_angles,)),
                "body_shape_prior_loss": (body_shape_prior_loss, (body_shape, )),
                "temp_pose_loss": (lambda ja, w: temp_pose_loss(ja, w, distance=1, rot_type="rotmat") + temp_pose_loss(ja, 5*w, distance=3, rot_type="rotmat"), (joint_angles, )),
                "angle_prior_loss": (self.angle_prior, (joint_angles, True, )),
                "rot_consistency_loss": (rot_consistency_loss, (joint_angles, )),
                "acceleration_loss": (acceleration_loss, (pts3d, )),  # pts3d
                "hand_reproj_loss": (proj_pts_loss, (hand_joints2d, joints3d_pred[:, HANDS_IDX], K, extrinsics, reprojection_loss_name, None)),
                "hand_acceleration_loss": (acceleration_loss, (joints3d_pred[:, HANDS_IDX], )),
                "hand_pts3d_temp_loss": (trans_temp_loss, (joints3d_pred[:, HANDS_IDX],)),
                "rotation_prior_hands_loss": (lambda ja, w: rotation_prior_loss(ja["left_hand"][...,1::3], w) + rotation_prior_loss(ja["left_hand"][...,0::3], w) + rotation_prior_loss(ja["right_hand"][...,1::3], w) + rotation_prior_loss(ja["right_hand"][...,0::3], w), (poses,)),
                "angular_acc_loss": (angular_acceleration_loss, (joint_angles, )),
                "sdf_loss": (self.sdf_loss.batch_multi_sdf_loss, (verts3d_pred,)),
                "l2_loss_3d_points": (lambda pts3d_, tri3d_, w: w*torch.nn.functional.mse_loss(pts3d_, tri3d_), (pts3d, triangulated_3d_points)) if triangulated_3d_points is not None else (lambda w: torch.tensor(0., device=joints3d_pred.device), ()),
                }


        all_losses = {}
        for loss_name in losses_names:
            if loss_name in losses_functions:
                loss_fn, inputs = losses_functions[loss_name]
                full_inputs = inputs + (float(weights_cfg[loss_name]["weight"]),)
                all_losses[loss_name] = loss_fn(*full_inputs)

        return all_losses


    def loss_functions_multi_people(self, vertices: List[torch.Tensor], pts2d, K, extrinsics,
                                    bodies_visibility, bodies_sampled_verts,
                                    bodies_contacts,
                                    losses_names, weights_cfg, ignore_first_t_frames=10):
        losses_functions = {
            "intersection_loss": (self.sdf_loss.multi_people_sdf_loss, (vertices, bodies_sampled_verts, bodies_contacts, ignore_first_t_frames)),
            "visibility_loss": (visibility_loss, (pts2d, bodies_sampled_verts, self.faces,
                                                  K, extrinsics, self.cam_heights, self.cam_widths, bodies_visibility, self.downsampled_verts_mat)),
        }

        all_losses = {}
        for loss_name in losses_names:
            if loss_name in losses_functions:
                loss_fn, inputs = losses_functions[loss_name]
                full_inputs = inputs + (float(weights_cfg[loss_name]["weight"]),)
                all_losses[loss_name] = loss_fn(*full_inputs)

        if not all_losses:
            return None

        return all_losses


    @staticmethod
    def _resolve_frame_slice(skip_frames: int):
        """
        Compute the [start, end) frame slice used in optimization/triangulation.
        """
        skip_frames = max(0, int(skip_frames))
        return skip_frames, None


    def process_data(self, pts3d_pred, joints3d_pred, lndmks2d, lndmks2d_vis, hand_joints2d, triangulated_3d_points,
                     triangulated_contacts,
                     skip_first_t_frames=0, valid_verts_idx=None):
        start_frame, end_frame = self._resolve_frame_slice(
            skip_frames=skip_first_t_frames,
        )


        pts3d_pred = pts3d_pred[start_frame:end_frame, valid_verts_idx]
        joints3d_pred = joints3d_pred[start_frame:end_frame, :]

        pts2d = [pts2d[start_frame:end_frame, valid_verts_idx] for pts2d in lndmks2d]
        pts2d_vis_weight = [pts2d_ldmks[start_frame:end_frame, valid_verts_idx] for pts2d_ldmks in lndmks2d_vis] if lndmks2d_vis is not None else None
        K = [K[start_frame:end_frame, :] for K in self.K]
        extrinsics = [extr[start_frame:end_frame, :] for extr in self.extrinsics]
        hand_joints2d = [hand_joints2d[start_frame:end_frame, :] for hand_joints2d in hand_joints2d] if hand_joints2d is not None else None
        triangulated_3d_points_ = triangulated_3d_points[start_frame:end_frame, :] if triangulated_3d_points is not None else None
        triangulated_contacts_ = triangulated_contacts[start_frame:end_frame, :] if triangulated_contacts is not None else None
        return pts2d, K, extrinsics, pts3d_pred, joints3d_pred, hand_joints2d, pts2d_vis_weight, triangulated_3d_points_, triangulated_contacts_


    def fit_smplx_to_2d_points(self, losses_names_iter:List[str], params: List[torch.Tensor],
                               num_iters:int = 100, optim_verts:bool = False,
                               tol:float = 1e-5, patience:int = 10,
                               reprojection_loss_name:str = "mse",
                               loss_cfg=None, use_sparse_body_ldmks=False,
                               scale_uncertainties:bool=False,
                               vis_clip_value:float=0.8,
                               occlusion_gating: Optional[dict] = None,
                               skip_first_t_frames: int = 0):

        optimizer = torch.optim.LBFGS(
                                    params,
                                    lr=0.3,                  # 0.2–0.5 is a sweet spot; 1.0 is often too big here
                                    max_iter=12,             # per .step() call
                                    max_eval=None,
                                    tolerance_grad=1e-7,
                                    tolerance_change=1e-9,
                                    history_size=10,         # shorter history helps with nonconvex/robust
                                    line_search_fn='strong_wolfe'
                                )
        prev_loss = float("inf")
        no_improvement = 0
        self.loss_terms = {}
        self.errors = {}
        self.scale_once = scale_uncertainties #True

        for iter in range(num_iters):
            def closure() -> torch.Tensor:
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                total_loss = torch.Tensor([0]).to(params[0].device)
                bodies_vertices = []
                bodies_pts2d = []
                bodies_visibility = []
                bodies_sampled_verts = []
                bodies_triangulated_3d_points = []
                bodies_contacts = []
                for body_id, poses in self.poses.items():
                    smplx_out = get_smplx_forward_per_parts(poses["global"], poses["body"], poses["left_hand"],
                                                            poses["right_hand"], poses["jaw"], poses["leye"], poses["reye"],
                                                            self.smplx_betas[body_id].repeat((poses["body"].shape[0], 1)),
                                                            self.smplx_trans[body_id],
                                                            gender="neutral",
                                                            smplx_models=self.smplx_model[body_id])

                    pts3d_pred = smplx_out.vertices if optim_verts else smplx_out.joints
                    joints3d_pred = smplx_out.joints
                    if optim_verts and self.downsampled_verts_mat is not None:
                        pts3d_pred = torch.einsum("ij,bjk->bik", self.downsampled_verts_mat, pts3d_pred)
                    verts3d_pred = smplx_out.vertices

                    if use_sparse_body_ldmks:
                        valid_verts_idx = self.body_idx
                    else:
                        valid_verts_idx = np.arange(pts3d_pred.shape[1])
                    stage_skip_first_t_frames = max(0, int(skip_first_t_frames))
                    pts2d, K, extrinsics, pts3d_pred, joints3d_pred, hand_joints2d, pts2d_vis_weight, triangulated_3d_points, triangulated_contacts_ = self.process_data(pts3d_pred, joints3d_pred,
                                                                                                                        self.pts2d[body_id],
                                                                                                                        self.pts2d_vis_weight[body_id],
                                                                                                                        self.hand_joints2d[body_id],
                                                                                                                        self.triangulated_3d_points[body_id] if self.triangulated_3d_points is not None else None,
                                                                                                                        self.triangulated_contact[body_id],
                                                                                                                        skip_first_t_frames=stage_skip_first_t_frames,
                                                                                                                        valid_verts_idx=valid_verts_idx)

                    # Slice vertices/sampled_verts to match the frame range from process_data
                    start_f = stage_skip_first_t_frames
                    bodies_vertices.append(verts3d_pred[start_f:])
                    bodies_sampled_verts.append(pts3d_pred)  # already sliced by process_data

                    bodies_pts2d.append(pts2d)
                    bodies_visibility.append(pts2d_vis_weight)
                    bodies_triangulated_3d_points.append(triangulated_3d_points)
                    bodies_contacts.append(triangulated_contacts_)

                    loss_terms = self.loss_functions(pts2d, K, extrinsics, pts3d_pred, joints3d_pred, torch.cat([poses[key] for key in poses.keys()], dim=-1),
                                                        self.smplx_betas[body_id], losses_names_iter,
                                                        reprojection_loss_name=reprojection_loss_name, hand_joints2d=hand_joints2d,
                                                        weights_cfg=loss_cfg,
                                                        pts2d_vis_weight=pts2d_vis_weight,
                                                        verts3d_pred=verts3d_pred,
                                                        poses=poses,
                                                        triangulated_3d_points=triangulated_3d_points,
                                                        vis_clip_value=vis_clip_value,
                                                        occlusion_gating=occlusion_gating)

                    loss = torch.sum(torch.stack(list(loss_terms.values())))
                    total_loss = total_loss + loss
                    self.loss_terms.update({f"{k}_{body_id:02}": v for k,v in loss_terms.items()})


                    with torch.no_grad():
                        old_pts2d = list(pts2d)
                        errors, pts2d = proj_pts_error(pts2d, pts3d_pred, K, extrinsics, 'mse', pts2d_vis_weight, scale_uncertainties=self.scale_once)
                        if self.scale_once: #num_iters > 500 and scale_uncertainties:
                            self.scale_once = True

                        self.errors.update({f"error: cam_{k}_body_{body_id:2d}": np.round(v.nanmean().item()) for k,v in errors.items()})

                multi_loss = self.loss_functions_multi_people(bodies_vertices, bodies_pts2d, K, extrinsics, bodies_visibility, bodies_sampled_verts,
                                                              bodies_contacts,
                                                              losses_names_iter, loss_cfg, ignore_first_t_frames=max(0, int(skip_first_t_frames)))
                if multi_loss is not None:
                    loss = torch.sum(torch.stack(list(multi_loss.values())))
                    total_loss = total_loss + loss
                    self.loss_terms.update(multi_loss)

                if total_loss.requires_grad:
                    total_loss.backward(retain_graph=True)

                # FIXME: should be `total_loss` (the backpropped scalar). Returning
                # `loss` makes LBFGS line search pick step sizes from a partial
                # objective — direction stays correct, step sizes mildly suboptimal.
                return loss

            optimizer.step(closure)

            show_log = True
            if show_log:
                with torch.no_grad():
                    log_string = f"Iter. {iter+1} of {num_iters} | Loss: {[f'{k}: {np.round(v.item(), 3)}' for k,v in self.loss_terms.items()]} | "
                    print(log_string)
                    print(f"Errors: {[f'{k}: {v}' for k,v in self.errors.items()]}")
            with torch.no_grad():
                current_loss = sum(self.loss_terms.values())
                if prev_loss - current_loss < tol:
                    no_improvement += 1
                else:
                    no_improvement = 0

                if no_improvement >= patience:
                    print(f"Stopping early at iteration {iter} due to lack of improvement.")
                    break

                prev_loss = current_loss


    def triangulate_3d_from_2d_points(self, skip_first_t_frames=5,
                                    use_uncertainties=False, img_size_px=512, channel_is_logvar=False):
        """
        Triangulate per-frame, per-point across all cameras.
        If use_uncertainties=True, rows are weighted by 1/sigma^2 (sigma in px).
        Assumes self.pts2d[cam]: [T, N, 2 or 3] in PIXELS (your codebase uses pixel space).
                self.pts2d_vis_weight[cam]: [T, N] in [0,1]
                self.K[cam]: [T, 3,3], self.extrinsics[cam]: [T, 4,4] (world->cam)
        Returns:
            Xw_all: [T_slice, N, 3] triangulated world points
            valid_mask: [T_slice, N] bool
        """
        device = self.K[0].device
        dtype  = self.K[0].dtype

        triangulated_points_world = []
        triangulated_valid_mask = []

        floor_contact_world = []
        contact_world = []

        triangulated_points_world_dict = {}
        triangulated_valid_mask_dict = {}
        floor_contact_world_dict = {}
        contact_world_dict = {}


        for body_id, poses in self.poses.items():
            lndmks2d = self.pts2d[body_id]            # list over cams: tensors [T,N,2 or 3]
            lndmks2d_vis = self.pts2d_vis_weight[body_id]  # list over cams: [T,N]
            ldmks2d_contact = self.contacts[body_id]
            ldmks2d_floor_contact = self.floor_contacts[body_id]
            valid_verts_idx = np.arange(lndmks2d[0].shape[1])

            start_frame, end_frame = self._resolve_frame_slice(
                skip_frames=skip_first_t_frames,
            )

            pts2d = [p[start_frame:end_frame, valid_verts_idx] for p in lndmks2d]
            vis = [v[start_frame:end_frame, valid_verts_idx] for v in lndmks2d_vis] if lndmks2d_vis is not None else None

            if ldmks2d_contact is not None:
                visible = torch.stack(vis, dim=0) > 0.5
                contact = [c[start_frame:end_frame, valid_verts_idx] for c in ldmks2d_contact]
                floor_contact = [fc[start_frame:end_frame, valid_verts_idx] for fc in ldmks2d_floor_contact]
                contact = torch.stack(contact, dim=0)
                contact[visible] = 0.0  # visible landmarks should have contact
                floor_contact = torch.stack(floor_contact, dim=0)

                floor_contact_prob = 1 - (1-floor_contact).prod(dim=0)
                floor_contact_prob = floor_contact.mean(dim=0)

                contact_prob = 1 - (1-contact).prod(dim=0)
                contact_prob = contact.mean(dim=0)
                contact_prob[contact_prob < 0.25] = 0.0
            else:
                contact = None
                floor_contact_prob = None
                contact_prob = None


            Ks = [K[start_frame:end_frame, :] for K in self.K]
            EX = [E[start_frame:end_frame, :] for E in self.extrinsics]


            # The --tri-irls uses triangulate_batch_irls.
            tri_fn = triang_funcs.triangulate_batch_irls if self.tri_irls else triang_funcs.triangulate_batch
            tri_kwargs = dict(vis_list=vis, use_uncertainties=use_uncertainties,
                              img_size_px=512, channel_is_logvar=False)
            if self.tri_irls:
                # Per-camera image sizes let IRLS rescale its residual scale r0
                # by each camera's diagonal (resolution-independent weighting).
                tri_kwargs["img_whs"] = [
                    (float(np.asarray(w).reshape(-1)[0]), float(np.asarray(h).reshape(-1)[0]))
                    for w, h in zip(self.cam_widths, self.cam_heights)]
            pts3d_world, valid_mask = tri_fn(pts2d, Ks, EX, **tri_kwargs)
            from utils.epipolar_association import mvpose_style_associate_and_triangulate, mvpose_style_associate_and_triangulate_temporal, animate_pointcloud_bodies_body_first

            triangulated_points_world.append(pts3d_world)
            triangulated_valid_mask.append(valid_mask)
            floor_contact_world.append(floor_contact_prob)
            contact_world.append(contact_prob)
            triangulated_points_world_dict[body_id] = pts3d_world
            triangulated_valid_mask_dict[body_id] = valid_mask
            floor_contact_world_dict[body_id] = floor_contact_prob
            contact_world_dict[body_id] = contact_prob

        # Legacy Plotly contact HTMLs — superseded by the Rerun (.rrd) view below
        # (faster, smaller, clearer). Kept commented out for easy fallback; the
        # `animate_pointcloud_bodies_body_first` helper still exists in
        # utils/epipolar_association.py if these need to be re-enabled.
        # animate_pointcloud_bodies_body_first(triangulated_points_world, floor_contact_world, out_html=os.path.join(self.save_prediction_fn, "_floor_contact.html"))
        # animate_pointcloud_bodies_body_first(triangulated_points_world, contact_world, out_html=os.path.join(self.save_prediction_fn, "_contact.html"))
        # Rerun (.rrd) view of the triangulated contact points. Best-effort.
        try:
            from utils.contact_rrd import write_contact_rrd
            write_contact_rrd(
                os.path.join(self.save_prediction_fn, "intermediate_triangulated_points.rrd"),
                triangulated_points_world, contact_world, floor_contact_world,
                valid_mask=triangulated_valid_mask,
                up_axis=self.up_axis,
            )
        except Exception as e:
            print(f"[contact_rrd] skipped: {e}")
        return triangulated_points_world_dict, triangulated_valid_mask_dict, floor_contact_world_dict, contact_world_dict


    def get_params_from_cfg(self, run_cfg, use_v_template=False):
        params = []
        for body_id, poses in self.poses.items():
            if run_cfg["optim_var"]["pose"]: #"pose" in run_cfg["optim_var"]:
                if "optimize_only_global_rotation" in run_cfg and run_cfg["optimize_only_global_rotation"]:
                    print("optimizing poses: ", "[global]")
                    params.append(poses["global"])
                elif run_cfg["optimize_only_limbs"]:
                    print("optimizing poses: ", "[global, body (limbs only)]")
                    params.append(poses["body"])
                    self.register_limbs_grad_masks()

                elif run_cfg["optimize_only_body"]:
                    print("optimizing poses: ", "[global, body]")
                    params.append(poses["global"])
                    params.append(poses["body"])
                    if run_cfg["optimize_hand_joints"]:
                        params.append(poses["left_hand"])
                        params.append(poses["right_hand"])
                elif run_cfg["optimize_only_hand_joints"]:
                    print("optimizing poses: ", "[left_hand, right_hand]")
                    params.append(poses["left_hand"])
                    params.append(poses["right_hand"])
                else:
                    print("optimizing poses: ", poses.keys())
                    for key in poses.keys():
                        if key == "jaw":
                            continue
                        print(f"optimizing {key}")
                        params.append(poses[key])

            if run_cfg["optim_var"]["betas"]: #"betas" in run_cfg["optim_var"]:
                if use_v_template:
                    print("betas are not optimized because v_template is used")
                else:
                    params.append(self.smplx_betas[body_id])
            if run_cfg["optim_var"]["trans"]: #"trans" in run_cfg["optim_var"]:
                params.append(self.smplx_trans[body_id])
        return params


    def _apply_reid_corrections(self, groups_per_t):
        """Re-assign pts2d/visibility/contacts according to the cross-view
        re-ID groups, then clear orphaned donor slots and swap the corrected
        dicts in. group['body_ids'] maps cam_id -> the original body_id
        detected as this physical person in that camera.

        Returns (corrections, canonical_ids, quarantined, pts2d_before,
        pts2d_vis_before). The *_before dicts are the pre-correction
        assignments, kept for the before/after visualisation; quarantined
        holds bodies whose detections the association double-booked (to be
        excluded from the fit by _exclude_unfittable_bodies).
        """
        pts2d_before     = self.pts2d
        pts2d_vis_before = self.pts2d_vis_weight

        new_pts            = copy.deepcopy(self.pts2d)
        new_pts_vis        = copy.deepcopy(self.pts2d_vis_weight)
        new_contacts       = copy.deepcopy(self.contacts)
        new_floor_contacts = copy.deepcopy(self.floor_contacts)
        corrections        = []   # each swap: {cam_id, group_idx, from_body_id, to_body_id}

        assigned = set()        # (body_id, cam_id) slots that receive data in the re-assignment
        canonical_ids = set()   # bodies adopted as a physical person by re-ID
        claimed_sources = {}    # (orig_body_id, cam_id) slot -> canonical_id of the group that consumed it
        quarantined = set()     # bodies whose detections the association double-booked
        for group_idx, group in enumerate(groups_per_t):
            # Canonical person_id: majority vote across cameras, tie-break by min id.
            counts       = Counter(group['body_ids'].values())
            canonical_id = min(counts, key=lambda bid: (-counts[bid], bid))
            # A duplicate canonical election alone is only a warning: a person's
            # group can legitimately split across disjoint camera sets, and the
            # per-slot tripwires below still guarantee no detection set is
            # double-booked. But it can also mean two people share one label
            # upstream, so say it loudly.
            if canonical_id in canonical_ids:
                print(f"ma_3d re-ID: WARNING: two cross-view groups elected the same canonical "
                      f"body {canonical_id} — benign if one person's group split across "
                      f"disjoint cameras, suspicious if two people share this label upstream. "
                      f"Groups: {[g['body_ids'] for g in groups_per_t]}")
            canonical_ids.add(canonical_id)
            print(f"Re-ID group {group_idx}: canonical_id={canonical_id}, "
                  f"body_ids per cam={group['body_ids']}")
            for cam_id, orig_body_id in group['body_ids'].items():
                # Tripwires: each detection slot may be consumed by at most one
                # group, and each receiving slot written by at most one group.
                # A violation means the association handed one detection set to
                # two people, or feeds one identity from two conflicting
                # sources — no bookkeeping can repair that, so quarantine the
                # bodies involved (they are excluded from the fit) and keep
                # everyone else. Partial writes into a quarantined body's
                # slots are moot: the body is popped before the fit. Never
                # observed on real data; guarded so it can only ever cost the
                # bodies involved, not the whole scene.
                if (orig_body_id, cam_id) in claimed_sources:
                    culprits = {orig_body_id, canonical_id,
                                claimed_sources[(orig_body_id, cam_id)]}
                    quarantined |= culprits
                    print(f"ma_3d re-ID: WARNING: detection slot (body {orig_body_id}, "
                          f"camera {cam_id}) is claimed by two cross-view groups — the "
                          f"association handed one detection set to two people. Quarantining "
                          f"bodies {sorted(culprits)}; the remaining people are unaffected. "
                          f"Groups: {[g['body_ids'] for g in groups_per_t]}")
                    continue
                claimed_sources[(orig_body_id, cam_id)] = canonical_id
                if (canonical_id, cam_id) in assigned:
                    culprits = {orig_body_id, canonical_id}
                    quarantined |= culprits
                    print(f"ma_3d re-ID: WARNING: receiving slot (body {canonical_id}, "
                          f"camera {cam_id}) is written by two cross-view groups — the "
                          f"association feeds one identity from two conflicting sources. "
                          f"Quarantining bodies {sorted(culprits)}; the remaining people are "
                          f"unaffected. Groups: {[g['body_ids'] for g in groups_per_t]}")
                    continue
                if orig_body_id != canonical_id:
                    corrections.append({
                        "cam_id":       cam_id,
                        "group_idx":    group_idx,
                        "from_body_id": orig_body_id,
                        "to_body_id":   canonical_id,
                    })
                assigned.add((canonical_id, cam_id))
                new_pts[canonical_id][cam_id]     = self.pts2d[orig_body_id][cam_id]
                new_pts_vis[canonical_id][cam_id] = self.pts2d_vis_weight[orig_body_id][cam_id]
                if self.contacts[orig_body_id] is not None:
                    new_contacts[canonical_id][cam_id]       = self.contacts[orig_body_id][cam_id]
                    new_floor_contacts[canonical_id][cam_id] = self.floor_contacts[orig_body_id][cam_id]

        # If a body's detections were given to another body and it received
        # none in return, blank its old slot (set its 2D points and visibility
        # weights to zero, so this camera contributes no evidence for it):
        # after a complete swap or rotation every slot has a new owner and
        # nothing is cleared, but a donor outside every re-ID group (e.g. an
        # extra body) would otherwise keep a copy of the detections it gave
        # away, and the same pixels constraining two bodies destabilises the
        # joint optimization.
        for c in corrections:
            b, cam = c["from_body_id"], c["cam_id"]
            if (b, cam) in assigned:
                continue
            new_pts[b][cam]     = torch.zeros_like(new_pts[b][cam])
            new_pts_vis[b][cam] = torch.zeros_like(new_pts_vis[b][cam])
            if new_contacts[b] is not None:
                new_contacts[b][cam]       = torch.zeros_like(new_contacts[b][cam])
                new_floor_contacts[b][cam] = torch.zeros_like(new_floor_contacts[b][cam])
            print(f"ma_3d re-ID: camera {cam}: body {b}'s detections here were re-assigned "
                  f"to body {c['to_body_id']} and nothing was re-assigned back — clearing "
                  f"body {b}'s stale duplicate in this camera (a detection belongs to at "
                  f"most one body).")

        print(f"Re-ID: {len(corrections)} label-swap corrections across "
              f"{len({c['cam_id'] for c in corrections})} camera(s).")

        self.pts2d            = new_pts
        self.pts2d_vis_weight = new_pts_vis
        self.contacts         = new_contacts
        self.floor_contacts   = new_floor_contacts
        return corrections, canonical_ids, quarantined, pts2d_before, pts2d_vis_before

    def _exclude_unfittable_bodies(self, canonical_ids, quarantined):
        """Remove bodies the fit cannot constrain from every per-body dict.

        A body with < 2 cameras of evidence (e.g. an extra body from an
        over-counted init) cannot be triangulated; fitting it produces a
        depth-unconstrained ghost whose chaotic parameters destabilise the
        OTHER bodies through the joint optimization. Exclusion requires two
        independent signals: (a) cross-view re-ID never adopted the body and
        (b) fewer than 2 cameras reach gate-level support. Quarantined bodies
        (double-booked by the association) are excluded unconditionally —
        their data is ambiguous, and a body constrained by contradictory
        detections corrupts every other body through the shared solve.
        """
        canonical_ids = canonical_ids - quarantined   # quarantine overrides re-ID immunity
        removable = []
        for body_id in self.pts2d.keys():
            if body_id in quarantined:
                removable.append((body_id, "its detections were double-booked by the "
                                           "cross-view association (quarantined)"))
                continue
            if body_id in canonical_ids:
                continue
            supports = self._view_support_fractions(body_id) or []
            n_supported = sum(1 for s in supports if s >= self._VIEW_SUPPORT_MIN_FRAME_FRAC)
            if n_supported < 2:
                removable.append((body_id, f"cross-view re-ID never adopted it as a person "
                                           f"and only {n_supported} camera(s) reach gate-level "
                                           f"support for it, so it cannot be meaningfully "
                                           f"triangulated"))
        if removable and len(removable) == len(self.poses):
            if quarantined:
                raise RuntimeError(
                    "ma_3d: every body is either quarantined by a re-ID conflict or has "
                    "fewer than 2 cameras of evidence — nothing is left to fit. Inspect the "
                    "cross-view association (groups logged above) and this sequence's 2D inputs.")
            print("ma_3d: every body has fewer than 2 cameras of evidence — keeping them all "
                  "(refusing to empty the scene); check this sequence's 2D inputs.")
            return
        for body_id, reason in removable:
            for attr in ("pts2d", "pts2d_vis_weight", "contacts", "floor_contacts",
                         "hand_joints2d", "poses", "smplx_betas", "smplx_trans"):
                getattr(self, attr).pop(body_id, None)
            for attr in ("triangulated_3d_points", "triangulated_contact",
                         "triangulated_floor_contact"):
                d = getattr(self, attr)
                if isinstance(d, dict):
                    d.pop(body_id, None)
            print(f"ma_3d: excluding body {body_id} from the fit — {reason}; fitting it "
                  f"would produce a wrongly-constrained body that also degrades the other "
                  f"bodies through the joint optimization.")

    def _gate_triangulation_outliers(self, valid_mask_dict, cfg):
        """Bridge transient triangulation "teleports" on the 3D seed (opt-in via
        --tri-outlier-gate). Bad masks / re-ID can make a majority of cameras
        agree on a wrong 3D point for one or two isolated frames; the fit then
        locks onto it and the body jumps metres away and back.

        For each detected transient excursion, for that body, we:
          (1) interpolate the triangulated points across the gap, so first_run's
              l2_loss_3d_points pin lands at the correct location; and
          (2) zero the 2D targets there for every camera, so no stage's
              reprojection can pull the body back to the wrong place. We zero the
              points themselves (not just the visibility weight, which is clipped
              up to vis_clip_value and so cannot suppress a view) exactly as the
              view-support gate does; proj_pts_loss masks a landmark by
              pts2d.sum(-1) > 0.

        The frame is then treated as "this body is unobserved here" and bridged by
        interpolation plus pose-continuity priors. Abstains on trajectories that
        are too broken to interpolate. Bit-identical to before when the flag is off
        (this method is never called)."""
        k           = float(cfg.get("k", 8.0))
        abs_floor   = float(cfg.get("abs_floor", 0.30))
        half_window = int(cfg.get("half_window", 7))
        max_gap     = int(cfg.get("max_gap", 4))
        max_frac    = float(cfg.get("max_flagged_frac", 0.15))

        for body_id, X in self.triangulated_3d_points.items():
            vm = valid_mask_dict[body_id]
            runs, flagged, info = triang_funcs.detect_translation_outlier_runs(
                X, vm, k=k, abs_floor=abs_floor, half_window=half_window, max_gap=max_gap)
            T = X.shape[0]
            if info["n_flagged"] > max_frac * T:
                print(f"[tri-gate] body {body_id}: {info['n_flagged']}/{T} frames flagged "
                      f"(> {max_frac:.0%}, thr={info['thr']:.2f} m); trajectory too broken, "
                      f"abstaining (no repair).")
                continue
            if not runs:
                if info["n_flagged"]:
                    print(f"[tri-gate] body {body_id}: {info['n_flagged']} flagged frame(s) "
                          f"but none interpolatable (clip edge or run > {max_gap}); left as-is.")
                continue
            with torch.no_grad():
                triang_funcs.interpolate_points_over_runs(X, runs)
                n_cams = len(self.pts2d[body_id])
                for a, b in runs:
                    for cam in range(n_cams):
                        self.pts2d[body_id][cam][a:b + 1] = 0.0
                        if self.pts2d_vis_weight[body_id] is not None:
                            self.pts2d_vis_weight[body_id][cam][a:b + 1] = 0.0
                        if self.contacts[body_id] is not None:
                            self.contacts[body_id][cam][a:b + 1] = 0.0
                        if self.floor_contacts[body_id] is not None:
                            self.floor_contacts[body_id][cam][a:b + 1] = 0.0
            frames = ", ".join(f"{a}..{b}" if b > a else f"{a}" for a, b in runs)
            n_frames = sum(b - a + 1 for a, b in runs)
            print(f"[tri-gate] body {body_id}: bridged {n_frames} frame(s) [{frames}] "
                  f"(thr={info['thr']:.2f} m, mad={info['mad']*1000:.0f} mm).")

        # Second .rrd from the REPAIRED seed, alongside the pre-gate
        # intermediate_triangulated_points.rrd (written inside triangulation), so
        # the raw and gated seed can be compared before/after in Rerun. Best-effort.
        try:
            from utils.contact_rrd import write_contact_rrd
            order = list(self.triangulated_3d_points.keys())
            write_contact_rrd(
                os.path.join(self.save_prediction_fn, "intermediate_triangulated_points_gated.rrd"),
                [self.triangulated_3d_points[b] for b in order],
                [self.triangulated_contact[b] for b in order],
                [self.triangulated_floor_contact[b] for b in order],
                valid_mask=[valid_mask_dict[b] for b in order],
                up_axis=self.up_axis,
            )
        except Exception as e:
            print(f"[tri-gate] gated .rrd skipped: {e}")

    def fit(self, optim_verts:bool=True, optim_cfg=None):
        time_start = time.time()
        use_v_template = optim_cfg["use_v_template"]

        # ── One-time pre-processing: cross-view re-ID and triangulation ───────
        # Intentionally placed BEFORE the per-stage loop so that:
        #   (A) the redundant first triangulate_3d_from_2d_points that previously
        #       preceded the re-ID inside the loop is eliminated, and
        #   (B) re-ID runs exactly once — running it on every optimization stage
        #       would be wasteful (it is idempotent once pts2d is corrected).
        from utils.epipolar_association import mvpose_style_associate_and_triangulate_temporal
        from utils.reid_viz import visualize_reid, plot_reid_stats

        # No triangulation happens before re-ID: the association below works on
        # 2D trajectories only, and the one triangulation the fit consumes runs
        # AFTER the corrections.

        # Multi-view temporal association to resolve label swaps across cameras.
        print("Running multi-view re-ID association...")
        _reid_t0 = time.time()
        groups_per_t, triX_per_t, valid_per_t = mvpose_style_associate_and_triangulate_temporal(
            self.poses, self.pts2d, self.pts2d_vis_weight,
            self.K, self.extrinsics,
            triang_funcs.triangulate_batch,
            skip_first_t_frames=0,
            img_size_px=512,
            channel_is_logvar=False,
            geom_scale=50.0,
            dg_thresh=80.0,
            min_visible_L=50,
            min_geom_joints=5,
            affinity_match_thresh=0.4,
            min_views_for_group=2,
        )
        print(f"Cross-view re-ID took {time.time() - _reid_t0:.2f} seconds.")

        # Re-assign pts2d (+vis/contacts) according to the re-ID groups; the
        # *_before snapshots feed the before/after visualisation below.
        (corrections, canonical_ids, quarantined,
         pts2d_before_viz, pts2d_vis_before_viz) = self._apply_reid_corrections(groups_per_t)

        # Triangulate with the corrected 2D assignments (the fit's only
        # full triangulation pass).
        triangulated_points_world_dict, triangulated_valid_mask_dict, floor_contact_world_dict, contact_world_dict = \
            self.triangulate_3d_from_2d_points(skip_first_t_frames=0)
        self.triangulated_3d_points     = triangulated_points_world_dict
        self.triangulated_contact       = contact_world_dict
        self.triangulated_floor_contact = floor_contact_world_dict

        # Opt-in (--tri-outlier-gate): bridge transient triangulation "teleports"
        # (a body jumping metres away for a frame or two on bad masks / re-ID)
        # before the fit locks onto them. No-op when the flag is off.
        if self.tri_outlier_gate:
            self._gate_triangulation_outliers(triangulated_valid_mask_dict, self.tri_outlier_gate_cfg)

        # Debug visualisations: before/after image overlays + per-camera stats.
        if self.cam_metadata_fns is not None:
            try:
                visualize_reid(
                    pts2d_before_viz,  pts2d_vis_before_viz,
                    self.pts2d,        self.pts2d_vis_weight,
                    corrections, self.cam_metadata_fns, self.imgs_pth,
                    self.save_prediction_fn,
                )
                plot_reid_stats(
                    corrections,
                    n_cams=len(self.cam_metadata_fns),
                    n_bodies=len(self.poses),
                    save_dir=self.save_prediction_fn,
                )
            except Exception as e:
                print(f"[reid_viz] Visualisation failed (non-fatal): {e}")

        # Drop bodies the fit cannot constrain (quarantined or < 2 cameras).
        self._exclude_unfittable_bodies(canonical_ids, quarantined)

        # ── Per-stage optimization loop ───────────────────────────────────────

        for run_name in optim_cfg["optim"].keys():
            run_cfg = optim_cfg["optim"][run_name]
            losses_names = run_cfg["losses"].keys()
            print("RUN: ", run_name)
            print(f"optimizing variables: ")
            print(f"pose: {run_cfg['optim_var']['pose']}")
            print(f"betas: {run_cfg['optim_var']['betas']}")
            print(f"trans: {run_cfg['optim_var']['trans']}")
            print(f"Optimizing with losses: {losses_names}")

            params = self.get_params_from_cfg(run_cfg, use_v_template)
            if run_cfg.get("only_ten_first_frames", False):
                print("Ignoring legacy only_ten_first_frames=True; full-sequence mode is enforced.")

            if params:
                self.fit_smplx_to_2d_points(losses_names, params, run_cfg["num_iters"],
                                            optim_verts, reprojection_loss_name=run_cfg["reproj_loss_name"],
                                            loss_cfg=run_cfg["losses"],
                                            use_sparse_body_ldmks=run_cfg["use_sparse_body_ldmks"],
                                            scale_uncertainties=run_cfg["scale_uncertainties"],
                                            vis_clip_value=run_cfg["vis_clip_value"],
                                            occlusion_gating=run_cfg.get("occlusion_gating", None),
                                            skip_first_t_frames=self.skip_start,
                                            )
            else:
                print("No parameters to optimize")

            smplx_pose = {}
            for key, poses in self.poses.items():
                smplx_pose[key] = torch.cat([poses[key] for key in poses.keys()], dim=-1)
        print(f"Optimization took {time.time()-time_start:.2f} seconds.")
        return smplx_pose, self.smplx_betas, self.smplx_trans, self.triangulated_3d_points, self.triangulated_contact, self.triangulated_floor_contact
