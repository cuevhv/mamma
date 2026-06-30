import torch
import numpy as np
from losses.rotation import aa_to_rotmat, aa_to_quaternion
from typing import List


def mse_loss(pts3d_pred, pts3d_gt, weight = 1., uncertainties=1.):
    return torch.mean(weight*((pts3d_pred - pts3d_gt)/uncertainties)**2)


def tukey_loss(y_true, y_pred, delta=40.0, weight=1., uncertainties=1.):
    residual = torch.abs(y_true - y_pred)/uncertainties
    tukey_loss = weight*torch.where(residual <= delta,
                             (1 - (1 - (residual / delta)**2)**3) * (delta ** 2) / 6,
                             (delta ** 2) / 6)
    return tukey_loss.mean()  # Return the mean loss across all samples


def huber_loss(y_true, y_pred, delta=3000.0, weight=1., uncertainties=1.):
    residual = torch.abs(y_true - y_pred)/uncertainties
    huber_loss = weight*torch.where(residual <= delta,
                             0.5 * residual ** 2,
                             delta * (residual - 0.5 * delta))
    return huber_loss.mean()  # Return the mean loss across all samples


def geman_mcclure_loss(y_true, y_pred, delta=500.0, weight=1., uncertainties=1.):
    residual = (y_true - y_pred)
    x_squared = (residual/uncertainties) ** 2
    sigma_squared = delta ** 2
    denominator = sigma_squared + x_squared
    numerator = x_squared
    loss = weight * numerator / denominator
    return loss.mean()


def world_to_camera(world_pts, extrinsics):
    """
    Convert world coordinates to camera coordinates using the extrinsics matrix.
    Args:
        world_pts: Tensor of shape [B, N, 3] representing the world coordinates.
        extrinsics: Tensor of shape [B, 4, 4] representing the extrinsics matrix.
    Returns:
        pts_cam: Tensor of shape [B, 3, N] representing the camera coordinates.
    """

    pts_cam = world_pts.transpose(-2,-1)  # [3, N]
    pts_cam = extrinsics[:, None, :3, :3] @ pts_cam  + extrinsics[:, None, :3, 3].unsqueeze(-1)  # [B, 3, N]
    return pts_cam  # [B, 3, N]


def visibility_loss(pts2ds: List[torch.Tensor], verts3d_preds_list: List[torch.Tensor], faces,
                    intrinsics, extrinsics, height, width, per_point_weight: List[torch.Tensor],
                    vertex_subsampling_map,
                    weight=1.):

    verts3d_preds = torch.cat([verts3d[:, None] for verts3d in verts3d_preds_list], dim=1)
    device = verts3d_preds.device
    loss = 0
    for cam_id in range(len(pts2ds[-1])):
        visibility = torch.stack([per_point_weight[people_id][cam_id] for people_id in range(len(pts2ds))], dim=1)
        pts_cam = world_to_camera(verts3d_preds, extrinsics[cam_id]).transpose(-1,-2)
        pts_cam_z = pts_cam[..., 2]
        visibility_mask = visibility > 0.7
        loss_cam = (pts_cam_z * visibility_mask).pow(2).sum() / (visibility_mask.sum() + 1e-6)
        loss = loss + loss_cam/len(pts2ds[-1])

    return weight*loss


def proj_pts_loss(pts2d, pts3d_pred, intrinsics, extrinsics, loss='mse',
                  per_point_weight=None, vis_clip_value=0.8, occlusion_gating=None, weight=1.):

    res = 0
    _occ_weights = None
    # [optional] With occlusion_gating on, landmarks are falling toward zero for occluded views.
    if occlusion_gating is not None and occlusion_gating.get("enabled", False) and per_point_weight is not None:
        from utils.occlusion_gating import compute_occlusion_gates
        _occ_weights = compute_occlusion_gates(per_point_weight, occlusion_gating)

    for cam_id in range(len(pts2d)):
        r, t = extrinsics[cam_id][:, :3, :3], extrinsics[cam_id][:, :3, 3].unsqueeze(1)
        pts3d_o2w = torch.matmul(pts3d_pred, r.permute((0,2,1))) + t
        pts2d_proj = torch.matmul(pts3d_o2w, intrinsics[cam_id].permute((0,2,1)))
        pts2d_proj = pts2d_proj / pts2d_proj[:, :, 2][:,:,None]
        pred_in_img = pts2d[cam_id][:,:,:2].sum(-1, keepdims=True) > 0
        if pts2d[cam_id].shape[-1] == 3:
            img_size = 512
            uncertainties = (pts2d[cam_id][...,2:])/ 2 * img_size  # scaled because sigma was [-1, 1]
            if uncertainties.min().item() > 15:
                uncertainties = uncertainties / 2.
            uncertainties = torch.clamp(uncertainties, min=1., max=50.)
        else:
            uncertainties = 1
        pred_in_img = weight * pred_in_img

        if per_point_weight is not None:
            default_w = torch.clip(per_point_weight[cam_id][..., None], min=vis_clip_value)
            if _occ_weights is not None:
                pred_in_img = _occ_weights[cam_id][..., None] * pred_in_img
            else:
                pred_in_img = default_w * pred_in_img

        if loss == "mse":
            error = mse_loss(pts2d_proj[:,:,:2], pts2d[cam_id][...,:2], weight=pred_in_img, uncertainties=2*uncertainties)
        elif loss == "tukey":
            error = tukey_loss(pts2d_proj[:,:,:2], pts2d[cam_id][...,:2], weight=pred_in_img, uncertainties=uncertainties)
        elif loss == "huber":
            error = huber_loss(pts2d_proj[:,:,:2], pts2d[cam_id][...,:2], weight=pred_in_img, uncertainties=uncertainties, delta=6.)
        elif loss == "geman_mcclure":
            # we do a dynamic covariance scaling https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-08-robust-slam.pdf
            # basically we scale the uncertainty by a scaling parameters
            error = geman_mcclure_loss(pts2d_proj[:,:,:2], pts2d[cam_id][...,:2], weight=pred_in_img, uncertainties=uncertainties, delta=8.)
        else:
            raise ValueError(f"Unknown loss type: {loss}")
        if torch.isnan(error).any() or torch.isinf(error).any():
            error = torch.tensor(0).to(error.device)

        res = res + error/len(pts2d)

    return res


def proj_pts_error(pts2d, pts3d_pred, intrinsics, extrinsics, loss='mse',
                   per_point_weight=None, scale_uncertainties=True, weight=1.):
    errors = {}
    for cam_id in range(len(pts2d)):
        r, t = extrinsics[cam_id][:, :3, :3], extrinsics[cam_id][:, :3, 3].unsqueeze(1)
        pts3d_o2w = torch.matmul(pts3d_pred, r.permute((0,2,1))) + t

        pts2d_proj = torch.matmul(pts3d_o2w, intrinsics[cam_id].permute((0,2,1)))
        pts2d_proj = pts2d_proj / pts2d_proj[:, :, 2][:,:,None]
        pred_in_img = pts2d[cam_id][:,:,:2].sum(-1, keepdims=True) > 0

        mask = pred_in_img.squeeze(-1)
        error = torch.linalg.norm(pts2d_proj[:,:,:2] - pts2d[cam_id][...,:2], axis=-1)
        if scale_uncertainties and pts2d[cam_id].shape[-1] == 3:
            uncertainties = pts2d[cam_id][...,2:] / 2 * 512  # scaled because sigma was [-1, 1]
            uncertainties = uncertainties.squeeze(-1)
            # scale uncertainty based on error
            t = 10.0
            error_as_uncertainty = error/t
            scale = torch.clamp(error_as_uncertainty, min=0.1, max=1.)
            pts2d[cam_id][...,2] = pts2d[cam_id][...,2]*scale
            # going back to original scale
            pts2d[cam_id][...,2] = pts2d[cam_id][...,2] * 2 / 512
        else:
            uncertainties = 1
        error = (error*mask).sum(dim=-1)/mask.sum(dim=-1)
        errors[cam_id] = error
    return errors, pts2d


def temp_pose_loss(pose, weights=1., distance=1, rot_type="rotmat"):
    if rot_type == "rotmat":
        b, j = pose.shape
        pose_rotmat = aa_to_rotmat(pose.reshape(-1, 3))
        pose_rotmat = pose_rotmat.reshape(b, -1, 3, 3)
        p_prev = pose_rotmat[:-distance, :]
        p_next = pose_rotmat[distance:, :]
        p_next = torch.transpose(p_next, -1, -2)
        diff = torch.matmul(p_prev, p_next).reshape(-1, 3, 3)
        I = torch.eye(3).to(diff.device)
        dev_from_identity = torch.linalg.matrix_norm(diff - I, ord=2)
        return 5*weights*torch.mean(dev_from_identity)
    elif rot_type == "aa":
        p_prev = pose[:-distance, :]
        p_next = pose[distance:, :]
        return 5*weights*torch.mean((p_prev-p_next)**2)


def trans_temp_loss(trans, weights=1.):
    offset = 1
    t_prev = trans[:-offset, :]
    t_next = trans[offset:, :]
    displacement = t_prev - t_next
    tau = 0.02**2  # 0.02 m
    squared_displacement = torch.sqrt(displacement.pow(2).sum(dim=-1) + 1e-6)
    masks = squared_displacement > tau
    return weights*squared_displacement.mean() + 5*weights*(squared_displacement[masks]).mean()


def rotation_prior_loss(joint_angles, weight=1):
    return weight * (joint_angles.pow(2).mean())


def body_shape_prior_loss(body_shape, weight=1):
    return weight*torch.mean(body_shape**2)


def rot_consistency_loss(joint_angles, weight=1.):
    b, j = joint_angles.shape
    pose_quat = aa_to_quaternion(joint_angles.reshape(-1, 3))
    pose_quat = pose_quat.reshape(b, -1, 4)
    pose_prev = pose_quat[:-1, :]
    pose_next = pose_quat[1:, :]
    quat_dot_product = (pose_prev*pose_next).sum(-1)
    return weight*torch.clamp(-quat_dot_product, min=0).abs().sum()


def acceleration_loss(joint_positions, weight=1., dt=1/20.0):
    v = (joint_positions[1:] - joint_positions[:-1]) / dt
    a = (v[1:] - v[:-1]) / dt
    return weight * (a.pow(2).mean())

class SMPLifyAnglePrior(torch.nn.Module):
    def __init__(self, dtype=torch.float32, device='cuda'):
        super(SMPLifyAnglePrior, self).__init__()

        # Indices for the roration angle of
        # 55: left elbow,  90deg bend at -np.pi/2
        # 58: right elbow, 90deg bend at np.pi/2
        # 12: left knee,   90deg bend at np.pi/2
        # 15: right knee,  90deg bend at np.pi/2
        angle_prior_idxs = np.array([55, 58, 12, 15], dtype=np.int64)
        angle_prior_idxs = torch.tensor(angle_prior_idxs, dtype=torch.long)
        self.register_buffer('angle_prior_idxs', angle_prior_idxs)

        angle_prior_signs = np.array([1, -1, -1, -1],
                                     dtype=np.float32 if dtype == torch.float32
                                     else np.float64)
        angle_prior_signs = torch.tensor(angle_prior_signs,
                                         dtype=dtype).to(device)
        self.register_buffer('angle_prior_signs', angle_prior_signs)

    def forward(self, pose, with_global_pose=False, weight=1.):
        ''' Returns the angle prior loss for the given pose

        Args:
            pose: (Bx[23 + 1] * 3) torch tensor with the axis-angle
            representation of the rotations of the joints of the SMPL model.
        Kwargs:
            with_global_pose: Whether the pose vector also contains the global
            orientation of the SMPL model. If not then the indices must be
            corrected.
        Returns:
            A sze (B) tensor containing the angle prior loss for each element
            in the batch.
        '''
        angle_prior_idxs = self.angle_prior_idxs - (not with_global_pose) * 3
        return weight*torch.mean(torch.exp(pose[:, angle_prior_idxs] *
                         self.angle_prior_signs).pow(2))


def so3_log(R):
    # R: [..., 3, 3]
    # numerically-stable log map → axis-angle vector
    trace = (R[..., 0,0] + R[...,1,1] + R[...,2,2]).clamp(-1+1e-7, 3-1e-7)
    theta = torch.acos(((trace - 1.0) * 0.5).clamp(-1+1e-7, 1-1e-7))  # [...,]
    # handle small angles with series
    sin_theta = torch.sin(theta)
    k = torch.where(sin_theta.abs() > 1e-6,
                    0.5 * theta / (sin_theta + 1e-8),
                    0.5 + (theta**2)/12.0)  # approx for small angles
    W = R - R.transpose(-1, -2)
    v = k.unsqueeze(-1) * torch.stack([W[...,2,1], W[...,0,2], W[...,1,0]], dim=-1)  # [...,3]
    return v  # axis-angle vector (so(3))


def angular_acceleration_loss(poses, weight=1., dt=1/20.0):
    B, _ = poses.shape
    poses = poses.reshape(B, -1, 3)
    B, J, _ = poses.shape
    poses = poses.reshape(-1, 3)
    rot_mat = aa_to_rotmat(poses)
    rot_mat = rot_mat.reshape(B, J, 3, 3)
    rots_relative = torch.einsum('abji, abjk -> abik', rot_mat[:-1], rot_mat[1:])

    v = so3_log(rots_relative)
    a = (v[1:] - v[:-1]) / (dt*dt)
    return weight * (a.pow(2).mean())