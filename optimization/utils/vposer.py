"""
VPoser manifold-distance pose prior for the SMPL-X body fit.

Wraps the pretrained VPoser VAE (loaded lazily and cached) and exposes
``vposer_recon_loss`` — a regulariser that penalizes how far a body pose is from
what VPoser can reconstruct (i.e. off the learned pose manifold), used to keep
data-unconstrained joints (e.g. occluded legs) on plausible configurations.

NOTE: this prior is a post-CVPR (v1.1.0) addition and is NOT used in the CVPR
submission — that pipeline runs without ``vposer_recon_loss``. It is opt-in,
enabled only by adding ``vposer_recon_loss`` to a config's stage losses.

Default weights dir is the standard V02_05 release under data/body_models/vposer
(override with $MAMMA_VPOSER_DIR).
"""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_VPOSER_DIR = os.environ.get(
    "MAMMA_VPOSER_DIR", os.path.join(_REPO_ROOT, "data", "body_models", "vposer", "V02_05"))
_VPOSER_CACHE = {}


_INSTALL_HINT = (
    "The VPoser pose prior needs the optional 'human_body_prior' package. "
    "Install it with:\n"
    "    pip install -r requirements/requirements-vposer.txt\n"
    "and fetch the V02_05 weights with: bash data/download_vposer.sh"
)


def check_available(expr_dir=None):
    """Preflight for the VPoser prior: importable package + weights on disk.

    Call this BEFORE the fit starts so a missing optional dependency fails in
    seconds with an actionable message, not minutes in (after re-ID and
    triangulation) with a bare ModuleNotFoundError.
    """
    expr_dir = expr_dir or _DEFAULT_VPOSER_DIR
    try:
        import human_body_prior  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    if not os.path.isdir(expr_dir):
        raise RuntimeError(
            f"VPoser weights not found at '{expr_dir}'. "
            "Fetch them with: bash data/download_vposer.sh "
            "(or set $MAMMA_VPOSER_DIR).")


def _get_vposer(expr_dir, device):
    key = (expr_dir, str(device))
    vp = _VPOSER_CACHE.get(key)
    if vp is None:
        try:
            from human_body_prior.tools.model_loader import load_model
            from human_body_prior.models.vposer_model import VPoser
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc
        vp, _ = load_model(expr_dir, model_code=VPoser,
                           remove_words_in_model_weights='vp_model.',
                           disable_grad=True,
                           comp_device='gpu' if device.type == 'cuda' else 'cpu')
        vp = vp.to(device).eval()           # eval() => BatchNorm uses running stats
        _VPOSER_CACHE[key] = vp
    return vp


def vposer_recon_loss(pose, weight=1., expr_dir=None):
    """VPoser manifold-distance prior: how far the body pose is from what VPoser
    can reconstruct.

        L = weight * mean( || theta - decode(encode(theta)) ||^2 )   (axis-angle)

    Unlike a ``||z||^2`` pose-embedding prior (which applies when optimizing in
    latent space), this does not pull the pose toward the latent origin (the
    dataset-typical pose); it penalizes the *distance to the learned pose manifold*
    instead. The reconstruction is detached so the loss acts as a stable
    projection-onto-manifold force, pulling the data-unconstrained legs onto
    plausible (reconstructable) configurations while the reprojection evidence
    still chooses which manifold pose to take.
    Because the decoder only emits bounded valid rotations (6D->matrot->aa, |aa|<=pi),
    out-of-manifold / impossible axis-angle rotations get a large penalty.

    pose: (T, 165) full SMPL-X pose; body_pose = pose[:, 3:66] (21 joints x 3, aa).
    """
    expr_dir = expr_dir or _DEFAULT_VPOSER_DIR
    vp = _get_vposer(expr_dir, pose.device)
    body_pose = pose[:, 3:66].reshape(-1, 21, 3)
    recon = vp.decode(vp.encode(body_pose).mean)["pose_body"].detach()   # (T,21,3) on-manifold target
    return weight * ((body_pose - recon) ** 2).sum(-1).mean()
