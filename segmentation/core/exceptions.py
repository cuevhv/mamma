"""Pipeline exception types."""


class CameraProcessingError(RuntimeError):
    """A single camera failed during segmentation (e.g. propagation error).

    Raised instead of silently skipping so the caller can decide: the
    per-sequence driver records the camera and keeps processing the others,
    then fails the run at the end. Crucially, no masks.npy is written for the
    failed camera — a transient failure (CUDA OOM, decode error) must not
    become a sticky wrong-camera cache that resume reuses.
    """

    def __init__(self, cam_name, reason):
        super().__init__(f"[{cam_name}] {reason}")
        self.cam_name = cam_name
        self.reason = reason
