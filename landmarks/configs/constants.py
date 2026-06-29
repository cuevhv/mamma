"""Project paths and constants.

Path roots are read from environment variables. A `.env` file in the repo root is
auto-loaded via python-dotenv when this module is imported; you can also export
the variables in your shell. `.env` is gitignored.

Required / optional environment variables
-----------------------------------------

| Variable                | Purpose                                                                 | Required by                                       |
| ----------------------- | ----------------------------------------------------------------------- | ------------------------------------------------- |
| MAMMA_EXTRA_DATA        | Legacy collaborator data (dataset npzs, vicon captures, backgrounds).   | Vicon / InteractionsTests evaluation              |
| MAMMA_SAM2_MASKS        | Root of SAM2-predicted masks for per-dataset evaluation.                | `lib/data_utils/process_real_data.py`             |
| WANDB_API_KEY           | Optional. Enables Weights & Biases logging.                             | `train.py` if you want wandb                      |

The webdataset root is no longer an env var — pass it as a Hydra CLI override
(`python train.py dataset_path=/your/root`), defaulting to `data/mammasyn`. The
BEDLAM masks webdataset is expected at `${dataset_path}/BEDLAM_MASKS_WD/`.

If a required variable is unset, importing the relevant code path raises
`EnvironmentError`.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Environment variable {name} is not set. "
            f"See the README \"Environment Variables\" section for the full list."
        )
    return value


class PATHS:
    # Paths below match the unified `data/` layout documented in
    # docs/INSTALL.md (the inference layout). Keeping training and
    # inference on the same on-disk locations means users who follow
    # INSTALL.md don't need a second set of directories for training.
    BODY_MODELS_PTH = "data/body_models"
    SUBSAMPLE_PTS_DIR = "data/body_models/downsampled_verts"
    SMPLX_LOCKHEAD_MODELS = "data/body_models/smplx_locked_head"

    PRETRAINED_VITPOSE_CKPT_PTH = "data/weights/vitpose/vitpose-b-multi-coco.pth"

    # Vars without sensible defaults — resolved lazily so importing this module
    # doesn't crash when an env var is unset; only the code paths that need
    # them fail, with a clear error.
    @classmethod
    def extra_data(cls) -> str:
        return _require_env("MAMMA_EXTRA_DATA")

    @classmethod
    def sam2_masks(cls) -> str:
        return _require_env("MAMMA_SAM2_MASKS")
