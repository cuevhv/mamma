"""Image normalization constants for the dense-landmark crops (ImageNet stats).

The former ``ViTDetDataset`` CPU crop pipeline (per-crop copy + blur + affine
warp) was replaced by on-GPU preprocessing in ``run_ma_2d.py`` (kornia warp /
blur / normalize). Only these two constants are still imported, so the dataset
class and its helpers were removed as dead code.
"""
import numpy as np

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
