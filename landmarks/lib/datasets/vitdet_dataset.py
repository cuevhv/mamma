from typing import Dict

import cv2
import numpy as np
from yacs.config import CfgNode
import torch

from .utils_eval import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

class ViTDetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_cv2: np.array,
                 mask_cv2: np.array,
                 boxes: np.array,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_cv2 = img_cv2
        self.mask_cv2 = mask_cv2

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = (cfg.data_cfg['image_size'][1], cfg.data_cfg['image_size'][0]) #cfg.model["backbone"]["img_size"] #[0]
        self.mean = DEFAULT_MEAN #255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = DEFAULT_STD # 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # Preprocess annotations
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.data_cfg["image_size"]
        bedlam_scale = 1.2 #1.6 #1.2  # defined by BEDLAM one
        bbox_size = expand_to_aspect_ratio(scale*200*bedlam_scale, target_aspect_ratio=BBOX_SHAPE)#.max()
        w_bbox_size, h_bbox_size = bbox_size
        patch_width = patch_height = self.img_size
        patch_height, patch_width = self.img_size

        # 3. generate image patch
        cvimg = self.img_cv2.copy()
        cvmask = self.mask_cv2.copy() if self.mask_cv2 is not None else None
        # Anti-alias before the downsampling warp. The blur only needs to cover
        # the region the warp samples — the bbox plus the gaussian's support — so
        # blur just that ROI with an equivalent cv2 gaussian instead of running a
        # skimage blur over the whole (e.g. 4K) frame. ~90x cheaper, and the
        # result inside the sampled patch is numerically equivalent.
        downsampling_factor = ((bbox_size.max() * 1.0) / patch_width) / 2.0
        if downsampling_factor > 1.1:
            sigma = (downsampling_factor - 1) / 2
            margin = int(np.ceil(4 * sigma)) + 2
            img_h, img_w = cvimg.shape[:2]
            rx0 = max(0, int(center_x - w_bbox_size / 2) - margin)
            rx1 = min(img_w, int(center_x + w_bbox_size / 2) + margin)
            ry0 = max(0, int(center_y - h_bbox_size / 2) - margin)
            ry1 = min(img_h, int(center_y + h_bbox_size / 2) + margin)
            if rx1 > rx0 and ry1 > ry0:
                ksize = 2 * int(np.ceil(3 * sigma)) + 1
                cvimg[ry0:ry1, rx0:rx1] = cv2.GaussianBlur(
                    cvimg[ry0:ry1, rx0:rx1], (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

        img_patch_cv, mask_patch, trans = generate_image_patch_cv2(cvimg, cvmask,
                                                    center_x, center_y,
                                                    w_bbox_size, h_bbox_size,
                                                    patch_width, patch_height,
                                                    False, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)
        mask_patch = convert_cvimg_to_tensor(mask_patch[:,:,None])/255.0 if mask_patch is not None else 0

        # apply normalization
        for n_c in range(min(self.img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,
            'mask': mask_patch,
            'personid': int(self.personid[idx]),
        }
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        return item
