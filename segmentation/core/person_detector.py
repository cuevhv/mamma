"""Person box detectors for ma_masks.

A detector is any object with ``detect(image, conf)`` that takes an RGB PIL image and returns
``[(xyxy, score), ...]``: float32 ``[x1, y1, x2, y2]`` pixel boxes inside the image for the
people scoring above ``conf``.

``conf`` is on the YOLO12x confidence scale, which the config thresholds
(``matching.yolo_person_conf`` and the lower-threshold retries) are expressed in. A detector
whose scores are calibrated differently maps it to its own scale.
"""
import importlib
import os
import platform

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_DETECTOR = "dfine"
DFINE_CHECKPOINT = "ustc-community/dfine-xlarge-coco"


class DFineDetector:
    """D-FINE (Apache-2.0 code and COCO-only weights) through Hugging Face transformers."""

    INPUT_SIZE = 640
    # YOLO12x confidence -> D-FINE-X score at matched precision.
    CONF_MAP = ((0.25, 0.4, 0.5, 1.0), (0.5, 0.6, 0.65, 1.0))

    def __init__(self, checkpoint=None, device="cuda"):
        try:
            from transformers import DFineForObjectDetection
            from transformers.utils import logging as hf_logging
        except ImportError as e:
            raise ImportError("The D-FINE person detector needs transformers: pip install transformers") from e
        hf_logging.disable_progress_bar()  # per-tensor loading bar would flood the run logs
        self.model = DFineForObjectDetection.from_pretrained(checkpoint or DFINE_CHECKPOINT).to(device).eval()
        self.person = self.model.config.label2id["person"]
        self.device = device

    @torch.inference_mode()
    def detect(self, image, conf):
        w, h = image.size
        x = torch.from_numpy(np.array(image)).to(self.device).permute(2, 0, 1)[None].float().div_(255)
        x = F.interpolate(x, (self.INPUT_SIZE,) * 2, mode="bilinear", align_corners=False, antialias=True)
        out = self.model(pixel_values=x)
        scores = out.logits[0, :, self.person].sigmoid()
        cx, cy, bw, bh = out.pred_boxes[0].unbind(-1)
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1).clamp(0, 1)
        boxes = boxes * boxes.new_tensor([w, h, w, h])
        keep = scores > float(np.interp(conf, *self.CONF_MAP))
        return list(zip(boxes[keep].cpu().numpy().astype(np.float32), scores[keep].tolist()))


class YoloDetector:
    """Ultralytics YOLO, opt-in only: AGPL-3.0, needs ``pip install ultralytics``."""

    def __init__(self, checkpoint=None, device="cuda"):
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"YOLO checkpoint file not found: {checkpoint}")
        from ultralytics import YOLO
        self.model = YOLO(checkpoint, verbose=False)
        self.model.to("cpu" if platform.system() == "Windows" else device)

    def detect(self, image, conf):
        boxes = self.model(image, verbose=False)[0].boxes
        return [(b.xyxy.cpu().numpy()[0].astype(np.float32), float(b.conf))
                for b in boxes if int(b.cls) == 0 and float(b.conf) > conf]


def build_person_detector(name=None, checkpoint=None, device="cuda"):
    """``name``: "dfine" (default), "yolo", or "package.module:Class" built as Class(checkpoint, device)."""
    name = name or DEFAULT_DETECTOR
    builtin = {"dfine": DFineDetector, "yolo": YoloDetector}
    if name in builtin:
        return builtin[name](checkpoint, device)
    if ":" in name:
        module, cls = name.split(":", 1)
        return getattr(importlib.import_module(module), cls)(checkpoint, device)
    raise ValueError(f"Unknown person detector {name!r}: use dfine, yolo or package.module:Class.")
