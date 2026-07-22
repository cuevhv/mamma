"""Labeled debug collage for the ma_2d per-body videos.

Replaces the old cv2.putText-on-the-pixels labeling, which had three
compounding problems: the Hershey bitmap text was drawn at full crop
resolution and then downscaled with the image (blurry), it sat on top of
the image content, and it was red — the worst possible color for JPEG +
H.264 4:2:0 encoding, where chroma is stored at quarter resolution and
red edges smear.

Here every tile gets a banner ABOVE the image instead: near-black strip,
off-white TrueType text (detail lives in the luma channel, which the
codecs keep at full resolution), rendered AFTER the tiles are downscaled
so the glyphs are never resampled. One font size is fitted so the longest
label fits its slot, shared by all banners for a consistent look.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# RGB (PIL order). Dark banner + light text keeps the label contrast in
# luma, which JPEG/H.264 preserve at full resolution (unlike chroma).
_BANNER_BG = (16, 17, 20)
_BANNER_FG = (242, 242, 242)
_BANNER_MUTED = (150, 155, 165)   # frame counter
_SEPARATOR_PX = 2                 # dark column between tiles

_FONT_CACHE: dict = {}


def _font_path() -> str | None:
    """DejaVuSans-Bold shipped with matplotlib (already a dependency)."""
    try:
        import matplotlib
        p = os.path.join(os.path.dirname(matplotlib.__file__),
                         "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf")
        return p if os.path.exists(p) else None
    except Exception:  # noqa: BLE001 — fall back to PIL's builtin font
        return None


def _font(size: int):
    from PIL import ImageFont
    if size not in _FONT_CACHE:
        p = _font_path()
        _FONT_CACHE[size] = (ImageFont.truetype(p, size) if p
                             else ImageFont.load_default())
    return _FONT_CACHE[size]


def _fit_font_size(labels_and_widths, lo: int = 9, hi: int = 22) -> int:
    """Largest size (<= hi) at which every label fits its own slot width."""
    from PIL import Image, ImageDraw
    probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    for size in range(hi, lo, -1):
        f = _font(size)
        if all(probe.textlength(t, font=f) <= max(w - 12, 10)
               for t, w in labels_and_widths):
            return size
    return lo


def labeled_collage(tiles, downscale: int = 2, corner_text: str | None = None) -> np.ndarray:
    """Compose [(bgr_image, label), ...] into one labeled row (BGR uint8).

    Tiles may have different widths but must share one height. Each is
    shrunk by ``downscale`` (INTER_AREA), then topped with its label
    banner; ``corner_text`` (e.g. the frame number) goes right-aligned in
    the first banner. Output dimensions are forced even for H.264.
    """
    from PIL import Image, ImageDraw

    assert tiles, "labeled_collage needs at least one tile"
    imgs, labels = zip(*tiles)
    if downscale > 1:
        imgs = [cv2.resize(im, (im.shape[1] // downscale, im.shape[0] // downscale),
                           interpolation=cv2.INTER_AREA) for im in imgs]
    h = imgs[0].shape[0]
    assert all(im.shape[0] == h for im in imgs), "tiles must share one height"
    widths = [im.shape[1] for im in imgs]

    size = _fit_font_size(list(zip(labels, widths)))
    banner_h = int(round(size * 2.0))

    total_w = sum(widths) + _SEPARATOR_PX * (len(imgs) - 1)
    # H.264 wants even dimensions; pad width with one extra separator column
    # and absorb height parity into the banner.
    pad_w = total_w % 2
    total_w += pad_w
    banner_h += (banner_h + h) % 2

    banner = Image.new("RGB", (total_w, banner_h), _BANNER_BG)
    draw = ImageDraw.Draw(banner)
    f = _font(size)
    x = 0
    for i, (label, w) in enumerate(zip(labels, widths)):
        ty = (banner_h - size) // 2 - 1
        draw.text((x + 6, ty), label, fill=_BANNER_FG, font=f)
        if i == 0 and corner_text:
            tw = draw.textlength(corner_text, font=f)
            # Keep the counter clear of a long first label.
            if 6 + draw.textlength(label, font=f) + 10 + tw <= w:
                draw.text((x + w - tw - 6, ty), corner_text, fill=_BANNER_MUTED, font=f)
        x += w + _SEPARATOR_PX

    row = np.full((h, total_w, 3), _BANNER_BG[::-1], dtype=np.uint8)  # BGR
    x = 0
    for im in imgs:
        row[:, x:x + im.shape[1]] = im
        x += im.shape[1] + _SEPARATOR_PX

    banner_bgr = np.asarray(banner, dtype=np.uint8)[:, :, ::-1]
    return np.vstack([banner_bgr, row])
