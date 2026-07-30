"""Learned inpainting for disocclusion holes (M5) via LaMa.

Uses the `simple-lama-inpainting` package (pip install simple-lama-inpainting),
which bundles the LaMa model (Suvorov et al., 2021) — a Fourier-conv inpainting
network that synthesizes plausible texture instead of diffusing boundary
colors like Telea.

Full 8K frames exceed LaMa's practical input size, so holes are filled per
connected component: each hole cluster gets a padded crop, LaMa runs on the
crop, and the filled region is pasted back.
"""

from __future__ import annotations

import numpy as np

_PAD = 64          # context ring around each hole component
_MAX_CROP = 1024   # cap crop size; larger components get more, capped

_lama = None


def _get_lama():
    global _lama
    if _lama is None:
        from simple_lama_inpainting import SimpleLama

        _lama = SimpleLama()
    return _lama


def inpaint_learned(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """img_rgb: (H, W, 3) uint8. mask: (H, W) uint8, nonzero = fill me.

    Returns a new uint8 image with masked regions filled by LaMa.
    """
    import cv2
    from PIL import Image

    if not mask.any():
        return img_rgb

    lama = _get_lama()
    out = img_rgb.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    h, w = mask.shape

    for i in range(1, n):
        x, y, cw, ch = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        x0 = max(0, x - _PAD)
        y0 = max(0, y - _PAD)
        x1 = min(w, x + cw + _PAD)
        y1 = min(h, y + ch + _PAD)
        if (x1 - x0) > _MAX_CROP or (y1 - y0) > _MAX_CROP:
            # Very large component: fall back to Telea for this crop.
            sub = mask[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = cv2.inpaint(
                out[y0:y1, x0:x1], sub, 3.0, cv2.INPAINT_TELEA)
            continue

        crop_img = Image.fromarray(out[y0:y1, x0:x1])
        crop_mask = Image.fromarray(mask[y0:y1, x0:x1])
        filled = np.asarray(lama(crop_img, crop_mask))
        # simple-lama may return a slightly different size (internal padding);
        # resize back to the crop before compositing.
        if filled.shape[:2] != (y1 - y0, x1 - x0):
            filled = cv2.resize(filled, (x1 - x0, y1 - y0),
                                interpolation=cv2.INTER_LINEAR)
        region = mask[y0:y1, x0:x1] > 0
        out[y0:y1, x0:x1][region] = filled[region]
    return out
