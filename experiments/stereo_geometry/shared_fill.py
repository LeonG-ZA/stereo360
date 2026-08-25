"""NEGATIVE RESULT, and the premise was wrong rather than the code.

Fill both eyes from one invented background, instead of inventing twice.

Disocclusion holes are filled per eye today, and `_directional_fill` mirrors
in opposite directions -- `from_right = baseline_sign > 0` -- so the left eye
and the right eye invent *different* content wherever content is invented.
That is binocular rivalry placed exactly where the geometry was least certain,
and unlike a hole it cannot be suppressed by fusion, because both eyes see
something plausible and they disagree.

The existing design defends holes landing in different places, and that
reasoning is right for *holes*: black is easy to suppress. It does not carry
over to filled content.

Filling once and sharing it is not as simple as inpainting the source, because
a hole has no source pixel by definition -- it is background the foreground
was covering. So the shared thing has to be a **background layer**: the source
with the foreground stripped away near each depth edge and inpainted from the
background around it, plus a depth map with the background continued behind
where the foreground used to be. That layer is warped for each eye exactly as
the source is, and wherever an eye holes, it takes that eye's *own* warp of
the shared layer.

Both eyes then show the same invention, and they show it at the background's
disparity rather than at whatever the filler happened to smear sideways. The
first is what stops the rivalry; the second is a bonus -- a directional fill
puts invented pixels at the foreground's disparity, which is wrong for
something that is meant to be behind it.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

REPO = os.environ.get(
    "STEREO360_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from stereo360 import warp  # noqa: E402


def max_shift_px(dn: np.ndarray, strength: float, width: int) -> int:
    """Widest longitude shift the warp can produce, in pixels.

    Sets how far behind an edge the background layer has to be invented: a
    hole can never be wider than the disparity step that opened it.
    """
    baseline = abs(strength) * warp._BASELINE_SCALE
    lam = 1.0 / (float(np.nanmax(dn)) + warp._MIN_INV_DEPTH)
    return int(np.ceil(np.degrees(np.arctan2(baseline, lam)) / 360.0 * width))


def background_layer(rgb: np.ndarray, dn: np.ndarray, reach: int,
                     min_step: float = 0.05):
    """The scene with near-edge foreground removed and inpainted behind.

    Returns (rgb, dn) for the layer. `reach` is how far behind an edge to
    invent, and should be the widest shift the warp can make.
    """
    k = np.ones((2 * reach + 1,) * 2, np.uint8)
    lo = cv2.erode(dn, k)                    # the local background's depth
    fg = (dn - lo) > min_step                # near side of a depth edge
    fg = cv2.dilate(fg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not fg.any():
        return rgb.copy(), dn.copy()
    bg_rgb = cv2.inpaint(rgb, fg.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    bg_dn = np.where(fg, lo, dn).astype(np.float32)
    return bg_rgb, bg_dn


def stereo_pair(frame, dn, strength, left_share=0.0, shared=True,
                reach=None, gradient_limit=0.0, fg_erode=0, **kw):
    """(left, right) with holes filled from a shared background layer.

    With `shared` false this falls back to the per-eye directional fill, so
    the two can be compared with everything else held identical.
    """
    h, w = dn.shape
    f = float(np.clip(left_share, 0.0, 1.0))
    if reach is None:
        reach = max(4, max_shift_px(dn, strength, w))
    bg_rgb, bg_dn = background_layer(frame, dn, reach) if shared else (None, None)

    def one(s):
        img, hole = warp.right_eye_from_disparity(
            frame, dn.copy(), s, normalize=False, inpaint=False,
            gradient_limit=gradient_limit, fg_erode=fg_erode, **kw)
        m = hole > 0
        if not m.any():
            return img
        if shared:
            # The layer is warped for *this* eye, so the invented pixels
            # arrive at their own disparity rather than being smeared
            # sideways from the foreground.
            bg, _ = warp.right_eye_from_disparity(
                bg_rgb, bg_dn.copy(), s, normalize=False, inpaint=False,
                gradient_limit=gradient_limit, fg_erode=fg_erode, **kw)
            img[m] = bg[m]
            return img
        return warp.fill_holes(img, hole, dn, directional=True,
                               baseline_sign=1.0 if s >= 0 else -1.0)

    if f <= 0.0:
        return frame, one(strength)
    if f >= 1.0:
        return one(-strength), frame
    return one(-f * strength), one((1.0 - f) * strength)
