"""Dilate the foreground depth, but stop at the picture's own edges.

Plain dilation grows the near region a fixed distance in every direction. That
fixes objects the depth map under-covers -- V3 Small leaves 15 px of the lamp
post and the table edge reading as background -- but it cannot tell the
difference between "still inside the object" and "already out on the floor",
so it drags a halo of background with it. Measured on the indoor frame: the
grout line under the table turned 3.44 against the source's 1.26, an elbow
right at the halo boundary, because the part of the line inside the halo moved
at the table's disparity and the rest did not.

The fix is to let the near depth flood outward but refuse to cross a strong
luminance edge. Inside an object there is no such edge, so the flood fills it
out to its own silhouette; at the silhouette the edge stops it. One pixel per
iteration, so the barrier is tested at every step rather than jumped over.
"""
from __future__ import annotations
import numpy as np
import cv2


def edge_barrier(rgb: np.ndarray, pct: float = 88.0, blur: int = 3,
                 mode: str = "luma") -> np.ndarray:
    """Pixels the flood may not enter.

    `luma` blocks on brightness, which is wrong for the features that need
    dilating most: a specular highlight along a rail or a rim is a strong
    luminance edge sitting *inside* the object, so the flood stops before it
    reaches the silhouette. Measured on the F12 chair, luma-barriered
    dilation left the bright top band at 11.3 px in the right eye against the
    left's 17.0 -- identical to not dilating at all.

    `chroma` blocks on colour instead. A highlight changes brightness while
    keeping its hue; a real silhouette usually changes hue as well, because
    it is a different material. Gradients are taken on Lab's a and b, which
    drops the brightness axis entirely.
    """
    if mode == "chroma":
        lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2Lab).astype(np.float32)
        chans = [lab[..., 1], lab[..., 2]]
    else:
        chans = [cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)]
    mag = np.zeros(rgb.shape[:2], np.float32)
    for c in chans:
        if blur:
            c = cv2.GaussianBlur(c, (2 * blur + 1,) * 2, 0)
        mag = np.maximum(mag, np.hypot(
            cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)))
    return mag > np.percentile(mag, pct)


def geodesic_dilate(depth: np.ndarray, rgb: np.ndarray, radius: int,
                    pct: float = 88.0, mode: str = "luma",
                    absorb: bool = True) -> np.ndarray:
    """Grey dilation of `depth` that will not propagate across image edges."""
    out = depth.astype(np.float32).copy()
    barrier = edge_barrier(rgb, pct, mode=mode)
    k = np.ones((3, 3), np.uint8)
    for _ in range(radius):
        if absorb:
            # An edge pixel may *receive* depth but never *emit* it. Freezing
            # them instead -- the obvious reading of "the flood stops at the
            # edge" -- is wrong, because the pixels the depth map gets wrong
            # are precisely the edge pixels: the chair rail's top few rows
            # carry the depth of what is behind it. Freezing them guarantees
            # the one thing that needed fixing never gets fixed, which is why
            # the first version measured identical to no dilation at all.
            emit = np.where(barrier, -np.inf, out)
            out = np.maximum(out, cv2.dilate(emit, k))
        else:
            out = np.where(barrier, out, np.maximum(out, cv2.dilate(out, k)))
    return out
