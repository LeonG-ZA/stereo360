"""Edge-aware depth smoothing (guided filter).

Per-frame depth models often produce noisy depth inside flat regions, which
shows up as wobble/flicker in the synthesized right eye. A guided filter
smooths the depth map using the RGB frame as the guide: depth is flattened
*within* regions of similar color but preserved across color edges, which is
exactly where depth discontinuities usually live.

This is the M2 stopgap for temporal/structural stability; full temporal
consistency arrives with the video depth model in M3.
"""

from __future__ import annotations

import numpy as np

from .base import DepthBackend


def guided_filter(guide_rgb: np.ndarray, src: np.ndarray,
                  radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Guided filter (He et al., 2010) via box filters — O(1) per pixel.

    guide_rgb: (H, W, 3) float32 in [0, 1]; src: (H, W) float32.
    Larger radius/eps = stronger smoothing.
    """
    import cv2

    r = radius
    size = (2 * r + 1, 2 * r + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, ddepth=-1, ksize=size, normalize=True,
                             borderType=cv2.BORDER_REFLECT)

    i_r, i_g, i_b = guide_rgb[..., 0], guide_rgb[..., 1], guide_rgb[..., 2]
    mean_r, mean_g, mean_b = box(i_r), box(i_g), box(i_b)
    mean_p = box(src)

    corr_rr = box(i_r * i_r)
    corr_rg = box(i_r * i_g)
    corr_rb = box(i_r * i_b)
    corr_gg = box(i_g * i_g)
    corr_gb = box(i_g * i_b)
    corr_bb = box(i_b * i_b)
    corr_rp = box(i_r * src)
    corr_gp = box(i_g * src)
    corr_bp = box(i_b * src)

    # 3x3 covariance of the guide + eps*I, solved per pixel by Cramer's rule.
    #
    # `np.linalg.solve` on an (H, W, 3, 3) stack was 33% of the whole program:
    # 20.3 s of a 62 s run, because it dispatches a general LAPACK solve per
    # pixel across 22 megapixels of face, six faces per frame. A symmetric 3x3
    # system has a closed form, so this becomes elementwise arithmetic -- and
    # it also drops the (H, W, 3, 3) float32 buffer (792 MB at face 1920) and
    # the float64 copy `solve` needed on top of it (1.6 GB).
    #
    # eps on the diagonal keeps the matrix positive definite, so the
    # determinant stays away from zero and float32 suffices; the guard below
    # only matters if a user asks for --smooth-eps 0.
    v_rr = corr_rr - mean_r * mean_r + eps
    v_rg = corr_rg - mean_r * mean_g
    v_rb = corr_rb - mean_r * mean_b
    v_gg = corr_gg - mean_g * mean_g + eps
    v_gb = corr_gb - mean_g * mean_b
    v_bb = corr_bb - mean_b * mean_b + eps

    # Cofactors of [[rr, rg, rb], [rg, gg, gb], [rb, gb, bb]].
    c00 = v_gg * v_bb - v_gb * v_gb
    c01 = v_rb * v_gb - v_rg * v_bb
    c02 = v_rg * v_gb - v_rb * v_gg
    c11 = v_rr * v_bb - v_rb * v_rb
    c12 = v_rb * v_rg - v_rr * v_gb
    c22 = v_rr * v_gg - v_rg * v_rg

    det = v_rr * c00 + v_rg * c01 + v_rb * c02
    inv_det = 1.0 / np.where(np.abs(det) < 1e-12, np.float32(1e-12), det)

    b_r = corr_rp - mean_r * mean_p
    b_g = corr_gp - mean_g * mean_p
    b_b = corr_bp - mean_b * mean_p

    a_r = (c00 * b_r + c01 * b_g + c02 * b_b) * inv_det
    a_g = (c01 * b_r + c11 * b_g + c12 * b_b) * inv_det
    a_b = (c02 * b_r + c12 * b_g + c22 * b_b) * inv_det

    mean_ar, mean_ag, mean_ab = box(a_r), box(a_g), box(a_b)
    mean_b_ = box(mean_p - (a_r * mean_r + a_g * mean_g + a_b * mean_b))
    return mean_ar * i_r + mean_ag * i_g + mean_ab * i_b + mean_b_


class EdgeAwareSmoothingBackend(DepthBackend):
    """Decorator: smooths another backend's depth with a guided filter
    using the input frame as the edge-preserving guide."""

    def __init__(self, backend: DepthBackend, radius: int = 8,
                 eps: float = 1e-3) -> None:
        self._backend = backend
        self.radius = radius
        self.eps = eps

    def estimate(self, frame_rgb: np.ndarray) -> np.ndarray:
        depth = self._backend.estimate(frame_rgb)
        # The filter can drift the range slightly; restore min/max so the
        # stereo strength stays calibrated to the raw backend's output.
        return self._smooth(frame_rgb, depth)

    def estimate_chunk(self, frames_rgb: list) -> list:
        # Smoothing is per-frame; delegate the chunk to the wrapped backend so
        # temporal consistency (if any) is preserved, then smooth each result.
        depths = self._backend.estimate_chunk(frames_rgb)
        return [self._smooth(f, d) for f, d in zip(frames_rgb, depths)]

    def _smooth(self, frame_rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        guide = frame_rgb.astype(np.float32) / 255.0
        smoothed = guided_filter(guide, depth, radius=self.radius, eps=self.eps)
        lo, hi = float(depth.min()), float(depth.max())
        return np.clip(smoothed, lo, hi).astype(np.float32)

    def close(self) -> None:
        self._backend.close()
