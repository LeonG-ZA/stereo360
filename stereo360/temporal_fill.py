"""Temporal hole filling: fill right-eye disocclusion holes from other frames.

The right-eye virtual camera is fixed, so background that is disoccluded in
one frame is often *visible* in the warped right-eye view of neighboring
frames (camera or foreground motion reveals it). Per-frame inpainting
(Telea/LaMa) hallucinates this content differently every frame — a major
source of flicker. This module instead gathers, for each hole pixel, the
valid colors other frames produced at the same pixel, and fills with their
median when they agree. Remaining holes (no consensus, or holes in every
frame) fall through to spatial inpainting.

Works best within a temporal chunk (see pipeline._convert_chunked): static
or slowly moving backgrounds with independently moving foreground.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .warp import _map_bands

_MAX_STD = 25.0     # max per-channel std across frames to accept a median
_ROW_CHUNK = 256    # process in row bands to bound memory


def temporal_fill(rights: List[np.ndarray], holes: List[np.ndarray],
                  row_chunk: int = _ROW_CHUNK) -> None:
    """Fill holes in-place across a list of warped right-eye frames.

    rights: list of K (H, W, 3) uint8 warped frames (unfilled, holes zeroed).
    holes:  list of K (H, W) uint8 masks (255 = hole). Updated in place:
            filled pixels are cleared from the mask.
    """
    if len(rights) < 2:
        return
    h, w = rights[0].shape[:2]
    k = len(rights)

    # Only pixels that are a hole in at least one frame can ever be filled, and
    # holes are a fraction of a percent of the frame. Computing the median and
    # spread for *every* pixel therefore threw almost all of the work away: at
    # 8K over a 6-frame chunk that cost 17 s, 2.84 s per frame, the single
    # largest item in the pipeline. Gathering the candidates first makes the
    # median a few thousand values instead of 88 million.
    for y0 in range(0, h, row_chunk):
        y1 = min(y0 + row_chunk, h)
        bands = [hm[y0:y1] for hm in holes]
        need = bands[0] > 0
        for hm in bands[1:]:
            need = need | (hm > 0)
        if not need.any():
            continue
        ys, xs = np.nonzero(need)                            # (M,) candidates

        vals = np.stack([r[y0:y1][ys, xs] for r in rights])   # (K, M, 3)
        valid = np.stack([hm[ys, xs] == 0 for hm in bands])   # (K, M)
        count = valid.sum(axis=0)                            # (M,)

        v = vals.astype(np.float32)
        v[~valid] = np.nan
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(v, axis=0)                    # (M, 3)
            std = np.nanstd(v, axis=0).mean(axis=-1)         # (M,)

        accept = (count == 1) | ((count >= 2) & (std <= _MAX_STD))
        # Only fill pixels that are holes in the *current* frame.
        for i in range(k):
            fill = np.nonzero((bands[i][ys, xs] > 0) & accept)[0]
            if fill.size:
                fy, fx = ys[fill], xs[fill]
                rights[i][y0:y1][fy, fx] = np.clip(
                    med[fill], 0, 255).astype(np.uint8)
                bands[i][fy, fx] = 0


def stabilize_depth(maps: List[np.ndarray], tau: float = 0.02,
                    row_chunk: int = _ROW_CHUNK) -> None:
    """Suppress warp edge-flapping: blend each pixel's depth toward the chunk
    median by an amount that ramps down as its temporal std approaches 2*tau
    (normalized units).

    Nearest-pixel forward warping rounds sub-pixel positions; tiny temporal
    depth noise at silhouettes (std ~0.005) flips the rounding between the
    foreground and background disparity, making edges shimmer. Median-locking
    stable pixels removes the flap; pixels with genuinely varying depth
    (moving objects) keep their per-frame values. A hard std threshold makes
    borderline pixels flip between "locked" and "per-frame" states across
    frames (visible popping), so the lock is a soft blend instead:
    w = clip((2*tau - std)/tau, 0, 1); m <- w*median + (1-w)*m. In place.
    """
    if len(maps) < 3:
        return
    h, w = maps[0].shape
    k = len(maps)

    def band_fn(y0: int, y1: int) -> None:
        stack = np.stack([m[y0:y1] for m in maps])  # (K, B, W)
        std = stack.std(axis=0)
        med = np.median(stack, axis=0)
        wgt = np.clip((2.0 * tau - std) / tau, 0.0, 1.0)
        if (wgt > 0).any():
            for i in range(k):
                band = maps[i][y0:y1]
                band += wgt * (med - band)

    # Bands touch disjoint rows of every map, so this parallelises outright.
    # Worth it because np.median selects per pixel across the chunk axis over
    # the whole frame: profiled at 8K this was 3.1 s per 6-frame chunk, the
    # largest cost in the chunked path outside the warp itself.
    _map_bands(band_fn, h, row_chunk)
