"""Tests for tiled depth inference (--depth-tiles)."""

import numpy as np

from stereo360.depth.base import DepthBackend
from stereo360.pipeline import _tile_boxes, estimate_tiled


class ConstBackend(DepthBackend):
    def __init__(self, value):
        self.value = value
        self.tile_sizes = []

    def estimate(self, frame_rgb):
        return np.full(frame_rgb.shape[:2], self.value, dtype=np.float32)

    def estimate_chunk(self, frames_rgb):
        self.tile_sizes.append(frames_rgb[0].shape[:2])
        return [self.estimate(f) for f in frames_rgb]


def test_tile_boxes_cover_face():
    boxes = _tile_boxes(1920, 4)
    assert len(boxes) == 16
    covered = np.zeros((1920, 1920), bool)
    for y0, y1, x0, x1 in boxes:
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_tiled_estimate_reconstructs_constant():
    backend = ConstBackend(0.7)
    faces = [np.zeros((128, 128, 3), np.uint8) for _ in range(2)]
    out = estimate_tiled(backend, faces, split=2)
    assert len(out) == 2
    for d in out:
        assert d.shape == (128, 128)
        np.testing.assert_allclose(d, 0.7, rtol=1e-5)


def test_tiled_uses_smaller_inputs():
    backend = ConstBackend(0.5)
    faces = [np.zeros((256, 256, 3), np.uint8)]
    estimate_tiled(backend, faces, split=4)
    # First call is the full-face scale reference (256^2); the rest are
    # tiles of stride+2*pad = 64 + 16 = 80px.
    for sizes in backend.tile_sizes[1:]:
        assert max(sizes) <= 80


def test_feather_blend_is_convex_and_gapless():
    """Blended output must be a convex combination of tile values (within
    their min/max) with no zero-weight gaps anywhere on the face."""

    class MeanBackend(DepthBackend):
        def estimate(self, frame_rgb):
            return np.full(frame_rgb.shape[:2], frame_rgb.mean(),
                           dtype=np.float32)

        def estimate_chunk(self, frames_rgb):
            return [self.estimate(f) for f in frames_rgb]

    rng = np.random.default_rng(0)
    face = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    out = estimate_tiled(MeanBackend(), [face], split=4)[0]
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 255.0
    # No zero-weight gaps: output is nonzero almost everywhere.
    assert (out > 0).mean() > 0.99


def test_tiles_scale_aligned_to_reference():
    """Tiles with wildly different per-tile scales must be aligned to the
    full-face reference before blending, else the blend is mush."""
    rng = np.random.default_rng(0)

    class RandomScaleBackend(DepthBackend):
        def estimate(self, frame_rgb):
            # Vary per-tile scale within the alignment clamp range (0.25-4x).
            return np.full(frame_rgb.shape[:2], rng.uniform(0.5, 2.0),
                           dtype=np.float32)

        def estimate_chunk(self, frames_rgb):
            # Full-face reference call (square input at face size): scale 1.
            if frames_rgb[0].shape[0] == 128 and frames_rgb[0].shape[1] == 128:
                return [np.full((128, 128), 1.0, dtype=np.float32)
                        for _ in frames_rgb]
            return [self.estimate(f) for f in frames_rgb]

    face = np.zeros((128, 128, 3), np.uint8)
    out = estimate_tiled(RandomScaleBackend(), [face], split=2)[0]
    # After alignment every tile is ~1.0, so the blend is ~1.0 everywhere.
    np.testing.assert_allclose(out, 1.0, rtol=0.2)
