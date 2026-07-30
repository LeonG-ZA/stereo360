"""Tests for edge-aware depth smoothing (guided filter)."""

import numpy as np
import pytest

from stereo360.depth.base import DepthBackend
from stereo360.depth.smoothing import EdgeAwareSmoothingBackend, guided_filter


class NoisyBackend(DepthBackend):
    """Returns a step depth map plus strong noise."""

    def estimate(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        depth = np.where(np.arange(w)[None, :] < w // 2, 1.0, 0.2).astype(
            np.float32) * np.ones((h, w), dtype=np.float32)
        rng = np.random.default_rng(0)
        return depth + rng.normal(0, 0.1, size=(h, w)).astype(np.float32)


def _step_frame(h=64, w=64):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, : w // 2] = 255
    return frame


def test_smooths_flat_regions_but_keeps_edge():
    frame = _step_frame()
    raw = NoisyBackend().estimate(frame)
    smooth = EdgeAwareSmoothingBackend(NoisyBackend(), radius=8).estimate(frame)

    assert smooth.shape == raw.shape and smooth.dtype == np.float32
    # Interior noise variance reduced on both sides of the step.
    left_var = smooth[8:-8, 4:24].var()
    right_var = smooth[8:-8, -24:-4].var()
    assert left_var < raw[8:-8, 4:24].var() * 0.2
    assert right_var < raw[8:-8, -24:-4].var() * 0.2
    # The step edge survives: means on each side remain well separated.
    assert smooth[8:-8, 4:24].mean() - smooth[8:-8, -24:-4].mean() > 0.5


def test_guided_filter_identity_on_constant_input():
    guide = np.random.default_rng(1).random((32, 32, 3)).astype(np.float32)
    src = np.full((32, 32), 0.7, dtype=np.float32)
    out = guided_filter(guide, src)
    np.testing.assert_allclose(out[8:-8, 8:-8], 0.7, atol=1e-4)


def test_output_clipped_to_raw_range():
    frame = _step_frame()
    raw = NoisyBackend().estimate(frame)
    smooth = EdgeAwareSmoothingBackend(NoisyBackend()).estimate(frame)
    assert smooth.min() >= raw.min() - 1e-6
    assert smooth.max() <= raw.max() + 1e-6


def test_close_delegates():
    class B(DepthBackend):
        closed = False

        def estimate(self, f):
            return np.zeros(f.shape[:2], dtype=np.float32)

        def close(self):
            self.closed = True

    b = B()
    EdgeAwareSmoothingBackend(b).close()
    assert b.closed
