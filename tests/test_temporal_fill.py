"""Tests for temporal hole filling."""

import numpy as np

from stereo360.temporal_fill import stabilize_depth, temporal_fill


def _frame(color, h=16, w=32):
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_fills_hole_from_consensus():
    # Frame 0 has a hole block; frames 1 and 2 show gray background there.
    rights = [_frame(0), _frame(128), _frame(130)]
    holes = [np.zeros((16, 32), np.uint8) for _ in range(3)]
    holes[0][4:12, 8:24] = 255
    temporal_fill(rights, holes)
    filled = rights[0][8, 16]
    assert 120 < filled.mean() < 140
    assert (holes[0] == 0).all()


def test_no_consensus_leaves_hole():
    # Every frame shows a different color at the pixel -> std too high.
    rights = [_frame(0), _frame(100), _frame(255)]
    holes = [np.full((16, 32), 255, np.uint8),
             np.zeros((16, 32), np.uint8),
             np.zeros((16, 32), np.uint8)]
    temporal_fill(rights, holes)
    # Frame 0's hole remains: values 0/100/255 disagree (std > 25).
    assert (holes[0] > 0).any()


def test_single_valid_frame_fills():
    rights = [_frame(0), _frame(77)]
    holes = [np.full((16, 32), 255, np.uint8),
             np.zeros((16, 32), np.uint8)]
    temporal_fill(rights, holes)
    assert (rights[0][8, 16] == 77).all()
    assert (holes[0] == 0).all()


def test_single_frame_list_is_noop():
    rights = [_frame(0)]
    holes = [np.full((16, 32), 255, np.uint8)]
    temporal_fill(rights, holes)
    assert (holes[0] > 0).all()


def test_stabilize_depth_locks_stable_pixels():
    rng = np.random.default_rng(0)
    base = np.full((16, 32), 0.5, dtype=np.float32)
    maps = [base + rng.normal(0, 0.003, base.shape).astype(np.float32)
            for _ in range(4)]
    stabilize_depth(maps, tau=0.02)
    # All four maps should now be (nearly) identical -> zero temporal flap.
    for m in maps[1:]:
        np.testing.assert_allclose(m, maps[0], atol=1e-6)


def test_stabilize_depth_preserves_real_motion():
    maps = [np.full((16, 32), 0.2 + 0.2 * i, dtype=np.float32)
            for i in range(4)]
    stabilize_depth(maps, tau=0.02)
    # Strongly varying pixels (std ~0.26) must keep per-frame values.
    for i, m in enumerate(maps):
        np.testing.assert_allclose(m, 0.2 + 0.2 * i, rtol=1e-6)


def test_stabilize_depth_short_sequence_noop():
    maps = [np.full((8, 8), 0.5, np.float32),
            np.full((8, 8), 0.6, np.float32)]
    stabilize_depth(maps)
    assert abs(float(maps[1][0, 0]) - 0.6) < 1e-6


def test_valid_pixels_untouched():
    rights = [_frame(10), _frame(200)]
    holes = [np.zeros((16, 32), np.uint8), np.zeros((16, 32), np.uint8)]
    temporal_fill(rights, holes)
    assert (rights[0] == 10).all() and (rights[1] == 200).all()
