"""Tests for learned inpainting (M5)."""

import numpy as np
import pytest

from stereo360 import warp
from stereo360.inpaint import inpaint_learned


def test_inpaint_mode_rejects_unknown():
    img = np.zeros((32, 64, 3), np.uint8)
    disp = np.full((32, 64), 0.5, np.float32)
    with pytest.raises(ValueError):
        warp.right_eye_from_disparity(img, disp, inpaint_mode="bogus")


@pytest.mark.slow
def test_inpaint_learned_fills_mask():
    pytest.importorskip("simple_lama_inpainting")
    img = np.full((128, 128, 3), 120, np.uint8)
    mask = np.zeros((128, 128), np.uint8)
    mask[48:80, 48:80] = 255
    out = inpaint_learned(img, mask)
    assert out.shape == img.shape and out.dtype == np.uint8
    # Uniform gray image: fill should stay roughly gray everywhere.
    assert out[64, 64].mean() > 80
    # Pixels outside the mask untouched.
    assert (out[0:10] == img[0:10]).all()


def test_inpaint_learned_noop_on_empty_mask():
    pytest.importorskip("simple_lama_inpainting")
    img = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), np.uint8)
    out = inpaint_learned(img, mask)
    assert (out == img).all()
