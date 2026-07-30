"""The GPU warp must agree with the numpy warp it replaces."""

import numpy as np
import pytest

from stereo360 import warp, warp_torch


def _scene(h, w, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    dn = np.full((h, w), 0.10, np.float32)
    dn[h // 4:3 * h // 4, w // 2 - 40:w // 2 + 40] = 0.90    # a near slab
    dn[h // 2 - 3:h // 2 + 3, :] = 0.75                      # a horizontal rail
    return img, dn


requires_gpu = pytest.mark.skipif(
    warp_torch.device_available() is None,
    reason="torch reports no usable GPU (expected on AMD/Windows: no ROCm)")


@requires_gpu
@pytest.mark.parametrize("gradient_limit", [0.0, 1.0])
@pytest.mark.parametrize("strength", [1.0, -0.5])
def test_gpu_warp_matches_numpy(monkeypatch, strength, gradient_limit):
    """Same geometry, same holes. The GPU path is an optimisation only.

    Exact equality is not the bar: `grid_sample` and `cv2.remap` round their
    bilinear taps differently, so isolated pixels on a high-contrast edge can
    differ by a few levels. On a real 8K frame that was 223 pixels out of 29.5
    million. What must match is the geometry and the hole mask, since those
    drive everything downstream.
    """
    img, dn = _scene(256, 512)
    dev = warp_torch.device_available()

    monkeypatch.setenv("STEREO360_GPU_WARP", "0")
    warp._gpu_probe.clear()
    cpu, cpu_hole = warp.right_eye_from_disparity(
        img, dn.copy(), strength, inpaint=False, normalize=False,
        gradient_limit=gradient_limit)

    gpu, gpu_hole = warp_torch.right_eye_from_disparity(
        img, dn.copy(), strength, warp._BASELINE_SCALE, warp._MIN_INV_DEPTH,
        warp._VIS_RATIO, warp._CRACK_MARGIN, 2, gradient_limit, dev)

    assert np.array_equal(cpu_hole > 0, gpu_hole > 0), "hole masks differ"
    diff = np.abs(cpu.astype(int) - gpu.astype(int)).max(axis=-1)
    assert diff.mean() < 0.05, f"mean difference {diff.mean():.3f} too large"
    assert (diff > 2).mean() < 0.001, (
        f"{100 * (diff > 2).mean():.3f}% of pixels differ by >2 levels")


@requires_gpu
def test_gpu_warp_is_used_by_default_and_can_be_forced_off(monkeypatch):
    monkeypatch.delenv("STEREO360_GPU_WARP", raising=False)
    warp._gpu_probe.clear()
    assert warp.gpu_device() is not None

    monkeypatch.setenv("STEREO360_GPU_WARP", "0")
    assert warp.gpu_device() is None


def test_forcing_the_gpu_on_a_machine_without_one_says_so(monkeypatch):
    """Silent CPU fallback is the failure mode worth being loud about.

    The message also has to be accurate about AMD: torch-directml *does* give
    a device, so "AMD cannot have a torch GPU" was wrong. It is not used
    because scatter_reduce_ falls back to the CPU there and the results differ
    from the reference, which is a different claim and the one that is true.
    """
    monkeypatch.setenv("STEREO360_GPU_WARP", "1")
    monkeypatch.setattr(warp_torch, "device_available", lambda: None)
    warp._gpu_probe.clear()
    with pytest.raises(RuntimeError, match="no CUDA or MPS device"):
        warp.gpu_device()
    warp._gpu_probe.clear()
