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

    Exact equality is not the bar, and neither is per-pixel closeness. The two
    paths sample the left image at *different sub-pixel positions*: cv2.remap
    snaps the sample coordinate to a 1/32-pixel fixed-point grid (OpenCV's
    INTER_BITS=5) before interpolating, while grid_sample works in float32.
    Snapping grid_sample's coordinates to the same 1/32 grid collapses the
    disagreement to 0.005 levels, i.e. pure float-to-uint8 rounding, which is
    how we know that grid is the whole of it.

    The resulting error is `coordinate error x local gradient`, so it scales
    with scene contrast rather than staying on edges: on the maximum-contrast
    noise below it is ~1.3 levels almost everywhere, on real footage a tenth
    of that. A mean-difference bound is therefore the wrong instrument -- it
    measures scene contrast more than it measures agreement.

    The noise scene is kept precisely *because* it is maximum contrast, which
    is what gives the geometry checks their teeth: a one-pixel misalignment
    takes 99.8% of pixels past the outlier bound below, where on smooth
    content it would hide. So this asserts what actually has to hold --
    identical holes, no systematic brightness shift, and no pixel meaningfully
    relocated -- and leaves sub-level sampling noise alone.
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

    signed = cpu.astype(np.float64) - gpu.astype(np.float64)
    # Sampling noise is symmetric and cancels; anything that survives the mean
    # is one path shifting the whole picture. Measures ~0.06 from rounding
    # asymmetry alone, and a 1-level brightness shift scores 0.94.
    assert abs(signed.mean()) < 0.1, (
        f"systematic brightness shift of {signed.mean():+.3f} levels")

    # Relocation, not resampling: 8 levels is far above the ~5-level worst case
    # the 1/32-px grid can produce on this scene, and far below what any real
    # geometry error costs on it.
    diff = np.abs(signed).max(axis=-1)
    assert (diff > 8).mean() < 0.0005, (
        f"{100 * (diff > 8).mean():.3f}% of pixels differ by >8 levels; "
        f"worst {diff.max():.0f}")


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
