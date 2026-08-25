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


@requires_gpu
def test_a_signed_detail_layer_survives_the_gpu_warp():
    """The GPU warp must not assume its input is an 8-bit image.

    `warp.right_eye_banded` sends two things through the warp: the blurred
    base, which is an ordinary uint8 frame, and the *detail* layer, which is
    the frame minus its own blur -- signed float centred on zero, with about
    half its samples negative. Rounding and clamping that to uint8 deletes
    every negative one, leaving a residual that can only ever add. The
    recombined frame then reads bright and overexposed exactly where it has
    texture, which is the whole point of the band split.

    The numpy path gets this right for free (`np.zeros_like(left_rgb)`); the
    torch path has to be asked. Measured with the clamp unconditional: the
    warped layer came back uint8, 0% negative, mean +9.2 instead of ~0.
    """
    img, dn = _scene(256, 512)
    dev = warp_torch.device_available()

    # The real production split, not a hand-rolled stand-in.
    bands = warp.detail_bands(img, dn, detail_sigma=3.0, depth_sigma=40.0)
    detail = bands.detail
    assert detail.dtype == np.float32
    assert (detail < 0).mean() > 0.4, "test scene has no darkening detail"

    out, _ = warp_torch.right_eye_from_disparity(
        detail, dn.copy(), 1.0, warp._BASELINE_SCALE, warp._MIN_INV_DEPTH,
        warp._VIS_RATIO, warp._CRACK_MARGIN, 2, 0.0, dev)

    assert out.dtype == np.float32, (
        f"signed input came back as {out.dtype}; the warp quantised it")
    assert (out < 0).mean() > 0.4, (
        f"only {100 * (out < 0).mean():.1f}% of the warped detail is negative; "
        "the darkening half was clamped away")
    assert abs(float(out.mean())) < 1.0, (
        f"warped detail has mean {float(out.mean()):+.2f}, not ~0; it is "
        "one-sided and will brighten whatever it is added to")


@requires_gpu
def test_the_banded_warp_does_not_brighten_on_the_gpu(monkeypatch):
    """The band split must not shift exposure, on either path.

    This is the symptom the check above catches at its source, asserted where
    a viewer would actually see it: the same frame through `right_eye_banded`
    on the CPU and on the GPU must land at the same brightness. With the
    detail layer clamped, the GPU came out 9.8 levels brighter than the CPU
    and 9.1 above the source frame.

    A bias rather than a per-pixel bound, for the reason given in
    `test_gpu_warp_matches_numpy`: the two paths sample on different sub-pixel
    grids, so they differ by a fraction of a level everywhere and only a
    systematic shift means anything.
    """
    img, dn = _scene(256, 512)

    def banded():
        warp._gpu_probe.clear()
        return warp.right_eye_banded(img, dn.copy(), 1.0, 3.0, 40.0,
                                     inpaint=False, normalize=False)[0]

    monkeypatch.setenv("STEREO360_GPU_WARP", "0")
    cpu = banded()
    monkeypatch.setenv("STEREO360_GPU_WARP", "1")
    gpu = banded()
    warp._gpu_probe.clear()

    bias = (cpu.astype(np.float64) - gpu.astype(np.float64)).mean()
    assert abs(bias) < 0.5, (
        f"GPU banded warp sits {-bias:+.2f} levels from the CPU one")
