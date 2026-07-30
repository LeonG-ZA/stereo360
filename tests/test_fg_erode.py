"""Tests for foreground erosion at depth edges in the warp stage."""

import numpy as np

from stereo360.warp import _erode_foreground, right_eye_from_disparity


def _railing_depth(h=256, w=512):
    """Far wall, four thin near railings, and one wide near object."""
    dn = np.full((h, w), 0.15, np.float32)
    for x in (150, 200, 250, 300):
        dn[60:200, x:x + 3] = 0.80
    dn[100:180, 380:460] = 0.55
    return dn


def test_erode_is_lipschitz_in_depth():
    """The dominant cause of thin structures changing shape between frames.

    Monocular depth carries per-frame noise around 1e-3 in normalized units.
    An operator that decides *binarily* from depth — a threshold, a connected
    component, a mask intersection — converts that noise into a finite,
    frame-varying change in which pixels it rewrites, and no amount of
    temporal depth smoothing downstream can undo it. So the requirement is
    quantitative, not cosmetic: the response must scale with the perturbation
    rather than saturate.

    The previous implementation labelled connected components of
    `dn >= local_max - 1e-6`; a 1e-6 tolerance against 1e-3 noise shatters
    those plateaus into speckle. Its edited region flipped ~30% of its own
    area at sigma=5e-4 and flipped by *the same amount* at sigma=2e-3 —
    saturation, i.e. a bifurcation rather than noise propagation.
    """
    rng = np.random.default_rng(0)
    dn0 = _railing_depth()
    base = dn0.copy()
    _erode_foreground(base, k=2)

    sigmas = (5e-4, 2e-3, 8e-3)
    peak, mean = [], []
    for sigma in sigmas:
        dn = dn0 + rng.normal(0, sigma, dn0.shape).astype(np.float32)
        _erode_foreground(dn, k=2)
        d = np.abs(dn - base)
        peak.append(float(d.max()))
        mean.append(float(d.mean()))

    # Bounded response. The constant is ~45 here, set by 1/thresh in the two
    # ramp terms; the assertion is on the order of magnitude, not the value.
    for sigma, p in zip(sigmas, peak):
        assert p < 60 * sigma, (
            f"response {p:.5f} to sigma={sigma} is out of proportion — "
            "the operator is making a discontinuous decision")
    # Proportionality is the real discriminator: a thresholding operator jumps
    # to a fixed response magnitude and then stays there (the old one flipped
    # the same ~1680 px at sigma=5e-4 and at 2e-3 alike), whereas a continuous
    # one tracks the input. Each 4x in sigma must give ~4x the response.
    for a, b in zip(mean, mean[1:]):
        assert 2.5 < b / a < 6.0, f"response does not track input: {mean}"


def test_erode_preserves_thin_structures():
    """A thin railing must keep its own depth; a wide object's near-side
    boundary must still be pulled back to the background."""
    dn0 = _railing_depth()
    dn = dn0.copy()
    _erode_foreground(dn, k=2)
    # Railing cores keep near depth (they would warp with the background,
    # losing their parallax, if erosion claimed them).
    for x in (150, 200, 250, 300):
        assert dn[130, x + 1] > 0.7, f"railing at x={x} was eroded away"
    # Wide-object interior untouched, its near-side boundary eroded.
    assert abs(float(dn[140, 420]) - 0.55) < 1e-6
    assert dn[140, 381] < 0.5


def test_erode_pulls_near_edge_to_background():
    # Near object (1.0) on the left half, background (0.1) on the right.
    dn = np.zeros((32, 64), dtype=np.float32)
    dn[:, :32] = 1.0
    dn[:, 32:] = 0.1
    _erode_foreground(dn, k=2)
    # Far interior untouched.
    assert dn[16, 8] == 1.0
    assert abs(float(dn[16, 56]) - 0.1) < 1e-6
    # Near-side boundary pixels pulled down toward the background level.
    assert dn[16, 31] < 0.5


def test_erode_no_edges_is_noop():
    dn = np.full((16, 16), 0.5, dtype=np.float32)
    _erode_foreground(dn, k=2)
    assert (dn == 0.5).all()


def test_warp_holes_fill_from_background():
    # Bright near band on dark background. Disocclusion holes must be filled
    # from the dark background, not smeared with the bright foreground:
    # background-biased inpainting recolors the near-side ring before Telea.
    h, w = 64, 128
    left = np.zeros((h, w, 3), dtype=np.uint8)
    left[:, 20:60] = 255  # bright near band
    disp = np.full((h, w), 0.05, dtype=np.float32)
    disp[:, 20:60] = 1.0

    # strength must be large enough that the disparity gap outruns the
    # fg_erode kernel -- erosion deliberately makes the near-side boundary
    # warp with the background, so a gap narrower than the kernel is closed
    # entirely and there is no hole left to test. (At strength=2 the gap is
    # ~1.3 px against a 7 px kernel; this only ever produced holes via the
    # untapered baseline's polar fold.)
    right, hole = right_eye_from_disparity(left, disp.copy(), strength=6.0,
                                           fg_erode=3)
    assert (hole > 0).any()
    frac_bright = (right[hole > 0].mean(axis=1) > 200).mean()
    assert frac_bright < 0.2
