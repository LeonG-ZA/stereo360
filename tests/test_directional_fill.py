"""Tests for background-directional disocclusion fill and baseline splitting."""

import numpy as np

from stereo360 import pipeline, warp


def _textured(h=256, w=512):
    """Background with strong horizontal texture, plus a near bar."""
    x = np.arange(w)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = (100 + 80 * np.sin(x / 5.0)).astype(np.uint8)
    img[:, :, 1] = (100 + 80 * np.sin(x / 11.0)).astype(np.uint8)
    dn = np.full((h, w), 0.10, np.float32)
    img[60:200, w // 2:w // 2 + 4] = (255, 255, 255)
    dn[60:200, w // 2:w // 2 + 4] = 0.90
    return img, dn


def test_directional_fill_leaves_valid_pixels_untouched():
    img, dn = _textured()
    right, hole = warp.right_eye_from_disparity(
        img, dn.copy(), strength=4.0, inpaint=False, normalize=False)
    assert (hole > 0).any()

    filled, done = warp._directional_fill(right, hole, baseline_sign=1.0)
    outside = hole == 0
    assert np.array_equal(filled[outside], right[outside])
    assert not (done & outside).any(), "marked a non-hole pixel as filled"


def test_directional_fill_is_deterministic():
    """The whole point: a stable input must give a stable fill.

    Telea re-solves a diffusion each frame, so identical inputs are the only
    case where it agrees with itself. The directional fill is a pure function
    of (image, hole mask, geometry), which is what makes it stop crawling.
    """
    img, dn = _textured()
    right, hole = warp.right_eye_from_disparity(
        img, dn.copy(), strength=4.0, inpaint=False, normalize=False)
    a, _ = warp._directional_fill(right, hole, 1.0)
    b, _ = warp._directional_fill(right.copy(), hole.copy(), 1.0)
    assert np.array_equal(a, b)


def test_directional_fill_draws_from_the_background_not_the_foreground():
    """Holes must not be filled with the bright near bar.

    The bar is the only bright thing in the scene, so any fill that pulls from
    the foreground side shows up immediately as bright hole pixels.
    """
    img, dn = _textured()
    right, hole = warp.right_eye_from_disparity(
        img, dn.copy(), strength=4.0, inpaint=False, normalize=False)
    filled, done = warp._directional_fill(right, hole, 1.0)
    got = filled[done]
    assert got.size > 0
    bright = (got.mean(axis=1) > 200).mean()
    assert bright < 0.05, f"{bright:.1%} of the fill came from the foreground"


def test_split_baseline_delivers_the_geometric_disparity():
    """Splitting must keep the 3D effect: total disparity is what fuses.

    The reference is the geometry (b / lambda converted to pixels), NOT the
    single-eye pipeline. Measured against a 4-px bar, the single-eye path
    comes out ~15% short of the ideal because the splat footprint and the
    sub-pixel resample bias the rendered position of a structure only a few
    pixels wide. That bias is common to both eyes, so it cancels in a
    symmetric pair -- the split result is the more faithful of the two, which
    is why asserting "split == single" would enshrine the error.
    """
    img, dn = _textured()
    strength = 4.0
    lam = 1.0 / (0.90 + warp._MIN_INV_DEPTH)
    ideal = -(strength * warp._BASELINE_SCALE) / lam / (2 * np.pi) * img.shape[1]

    left, right = pipeline.stereo_pair(
        img, dn.copy(), strength, split=True, inpaint=False, normalize=False)

    def bar_x(im, row=130):
        line = (im[row, :, 0] > 200).astype(np.float64)
        assert line.sum() > 0
        return float((line * np.arange(im.shape[1])).sum() / line.sum())

    got = bar_x(right) - bar_x(left)
    assert abs(got - ideal) < 0.15 * abs(ideal), (
        f"disparity {got:.2f} px is not the geometric {ideal:.2f} px")


def test_split_baseline_shrinks_holes_per_eye():
    """Each eye of a split pair holes less than one eye carrying it all.

    Deliberately not asserting the size of the reduction. On real footage the
    hole area measured roughly cubic in warp distance (0.002% of an 8K frame
    at strength 0.3 against 0.071% at 1.0), but that comes from more and more
    depth edges crossing the one-pixel threshold at which a hole survives the
    splat at all. A synthetic scene with a single isolated edge cannot show
    it: there the hole area saturates once the structure has cleared its own
    width. So the scene-dependent factor stays in the docs, and the test
    asserts only the direction, which is geometry.
    """
    img, dn = _textured()
    _, full = warp.right_eye_from_disparity(
        img, dn.copy(), 4.0, inpaint=False, normalize=False)
    _, half = warp.right_eye_from_disparity(
        img, dn.copy(), 2.0, inpaint=False, normalize=False)
    assert (half > 0).sum() < (full > 0).sum()


def test_split_baseline_holes_are_monocular_once_resolvable():
    """The fusion benefit: the eyes must not hole in the same place.

    This holds once the per-eye displacement exceeds the occluding
    structure's own width -- below that the two holes land on the same few
    columns and nothing is gained. A 4-px bar needs roughly +-6 here; on real
    8K footage at +-0.5 the displacement is far larger than any thin
    structure, and the holes measured 100% disjoint (1 shared pixel in the
    whole frame).
    """
    img, dn = _textured()
    _, hl = warp.right_eye_from_disparity(
        img, dn.copy(), -6.0, inpaint=False, normalize=False, fg_erode=0)
    _, hr = warp.right_eye_from_disparity(
        img, dn.copy(), 6.0, inpaint=False, normalize=False, fg_erode=0)
    both = ((hl > 0) & (hr > 0)).sum()
    either = ((hl > 0) | (hr > 0)).sum()
    assert either > 0
    assert both / either < 0.05, (
        f"{100 * both / either:.0f}% of holes are in both eyes; fusion cannot "
        "hide those")


def test_gradient_limit_removes_holes_and_keeps_depth_range():
    """Clamping the depth gradient prevents holes instead of filling them.

    Holes are exactly where the warp x -> x + s(x) stops being injective,
    i.e. where 1 + ds/dx <= 0. The shift is proportional to depth, so bounding
    the depth gradient bounds ds/dx and the fold cannot occur. The clamp only
    lowers values, so depth ORDERING and the overall depth RANGE survive --
    which is what separates it from simply lowering --strength.
    """
    img, dn = _textured()
    strength = 6.0
    _, before = warp.right_eye_from_disparity(
        img, dn.copy(), strength, inpaint=False, normalize=False)
    assert (before > 0).any()

    _, after = warp.right_eye_from_disparity(
        img, dn.copy(), strength, inpaint=False, normalize=False,
        gradient_limit=0.5)
    assert (after > 0).sum() < 0.25 * (before > 0).sum(), (
        f"holes {(before > 0).sum()} -> {(after > 0).sum()}")

    clamped = warp.limit_disparity_gradient(dn.copy(), 0.01)
    assert clamped.max() <= dn.max() + 1e-6, "clamp must not move anything nearer"
    assert clamped.min() >= dn.min() - 1e-6


def test_gradient_limit_actually_bounds_the_gradient():
    rng = np.random.default_rng(0)
    dn = rng.random((32, 256)).astype(np.float32)
    for slope in (0.05, 0.01, 0.002):
        out = warp.limit_disparity_gradient(dn.copy(), slope)
        grad = np.abs(np.diff(out, axis=1)).max()
        assert grad <= slope + 1e-5, f"gradient {grad} exceeds limit {slope}"


def test_vertical_clamp_catches_near_horizontal_depth_edges():
    """A horizontal-only clamp misses edges whose cliff runs vertically.

    A handrail crossing the frame has depth that changes fast going up/down
    and slowly going left/right, so clamping rows alone leaves it untouched.
    The warp is mostly but not purely horizontal -- the pole taper shifts
    latitude too -- so that cliff still folds, as a thin dashed line hugging
    the rail. On a real 8K frame this was the entire residual: 111 hole px
    with rows-only clamping, 0 once the vertical direction is clamped as well.
    """
    h, w = 256, 512
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = 120
    # A near band that is wide in x and thin in y: its depth cliff is vertical.
    dn = np.full((h, w), 0.10, np.float32)
    dn[120:126, :] = 0.90
    img[120:126, :] = (255, 255, 255)

    slope = 0.004
    rows_only = warp.limit_disparity_gradient(dn.copy(), slope)
    both = warp.limit_disparity_gradient(dn.copy(), slope, slope * 2.0)

    gy_rows = np.abs(np.diff(rows_only, axis=0)).max()
    gy_both = np.abs(np.diff(both, axis=0)).max()
    assert gy_rows > 0.5, "rows-only clamp should leave the vertical cliff intact"
    assert gy_both <= slope * 2.0 + 1e-5, f"vertical gradient {gy_both} not bounded"
    # and it still only ever lowers values
    assert both.max() <= dn.max() + 1e-6
