"""Unit tests for DIBR warping and inpainting (no torch/model required)."""

import cv2
import numpy as np

from stereo360 import warp


def make_gradient_equirect(w=512, h=256) -> np.ndarray:
    """Horizontal gradient: makes horizontal parallax shifts easy to measure."""
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    img = np.stack([u * 255.0 / w, v * 255.0 / h, np.full_like(u, 128)],
                   axis=-1)
    return img.astype(np.uint8)


def test_flat_depth_minimal_holes():
    """Uniform depth => no true disocclusions; only sub-pixel splat gaps.

    Forward point-splatting of a smooth remap leaves scattered single-pixel
    gaps where the mapping locally expands; those are what inpainting fills.
    """
    img = make_gradient_equirect()
    disp = np.full(img.shape[:2], 0.5, dtype=np.float32)
    _, hole = warp.right_eye_from_disparity(img, disp, strength=1.0, inpaint=False)
    assert (hole > 0).mean() < 0.005


def test_near_objects_shift_more():
    """Closer regions must show larger horizontal parallax than farther ones
    in the forward (+Z) viewing direction."""
    w, h = 512, 256
    img = make_gradient_equirect(w, h)
    cx = w // 2  # lon = 0 (forward), perpendicular to the +X baseline =>
    # maximum angular parallax. (At lon = +/-90 the view is parallel to the
    # baseline and parallax is near zero by geometry.)

    # Disparity varies vertically (near top, far bottom) so the sample column
    # at lon=0 stays far from the disparity discontinuity.
    disp = np.full((h, w), 0.1, dtype=np.float32)
    disp[:h // 2, :] = 0.9
    right, _ = warp.right_eye_from_disparity(img, disp, strength=10.0,
                                             inpaint=False)

    # Horizontal gradient channel turns pixel shift into value difference.
    near_shift = abs(int(right[h // 4, cx, 0]) - int(img[h // 4, cx, 0]))
    far_shift = abs(int(right[3 * h // 4, cx, 0]) - int(img[3 * h // 4, cx, 0]))
    assert near_shift > 2, f"expected measurable near parallax, got {near_shift}"
    assert near_shift > far_shift * 2, (
        f"near shift {near_shift} should greatly exceed far shift {far_shift}")


def _near_block_disp(w, h):
    """Far background with a near block straddling lon=0.

    The block edges must sit where the baseline actually produces parallax.
    At lon=+/-90deg the view direction is parallel to the +X baseline, so the
    translation is purely radial and opens no disocclusion at all -- putting
    the edges there (as this test once did, at w//4 and 3w//4) measures
    nothing. It only appeared to work because the untapered baseline used to
    fold the polar caps and manufacture holes there; see warp._pole_taper.
    """
    disp = np.full((h, w), 0.05, dtype=np.float32)   # far background
    disp[:, w // 2 - w // 8:w // 2 + w // 8] = 0.95  # near block around lon=0
    return disp


def test_disocclusion_holes_at_depth_edge():
    """A near block in front of far background must create holes at its edge."""
    w, h = 512, 256
    img = make_gradient_equirect(w, h)
    _, hole = warp.right_eye_from_disparity(img, _near_block_disp(w, h),
                                            strength=3.0, inpaint=False)
    hole_frac = (hole > 0).mean()
    assert hole_frac > 0.001, "expected disocclusion holes at the depth edge"


def test_a_depth_edge_disoccludes_at_every_longitude():
    """The eye offset is sideways of the *view* direction, so a depth edge
    opens a hole wherever on the sphere it happens to sit.

    This test used to assert the opposite -- that an edge at lon +/-90deg opens
    no hole "because there the eye offset lies along the line of sight". That
    was a true description of an offset along a fixed world +X, and it was the
    bug: no hole there because there was no parallax there. The same geometry
    gave a full-size disparity of the *opposite sign* behind the viewer, so
    near read as far while occlusion and perspective still said near, and the
    far half of a scene was painful to look at.
    """
    w, h = 512, 256
    img = make_gradient_equirect(w, h)
    fracs = {}
    for name, centre in (("front", 0.5), ("side", 0.25), ("behind", 0.0)):
        disp = np.full((h, w), 0.05, dtype=np.float32)
        c = int(centre * w)
        cols = (np.arange(c - w // 8, c + w // 8) % w)
        disp[:, cols] = 0.95
        _, hole = warp.right_eye_from_disparity(img, disp, strength=3.0,
                                               inpaint=False)
        fracs[name] = float((hole > 0).mean())
    assert min(fracs.values()) > 0.001, fracs
    # Same edge, same depth step, so the same amount of disocclusion whichever
    # way it is facing.
    assert max(fracs.values()) < 1.6 * min(fracs.values()), fracs


def test_inpainting_fills_holes():
    """With inpainting enabled, hole regions should be filled (non-black) and
    the returned mask still reports where holes were."""
    w, h = 512, 256
    img = make_gradient_equirect(w, h)
    right, hole = warp.right_eye_from_disparity(img, _near_block_disp(w, h),
                                                strength=3.0, inpaint=True)
    assert (hole > 0).any()
    # Inpainted pixels should not remain pure black (gradient image has no
    # pure-black regions away from column 0).
    hole_region = right[hole > 0]
    assert (hole_region.sum(axis=1) > 0).mean() > 0.9


def test_strength_zero_is_identity():
    """strength=0 => right eye identical to left (no baseline)."""
    img = make_gradient_equirect()
    disp = np.random.rand(*img.shape[:2]).astype(np.float32)
    right, hole = warp.right_eye_from_disparity(img, disp, strength=0.0,
                                                inpaint=False)
    assert hole.sum() == 0
    assert np.array_equal(right, img)


def test_pole_baseline_taper_prevents_cap_collapse():
    """Nadir/zenith detail must survive the warp.

    A constant world-space translation gives roughly constant *angular*
    parallax b/lambda at every latitude, but equirect meridians converge, so
    the same angle costs 1/cos(lat) pixels of longitude — 58x the equator's
    shift at lat 89deg. And because the offset direction is fixed in world
    space while the meridians fan out, the cap does not translate, it folds:
    opposite longitudes are driven onto the same meridian, many-to-one, and
    the z-buffer keeps one of each colliding group. That is why detail near
    the bottom pole vanished rather than merely shifting.
    """
    w, h = 512, 256
    img = make_gradient_equirect(w, h)
    disp = np.full((h, w), 0.5, dtype=np.float32)

    right, hole = warp.right_eye_from_disparity(img, disp, strength=1.0,
                                                inpaint=False)
    cap = slice(int(h * 0.99), h)          # bottom ~1.8 deg of latitude

    # No holes torn open in the cap ...
    assert (hole[cap] > 0).mean() < 0.01
    # ... and the cap still carries the full range of source detail: the
    # gradient's distinct longitudes must survive rather than be squashed
    # onto one meridian.
    src_levels = np.unique(img[cap, :, 0]).size
    out_levels = np.unique(right[cap, :, 0]).size
    assert out_levels > 0.9 * src_levels, (
        f"pole detail collapsed: {src_levels} source levels -> {out_levels}")


def test_pole_taper_leaves_equator_parallax_intact():
    """The taper must not water down the stereo effect where it matters."""
    w, h = 512, 256
    img = make_gradient_equirect(w, h)
    disp = np.full((h, w), 0.9, dtype=np.float32)
    disp[:, :] = 0.9
    right, _ = warp.right_eye_from_disparity(img, disp, strength=10.0,
                                             inpaint=False, normalize=False)
    # At lon=0 (max parallax direction) the equator row must still shift.
    shift = abs(int(right[h // 2, w // 2, 0]) - int(img[h // 2, w // 2, 0]))
    assert shift > 2, f"equator parallax lost: {shift}"


def test_inpainting_never_overwrites_valid_foreground():
    """Only genuine holes may be repainted.

    fill_holes grows the inpaint mask over the whole foreground component
    touching a hole, so that Telea's fill boundary lands on background instead
    of smearing bright foreground inward. That expansion governs where the
    inpainter reads, not what it is allowed to destroy — the expanded pixels
    are correctly warped image content. Because the mask grows by connected
    component, a single near-camera object (a hand holding the rig at the
    nadir) reaches every nearby hole and once swallowed 17% of the frame and
    89% of the bottom cap, against 0.02% genuine holes.
    """
    w, h = 512, 256
    img = make_gradient_equirect(w, h)
    # Near foreground with distinctive texture, against far background.
    img[80:180, 200:320] = np.random.default_rng(0).integers(
        0, 255, (100, 120, 3), dtype=np.uint8)
    disp = np.full((h, w), 0.05, dtype=np.float32)
    disp[80:180, 200:320] = 0.95

    unfilled, hole = warp.right_eye_from_disparity(
        img, disp.copy(), strength=3.0, inpaint=False)
    filled, _ = warp.right_eye_from_disparity(
        img, disp.copy(), strength=3.0, inpaint=True)

    assert (hole > 0).any(), "test needs real holes to be meaningful"
    # Every non-hole pixel must survive the inpaint stage byte for byte,
    # allowing only the small deliberate dilation around each hole.
    protected = cv2.dilate(hole, np.ones((7, 7), np.uint8)) == 0
    assert np.array_equal(filled[protected], unfilled[protected]), (
        f"{(filled[protected] != unfilled[protected]).any(axis=-1).sum()} "
        "valid pixels were overwritten by inpainting")


def test_cropped_inpaint_matches_whole_frame():
    """Cropping is an optimisation, not an approximation.

    Telea's cost follows the image it is given, not the mask. fill_holes hands
    it a component-expanded mask (18% of the frame at 8K) purely to put the
    fill boundary on background, then keeps only the genuine holes (0.4%) --
    so ~99% of the work was discarded. Crops are placed on the holes and
    painted with the expanded mask, which must leave the kept pixels bitwise
    unchanged.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (256, 512, 3), dtype=np.uint8)
    paint = np.zeros((256, 512), np.uint8)
    paint[60:200, 100:300] = 255          # a big "foreground component"
    seed = np.zeros((256, 512), np.uint8)
    seed[120:130, 150:160] = 255          # the actual hole inside it
    seed[80:84, 260:264] = 255            # and a second, separate one
    paint = np.maximum(paint, seed)

    cropped = warp._inpaint_telea_cropped(img, paint, seed > 0, 3.0)
    whole = cv2.inpaint(img, paint, 3.0, cv2.INPAINT_TELEA)

    keep = seed > 0
    assert np.array_equal(cropped[keep], whole[keep]), (
        f"{(cropped[keep] != whole[keep]).any(axis=-1).sum()} kept pixels "
        "differ from the whole-frame result")


def test_splat_erosion_matches_four_way_splat():
    """The 2x2 splat footprint is built by erosion, and must stay exact.

    Splatting each sample to all four integer neighbours and taking the min
    equals splatting to the floor position alone then taking, per output pixel,
    the min over the 2x2 block ending at it -- a separable erosion, two shifted
    minimums instead of four scattered ones. `np.minimum.at` is unbuffered and
    cannot be threaded (arbitrary targets, shared buffer), so it set the floor
    on pass 1.

    The subtlety is the latitude clamp. The four-way splat clamps *after*
    adding dv, so a sample at v0 = -1 reaches row 0 and stops. Clamping to row
    0 before eroding would let it carry into row 1 as well, so the buffer
    carries a guard row at each pole.
    """
    from stereo360.projection import (equirect_rows_to_dir,
                                      _dir_to_equirect_uv)
    rng = np.random.default_rng(0)
    for h, w, strength in ((64, 128, 2.0), (96, 192, 8.0), (120, 240, 0.5)):
        dn = rng.random((h, w)).astype(np.float32)
        b = warp._BASELINE_SCALE * strength
        lam = 1.0 / (dn + warp._MIN_INV_DEPTH)
        d = equirect_rows_to_dir(0, h, w, h)
        p = lam[..., None] * d
        p[..., 0], p[..., 2] = warp._eye_offset(lam, d, -b)
        nrm = np.linalg.norm(p, axis=-1)
        tu, tv = _dir_to_equirect_uv(p / nrm[..., None], w, h)
        lam_r = nrm.astype(np.float32).ravel()
        u0 = np.floor(tu).astype(np.int64).ravel()
        v0 = np.floor(tv).astype(np.int64).ravel()

        four = np.full(h * w, np.inf, np.float32)
        for du in (0, 1):
            for dv in (0, 1):
                np.minimum.at(four, np.clip(v0 + dv, 0, h - 1) * w
                              + ((u0 + du) % w), lam_r)

        zp = np.full((h + 2) * w, np.inf, np.float32)
        np.minimum.at(zp, (np.clip(v0, -1, h - 1) + 1) * w + (u0 % w), lam_r)
        zpad = zp.reshape(h + 2, w)
        sh = np.empty_like(zpad)
        sh[:, 0] = zpad[:, -1]
        sh[:, 1:] = zpad[:, :-1]
        np.minimum(zpad, sh, out=zpad)
        sh[0] = zpad[0]
        sh[1:] = zpad[:-1]
        np.minimum(zpad, sh, out=zpad)

        assert np.array_equal(four.reshape(h, w), zpad[1:h + 1]), (
            f"erosion splat differs from the four-way splat at {h}x{w}")


def test_warp_is_invariant_to_band_size_and_thread_count():
    """Banding and threading are memory/speed controls, never semantics.

    Both the pass-1 scatter and the separable erosion that completes the 2x2
    splat footprint run band by band against a reused band-sized scratch
    buffer, so a full frame is never duplicated. The vertical erosion pass is
    the delicate one: its bands are *not* independent (row y reads row y-1),
    so they run bottom-up with the source rows copied first.

    The default test resolution fits in a single band, which hides exactly
    that class of bug -- an off-by-one in the band range once left every row
    but the last untouched, and only one unrelated assertion noticed. So this
    sweeps band sizes that do not divide the height, plus serial and threaded
    execution, and demands bitwise identical output.
    """
    rng = np.random.default_rng(0)
    h, w = 300, 600                     # 300 is indivisible by most bands
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    # Sharp depth discontinuities, so pass 1 really does leave holes and the
    # erosion has something to close. A smoothly varying depth map produces
    # none, and would let a broken band loop pass unnoticed.
    dn = np.full((h, w), 0.05, np.float32)
    dn[:, w // 2 - 60:w // 2 + 60] = 0.95        # near block around lon=0
    dn[40:260, w // 2 + 120:w // 2 + 124] = 0.80  # a thin structure too

    original = warp._WORKERS
    try:
        warp._WORKERS = 1
        ref, ref_hole = warp.right_eye_from_disparity(
            img, dn.copy(), strength=3.0, inpaint=False, normalize=False,
            chunk_rows=h + 8)            # single band, whole frame
        assert (ref_hole > 0).any(), "test needs real holes to be meaningful"

        for chunk_rows in (256, 128, 100, 64, 37, 7):
            for workers in (1, 4):
                warp._WORKERS = workers
                got, hole = warp.right_eye_from_disparity(
                    img, dn.copy(), strength=3.0, inpaint=False,
                    normalize=False, chunk_rows=chunk_rows)
                assert np.array_equal(got, ref), (
                    f"image differs at chunk_rows={chunk_rows}, "
                    f"workers={workers}")
                assert np.array_equal(hole, ref_hole), (
                    f"hole mask differs at chunk_rows={chunk_rows}, "
                    f"workers={workers}")
    finally:
        warp._WORKERS = original
