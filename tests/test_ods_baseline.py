"""The stereo baseline must follow the viewing direction, not a fixed axis.

Reported from a Quest 3: two regions were painful to look at -- the right arm
of the bench, and the gaps between the trunks of the tree behind it -- while
the same scene straight ahead was fine. Both sat behind the viewer (lon +142
and +172 degrees), and both are high-contrast occluding edges.

The cause was the right eye being offset along a fixed world +X. For a point at
distance L in direction (lon, lat) that gives a longitude shift of
-b*cos(lon)/L: full size at the equirect centre, ZERO at lon +/-90 degrees, and
full size with the OPPOSITE SIGN behind the viewer. Opposite sign is
pseudoscopic -- near reads as far -- while occlusion, perspective and texture
all still say near, so the eyes were handed a depth contradicting every other
cue, worst exactly at an occluding edge.

Measured before the fix, at 3840 wide with a full shift of 10.08 px:
    lon    0 deg  ->  -10.08 px   (1.00x)
    lon   90 deg  ->   +0.00 px   (0.00x)
    lon  180 deg  ->  +10.08 px  (-1.00x)
"""

import numpy as np
import pytest

from stereo360 import warp
from stereo360.projection import equirect_rows_to_dir

LONGITUDES = [-180, -135, -90, -45, 0, 45, 90, 135]
LATITUDES = [-60, -30, 0, 30, 60]


def _shift(lon_deg, lat_deg, dn=0.5, baseline=None, w=3840, h=1920):
    """Longitude shift in pixels that the warp geometry gives one point."""
    b = warp._BASELINE_SCALE if baseline is None else baseline
    lam = 1.0 / (dn + warp._MIN_INV_DEPTH)
    lon, lat = np.radians(float(lon_deg)), np.radians(float(lat_deg))
    cl = np.cos(lat)
    d = np.array([[[cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)]]])
    px, pz = warp._eye_offset(lam, d, -b)
    moved = (np.arctan2(px, pz).item() - lon + np.pi) % (2 * np.pi) - np.pi
    return moved * w / (2 * np.pi)


def _vertical_shift(lon_deg, lat_deg, dn=0.5, w=3840, h=1920):
    b = warp._BASELINE_SCALE
    lam = 1.0 / (dn + warp._MIN_INV_DEPTH)
    lon, lat = np.radians(float(lon_deg)), np.radians(float(lat_deg))
    cl = np.cos(lat)
    d = np.array([[[cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)]]])
    px, pz = warp._eye_offset(lam, d, -b)
    py = lam * d[..., 1]
    norm = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    lat2 = np.arcsin(np.clip(py / norm, -1, 1)).item()
    return (lat2 - lat) * (h / np.pi)


@pytest.mark.parametrize("lon", LONGITUDES)
def test_disparity_is_the_same_size_at_every_longitude(lon):
    front = _shift(0, 0)
    here = _shift(lon, 0)
    assert here == pytest.approx(front, rel=1e-6), (
        f"lon {lon}: {here:.3f} px against {front:.3f} px straight ahead")


@pytest.mark.parametrize("lon", LONGITUDES)
def test_disparity_never_reverses_sign(lon):
    """The failure that hurt. A sign flip is pseudoscopic: stereo says far
    where occlusion says near, and the two cannot be reconciled."""
    assert np.sign(_shift(lon, 0)) == np.sign(_shift(0, 0)), lon


@pytest.mark.parametrize("lon", LONGITUDES)
def test_no_longitude_is_a_dead_zone(lon):
    """At lon +/-90 the old geometry had no parallax at all -- flat, not just
    wrong."""
    assert abs(_shift(lon, 0)) > 0.9 * abs(_shift(0, 0)), lon


@pytest.mark.parametrize("lat", LATITUDES)
def test_disparity_holds_across_latitude_too(lat):
    """The cos(latitude) taper is still in there, and still has to leave the
    longitude shift alone: equirect meridians converge, so an untapered offset
    costs 1/cos(lat) pixels of longitude and folds the polar caps."""
    assert _shift(0, lat) == pytest.approx(_shift(0, 0), rel=1e-6)


@pytest.mark.parametrize("lat", LATITUDES)
def test_vertical_disparity_stays_negligible(lat):
    """Vertical disparity cannot be fused at all, so it has to stay far below
    a pixel. It is second order in baseline/distance."""
    for lon in LONGITUDES:
        assert abs(_vertical_shift(lon, lat)) < 0.1, (lon, lat)


def test_the_world_offset_still_tapers_to_nothing_at_the_poles():
    """The taper predates this fix and has to survive it.

    Without it the polar cap does not translate but *folds*: opposite
    longitudes are driven onto one meridian, many-to-one, and the z-buffer
    keeps one of each colliding group, so nadir detail vanishes rather than
    shifting. The offset is `baseline * cos(lat)` in world units, which the
    convergence of the meridians turns back into a uniform longitude shift --
    which is why the two tests above hold at every latitude as well.
    """
    b = warp._BASELINE_SCALE
    mags = []
    for lat in (0, 45, 80, 89, 89.99):
        la = np.radians(float(lat))
        d = np.array([[[0.0, np.sin(la), np.cos(la)]]])
        px, pz = warp._eye_offset(1.0, d, -b)
        # The offset is what the transform added beyond dist * direction.
        mags.append(float(np.hypot(px - d[..., 0], pz - d[..., 2]).item()))
    assert mags[0] == pytest.approx(b, rel=1e-6)
    assert mags == sorted(mags, reverse=True), mags
    assert mags[-1] < 1e-4, mags


def test_the_polar_cap_shifts_rather_than_folding():
    """The consequence of that taper: every row stays a bijection in
    longitude, so no two source columns land on one target column."""
    w, h = 512, 256
    b = warp._BASELINE_SCALE * 3.0
    d = equirect_rows_to_dir(0, h, w, h)
    px, pz = warp._eye_offset(1.8, d, -b)
    u = np.arctan2(px, pz)
    for row in (0, 1, h // 8, h // 2, h - 1):
        # Unwrapped, the mapping must be strictly increasing across the row.
        step = np.diff(np.unwrap(u[row]))
        assert (step > 0).all(), f"row {row} folds"


def test_the_two_passes_agree_on_longitude():
    """Pass 1 moves a point to the right eye; pass 2 takes the distance the
    z-buffer resolved and brings it back. They do not invert exactly -- pass 2
    only knows the *new* distance, which is `hypot(dist, baseline)` -- but the
    residual is third order in baseline/distance and has to stay far below a
    pixel, or the sampled colour comes from the wrong place."""
    w, h = 512, 256
    b = warp._BASELINE_SCALE * 3.0
    d = equirect_rows_to_dir(0, h, w, h)
    dist = np.full((h, w), 1.8, np.float32)

    px, pz = warp._eye_offset(dist, d, -b)
    forward = np.stack([px, dist * d[..., 1], pz], axis=-1)
    fnorm = np.linalg.norm(forward, axis=-1, keepdims=True)
    back_px, back_pz = warp._eye_offset(fnorm[..., 0], forward / fnorm, b)

    err = np.abs(np.arctan2(back_px, back_pz)
                 - np.arctan2(d[..., 0], d[..., 2]))
    err = np.minimum(err, 2 * np.pi - err) * w / (2 * np.pi)
    assert err.max() < 0.02, f"{err.max():.4f} px of round-trip error"


def test_measured_disparity_is_uniform_in_a_real_warp():
    """End to end through `right_eye_from_disparity`, no analysis of the
    geometry: warp a texture at one constant depth and cross-correlate the two
    eyes at several longitudes."""
    w, h = 1920, 960
    rng = np.random.default_rng(0)
    import cv2

    tex = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    left = cv2.GaussianBlur(cv2.resize(tex, (w, h),
                                       interpolation=cv2.INTER_LANCZOS4),
                            (5, 5), 0)
    dn = np.full((h, w), 0.5, np.float32)
    right, _ = warp.right_eye_from_disparity(left, dn.copy(), strength=1.0,
                                             inpaint=False, normalize=False)
    lg = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)[h // 2 - 16:h // 2 + 16]
    rg = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)[h // 2 - 16:h // 2 + 16]

    shifts = []
    for lon in LONGITUDES:
        cx = int((lon / 360.0) * w + w / 2) % w
        cols = (np.arange(cx - 64, cx + 64) % w)
        a = lg[:, cols].astype(np.float32)
        a = a - a.mean()
        best, peak = 0, -2.0
        for s in range(-20, 21):
            bcol = rg[:, (cols + s) % w].astype(np.float32)
            bcol = bcol - bcol.mean()
            v = float((a * bcol).sum()
                      / max(np.linalg.norm(a) * np.linalg.norm(bcol), 1e-9))
            if v > peak:
                best, peak = s, v
        assert peak > 0.9, f"lon {lon}: eyes do not even correlate ({peak:.2f})"
        shifts.append(best)

    assert len(set(shifts)) == 1, dict(zip(LONGITUDES, shifts))
    assert shifts[0] < 0, shifts


def test_hole_fill_pulls_background_from_one_side_everywhere():
    """The fill direction followed from the old geometry too: it read
    `cos(lon) * baseline_sign`, which reversed across half the sphere and went
    to zero near lon +/-90 -- so where disocclusions are widest the background
    was continued in from the foreground's side."""
    import inspect

    src = inspect.getsource(warp._directional_fill)
    assert "np.cos(lon)" not in src
    assert "from_right = bool(baseline_sign > 0)" in src
