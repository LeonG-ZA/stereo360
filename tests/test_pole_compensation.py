"""Cancelling the depth distortion ODS imposes away from the equator.

The artifact: omnidirectional stereo separates the eyes along the local
horizontal, so the angular disparity it delivers for a point at latitude
`lat` falls short of that point's distance by `cos(lat)`. A viewer reading
it back places everything at `R / cos(lat)`, which turns a level floor into
a funnel -- correct at the horizon, collapsing underfoot.

Geometry decides what right looks like here, so the fixtures are exact: a
level plane has a closed form, and so does the disparity the warp produces
for it.
"""

import numpy as np
import pytest

from stereo360 import projection, warp

H, W = 256, 512


def _lat(height=H):
    return (np.linspace(np.pi / 2, -np.pi / 2, height, endpoint=False)
            - (np.pi / height) / 2)


def test_off_by_default_and_a_no_op_when_off():
    """Every existing render must be untouched unless the flag is passed."""
    assert projection.POLE_COMPENSATION == 1.0
    d = np.random.default_rng(0).random((H, W)).astype(np.float32) + 0.1
    for cap in (None, 0.0, 1.0):
        assert projection.apply_pole_compensation(d, cap) is d, (
            f"cap={cap} should return the map untouched")


def test_the_gain_is_one_over_cos_latitude_up_to_the_cap():
    gain = projection.pole_compensation_gain(H, cap=3.0)[:, 0]
    lat = _lat()
    below = lat < np.radians(-8.0)          # past the ramp
    want = np.minimum(1.0 / np.cos(lat[below]), 3.0)
    assert np.allclose(gain[below], want, rtol=1e-5)


def test_nothing_above_the_horizon_moves():
    """The same error exists up there; correcting it is a different change."""
    gain = projection.pole_compensation_gain(H, cap=5.0)[:, 0]
    assert np.allclose(gain[_lat() > 0], 1.0)


def test_the_cap_is_what_bounds_it():
    for cap in (2.0, 3.0, 5.0):
        gain = projection.pole_compensation_gain(H, cap=cap)[:, 0]
        assert gain.max() <= cap + 1e-6
        assert np.isfinite(gain).all(), "the pole term must not reach infinity"


def _apparent_depth(lat_deg, cam_h, cap):
    """Depth a viewer reads back for level ground, through the shipped warp."""
    lat = np.radians(-lat_deg)
    R = cam_h / np.sin(-lat)
    gain = min(1.0 / np.cos(lat), cap)
    R_fed = R / gain
    b = warp._BASELINE_SCALE
    d = np.array([[0.0, np.sin(lat), np.cos(lat)]])
    d = d / np.linalg.norm(d)
    px, pz = warp._eye_offset(np.array([R_fed]), d, b)
    dlon = np.arctan2(px[0], pz[0]) - np.arctan2(d[0, 0] * R_fed, d[0, 2] * R_fed)
    # Meridians converge: a longitude shift subtends cos(lat) of itself.
    ang = abs(dlon * np.cos(lat))
    return (b / np.tan(ang)) * np.sin(-lat)


@pytest.mark.parametrize("lat_deg,ratio", [
    (30, 1.15), (45, 1.41), (60, 2.00), (75, 3.86)])
def test_uncompensated_ods_turns_flat_ground_into_a_funnel(lat_deg, ratio):
    """The artifact itself, so the fix below is measured against a real number."""
    got = _apparent_depth(lat_deg, 1.6, cap=1.0) / 1.6
    assert got == pytest.approx(ratio, abs=0.01)
    assert got == pytest.approx(1.0 / np.cos(np.radians(lat_deg)), rel=1e-3)


@pytest.mark.parametrize("lat_deg", [10, 20, 30, 45, 60, 70])
def test_compensation_puts_the_floor_back_on_its_plane(lat_deg):
    """Level to 70 degrees below the horizon at cap 3, which is the claim."""
    assert _apparent_depth(lat_deg, 1.6, cap=3.0) == pytest.approx(1.6, abs=0.01)


def test_it_scales_rather_than_replaces():
    """A bollard must keep its depth relative to the ground it stands on.

    Replacing the lower hemisphere with a fitted plane flattens the floor
    better and takes every upright object with it, which is why this scales.
    """
    lat = _lat()
    ground = np.tile(np.maximum(-np.sin(lat), 0.05)[:, None], (1, W))
    scene = ground.copy()
    scene[H // 2 + 20:H // 2 + 40, 100:110] *= 3.0      # something standing up
    out = projection.apply_pole_compensation(scene.astype(np.float32), 3.0)
    ratio = out[H // 2 + 20:H // 2 + 40, 100:110] / out[H // 2 + 20:H // 2 + 40, 200:210]
    assert np.allclose(ratio, 3.0, rtol=1e-4), (
        "the object's depth relative to its own ground row changed")
