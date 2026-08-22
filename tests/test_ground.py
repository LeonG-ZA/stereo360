"""Pulling the ground onto its plane.

The artifact this exists for: on a road, the estimated ground bends by 14% of
its own depth between the camera's feet and one camera height out, which the
angular correction only partly straightens because the bend is not symmetric
about a face axis.

Geometry decides what right looks like. Inverse depth over any plane is exactly
linear in the ray direction, so a synthetic plane is an exact fixture rather
than an approximation, and anything the code does to one is measurable to
float precision.
"""

import numpy as np
import pytest

from stereo360 import ground, projection

W, H = 256, 128


def plane_map(q, w=W, h=H):
    """An equirect inverse-depth map that is exactly the plane `q`."""
    out = np.zeros((h, w), np.float32)
    for y in range(h):
        d = projection.equirect_rows_to_dir(y, y + 1, w, h)[0]
        out[y] = d @ np.asarray(q, np.float64)
    # Above the plane's horizon the expression goes negative; the sky is not
    # part of any of this, so give it something small and positive.
    return np.maximum(out, 0.05).astype(np.float32)


LEVEL = np.array([0.0, -1.0, 0.0])          # flat ground, camera height 1


def test_a_flat_ground_is_recovered_exactly():
    fit = ground.fit_plane(plane_map(LEVEL))
    assert fit.ok, fit.why
    assert fit.tilt_deg == pytest.approx(0.0, abs=0.5)
    assert fit.height == pytest.approx(1.0, rel=0.05)
    assert fit.residual < 0.02


def test_a_tilted_ground_is_recovered_as_tilted():
    q = np.array([0.12, -1.0, 0.0])          # about 7 degrees off level
    fit = ground.fit_plane(plane_map(q))
    assert fit.ok, fit.why
    assert fit.tilt_deg == pytest.approx(6.8, abs=1.5)


def test_level_holds_the_normal_up_even_when_the_data_tilts():
    """The free fit will absorb depth error as tilt -- on a real road it
    claimed 6.7 degrees where the picture's own horizon said about one -- so
    there has to be a way to say the ground is level and mean it."""
    q = np.array([0.12, -1.0, 0.0])
    assert ground.fit_plane(plane_map(q), level=True).tilt_deg == \
        pytest.approx(0.0, abs=1e-6)


def test_flattening_a_plane_changes_nothing():
    disp = plane_map(LEVEL)
    out = ground.flatten(disp, 1.0)
    assert np.abs(out - disp).max() < 0.01


def test_a_bend_is_removed():
    """A smooth bow over the whole ground is what this is for."""
    disp = plane_map(LEVEL)
    d = np.stack([projection.equirect_rows_to_dir(y, y + 1, W, H)[0]
                  for y in range(H)])
    down = np.clip(-d[..., 1], 0, 1)
    bent = (disp * (1.0 + 0.25 * (1.0 - down) ** 2)).astype(np.float32)

    ground_only = down > 0.35
    before = np.abs(bent - disp)[ground_only].mean()
    after = np.abs(ground.flatten(bent, 1.0) - disp)[ground_only].mean()
    assert after < before / 3, (before, after)


def test_a_kerb_sized_step_survives():
    """Local detail must not be flattened along with the bend: the low-pass is
    the whole reason this does not simply snap everything onto the plane."""
    disp = plane_map(LEVEL)
    bump = disp.copy()
    # Narrow against the smoothing kernel, which is what a kerb is at 8K:
    # 15 degrees of arc is 160 px on a 3840-wide frame and a kerb is a few.
    bump[H - 12:H - 4, 104:107] *= 1.5
    out = ground.flatten(bump, 1.0)
    kept = (out[H - 12:H - 4, 104:107] - out[H - 12:H - 4, 130:133]).mean()
    original = (bump[H - 12:H - 4, 104:107]
                - bump[H - 12:H - 4, 130:133]).mean()
    assert kept > 0.5 * original, (original, kept)


def test_it_refuses_when_there_is_no_ground():
    rng = np.random.default_rng(0)
    noise = rng.uniform(0.2, 2.0, (H, W)).astype(np.float32)
    fit = ground.fit_plane(noise)
    assert not fit.ok
    assert fit.why


def test_a_refusal_leaves_the_depth_alone():
    rng = np.random.default_rng(1)
    noise = rng.uniform(0.2, 2.0, (H, W)).astype(np.float32)
    assert np.array_equal(ground.flatten(noise, 1.0), noise)


def test_zero_strength_is_the_identity():
    disp = plane_map(LEVEL)
    assert ground.flatten(disp, 0.0) is disp


def test_the_sky_is_never_touched():
    """The correction fades out before the horizon, so nothing above it moves
    and no seam lands at a fixed latitude."""
    disp = plane_map(LEVEL)
    bent = (disp * 1.2).astype(np.float32)
    out = ground.flatten(bent, 1.0)
    top = slice(0, H // 2 - 4)
    assert np.array_equal(out[top], bent[top])
