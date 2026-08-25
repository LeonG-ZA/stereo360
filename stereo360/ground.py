"""Pull the ground onto one plane, where it agrees that it is one.

The angular correction straightens the *radially symmetric* part of the depth
model's error -- the dome, where flat ground rises the further it gets from the
face axis. Measured by the median over every direction it does that well: on a
road it took the error at one camera height out from +0.200 to +0.008.

That median hides the rest. Broken out by direction the same road is not domed,
it is *tilted*: at one camera height, +0.13 behind and -0.10 in front, a spread
of 0.24 camera heights that the correction leaves untouched because it is not
a function of angle off the face axis. In a headset that reads as the road
still bulging.

Geometry pins what the answer should be. For any plane, inverse depth along a
ray is exactly linear in the ray direction:

    1/R(d) = q . d          q = -n/h for unit normal n at distance h

three numbers, no offset term, and no assumption that the plane is level. So
fit q to the ground, and blend the ground toward it.

The blend is the whole design. Replacing outright would flatten the kerb, the
speed bump and the pothole along with the error, so a pixel's correction fades
out as its disagreement with the plane grows: a road surface that misses the
plane by a few percent is pulled onto it, a kerb that misses it by a lot keeps
its own depth. The fade is on the *relative* residual, because a fixed
tolerance in inverse depth means something different at two metres and at
twenty.

Off unless asked for. It measured well on one scene, which is not enough to
change what every existing render produces, and it can only help where there
genuinely is a dominant plane in view.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from . import projection

#: Directions this far below the horizon are candidates for the fit. Generous:
#: the trimming below decides what is really the plane, and starting from a
#: tight cone throws away the far ground that pins the fit's tilt.
BELOW_HORIZON_DEG = 8.0

#: Fraction of samples kept at each refit.
KEEP = 0.6

#: A plane further off level than this is not the ground of a 360 shot, and
#: fitting to it would drag the whole lower hemisphere.
MAX_TILT_DEG = 35.0

#: How badly the kept samples may still miss the plane before the fit is not
#: describing a plane at all. Least squares always returns *a* plane, even for
#: noise -- on random data the tilt and inlier guards both pass happily, so
#: without this the correction would fire on a scene that has no ground in it.
#: Real ground measures 2% free and 5% held level; 15% is far outside both.
MAX_RESIDUAL = 0.15

#: Where the correction fades out near the horizon, in degrees below it. A hard
#: edge there would put a seam across the frame at a fixed latitude.
FADE_LO_DEG = 4.0
FADE_HI_DEG = 12.0


class GroundPlane(NamedTuple):
    """The fitted plane, and whether it is worth applying."""

    q: np.ndarray               # (3,) with q . d = inverse depth on the plane
    normal: np.ndarray          # unit, pointing up out of the plane
    height: float               # camera height above it, in inverse-depth units
    tilt_deg: float
    inliers: int
    residual: float             # robust relative residual of the fit
    ok: bool
    why: str

    def describe(self) -> str:
        if not self.ok:
            return f"no ground plane ({self.why})"
        return (f"tilt {self.tilt_deg:.1f} deg, residual "
                f"{self.residual * 100:.1f}%, {self.inliers} inliers")


def _samples(disp: np.ndarray, stride: int):
    """Directions and inverse depths on a stride grid of the lower hemisphere."""
    h, w = disp.shape
    ys = np.arange(0, h, stride)
    dirs, vals = [], []
    for y0 in ys:
        d = projection.equirect_rows_to_dir(y0, y0 + 1, w, h)[0, ::stride]
        dirs.append(d)
        vals.append(disp[y0, ::stride])
    return (np.concatenate(dirs).astype(np.float64),
            np.concatenate(vals).astype(np.float64))


def fit_plane(disp: np.ndarray, samples: int = 240,
              below_deg: float = BELOW_HORIZON_DEG, keep: float = KEEP,
              iters: int = 5, level: bool = False) -> GroundPlane:
    """Least squares q for `inverse depth = q . d`, trimmed to the inliers.

    `samples` is a grid resolution, not a pixel stride, so the sample count
    does not depend on how large the frame happens to be -- the same reasoning
    as `align_overlapping_faces`.
    """
    bad = GroundPlane(np.zeros(3), np.array([0.0, 1.0, 0.0]), 0.0, 0.0, 0,
                      0.0, False, "")
    h, w = disp.shape
    stride = max(1, min(h, w) // max(samples, 1))
    d, v = _samples(disp, stride)

    cand = (d[:, 1] < -np.sin(np.radians(below_deg))) & (v > 1e-6) \
        & np.isfinite(v)
    if int(cand.sum()) < 200:
        return bad._replace(why="too little below the horizon")
    d, v = d[cand], v[cand]

    # With `level`, only the camera's height is solved for and the normal is
    # held straight up. Worth having because the free fit will absorb depth
    # error as tilt: on a road where the picture's own horizon says the camera
    # is about a degree off level, the free fit claimed 6.7 degrees, and
    # flattening onto that lays the ground perfectly onto a ramp.
    basis = -d[:, 1:2] if level else d

    w_mask = np.ones(len(v), bool)
    q = None
    for _ in range(iters):
        try:
            sol, *_ = np.linalg.lstsq(basis[w_mask], v[w_mask], rcond=None)
        except np.linalg.LinAlgError:
            return bad._replace(why="singular system")
        q = np.array([0.0, -sol[0], 0.0]) if level else sol
        if not np.all(np.isfinite(q)):
            return bad._replace(why="non-finite solution")
        pred = basis[:, 0] * sol[0] if level else d @ q
        rel = np.abs(v - pred) / np.maximum(np.abs(pred), 1e-6)
        w_mask = rel <= np.quantile(rel, keep)

    qn = float(np.linalg.norm(q))
    if qn < 1e-9:
        return bad._replace(why="degenerate plane")
    normal = -q / qn
    tilt = float(np.degrees(np.arccos(np.clip(normal[1], -1.0, 1.0))))
    resid = float(np.median(rel[w_mask]))
    n_in = int(w_mask.sum())

    out = GroundPlane(q, normal, 1.0 / qn, tilt, n_in, resid, True, "")
    if tilt > MAX_TILT_DEG:
        return out._replace(ok=False,
                            why=f"fitted plane is {tilt:.0f} deg off level")
    if n_in < 200:
        return out._replace(ok=False, why="too few inliers")
    if resid > MAX_RESIDUAL:
        return out._replace(
            ok=False, why=f"no plane there (residual {resid * 100:.0f}%)")
    return out


#: How much of the sphere the residual is smoothed over, in degrees of arc.
#: The bend being removed spans the whole ground; a kerb spans a fraction of a
#: degree. Anything between the two works, and 15 is comfortably between.
SMOOTH_DEG = 15.0


def flatten(disp: np.ndarray, strength: float,
            plane: Optional[GroundPlane] = None,
            smooth_deg: float = SMOOTH_DEG,
            level: bool = False) -> np.ndarray:
    """Remove the ground's smooth departure from its plane. Returns a new array.

    Not a blend toward the plane. The first version of this snapped each pixel
    onto the plane in proportion to how well it already agreed, which sounds
    reasonable and produces bumps: a pixel just inside the tolerance is moved
    the whole way and its neighbour just outside is not moved at all, so the
    threshold itself prints into the depth map. Judged in a headset the result
    was bumpier than doing nothing, and worst where the error was largest --
    which is exactly where most pixels sit near the boundary.

    What is actually wrong is *smooth*: the ground bends over tens of degrees.
    What must be preserved is *local*: a kerb, a pothole, a stone. So take the
    residual against the plane, low-pass it over `smooth_deg`, and subtract
    only that. Large-scale bend goes, local detail stays, and there is no
    threshold anywhere to print an edge.

    The smoothing is masked to the ground: sky residuals are meaningless and
    must not leak downward across the horizon, so the blur is normalised by a
    blurred mask rather than run over the whole frame.

    `strength` scales what is subtracted: 1.0 removes the bend, 0.5 halves it.
    """
    if strength <= 0.0:
        return disp
    plane = plane if plane is not None else fit_plane(disp, level=level)
    if not plane.ok:
        return disp

    import cv2

    h, w = disp.shape
    q = plane.q.astype(np.float32)
    lo = np.sin(np.radians(FADE_LO_DEG))
    hi = np.sin(np.radians(FADE_HI_DEG))

    ideal = np.empty((h, w), np.float32)
    fade = np.empty((h, w), np.float32)
    for y0 in range(0, h, 256):
        y1 = min(y0 + 256, h)
        d = projection.equirect_rows_to_dir(y0, y1, w, h)
        ideal[y0:y1] = d @ q
        down = -d[..., 1]
        f = np.clip((down - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        fade[y0:y1] = f * f * (3.0 - 2.0 * f)

    mask = ((ideal > 1e-6) & (fade > 0.0)).astype(np.float32)
    resid = (disp - ideal) * mask

    # A box blur is separable and O(1) per pixel, and the shape of the kernel
    # does not matter for something this wide. Rows wrap: the equirect is
    # continuous across +/-180 degrees and a reflected border would fold the
    # scene back on itself there.
    k = max(3, int(round(smooth_deg / 360.0 * w)) | 1)
    pad = k // 2
    def blur(a):
        wide = np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
        out = cv2.blur(wide, (k, k), borderType=cv2.BORDER_REPLICATE)
        return out[:, pad:pad + w]

    num = blur(resid)
    den = blur(mask)
    bend = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 0.0)

    return (disp - strength * fade * mask * bend).astype(np.float32)
