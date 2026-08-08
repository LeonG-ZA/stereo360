"""Score a depth map against the three artifacts we have actually measured.

Every one of these compares the depth map to something the world guarantees,
rather than to my judgement -- which has been wrong repeatedly. If a change
does not move these numbers it has not done anything, whatever it looks like.

  chair_gap   The gaps between the chair's back slats show a wall metres
              behind. Reported as how many times nearer the gaps read than
              that wall. 1.0 would be correct; the pipeline currently gives
              1.8 without tiling and 2.5 with tiles 3.

  wall_wobble A vertical plane at horizontal distance R puts a point at
              elevation phi at R/cos(phi), so inverse depth along a vertical
              wall must follow cos(phi) -- smooth and gentle. Reported as the
              peak-to-peak departure from that, in percent. 62% currently,
              which is the bowed edge by the entrance door.

  floor_rms   A camera h above a flat floor sees latitude t below the horizon
              at h/sin(t), so inverse depth must follow sin(t). Reported as
              rms departure, in percent.
"""

import os

import cv2
import numpy as np

#: The photo every coordinate below was measured against, expected beside the
#: repository rather than in it: it is 15 MB of somebody's living room, which
#: is not something to publish, and a checkout has no use for it unless you
#: are re-running these scores. STEREO360_INDOOR points somewhere else.
#:
#: Substituting another photo does not work, and fails quietly if you try.
#: Every landmark below is a pixel coordinate in *this* 11904x5952 frame, so a
#: different image scores whatever happens to sit at those coordinates --
#: meaningless numbers rather than wrong ones, which is harder to notice.
INDOOR = os.environ.get(
    "STEREO360_INDOOR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "indoor.jpg"))


def source():
    """The reference photo, or an explanation of where to put it.

    `cv2.imread` returns None for a missing file, so without this the first
    symptom is a TypeError several frames deep in a scoring function, which
    says nothing about the actual problem.
    """
    rgb = cv2.imread(INDOOR)
    if rgb is None:
        raise FileNotFoundError(
            f"The reference photo is not at {INDOOR}.\n"
            f"It is deliberately not in the repository -- it is a 360 photo "
            f"of a private house. Put your copy there, or set "
            f"STEREO360_INDOOR to it.\n"
            f"Only that exact frame is meaningful: the landmarks below are "
            f"pixel coordinates in it.")
    return rgb

# Located by hand against the source, and fixed so scores stay comparable.
CHAIR_ROW = 4300
CHAIR_X = (8400, 8560)
CHAIR_WALL_X = (8900, 9200)
DOOR_EDGE_X = 4209
FLOOR_LAT = (0.62, 0.95)          # fraction of image height: below the horizon


def _gray_row(rgb, y, x0, x1):
    return cv2.cvtColor(rgb[y, x0:x1].reshape(1, -1, 3),
                        cv2.COLOR_BGR2GRAY).ravel().astype(int)


def chair_gap(disp, rgb):
    """How many times nearer the see-through gaps read than the wall behind."""
    row = _gray_row(rgb, CHAIR_ROW, *CHAIR_X)
    gaps = disp[CHAIR_ROW, CHAIR_X[0]:CHAIR_X[1]][row < 70]
    wall = disp[CHAIR_ROW, CHAIR_WALL_X[0]:CHAIR_WALL_X[1]].mean()
    if wall <= 0 or not len(gaps):
        return float("nan")
    return float(gaps.mean() / wall)


def wall_wobble(disp):
    """Peak-to-peak departure from the vertical-plane law, in percent."""
    H = disp.shape[0]
    ys = np.arange(2600, 4500, 25)
    phi = np.deg2rad((ys / H - 0.5) * 180.0)
    meas = np.array([np.median(disp[y, DOOR_EDGE_X + 40:DOOR_EDGE_X + 140])
                     for y in ys])
    k = float(np.sum(meas * np.cos(phi)) / np.sum(np.cos(phi) ** 2))
    err = (meas - k * np.cos(phi)) / (k * np.cos(phi)) * 100
    return float(err.max() - err.min())


def floor_rms(disp):
    """Rms departure from the flat-floor law, in percent.

    Median across azimuth per row, so furniture standing on the floor is
    outvoted by the tiles around it.
    """
    H = disp.shape[0]
    y0, y1 = int(H * FLOOR_LAT[0]), int(H * FLOOR_LAT[1])
    rows = np.arange(y0, y1, 8)
    lat = np.deg2rad((rows / H - 0.5) * 180.0)
    prof = np.array([np.median(disp[y]) for y in rows])
    k = float(np.sum(prof * np.sin(lat)) / np.sum(np.sin(lat) ** 2))
    err = (prof - k * np.sin(lat)) / (k * np.sin(lat)) * 100
    return float(np.sqrt(np.mean(err ** 2)))


def depth_span(disp):
    """1st-99th percentile spread of the depth map.

    The one score here that is NOT a ratio, and it exists because the other
    three are. Being scale-invariant made them immune to --strength, which
    was the point -- but it also made them blind to the stereo collapsing
    altogether. A map multiplied by 0.001 scores identically on all three.

    That is not hypothetical: a damped render went out for review with the
    depth backend accidentally omitted, so it was a passthrough with no
    stereo at all, and the scorecard had nothing to say about it. Anything
    that shrinks this is taking depth away, however good the ratios look.
    """
    lo, hi = np.percentile(disp[::16, ::16], [1, 99])
    return float(hi - lo)


def score(disp, rgb=None):
    if rgb is None:
        rgb = source()
    return {"chair_gap": chair_gap(disp, rgb),
            "wall_wobble": wall_wobble(disp),
            "floor_rms": floor_rms(disp),
            "depth_span": depth_span(disp)}


def report(name, s, base=None):
    def d(key, fmt):
        v = s[key]
        if base is None:
            return f"{v:{fmt}}"
        delta = v - base[key]
        arrow = "  " if abs(delta) < 1e-9 else (" v" if delta < 0 else " ^")
        return f"{v:{fmt}}{arrow}"
    print(f"  {name:<34}{d('chair_gap','>9.2f')}{d('wall_wobble','>12.1f')}"
          f"{d('floor_rms','>11.1f')}{d('depth_span','>12.2f')}")


_H1 = (f"  {'':34}{'chair gap':>9}{'wall wobble':>12}{'floor rms':>11}"
       f"{'depth span':>12}")
_H2 = (f"  {'':34}{'(-> 1.0)':>9}{'(-> 0%)':>12}{'(-> 0%)':>11}"
       f"{'(keep!)':>12}")
HEADER = _H1 + chr(10) + _H2
