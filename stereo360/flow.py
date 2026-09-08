"""Frame interpolation by optical flow: the option that downloads nothing.

RIFE needs a 20 MB graph and a runtime to execute it. This needs neither --
OpenCV is already a dependency, and DIS is its fast classical flow estimator.
The middle frame is made the way it was made before neural interpolation:
estimate the motion both ways, carry each neighbour along it, blend.

Measured against RIFE v4.25 on the three tests this project has:

    test                        flow    RIFE
    real footage, hold-out     40.88   40.90
    synthetic 8K yaw pan       97.64   57.07
    pan with an occluder       43.59   42.98
      the moving object        35.05   37.32
      the trail behind it      22.47   23.26
    8K seconds a frame          0.53    1.21

Level on real footage, far ahead on rotation, behind on occlusion, and about
twice as fast. The pattern is what the two methods are: a warp reconstructs
motion it can see and cannot invent what was hidden, so it is near exact on a
pan -- 99 dB at the seam, which is bit-exact -- and loses where an object
uncovers background that appears in neither neighbouring frame. RIFE guesses
that content and wins there by a decibel or two.

For 360 the balance leans this way more than the table suggests: a camera
that yaws produces near-pure horizontal translation in equirectangular, which
is the case flow solves exactly.

Two things about the implementation are load-bearing:

**The sign.** OpenCV gives `b(x + flow) = a(x)`, so the frame a fraction `t`
along is `a(x - t*flow)`. Sampling at `+t` carries the content away from
where it belongs by the whole motion rather than none of it, and scores 22 dB
on a pan where the right sign scores 97. It looks plausible enough in a still
to pass a glance.

**The seam.** The frame is wrapped sideways before the flow is estimated, so
the +/-180 join has context on both sides. Without it a panning 360 frame
tears there, which is the failure ffmpeg's block matching shows at 30.71 dB
against RIFE's 52.34 in the table in interpolate.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

CODE = "flow"
NAME = "Optical flow"
DESC = ("Classical motion estimation, no model to download and nothing to "
        "install. Level with RIFE on real footage, better on camera pans, "
        "weaker where one thing passes in front of another, and about twice "
        "as fast.")

#: How much of the far side to wrap in before estimating. Wider than any
#: motion the estimator will be asked to follow, so the join has real context
#: rather than a mirrored guess.
WRAP = 256

#: DIS presets, cheapest first. `ultrafast` measured level with `fast`
#: everywhere except the frame as a whole, and better on a moving object, at
#: two thirds the cost -- so it is the default rather than the fallback.
PRESETS = ("ultrafast", "fast", "medium")
DEFAULT_PRESET = "ultrafast"


class FlowError(RuntimeError):
    """Raised when this machine cannot do it at all."""


def _preset(name: str) -> int:
    import cv2

    table = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
             "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
             "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
    if name not in table:
        raise FlowError(f"unknown preset {name!r}; expected one of "
                        f"{', '.join(PRESETS)}")
    return table[name]


def available() -> bool:
    """Whether OpenCV here carries DIS. It has since 4.x, but say rather
    than assume: the interface offers what this answers."""
    try:
        import cv2

        return hasattr(cv2, "DISOpticalFlow_create")
    except ImportError:
        return False


def describe() -> dict:
    if not available():
        return {"available": False,
                "reason": "this OpenCV has no DISOpticalFlow"}
    return {"available": True, "reason": ""}


class Engine:
    """Holds the estimator so a run makes one rather than one per frame."""

    def __init__(self, preset: str = DEFAULT_PRESET) -> None:
        if not available():
            raise FlowError("OpenCV here has no DISOpticalFlow, so "
                            "--interpolate flow cannot run.")
        import cv2

        self._cv2 = cv2
        self._dis = cv2.DISOpticalFlow_create(_preset(preset))
        self.preset = preset
        self.provider = f"OpenCV DIS, {preset}"

    def between(self, a: np.ndarray, b: np.ndarray,
                t: float = 0.5) -> np.ndarray:
        """The frame `t` of the way from `a` to `b`."""
        cv2 = self._cv2
        # Never wrap more than the frame has. A 360 frame is thousands of
        # pixels wide and this never binds in a render -- but a narrow one
        # would otherwise slice past both ends and come back empty, which is
        # a silent nothing rather than an error.
        pad = min(WRAP, a.shape[1])
        aw = np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
        bw = np.concatenate([b[:, -pad:], b, b[:, :pad]], axis=1)
        ga = cv2.cvtColor(aw, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(bw, cv2.COLOR_RGB2GRAY)
        fwd = self._dis.calc(ga, gb, None)
        bwd = self._dis.calc(gb, ga, None)
        h, w = ga.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        # a(x - t*flow) and b(x - (1-t)*flow): each neighbour carried its own
        # share of the way, then mixed in the same proportion, so t=0.25 leans
        # on the frame it is nearer to.
        ha = cv2.remap(aw, gx - fwd[..., 0] * t, gy - fwd[..., 1] * t,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        hb = cv2.remap(bw, gx - bwd[..., 0] * (1.0 - t),
                       gy - bwd[..., 1] * (1.0 - t),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mixed = ha.astype(np.float32) * (1.0 - t) + hb.astype(np.float32) * t
        return np.clip(mixed + 0.5, 0, 255).astype(np.uint8)[:, pad:-pad]
