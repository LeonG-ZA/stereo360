"""Confidence-weighted disparity damping: does disagreement locate the error?

The premise is that where two depth estimates disagree, the model is guessing
-- and that is exactly where the artifacts live. If it holds, the disagreement
map can be used to damp only the untrustworthy disparity and leave the rest at
full strength, instead of turning the whole image down with --strength.

Nothing here touches the pipeline. It scores against score.py first, because
the premise might simply be false.
"""

import cv2
import numpy as np


def align(src, ref, sample=200_000):
    """Put `src` on `ref`'s scale: relative depth has no absolute units.

    Least squares on a random sample, after clipping the extremes -- the sky
    and the very near field would otherwise dominate the fit.
    """
    a, b = src.ravel(), ref.ravel()
    idx = np.random.default_rng(0).choice(a.size, min(sample, a.size), False)
    a, b = a[idx], b[idx]
    lo, hi = np.percentile(b, [2, 98])
    keep = (b > lo) & (b < hi)
    a, b = a[keep], b[keep]
    A = np.stack([a, np.ones_like(a)], axis=1)
    scale, offset = np.linalg.lstsq(A, b, rcond=None)[0]
    return src * scale + offset


def disagreement(d1, d2, blur=9):
    """Normalised |d1 - d2|, smoothed so single pixels do not dominate."""
    diff = np.abs(d1 - d2).astype(np.float32)
    diff = cv2.GaussianBlur(diff, (blur, blur), 0)
    scale = np.percentile(diff, 95)
    return np.clip(diff / max(scale, 1e-6), 0, 1)


def damp_toward_far(disp, conf, amount=1.0):
    """Reduce disparity where confidence is low.

    Toward *far* rather than toward a local average, on the observation that
    these errors are one-sided: the chair's gaps read too near, the floor
    ridge reads too near, the wall's lower half reads too near. Monocular
    depth hallucinates surfaces closer and more solid than they are, so
    pulling the doubtful parts back is the direction that helps.
    """
    return disp * (1.0 - amount * (1.0 - conf))


def damp_toward_smooth(disp, conf, radius=25):
    """Blend toward a locally smoothed depth where confidence is low."""
    k = radius * 2 + 1
    smooth = cv2.GaussianBlur(disp.astype(np.float32), (k, k), 0)
    return conf * disp + (1.0 - conf) * smooth


def cli_defaults():
    """The values the CLI actually passes, for anything rendering directly.

    The library's own defaults are not the CLI's. `preview_frame` defaults
    `gradient_limit` to 0.0 -- disabled -- while every real render gets 1.0.
    With it off, the warp is allowed past the slope where it stops being
    injective and fine structure tears into fragments; two renders went out
    for review with chairs disintegrating before this was noticed.

    Read from the parser rather than copied, so it cannot drift.
    """
    from stereo360 import cli, projection
    p = cli.build_parser()
    out = {}
    for name in ("gradient_limit", "fg_erode", "strength", "face_size"):
        v = p.get_default(name)
        if v is not None:
            out[name] = v
    out["inpaint_mode"] = p.get_default("inpaint")
    out["face_overlap"] = p.get_default("face_overlap") or projection.FACE_OVERLAP
    return out


def damp_gradient(disp, conf, floor=0.25, iters=12):
    """Limit the depth *gradient* where confidence is low, in place of scaling.

    Scaling the depth by a confidence factor was the first idea and it is
    self-defeating: d(d*f) = f.dd + d.df, so it manufactures gradient wherever
    the confidence map varies -- which is at object boundaries, exactly where
    it is supposed to be helping. The renders came back with every straight
    line turned to sawtooth.

    This can only ever remove gradient. Each step clamps the local difference
    to a ceiling that falls with confidence, so a surface the two passes agree
    about keeps its full relief and a boundary they argue about is flattened
    toward its surroundings. Absolute depth is left alone, so near objects do
    not lose their separation from far ones.
    """
    import cv2
    d = disp.astype(np.float32).copy()
    # The ceiling has to be on the same footing as the quantity being
    # clipped. Taking it from 8-pixel-step gradients while clipping 1-pixel
    # deviations made it orders of magnitude too large, and the clip never
    # bound once -- every score came back byte-identical to the baseline.
    excess0 = d - cv2.blur(d, (3, 3))
    scale = float(np.percentile(np.abs(excess0), 99))
    ceiling = (scale * (floor + (1.0 - floor) * conf)).astype(np.float32)
    for _ in range(iters):
        blur = cv2.blur(d, (3, 3))
        d = blur + np.clip(d - blur, -ceiling, ceiling)
    return d
