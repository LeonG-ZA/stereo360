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
