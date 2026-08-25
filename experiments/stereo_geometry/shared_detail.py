"""Carry fine detail on a smooth displacement, so it cannot shear.

Thin structures disagree between the eyes because they sit on depth
discontinuities: the boundary is misplaced by 10-25 px, each eye shears the
structure by a different amount, and the pair no longer fuses. Nothing in this
directory has managed to put those boundaries in the right place -- see
`../segment_depth/README.md` for six attempts and why each failed.

This gives up on placing them and changes what rides over them instead.

Split each frame into a blurred base and the detail removed by that blur.
Warp the base with the real depth, so the coarse depth percept keeps every
discontinuity it should have -- and the base is blurred, so a boundary error
of a few pixels moves smooth content and is nearly invisible. Warp the detail
with a *smoothed* depth instead. A smooth displacement field has no steps in
it, so it cannot tear or shear a thin structure: the detail arrives whole, and
it arrives the same way in both eyes because both were displaced by the same
smooth field.

What is given up is that fine detail sits at a slightly wrong depth -- it
follows the smoothed depth rather than the true one. That is the trade, and it
is deliberate: detail at a slightly wrong depth that both eyes agree on is
easier to fuse than detail at the right depth that they disagree about. Coarse
disparity, which the base still carries correctly, is what the depth percept
mostly rests on.

The obvious risk is that it reads as texture pasted flat onto the scene. That
is what `detail_sigma` trades: too large and the "detail" includes whole
objects, which then float at the wrong depth.

**Measured, at share 0.15, detail_sigma 12, depth_sigma 40.** On the indoor
frame the eyes' disagreement about the chair rail's bright band falls from
-5.83 px to -0.17 -- as good as plain dilation's -0.33 -- for the same
fidelity cost (-1.17 against dilation's -1.00), and the floor grout line
*improves*, 0.21 to 0.07, where dilation made it worse at 0.30. Sharpness is
preserved rather than spent: 5.78 against plain's 5.44.

**A warning about the road numbers.** The shape residual reports the lamp
finial getting worse, 5.58 to 7.30. The picture says the opposite and plainly
so: plain leaves the finial smeared and leaning with a ragged edge along the
cap, and this leaves it round and upright. The residual is a mean-abs
difference against the source, so it charges for the base/detail split as a
photometric change and never asks about shape. Do not read those three road
numbers as evidence against this; they are measuring the wrong thing for it.

`detail_mix` interpolates back toward the ordinary warp and was tried at 0.25
and 0.5. It is a dial between this and plain and resolves nothing: every
target moves monotonically back toward its plain value.

**With the limiter at its proper 0.6** the numbers settle differently and
better. Indoor: the chair rail's disagreement goes -7.50 to -0.83 and the
grout line 0.08 to 0.04. Road: sign post 5.43 to 4.71 and handrail 7.48 to
4.74, with the lamp finial a wash at 7.18 against 7.27 -- the penalty it
showed at limiter 0 was the limiter's absence, not this.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

REPO = os.environ.get(
    "STEREO360_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from stereo360 import warp  # noqa: E402


def _blur(img, sigma):
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def split_bands(rgb, detail_sigma):
    """(base, detail) with base + detail == rgb, in float32."""
    f = rgb.astype(np.float32)
    base = _blur(f, detail_sigma)
    return base, f - base


def smooth_depth(dn, depth_sigma, mix=0.0):
    """Depth with its discontinuities removed rather than relocated.

    The whole point is that this field has no steps: a displacement built from
    it cannot shear a structure, whatever the true boundary is doing.

    `mix` blends the true depth back in, and exists because fully smooth is
    wrong for small isolated objects. An extended structure -- a rail, a
    handrail -- stays inside its own smoothed depth, so its detail is placed
    about right. A lamp finial a few dozen pixels across does not: the smooth
    field there is dominated by the background behind it, so its detail is
    displaced at nearly the background's disparity and ends up worse than
    doing nothing. Measured on the road frame, fully smooth took the finial
    from 5.58 to 7.30 while improving the handrail from 7.48 to 4.74.
    """
    sm = _blur(dn.astype(np.float32), depth_sigma)
    if mix <= 0.0:
        return sm
    return ((1.0 - mix) * sm + mix * dn.astype(np.float32)).astype(np.float32)


def render_eye(base, detail, dn, dn_smooth, strength, **kw):
    """One eye: base on the real depth, detail on the smooth one."""
    b, hole = warp.right_eye_from_disparity(
        base, dn.copy(), strength, normalize=False, inpaint=False, **kw)
    d, dhole = warp.right_eye_from_disparity(
        detail, dn_smooth.copy(), strength, normalize=False, inpaint=False,
        **kw)
    # A hole in the detail layer means no detail, not black: adding nothing
    # leaves the base showing through, which is what should happen where
    # nothing is known.
    d[dhole > 0] = 0.0
    out = b + d
    return np.clip(out, 0, 255).astype(np.uint8), hole


def stereo_pair(frame, dn, strength, left_share=0.0, detail_sigma=6.0,
                depth_sigma=40.0, detail_mix=0.0, shared=True,
                gradient_limit=0.6, fg_erode=2, **kw):
    """(left, right). With `shared` false this is the ordinary warp.

    `gradient_limit` and `fg_erode` default to what the pipeline uses, and
    that matters more than it looks. An earlier version of this file carried
    0 and 0 over from the segmentation experiments, where the limiter has to
    be off because it re-creates the ramp a segment removes. Nothing here
    needs that, and without the limiter a strong vertical depth edge -- a
    curtain against a wall -- opens a disocclusion wide enough that the fill
    comes back as a blocky sawtooth several pixels across. It is plainly
    visible and it is not caused by the detail split: the ordinary warp shows
    the same artefact at 0. At 0.6 it disappears from both.
    """
    f = float(np.clip(left_share, 0.0, 1.0))
    common = dict(gradient_limit=gradient_limit, fg_erode=fg_erode, **kw)

    def plain(s):
        img, hole = warp.right_eye_from_disparity(
            frame, dn.copy(), s, normalize=False, inpaint=False, **common)
        return img, hole

    if shared:
        base, detail = split_bands(frame, detail_sigma)
        dns = smooth_depth(dn, depth_sigma, detail_mix)

        def one(s):
            return render_eye(base, detail, dn, dns, s, **common)
    else:
        one = plain

    def finish(img, hole, s):
        if hole is None or not (hole > 0).any():
            return img
        return warp.fill_holes(img, hole, dn, directional=True,
                               baseline_sign=1.0 if s >= 0 else -1.0)

    if f <= 0.0:
        return frame, finish(*one(strength), strength)
    if f >= 1.0:
        return finish(*one(-strength), -strength), frame
    return (finish(*one(-f * strength), -f * strength),
            finish(*one((1.0 - f) * strength), (1.0 - f) * strength))
