"""Give the untouched eye the same processing as its partner, without moving it.

At `left_share` 0 the left eye is the photograph and the right has been
resampled, hole-filled and sheared. Even with perfect depth the two differ in
*character* -- one is sharp, one is not -- and a sharp eye paired with a soft
one is a binocular mismatch in its own right.

The round trip warps the source by +b and then by -b, so it comes back where
it started carrying two passes of the warp's resampling and its fill. The
reverse pass needs the depth of the intermediate view, which is obtained by
warping the depth map as an image alongside the frame.

**What this can and cannot equalise.** Resampling softness and fill character:
yes, that is the point. Shear from depth error: no. Two warps in opposite
directions cancel their shear where the depth is right and leave a small
residual where it is wrong, which is not the same thing as the single large
shear the other eye carries. So this narrows the gap between the eyes on
sharpness while leaving the geometric disagreement alone -- worth measuring
precisely because it is easy to assume it does more.

The cost is deliberate and has to be judged: it degrades a pristine eye on
purpose. `bench`'s "L vs source" column is the one to watch, since a change
that improves agreement by damaging both eyes is not an improvement.

**Measured, on the indoor frame at share 0.** It halves the eyes'
disagreement about the chair rail's bright band, -6.83 px to -3.00, and pays
4.00 px of that band in the left eye against the source, which is exactly the
"agreement bought by damage" the column is there to expose.

At its stated purpose it is inefficient. The point was to match the eyes'
*sharpness*, and it overshoots: two warps leave the left eye softer (5.012)
than the right's single warp (5.292), so the gap only closes from 6.6% to
5.3% while 11% of the left eye's detail is spent. Matching one warp takes one
warp, and an even split gives exactly that -- measured 5.400 against 5.380, a
0.4% gap, thirteen times better than this and free.

So: keep for the disagreement result, do not reach for it to match sharpness.
An even split dominates it there.

**A caveat this turned up, worth more than the idea.** At share 0.5 the left
eye's band measures 3.17 px *thicker* than the source while the right's is
thinner. The two warp directions distort this feature in opposite senses, so
an even split mirrors the error rather than equalising it. "Both eyes wrong
identically" is not what a split delivers on an asymmetric feature, and the
default-share question in plans/todo.md should be settled with that in mind.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.environ.get(
    "STEREO360_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from stereo360 import warp  # noqa: E402


def _warp_depth(dn, strength, **kw):
    """The depth map as the warped view sees it.

    Carried through the same warp as the picture, as a three-channel float
    image, so the reverse pass has a depth map registered to what it is
    undoing rather than to where the frame started.
    """
    d3 = np.repeat(dn[:, :, None], 3, axis=2).astype(np.float32)
    out, hole = warp.right_eye_from_disparity(
        d3, dn.copy(), strength, normalize=False, inpaint=False, **kw)
    got = out[..., 0].astype(np.float32)
    # Holes have no depth; take the neighbouring background rather than zero,
    # which would read as infinitely far and shear the reverse pass.
    m = hole > 0
    if m.any():
        import cv2
        k = np.ones((9, 9), np.uint8)
        got = np.where(m, cv2.dilate(got, k), got)
    return np.clip(got, 0.0, 1.0)


def roundtrip(frame, dn, strength, gradient_limit=0.0, fg_erode=0, **kw):
    """`frame` put through the warp and brought back, net displacement zero."""
    fwd, hole = warp.right_eye_from_disparity(
        frame, dn.copy(), strength, normalize=False, inpaint=True,
        gradient_limit=gradient_limit, fg_erode=fg_erode, **kw)
    dnw = _warp_depth(dn, strength, gradient_limit=gradient_limit,
                      fg_erode=fg_erode)
    back, _ = warp.right_eye_from_disparity(
        fwd, dnw, -strength, normalize=False, inpaint=True,
        gradient_limit=gradient_limit, fg_erode=fg_erode, **kw)
    return back


def stereo_pair(frame, dn, strength, left_share=0.0, match=True,
                gradient_limit=0.0, fg_erode=0, **kw):
    """(left, right), optionally putting the source eye through a round trip.

    Only does anything at the extremes: in between, both eyes are already
    warped and already share the warp's character.
    """
    f = float(np.clip(left_share, 0.0, 1.0))
    common = dict(normalize=False, gradient_limit=gradient_limit,
                  fg_erode=fg_erode, **kw)
    if f <= 0.0:
        right, _ = warp.right_eye_from_disparity(frame, dn.copy(), strength,
                                                 **common)
        left = roundtrip(frame, dn, strength, gradient_limit=gradient_limit,
                         fg_erode=fg_erode, **kw) if match else frame
        return left, right
    if f >= 1.0:
        left, _ = warp.right_eye_from_disparity(frame, dn.copy(), -strength,
                                                **common)
        right = roundtrip(frame, dn, -strength, gradient_limit=gradient_limit,
                          fg_erode=fg_erode, **kw) if match else frame
        return left, right
    left, _ = warp.right_eye_from_disparity(frame, dn.copy(), -f * strength,
                                            **common)
    right, _ = warp.right_eye_from_disparity(frame, dn.copy(),
                                             (1.0 - f) * strength, **common)
    return left, right
