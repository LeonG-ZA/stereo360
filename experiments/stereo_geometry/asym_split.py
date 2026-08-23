"""How unevenly can the baseline be divided before the eyes stop agreeing?

Whole baseline, a reconstructed left eye, a round trip and an even split are
not four separate ideas. They are points on one axis: what fraction `f` of the
total separation is given to the left eye, with the right eye taking the rest.

    f = 0.00   whole baseline -- left eye never moves
    f = 0.15   a minor displacement
    f = 0.50   the even split

The total separation is `b` in every case, so the depth effect is identical
and only the division changes. Displacement error is proportional to how far
an eye is warped, so the prediction is that each eye carries distortion in
proportion to its share, disagreement runs with the difference between the
shares, and the minimum sits at an even split. A small displacement should
therefore buy a correspondingly small improvement -- not nothing, but not much.

Worth measuring rather than asserting, because the prediction assumes the
distortion grows smoothly with displacement, and thin structures may instead
have a threshold: intact until the warp exceeds a pixel or two, broken after.
If that is what happens the curve will have a knee rather than a slope, and
the useful setting is just past it rather than at 50/50.

Diagnostic only.
"""
from __future__ import annotations

import os
import sys
import time

REPO = os.environ.get("STEREO360_REPO",
                      r"C:\Users\leong\OneDrive\Documents\stereo360")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, OUT)

import cv2
import numpy as np

import da3_render as R
import mesh_warp
from stereo360 import ground, warp
from three_way import TARGETS, residual

FRACTIONS = (0.00, 0.15, 0.30, 0.50)


def main():
    eq = np.load(os.path.join(OUT, "v3_7680p_eq.npy")).astype(np.float32)
    pl = ground.fit_plane(eq)
    eq *= pl.height / 1.59
    d_near = float(1.0 / np.nanpercentile(eq, 99.9))
    S = d_near * 1.05 * 0.98
    R.metric_to_normalised(eq, S)
    b = R.strength_for(65.0, S) * warp._BASELINE_SCALE
    src = cv2.resize(cv2.imread(os.path.join(REPO, "7680p.jpg")),
                     (7680, 3840), interpolation=cv2.INTER_AREA)
    print(f"total baseline {b:.5f}, held constant\n", flush=True)

    rows = {}
    for f in FRACTIONS:
        t = time.time()
        L = src if f == 0.0 else \
            mesh_warp.render_full(eq, src, -f * b, subdiv=4)[0]
        Rr = mesh_warp.render_full(eq, src, (1.0 - f) * b, subdiv=4)[0]
        rows[f] = [(residual(src, L, *t4[1:]), residual(src, Rr, *t4[1:]))
                   for t4 in TARGETS]
        print(f"  f={f:.2f}  left {f * 100:.0f}% / right {(1 - f) * 100:.0f}%"
              f"   {time.time() - t:.0f}s", flush=True)

    print(f"\n{'feature':<14}" + "".join(f"{f:>18.2f}" for f in FRACTIONS))
    print(f"{'':<14}" + "".join(f"{'L / R  =  gap':>18}" for _ in FRACTIONS))
    for i, (lab, *_ ) in enumerate(TARGETS):
        cells = []
        for f in FRACTIONS:
            a, c = rows[f][i]
            cells.append(f"{a:.1f}/{c:.1f} = {abs(a - c):>4.2f}".rjust(18))
        print(f"{lab:<14}" + "".join(cells))
    print("\ndisagreement only:")
    print(f"{'feature':<14}" + "".join(f"{f:>10.2f}" for f in FRACTIONS))
    for i, (lab, *_) in enumerate(TARGETS):
        print(f"{lab:<14}" + "".join(
            f"{abs(rows[f][i][0] - rows[f][i][1]):>10.2f}" for f in FRACTIONS))


if __name__ == "__main__":
    main()
