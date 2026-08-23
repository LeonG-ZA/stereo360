"""Two ways to reconstruct the left eye without moving it, measured.

The question is whether a left eye that has been *through* the renderer, but
not displaced, agrees with the right eye better than a pristine one does.

    reconstructed   L = warp(source, 0)
    round trip      L = warp(warp(source, +b), -b)

They are not the same thing, and the difference is the whole point. At zero
baseline the displacement is zero for every pixel whatever the depth says, so
the first picks up the rasteriser's resampling character and none of the depth
map's errors -- it equalises texture, not geometry. The round trip goes out to
the right eye and comes back, so wherever the depth is wrong the two warps do
not cancel, and the left eye returns carrying the same distortion the right
eye has, at the same place, plus whatever was invented to fill the holes on
the way out.

Both keep the whole-baseline geometry: the left eye sits at the origin and the
right carries the full separation. Measured against split, where each eye is
displaced half as far in opposite directions.

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
from three_way import TARGETS, residual, to_dn


def main():
    eq = np.load(os.path.join(OUT, "v3_7680p_eq.npy")).astype(np.float32)
    pl = ground.fit_plane(eq)
    eq *= pl.height / 1.59
    d_near = float(1.0 / np.nanpercentile(eq, 99.9))
    S = d_near * 1.05 * 0.98
    R.metric_to_normalised(eq, S)
    strength = R.strength_for(65.0, S)
    b = strength * warp._BASELINE_SCALE
    src = cv2.resize(cv2.imread(os.path.join(REPO, "7680p.jpg")),
                     (7680, 3840), interpolation=cv2.INTER_AREA)
    print(f"baseline {b:.5f}\n", flush=True)

    t = time.time()
    right, cut_r, z_r = mesh_warp.render_full(eq, src, b, subdiv=4,
                                              want_depth=True)
    print(f"  right eye            {time.time() - t:.0f}s", flush=True)

    t = time.time()
    l_recon, _ = mesh_warp.render_full(eq, src, 0.0, subdiv=4)
    print(f"  left, reconstructed  {time.time() - t:.0f}s", flush=True)

    t = time.time()
    l_trip, _ = mesh_warp.render_full(to_dn(z_r, ~cut_r), right, -b, subdiv=4)
    print(f"  left, round trip     {time.time() - t:.0f}s\n", flush=True)

    np.save(os.path.join(OUT, "_lv_recon.npy"), l_recon)
    np.save(os.path.join(OUT, "_lv_trip.npy"), l_trip)
    np.save(os.path.join(OUT, "_lv_right.npy"), right)

    print(f"{'feature':<14}{'left eye is':<18}{'left':>7}{'right':>7}"
          f"{'  DISAGREEMENT':>16}")
    for lab, x0, x1, y0, y1 in TARGETS:
        rr = residual(src, right, x0, x1, y0, y1)
        for nm, L in (("source (pristine)", src),
                      ("reconstructed", l_recon),
                      ("round trip", l_trip)):
            ll = residual(src, L, x0, x1, y0, y1)
            print(f"{lab if nm.startswith('source') else '':<14}{nm:<18}"
                  f"{ll:>7.2f}{rr:>7.2f}{abs(ll - rr):>14.2f}")
        print()


if __name__ == "__main__":
    main()
