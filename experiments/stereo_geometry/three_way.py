"""Whole, split and chained baselines, measured on how much the eyes disagree.

The usual measure is each eye's error against the source, which answers "is
this eye right". The argument being tested here is that the pair matters more
than either half: two eyes that agree on a slightly wrong shape fuse into a
slightly wrong object, while one right eye and one wrong eye fuse into nothing
and the viewer keeps trying. So the number to watch is the *disagreement
between the eyes*, not either eye's fidelity.

    whole    L = source untouched,        R = warp(source, +b)
    split    L = warp(source, -b/2),      R = warp(source, +b/2)
    chained  R = warp(source, +b/2),      L = warp(R, -b)

Chaining is the point of the experiment. Under split the two eyes are warped
in opposite directions from one source, so whatever the depth map gets wrong
about a small feature comes out *mirrored* -- measured on a bollard finial,
leaning right in one eye and left in the other, which is the worst case for
fusion rather than the best. Under chaining the left eye starts from the right
eye, so it inherits that error instead of mirroring it.

Measuring disagreement needs care: the two eyes are separated by a real
parallax, so a naive difference would just measure that. Each feature is
therefore aligned by brute-force integer shift against the source first, and
what is compared is how far each eye had to be *distorted*, not where it sits.

Diagnostic only.
"""
from __future__ import annotations

import os
import sys
import time

REPO = os.environ.get("STEREO360_REPO",
                      os.path.dirname(os.path.dirname(
                          os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, OUT)

import cv2
import numpy as np

import da3_render as R
import mesh_warp
from stereo360 import ground, warp

#: Small features whose shape the depth map is known to get wrong, as
#: (label, x0, x1, y0, y1) on the 7680-wide equirect.
TARGETS = [
    ("lamp finial",   45,  115, 2700, 2770),
    ("sign post",   4230, 4290, 2080, 2220),
    ("handrail",    4180, 4240, 2500, 2650),
]


def to_dn(zbuf, filled):
    """A rendered eye's own normalised inverse depth, from its z-buffer.

    Holes have no sample and so no distance; they are filled from their
    neighbours, because a second warp needs *some* depth everywhere and the
    alternative is punching the hole through into the next view as well.
    """
    dn = np.where(np.isfinite(zbuf) & (zbuf > 1e-6),
                  1.0 / np.maximum(zbuf, 1e-6) - warp._MIN_INV_DEPTH, 0.0)
    dn = dn.astype(np.float32)
    bad = (~filled) | ~np.isfinite(dn)
    if bad.any():
        dn = cv2.inpaint((dn * 255).clip(0, 255).astype(np.uint8),
                         bad.astype(np.uint8), 3, cv2.INPAINT_TELEA
                         ).astype(np.float32) / 255.0
    return np.clip(dn, 0.0, 1.0)


def residual(src, img, x0, x1, y0, y1):
    """Lowest mean-abs-difference against the source over integer shifts.

    Divides out where the feature sits, leaving how far its shape had to bend.
    """
    w = src.shape[1]
    cols = np.arange(x0, x1)
    ref = src[y0:y1][:, cols % w].astype(np.float32)
    im = img.astype(np.float32)
    best = 1e9
    for dy in range(-8, 9):
        for dx in range(-60, 61):
            # Columns wrap: the lamp sits a few dozen pixels from +/-180, and
            # a plain slice would run off the array rather than round the
            # sphere.
            p = im[y0 + dy:y1 + dy][:, (cols + dx) % w]
            if p.shape != ref.shape:
                continue
            v = p.max(axis=2) > 0
            if v.mean() < 0.75:
                continue
            best = min(best, np.abs(p - ref).mean(axis=2)[v].mean())
    return best


def main():
    eq = np.load(os.path.join(OUT, "v3_7680p_eq.npy")).astype(np.float32)
    pl = ground.fit_plane(eq)
    eq *= pl.height / 1.59                      # anchor to the metric camera
    d_near = float(1.0 / np.nanpercentile(eq, 99.9))
    S = d_near * 1.05 * 0.98
    R.metric_to_normalised(eq, S)
    strength = R.strength_for(65.0, S)
    b = strength * warp._BASELINE_SCALE          # the whole baseline
    half = b * 0.5
    src = cv2.resize(cv2.imread(os.path.join(REPO, "7680p.jpg")),
                     (7680, 3840), interpolation=cv2.INTER_AREA)
    print(f"scale {S:.2f} m/unit, strength {strength:.3f}, baseline {b:.5f}\n",
          flush=True)

    eyes = {}

    t = time.time()
    r_whole, c_whole = mesh_warp.render_full(eq, src, b, subdiv=4)
    eyes["whole"] = (src, r_whole)
    print(f"  whole   right {time.time() - t:.0f}s", flush=True)

    t = time.time()
    l_split, _ = mesh_warp.render_full(eq, src, -half, subdiv=4)
    r_split, _ = mesh_warp.render_full(eq, src, half, subdiv=4)
    eyes["split"] = (l_split, r_split)
    print(f"  split   both  {time.time() - t:.0f}s", flush=True)

    t = time.time()
    r_ch, cut_ch, z_ch = mesh_warp.render_full(eq, src, half, subdiv=4,
                                               want_depth=True)
    dn_r = to_dn(z_ch, ~cut_ch)
    l_ch, _ = mesh_warp.render_full(dn_r, r_ch, -b, subdiv=4)
    eyes["chained"] = (l_ch, r_ch)
    print(f"  chained both  {time.time() - t:.0f}s\n", flush=True)

    for k, (L, Rr) in eyes.items():
        np.save(os.path.join(OUT, f"_3way_{k}_L.npy"), L)
        np.save(os.path.join(OUT, f"_3way_{k}_R.npy"), Rr)

    print(f"{'feature':<14}{'mode':<9}{'left':>8}{'right':>8}"
          f"{'  DISAGREEMENT':>16}")
    for lab, x0, x1, y0, y1 in TARGETS:
        for k, (L, Rr) in eyes.items():
            a = residual(src, L, x0, x1, y0, y1)
            c = residual(src, Rr, x0, x1, y0, y1)
            print(f"{lab if k == 'whole' else '':<14}{k:<9}"
                  f"{a:>8.2f}{c:>8.2f}{abs(a - c):>14.2f}")
        print()


if __name__ == "__main__":
    main()
