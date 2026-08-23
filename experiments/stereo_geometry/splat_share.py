"""Render a stereo pair with the splat path at an arbitrary baseline share.

`stereo_pair` offers two arrangements -- the whole baseline in one eye, or an
even split -- because those were the only two anyone wanted. The share sweep
found the useful point is neither: at 15/85 the left eye is nearly the original
and the eyes still agree to within a fraction of what a pristine eye costs. So
call the warp directly, once per eye, with each eye's own share.

`right_eye_from_disparity` multiplies its `strength` by `_BASELINE_SCALE` to
get a baseline, so passing `-f * total` and `(1 - f) * total` puts the eyes at
those shares of one separation.

Everything else is the splat path as it normally runs, including its mirrored
directional fill -- the point is to compare renderers, not to give this one the
mesh path's settings.

Diagnostic only.
"""
from __future__ import annotations

import argparse
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
from stereo360 import ground, pipeline, warp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--eq", required=True)
    ap.add_argument("--anchor-height", type=float, default=None)
    ap.add_argument("--mm", type=float, default=40.0)
    ap.add_argument("--left-share", type=float, default=0.15)
    ap.add_argument("--gradient-limit", type=float, default=0.6)
    ap.add_argument("--fg-erode", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = args.image if os.path.isabs(args.image) \
        else os.path.join(REPO, args.image)
    eq = np.load(args.eq).astype(np.float32)
    if args.anchor_height:
        pl = ground.fit_plane(eq)
        eq *= pl.height / args.anchor_height
        print(f"anchored: plane {pl.height:.2f} units -> "
              f"{args.anchor_height:.2f} m", flush=True)

    w, h = 7680, 3840
    frame = cv2.imread(path)
    if frame.shape[1] != w:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    d_near = float(1.0 / np.nanpercentile(eq, 99.9))
    S = d_near * 1.05 * 0.98
    R.metric_to_normalised(eq, S)
    total = R.strength_for(args.mm, S)
    f = args.left_share
    print(f"scale {S:.2f} m/unit, strength {total:.3f} -> {args.mm:.0f} mm, "
          f"left {f * 100:.0f}% / right {(1 - f) * 100:.0f}%", flush=True)

    eyes = []
    for name, s in (("left", -f * total), ("right", (1.0 - f) * total)):
        t = time.time()
        img, hole = warp.right_eye_from_disparity(
            frame, eq.copy(), s, normalize=False,
            gradient_limit=args.gradient_limit, fg_erode=args.fg_erode)
        eyes.append(img)
        print(f"  {name}: {time.time() - t:.0f}s, hole "
              f"{100.0 * (hole > 0).mean():.4f}%", flush=True)

    cv2.imwrite(args.out, np.vstack(eyes),
                pipeline.image_encode_params(os.path.splitext(args.out)[1]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
