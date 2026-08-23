"""Render a stereo pair from DA3METRIC-LARGE's metric depth.

Metric depth lets the baseline be set in millimetres instead of guessed. The
warp works in "relative depth units": it forms lam = 1 / (dn + _MIN_INV_DEPTH)
from the normalised inverse depth and shifts the eye by strength *
_BASELINE_SCALE in those units. Normalisation is affine -- dn = (disp - lo) /
(hi - lo) -- so choosing lo and hi pins what one relative unit means in metres.

Ask for lam = D / S, that is, one relative unit per S metres:

    1 / (dn + 0.05) = D / S
    dn = S / D - 0.05 = S * disp - 0.05

and matching that against dn = disp / (hi - lo) - lo / (hi - lo) gives

    hi - lo = 1 / S        lo = 0.05 / S        hi = 1.05 / S

after which the eye separation really is strength * _BASELINE_SCALE * S metres,
and asking for a human 65 mm is arithmetic rather than taste.

The clip to [0, 1] then sets what is representable: dn = 1 at D = S / 1.05 and
dn = 0 at D = 20 S. With S = 1 m that is 0.95 m to 20 m, and everything beyond
20 m is rendered at 20 m -- which costs nothing, because the parallax from a
65 mm baseline at 20 m is already under a pixel at this resolution.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = r"C:\Users\leong\OneDrive\Documents\stereo360"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, OUT)

import cv2
import numpy as np

from stereo360 import pipeline, projection, warp

OV = projection.FACE_OVERLAP

#: Metres per relative depth unit. 1.0 puts the representable range at
#: 0.95 m to 20 m, which covers a street scene from the kerb at your feet to
#: the point where a 65 mm baseline stops producing measurable parallax.
SCALE_M = 1.0

#: What we are actually trying to reproduce.
HUMAN_MM = 65.0


def rescue_outlier_faces(disp, tol=0.35, verbose=True):
    """Rescale only the faces whose metric scale disagrees with the rest.

    A metric model is supposed to make `align_overlapping_faces` unnecessary,
    and for five of the six faces it nearly does -- they agree to within 1.3x.
    The up face does not: shown nothing but sky it has no cue to judge
    distance from, and answers about 10 m, where the side faces put the same
    sky at 88 m. Left alone that paints a ring across the sky at the latitude
    where the faces meet, a step from 0.9 px of parallax to 8 px.

    Running the aligner over all six would fix the ring by moving everything,
    including the five faces whose metric scale is the thing we came for. So
    run it on a copy, read off the ratio it wanted per face, and apply only
    the ratios that are outliers -- measured against the median of the rest,
    which cancels the aligner's arbitrary global gauge.
    """
    probe = {k: v.copy() for k, v in disp.items()}
    projection.align_overlapping_faces(probe, OV)
    ratio = {}
    for k in projection.FACES:
        m = np.isfinite(disp[k]) & (disp[k] > 1e-6) & np.isfinite(probe[k])
        ratio[k] = float(np.median(probe[k][m] / disp[k][m]))

    gauge = float(np.median(list(ratio.values())))
    fixed = []
    for k, r in ratio.items():
        rel = r / gauge
        if abs(np.log(rel)) > np.log(1.0 + tol):
            disp[k] = (disp[k] * rel).astype(np.float32)
            fixed.append((k, rel))
    if verbose:
        for k, rel in fixed:
            print(f"  rescaled {k} by {rel:.3f} "
                  f"(it now reads {1.0 / rel:.1f}x further)")
        if not fixed:
            print("  all six faces agreed; nothing rescaled")
    return disp


def strength_for(mm, scale_m=SCALE_M):
    """--strength that yields `mm` of real eye separation."""
    return (mm / 1000.0) / (warp._BASELINE_SCALE * scale_m)


def metric_to_normalised(disp, scale_m=SCALE_M):
    """In-place affine map so one relative unit is `scale_m` metres."""
    lo = warp._MIN_INV_DEPTH / scale_m
    hi = (1.0 + warp._MIN_INV_DEPTH) / scale_m
    return warp.normalize_inv_depth_with(disp, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="7680p.jpg")
    ap.add_argument("--face", type=int, default=1920)
    ap.add_argument("--res", type=int, default=504)
    ap.add_argument("--faces", default=None,
                    help="explicit per-face depth .npz, e.g. a "
                         "fused one; overrides --face/--res")
    ap.add_argument("--width", type=int, default=7680)
    ap.add_argument("--mm", type=float, default=HUMAN_MM)
    # 5, not 8. The depth edge overhangs the van's silhouette by 7 px, so a
    # reach of 8 does remove the drag -- and takes the van's own bodywork with
    # it, because the white strip between its rear window and its outer edge is
    # only 10 px wide. Measured across the sweep: at 2-4 the strip is perfect
    # and 10 px of road travels with the van; at 8 the road is right and the
    # strip comes out 8 px in one eye against 13 in the other; and 6 and 7 are
    # worse than either end, reporting +4 and -13 px of disparity where the
    # truth is -4. 5 keeps the strip within a pixel and recovers part of the
    # drag. Neither defect is actually solved -- see findings.
    ap.add_argument("--fg-erode", type=int, default=5,
                    help="how far the foreground halo is pulled back to the "
                         "background depth, in px")
    ap.add_argument("--gradient-limit", type=float, default=0.6,
                    help="clamp depth slope as a fraction of the "
                         "injective limit; 0 disables")
    ap.add_argument("--whole-baseline", action="store_true",
                    help="put the entire baseline in the right eye, "
                         "leaving the left pristine")
    ap.add_argument("--scale", type=float, default=None,
                    help="metres per relative unit; default picks the "
                         "smallest that does not clip the nearest thing")
    ap.add_argument("--align", action="store_true",
                    help="run the relative-depth aligner over all six, "
                         "which discards the metric scale")
    ap.add_argument("--keep-sky-face", action="store_true",
                    help="do not rescue the face whose scale is an "
                         "outlier (see rescue_outlier_faces)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = args.image if os.path.isabs(args.image) \
        else os.path.join(REPO, args.image)
    stem = os.path.splitext(os.path.basename(path))[0]
    cache = args.faces or os.path.join(
        OUT, f"da3metric_{stem}_{args.face}_{args.res}.npz")
    if not os.path.exists(cache):
        sys.exit(f"no cached depth at {cache}; run da3_metric.py first")

    z = np.load(cache)
    disp = {k: (1.0 / np.maximum(z[k], 1e-3)).astype(np.float32)
            for k in z.files}
    if args.align:
        projection.align_overlapping_faces(disp, OV)
    elif not args.keep_sky_face:
        rescue_outlier_faces(disp)

    w = args.width
    h = w // 2
    frame = cv2.imread(path)
    if frame.shape[1] != w:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    eq = projection.overlapping_faces_to_equirect(disp, w, h, OV)
    d_near = float(1.0 / np.nanpercentile(eq, 99.9))
    d_far = float(1.0 / max(np.nanpercentile(eq, 0.1), 1e-6))
    print(f"depth at render res: {d_near:.2f} m nearest, {d_far:.1f} m "
          f"farthest (0.1st/99.9th percentile)")

    # The scale is a numerical encoding, not a geometric choice: it fixes what
    # one relative unit means, and `strength_for` divides it straight back out,
    # so the rendered eye separation is the same 65 mm whatever we pick. All it
    # decides is the window [S/1.05, 20 S] that survives the clip to [0, 1].
    # Pick the smallest S that keeps the nearest real surface inside it, which
    # on the road is about 1 m and in a room with a 2.4 m ceiling is less.
    # No cap. An earlier version clamped this to 1.0, which quietly broke
    # the warp's foreground erosion: that stage measures local depth contrast
    # against a fixed 0.05 in normalised units, and with the scale pinned at
    # 1.0 the van at 8.5 m and the road at 14.7 m came out 0.0676 and 0.0180 --
    # a contrast of 0.0496, just under the threshold, so the erosion computed
    # a weight of exactly zero and did nothing at all. The repo's constant is
    # tuned for its percentile normalisation, which spreads a scene across
    # most of [0, 1]; a metric encoding squeezes the midground toward zero and
    # walks under thresholds calibrated against the other convention.
    scale = args.scale if args.scale else d_near * 1.05 * 0.98
    near, far = scale / 1.05, 20.0 * scale
    print(f"scale {scale:.2f} m per unit -> "
          f"representable {near:.2f} m .. {far:.0f} m")
    if d_far > far:
        print(f"    (everything past {far:.0f} m renders at {far:.0f} m; "
              f"a {args.mm:.0f} mm baseline there is under a pixel)")

    metric_to_normalised(eq, scale)
    strength = strength_for(args.mm, scale)
    print(f"strength {strength:.3f} -> {args.mm:.0f} mm eye separation")

    # Split by default, which the fixed-strength renders did not need.
    # A true 65 mm is about twice the baseline those used, and disocclusion
    # area grows far faster than linearly with warp distance -- measured on
    # this scene, going from 36 mm to 65 mm tripled it, from 0.004% of the
    # frame to 0.014%. That does not sound like much until you see where it
    # lands: the largest hole here is a 7 x 147 px strip behind a handrail
    # post, and the directional fill turns it into a vertical streak of
    # stretched grass. Splitting warps each eye half as far in opposite
    # directions, which puts the hole area back to the 36 mm level and, more
    # importantly, puts the holes in *different* places in each eye, so
    # wherever one eye guesses the other has real pixels. The cost is that
    # the left eye is no longer the untouched original.
    split = not args.whole_baseline
    # And clamp the depth gradient, which the fixed-strength renders also
    # did not need. What looked like an unfilled hole beside a handrail post
    # was not a hole at all -- the depth map ramps *across* a thin upright
    # instead of giving it a plateau, so the warp treats post and grass as one
    # continuous surface and stretches the post's dark pixels down the gap.
    # The limit is in units of the slope at which the warp stops being
    # injective, and that slope is inversely proportional to the baseline: at
    # a true 65 mm it is half what it was at 36 mm, so ramps that were safe
    # before now smear. Measured on this scene it takes the residual hole area
    # from 0.0044% to 0.0000% and removes the smear outright.
    left, right = pipeline.stereo_pair(frame, eq, strength, split,
                                       normalize=False,
                                       gradient_limit=args.gradient_limit,
                                       fg_erode=args.fg_erode)
    print(f"split baseline: {split}, gradient limit: "
          f"{args.gradient_limit}, fg erode: {args.fg_erode}")
    tb = np.vstack([left, right])
    out = args.out or os.path.join(OUT, f"{stem}_da3metric_360_TB.jpg")
    # The repo's own settings: quality 100 and 4:4:4 chroma. Not fussiness
    # -- 4:2:0 halves the chroma resolution, and the previous renders this
    # is meant to be compared against were all written at 4:4:4.
    cv2.imwrite(out, tb,
                pipeline.image_encode_params(os.path.splitext(out)[1]))
    print(f"wrote {out}  {tb.shape[1]}x{tb.shape[0]}")


if __name__ == "__main__":
    main()
