"""Render a full stereo pair with the mesh warp, for headset comparison.

Same depth, same 65 mm baseline, same split and the same hole filler as the
splat path -- the only thing that differs is how the surface is rasterised, so
anything visible in the headset is attributable to the renderer.

Two differences from the splat path are deliberate and worth knowing while
judging it:

  * No `gradient_limit`. The splat needs it because it keeps the warp
    injective by rewriting depth values, which flattens small objects (on the
    lamp it spread the depth step over ~12 px and ate a fifth of a 63 px ball
    from each side). A mesh cuts geometry instead, so the depth is left alone.

  * `fg_erode` is applied once, up front, to the depth map itself rather
    than inside each eye's warp. Leaving it out was a mistake: the reasoning
    was that explicit cuts put the hole on the silhouette so nothing needs
    eroding, and that is true for the *cut*, but it does nothing about the
    depth map's halo. The depth edge at the van sits 7 px outside the van's
    actual silhouette, and a mesh renders that faithfully -- measured, road
    10 px past the colour edge still moved at the van's disparity, where the
    splat had released it by 5 px. It is applied to the shared depth, not per
    eye, because the erosion does not depend on which way the baseline goes.

Measured on three crops, the mesh's error above its own noise floor was 0.99
against the splat's 4.59 on the sign post, 2.63 against 6.11 on the lamp, and
5.53 against 5.05 on the handrail -- two clear wins and a tie. But this is a
prototype rasteriser: it interpolates to samples and then remaps, two
resamplings where a real barycentric rasteriser does one, and that costs it a
noise floor of its own the splat does not have (5.68 / 3.67 / 1.90 on those
same crops, against the splat's 0.00). Expect the mesh render to look slightly
softer overall. That softness is the prototype, not the idea.
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
import mesh_warp
import mesh_raster_gpu
from stereo360 import pipeline, projection, warp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="7680p.jpg")
    ap.add_argument("--faces", default=None)
    ap.add_argument("--eq", default=None,
                    help="a precomputed equirect inverse-depth .npy instead "
                         "of per-face metric depth")
    ap.add_argument("--anchor-height", type=float, default=None,
                    help="metres from camera to the ground plane. Relative "
                         "depth has no scale of its own, so anchoring the "
                         "fitted plane to a known height makes it comparable "
                         "with the metric model's render")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=7680)
    ap.add_argument("--mm", type=float, default=65.0)
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--left-share", type=float, default=0.5,
                    help="fraction of the separation the left eye "
                         "takes; 0.5 is an even split, 0.0 leaves "
                         "the left eye as the source")
    ap.add_argument("--whole-baseline", action="store_true",
                    help="leave the left eye as the untouched "
                         "source and put the whole baseline in "
                         "the right")
    ap.add_argument("--no-directional", action="store_true",
                    help="fill holes by inpainting rather than by "
                         "mirroring the neighbouring background")
    ap.add_argument("--cut", type=float, default=mesh_warp.CUT_RATIO)
    ap.add_argument("--raster", action="store_true",
                    help="use the scanline rasteriser on the GPU instead of "
                         "the subsampling scatter. Computes coverage per "
                         "output pixel, so it reproduces its input exactly "
                         "at zero baseline and has no hairlines; also ~11x "
                         "faster. `--subdiv` is ignored when it is on")
    # Off. Applying it cost far more than the halo it fixed: the railing
    # posts and the indoor chair posts came back almost entirely eaten. The
    # cut mask is not the cause -- measured, eroding *reduced* cut area over
    # the railings from 2.11% to 1.72% -- so it is not that the ramp gets cut.
    # What the probe does show is a 17 px post keeping its peak depth (0.439)
    # and contrast (0.181) while its transition band doubles from 6 px to
    # 12 px, so the post becomes a spike rather than a plateau and the
    # background can take the depth test through its flanks. That is a
    # hypothesis, not a proven mechanism. Either way the trade is bad: a
    # slightly thick van edge is worth far less than intact railings.
    ap.add_argument("--fg-erode", type=int, default=0,
                    help="pull the foreground halo back to the background "
                         "depth by this many px before meshing; 0 disables")
    args = ap.parse_args()

    path = args.image if os.path.isabs(args.image) \
        else os.path.join(REPO, args.image)
    w = args.width
    h = w // 2
    frame = cv2.imread(path)
    if frame.shape[1] != w:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    if args.eq:
        eq = np.load(args.eq).astype(np.float32)
    else:
        z = np.load(args.faces)
        disp = {k: (1.0 / np.maximum(z[k], 1e-3)).astype(np.float32)
                for k in z.files}
        R.rescue_outlier_faces(disp)
        eq = projection.overlapping_faces_to_equirect(disp, w, h,
                                                      projection.FACE_OVERLAP)

    if args.anchor_height:
        # Relative depth carries no scale. Scaling inverse depth by k scales
        # the fitted plane's height by 1/k, so pick k that puts the camera at
        # the height the metric model measured for this scene. Everything
        # downstream then means the same thing it does for the metric render,
        # and the 65 mm is a real 65 mm rather than a guess.
        from stereo360 import ground
        pl = ground.fit_plane(eq)
        if not pl.ok:
            sys.exit(f"cannot anchor: {pl.describe()}")
        k = pl.height / args.anchor_height
        eq *= k
        print(f"anchored: plane was {pl.height:.2f} units, scaled by "
              f"{k:.3f} to sit at {args.anchor_height:.2f} m", flush=True)

    d_near = float(1.0 / np.nanpercentile(eq, 99.9))
    scale = d_near * 1.05 * 0.98
    R.metric_to_normalised(eq, scale)
    strength = R.strength_for(args.mm, scale)
    half = strength * 0.5
    b_units = half * warp._BASELINE_SCALE
    print(f"scale {scale:.2f} m/unit, strength {strength:.3f} "
          f"-> {args.mm:.0f} mm, baseline {b_units:.5f} units", flush=True)

    eq_cut = None
    if args.fg_erode > 0:
        # Keep the un-eroded copy: it decides where geometry is cut, while the
        # eroded one decides where the surface goes. See render_full's dn_cut.
        eq_cut = eq.copy()
        warp._erode_foreground(eq, args.fg_erode)
        print(f"  eroded the foreground halo by {args.fg_erode} px "
              f"(cuts still decided from the un-eroded depth)", flush=True)

    eyes = []
    # Whole baseline: the left eye is the original, copied through rather
    # than rendered. With the scatter renderer it *has* to be copied: it
    # rounds its samples and so does not reproduce its input even when
    # nothing moves (a residual of 1.9 to 5.7 against the splat's 0.00), and
    # rendering an identity warp would soften a frame that needs no warping.
    # `--raster` no longer has that problem -- measured, it reproduces the
    # source exactly at zero baseline, 0 levels on every pixel -- but the
    # copy is still both correct and free, so it stays either way.
    #
    # Worth it where one eye's shift *occludes* rather than reveals: hiding
    # something behind a nearer surface costs nothing, while revealing what
    # was behind it has to be invented. Splitting halves each eye's warp but
    # gives *both* eyes something to invent, and on this indoor frame that is
    # what put a mirrored phantom of a tap into the left eye, which whole
    # baseline cannot do because the left eye is never touched.
    f = 0.0 if args.whole_baseline else args.left_share
    plan = (("right", 2.0 * (1.0 - f)),) if f <= 0.0         else (("left", -2.0 * f), ("right", 2.0 * (1.0 - f)))
    if f <= 0.0:
        eyes.append(frame)
    # b_units is half the separation, so the multipliers above are shares of
    # the whole: an even split gives -1 and +1, and 15/85 gives -0.3 and +1.7.
    for name, sgn in plan:
        t = time.time()
        render = (mesh_raster_gpu.render_full if args.raster
                  else mesh_warp.render_full)
        img, cut = render(eq, frame, sgn * b_units,
                          cut_ratio=args.cut, subdiv=args.subdiv,
                          dn_cut=eq_cut)
        print(f"  {name}: {time.time() - t:.0f}s, cut {100.0 * cut.mean():.2f}%",
              flush=True)
        if cut.any():
            # `_directional_fill` mirrors the neighbouring background
            # across the hole boundary, which continues texture rather than
            # smearing one colour -- right for grass or carpet, and wrong when
            # the neighbour is a recognisable object, which comes back
            # reversed beside itself.
            img = warp.fill_holes(img, (cut * 255).astype(np.uint8), eq,
                                  directional=not args.no_directional,
                                  baseline_sign=1.0 if sgn > 0 else -1.0)
        eyes.append(img)

    tb = np.vstack(eyes)
    cv2.imwrite(args.out, tb,
                pipeline.image_encode_params(os.path.splitext(args.out)[1]))
    print(f"wrote {args.out}  {tb.shape[1]}x{tb.shape[0]}")


if __name__ == "__main__":
    main()
