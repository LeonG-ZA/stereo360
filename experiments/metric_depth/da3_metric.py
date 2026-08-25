"""Run DA3METRIC-LARGE over the six faces and see what metric depth buys us.

Three questions, in order of how much they matter.

1. Is the metric scale real? Every stereo render so far has had an arbitrary
   per-scene scale -- the same --strength 1.2 gave a 49 mm eye separation on
   the road and 35 mm indoors, against a human 65 mm -- because a relative
   depth model has no idea how big anything is. A metric model claims to know.
   The ground-plane fit gives a direct check that needs no ground truth beyond
   common sense: `GroundPlane.height` is the camera's height above the road,
   in whatever units the depth is in. A camera on a tripod or a car roof is
   1.5-2.5 m up. If the fit says that, the metric scale survived being shown a
   98 degree crop; if it says 6 m, it did not.

2. Do the six faces agree without being told to? `align_overlapping_faces`
   exists because six independent relative-depth estimates each carry their
   own arbitrary scale. Six metric estimates should not need it. The spread of
   the six per-face scale factors is the measurement, and it is also a second,
   independent read on question 1.

3. Is the ground flat? The road's departure from its own fitted plane is
   0.141 under V3 and that is the number the headset reads as a bulge.

Everything here writes its depth out to .npy so the render step never has to
re-run inference.

Diagnostic only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.environ.get(
    "STEREO360_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, OUT)

import cv2
import numpy as np

import da3_shim
from stereo360 import ground, projection

MODEL = "depth-anything/DA3METRIC-LARGE"
OV = projection.FACE_OVERLAP

#: The model's own patch grid is 14 px, and 504 = 36 x 14 is what its API
#: defaults to. Feeding a larger face buys nothing -- it is resized to this
#: before the backbone sees it -- so the face size only has to be enough not
#: to throw away detail on the way in.
PROCESS_RES = 504

#: Match what the repo picks for a 7680-wide source (width / 4, its lossless
#: value). Sampling the faces at 512 straight from a 7680 equirect throws
#: away three pixels in four, and it throws them away worst on exactly the
#: thin uprights this is meant to measure. Extract at full rate and let the
#: model's own resize do the reduction.
FACE = 1920


def load_model(verbose=True):
    DA3, faked = da3_shim.import_api()
    if verbose:
        print("shimmed away:", ", ".join(faked), flush=True)
    import torch
    if da3_shim.patch_for_cpu(DA3) and verbose:
        print("no CUDA: running fp32 on the CPU", flush=True)
    torch.set_num_threads(os.cpu_count() or 4)
    t = time.time()
    model = DA3.from_pretrained(MODEL).eval()
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"loaded {n / 1e6:.0f}M params in {time.time() - t:.0f}s",
              flush=True)
    return model


def depth_faces(model, img, face=FACE, cache=None, res=PROCESS_RES):
    """Metric depth for each of the six widened faces, in metres."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        print(f"reusing {os.path.basename(cache)}", flush=True)
        return {k: z[k] for k in z.files}

    import torch
    faces = projection.equirect_to_overlapping_faces(img, face, OV)
    out = {}
    for i, name in enumerate(projection.FACES):
        t = time.time()
        rgb = cv2.cvtColor(faces[name], cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            # Nothing is written: the export is gated on export_dir, which
            # defaults to None. export_format still has to be a string --
            # inference does `"gs" in export_format` before looking at it.
            pred = model.inference([rgb], process_res=res)
        d = np.squeeze(np.asarray(pred.depth)).astype(np.float32)
        if d.shape != (face, face):
            d = cv2.resize(d, (face, face), interpolation=cv2.INTER_LINEAR)
        out[name] = d
        print(f"  [{i + 1}/6] {name:<6} {time.time() - t:5.1f}s  "
              f"depth {np.nanmin(d):6.2f} .. {np.nanmedian(d):6.2f} .. "
              f"{np.nanmax(d):7.2f} m", flush=True)
    if cache:
        np.savez_compressed(cache, **out)
    return out


def to_disp(metric):
    """Metric depth (metres) -> inverse depth (1/metres), which is what every
    stage downstream of here speaks."""
    return {k: (1.0 / np.maximum(v, 1e-3)).astype(np.float32)
            for k, v in metric.items()}


def face_agreement(disp):
    """How far apart the six faces are before anyone aligns them.

    `align_overlapping_faces` fits each face to its neighbours over the band
    they share. Run it on a copy and read off how much it had to move things:
    for a metric model that should be nearly nothing.
    """
    before = {k: v.copy() for k, v in disp.items()}
    after = {k: v.copy() for k, v in disp.items()}
    projection.align_overlapping_faces(after, OV)
    rows = []
    for k in projection.FACES:
        b, a = before[k], after[k]
        m = np.isfinite(b) & np.isfinite(a) & (b > 1e-6)
        rows.append((k, float(np.median(a[m] / b[m]))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="7680p.jpg")
    ap.add_argument("--face", type=int, default=FACE)
    ap.add_argument("--w", type=int, default=7680)
    ap.add_argument("--res", type=int, default=PROCESS_RES,
                    help="model input size; multiples of 14")
    args = ap.parse_args()

    path = args.image if os.path.isabs(args.image) \
        else os.path.join(REPO, args.image)
    stem = os.path.splitext(os.path.basename(path))[0]
    cache = os.path.join(OUT,
                         f"da3metric_{stem}_{args.face}_{args.res}.npz")

    face = max(args.face, args.res)
    img = cv2.imread(path)
    print(f"{os.path.basename(path)}  {img.shape[1]}x{img.shape[0]}\n",
          flush=True)

    model = None if (cache and os.path.exists(cache)) else load_model()
    metric = depth_faces(model, img, face, cache, args.res)
    disp = to_disp(metric)

    print("\n--- do the six faces agree on scale by themselves? ---")
    print("    (ratio the aligner had to apply; 1.000 = already agreed)")
    for name, r in face_agreement(disp):
        print(f"      {name:<6} {r:.3f}")

    w, h = args.w, args.w // 2
    for label, align in (("metric, as predicted", False),
                         ("metric, then aligned", True)):
        d = {k: v.copy() for k, v in disp.items()}
        if align:
            projection.align_overlapping_faces(d, OV)
        eq = projection.overlapping_faces_to_equirect(d, w, h, OV)
        pl = ground.fit_plane(eq)
        print(f"\n--- {label} ---")
        print(f"    ground plane: {pl.describe()}")
        if pl.ok:
            print(f"    camera height above the road: {pl.height:.2f} m")
        if not align:
            np.save(os.path.join(OUT,
                                 f"da3metric_{stem}_{args.res}_eq.npy"),
                    eq)
            print(f"    saved equirect metric inverse depth "
                  f"({w}x{h}) for the render step")


if __name__ == "__main__":
    main()
