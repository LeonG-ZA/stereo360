"""Keep the sharp edges of a high-res run and the metric scale of the calibrated one.

Running DA3METRIC at process_res 1008 instead of its default 504 sharpens the
depth edge across a thin upright from 6 px to 2 px -- which is the whole
thin-structure complaint -- and simultaneously breaks the metric scale: the
camera reads 0.84 m above the road where the 504 run said 1.58 m and straight
down measured 1.6 m. The input size is part of what the model was calibrated
against, not a free quality knob.

A single rescale does not repair it. The ratio between the two runs has a
median of 1.98 but an interquartile spread of 1.23, so dividing by a constant
leaves about 23% of depth-dependent distortion.

But look at what each run is good for. The scale error is *smooth*: it comes
from the model reading the wrong implied focal length, which varies slowly
across the sphere. The detail is *local*: it is the 2 px edge at the post. So
divide the 1008 map by a heavily low-passed version of its own ratio against
the 504 map. Low spatial frequencies -- the metric scale -- come from the
calibrated run; high spatial frequencies -- the edges -- come from the sharp
one. Same reasoning as `ground.flatten` subtracting only the smooth part of
the residual, and the same reason it does not print an edge anywhere.

Diagnostic only.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

#: Degrees of arc the scale correction is smoothed over. The calibration error
#: spans the whole sphere; the edge we are protecting spans a fraction of a
#: degree. Anything between works.
SMOOTH_DEG = 12.0


def _wrap_blur(a: np.ndarray, k: int) -> np.ndarray:
    """Box blur that wraps in longitude, as the equirect does."""
    pad = k // 2
    wide = np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
    out = cv2.blur(wide, (k, k), borderType=cv2.BORDER_REPLICATE)
    return out[:, pad:pad + a.shape[1]]


def fuse(sharp: np.ndarray, calibrated: np.ndarray,
         smooth_deg: float = SMOOTH_DEG) -> np.ndarray:
    """Sharp map's detail, calibrated map's scale. Both are inverse depth."""
    ok = ((sharp > 1e-6) & (calibrated > 1e-6)
          & np.isfinite(sharp) & np.isfinite(calibrated))
    # In the log domain a scale error is a constant offset, so the blur is
    # doing an average of ratios rather than a ratio of averages -- which is
    # what "smooth part of the scale error" actually means.
    lr = np.zeros_like(sharp, np.float32)
    lr[ok] = np.log(sharp[ok] / calibrated[ok])
    m = ok.astype(np.float32)

    k = max(3, int(round(smooth_deg / 360.0 * sharp.shape[1])) | 1)
    num, den = _wrap_blur(lr * m, k), _wrap_blur(m, k)
    smooth = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 0.0)
    return (sharp / np.exp(smooth)).astype(np.float32)


if __name__ == "__main__":
    OUT = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(
                          os.path.dirname(os.path.abspath(__file__)))))
    from stereo360 import ground

    sharp = np.load(os.path.join(OUT, "da3metric_7680p_1008_eq.npy"))


def fuse_faces(sharp_npz, cal_npz, out_npz, smooth_deg=SMOOTH_DEG,
               face_fov_deg=112.0):
    """Same fusion, per cubemap face, so the rest of the pipeline is unchanged.

    Working on the faces rather than the assembled equirect matters: the sky
    face still needs its scale rescued against its neighbours, and that runs
    on faces. Doing this first keeps both corrections in their right order.

    The faces hold metric *depth* in metres and everything above is in inverse
    depth, so convert on the way in and back on the way out. No wrap here --
    a face is a flat perspective image and its border genuinely is a border.
    """
    a, b = np.load(sharp_npz), np.load(cal_npz)
    out = {}
    for k in a.files:
        sharp = 1.0 / np.maximum(a[k], 1e-3)
        cal = 1.0 / np.maximum(b[k], 1e-3)
        px = max(3, int(round(smooth_deg / face_fov_deg * sharp.shape[1])) | 1)
        lr = np.log(np.maximum(sharp, 1e-9) / np.maximum(cal, 1e-9))
        smooth = cv2.blur(lr.astype(np.float32), (px, px),
                          borderType=cv2.BORDER_REPLICATE)
        out[k] = (1.0 / (sharp / np.exp(smooth))).astype(np.float32)
    np.savez_compressed(out_npz, **out)
    return out


if __name__ == "__main__":
    OUT = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.environ["STEREO360_REPO"])
    from stereo360 import ground, projection

    fuse_faces(os.path.join(OUT, "da3metric_7680p_1920_1008.npz"),
               os.path.join(OUT, "da3metric_7680p_1920_504.npz"),
               os.path.join(OUT, "da3metric_7680p_1920_fused.npz"))
    print("wrote da3metric_7680p_1920_fused.npz\n")

    for tag, name in (("504 calibrated", "da3metric_7680p_1920_504.npz"),
                      ("1008 sharp", "da3metric_7680p_1920_1008.npz"),
                      ("fused", "da3metric_7680p_1920_fused.npz")):
        z = np.load(os.path.join(OUT, name))
        d = {k: (1.0 / np.maximum(z[k], 1e-3)).astype(np.float32)
             for k in z.files}
        eq = projection.overlapping_faces_to_equirect(
            d, 7680, 3840, projection.FACE_OVERLAP)
        pl = ground.fit_plane(eq)
        row = 1.0 / eq[2510, 4165:4255]
        near, far = row.min(), np.percentile(row, 90)
        band = int(((row > near * 1.25) & (row < far * 0.8)).sum())
        h = f"{pl.height:.2f} m" if pl.ok else "no plane"
        print(f"  {tag:<16} camera {h:>8}   edge {band} px")
