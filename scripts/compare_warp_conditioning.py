"""A/B the OLD forward colour-splat warp against the NEW inverse warp.

Measures *conditioning*: how much the synthesized right eye changes when the
depth map is perturbed by a small amount, using a real frame and real depth.

Frame-to-frame depth noise on thin structures is exactly such a perturbation.
A well-conditioned renderer responds proportionally (sub-pixel shift -> small,
smooth change). Nearest-neighbour forward splatting responds in discrete +/-1
pixel jumps, which is what makes railings change shape between frames.

Run:  python scripts/compare_warp_conditioning.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from stereo360 import ffmpeg_io, pipeline, warp
from stereo360.projection import _dir_to_equirect_uv, _equirect_uv_to_dir

W, H = 1280, 640
FACE = 320
STRENGTH = 2.0
SEED = 0


def old_forward_warp(left_rgb, dn, strength=STRENGTH, chunk_rows=256):
    """The previous implementation: forward-splat COLOUR to the nearest
    integer target pixel, z-buffered. Reproduced verbatim for comparison."""
    h, w = dn.shape
    baseline = strength * warp._BASELINE_SCALE
    src_flat = left_rgb.reshape(-1, 3)
    n = h * w

    zbuf = np.full(n, np.inf, dtype=np.float64)
    winner = np.full(n, -1, dtype=np.int32)

    for y0 in range(0, h, chunk_rows):
        y1 = min(y0 + chunk_rows, h)
        rows = y1 - y0
        lam = 1.0 / (dn[y0:y1] + warp._MIN_INV_DEPTH)
        v, u = np.meshgrid(np.arange(y0, y1), np.arange(w), indexing="ij")
        d = _equirect_uv_to_dir(u.astype(np.float32), v.astype(np.float32), w, h)
        p = lam[..., None] * d
        p[..., 0] -= baseline
        norm = np.linalg.norm(p, axis=-1, keepdims=True)
        tu, tv = _dir_to_equirect_uv(p / norm, w, h)
        lam_r = norm[..., 0].astype(np.float64).ravel()
        src_idx = (y0 * w) + np.arange(rows * w, dtype=np.int32)
        ix = np.clip(np.round(tu).astype(np.int32), 0, w - 1).ravel()
        iy = np.clip(np.round(tv).astype(np.int32), 0, h - 1).ravel()
        dst = iy.astype(np.int64) * w + ix
        np.minimum.at(zbuf, dst, lam_r)
        won = lam_r == zbuf[dst]
        winner[dst[won]] = src_idx[won]

    right = np.zeros_like(left_rgb)
    hit = winner >= 0
    right.reshape(-1, 3)[hit] = src_flat[winner[hit]]
    hole = np.where(hit, 0, 255).astype(np.uint8).reshape(h, w)
    return right, hole


def new_inverse_warp(left_rgb, dn, strength=STRENGTH):
    return warp.right_eye_from_disparity(
        left_rgb, dn.copy(), strength=strength, inpaint=False,
        fg_erode=0, normalize=False)


def response(fn, left, dn, noise, roi):
    """Output change (grey levels) over `roi` pixels valid in both renders."""
    a, hole_a = fn(left, dn.copy())
    b, hole_b = fn(left, (dn + noise).astype(np.float32))
    both = (hole_a == 0) & (hole_b == 0) & roi
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=-1)
    sel = diff[both]
    return sel.mean(), (sel > 8).mean(), (sel > 32).mean()


def main():
    print("decoding frame...", flush=True)
    frame = next(iter(ffmpeg_io.decode_frames("input.mp4", max_frames=1)))
    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

    print("estimating depth (onnx)...", flush=True)
    from stereo360.depth.onnx_backend import OnnxDepthBackend

    backend = OnnxDepthBackend("models/depth_anything_v2_small.onnx")
    disp = pipeline.depth_map_for_frame(frame, FACE, backend)
    dn = warp.normalize_inv_depth(disp).astype(np.float32)

    # Thin structures / silhouettes = large depth gradient. This is where the
    # artifact lives; whole-frame averages are dominated by flat regions.
    gx = cv2.Sobel(dn, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dn, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    edge = cv2.dilate((grad > 0.05).astype(np.uint8),
                      np.ones((3, 3), np.uint8)).astype(bool)
    allpx = np.ones_like(edge)
    print(f"depth-edge pixels: {edge.mean():.2%} of frame")

    rng = np.random.default_rng(SEED)
    base = rng.standard_normal((H, W)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 3.0)
    base /= np.abs(base).max()

    for name, roi in (("WHOLE FRAME", allpx), ("DEPTH EDGES", edge)):
        print(f"\n=== {name} ===")
        print(f"{'noise':>7} | {'OLD forward splat':>30} | "
              f"{'NEW inverse warp':>30}")
        print(f"{'':>7} | {'mean':>8} {'>8GL':>9} {'>32GL':>10} | "
              f"{'mean':>8} {'>8GL':>9} {'>32GL':>10}")
        print("-" * 74)
        for eps in (0.005, 0.02, 0.05):
            noise = base * eps
            om, o8, o32 = response(old_forward_warp, frame, dn, noise, roi)
            nm, n8, n32 = response(new_inverse_warp, frame, dn, noise, roi)
            print(f"{eps:>7.3f} | {om:>8.3f} {o8:>8.2%} {o32:>9.2%} | "
                  f"{nm:>8.3f} {n8:>8.2%} {n32:>9.2%}")

    print("\nmean  = mean abs change in grey levels for that depth perturbation"
          "\n>8GL  = fraction of pixels jumping more than 8 grey levels"
          "\n>32GL = fraction jumping more than 32 grey levels (visible pops)"
          "\nSmooth sub-pixel response raises 'mean' slightly but must collapse"
          "\nthe jump tails -- those tails are the shape-changing artifact.")


if __name__ == "__main__":
    main()
