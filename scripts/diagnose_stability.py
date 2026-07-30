"""Diagnose right-eye temporal instability: measure per-pixel variance
introduced at each pipeline stage over a few frames of input.mp4.

Stages measured (per-pixel std across time, lower = more stable):
  A. temporal depth itself (VDA small, chunk of N equirect frames)
  B. warped right eye, no inpaint, per-frame depth
  C. warped right eye, no inpaint, FIXED depth (frame 0) -- isolates
     warp-induced variance from depth variance
  D. after background-biased Telea fill
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from stereo360 import ffmpeg_io, warp
from stereo360.depth.video_depth_anything import VideoDepthAnythingBackend

N = 4
W, H = 1280, 640  # downscaled for CPU speed

print("decoding...", flush=True)
frames = []
for i, f in enumerate(ffmpeg_io.decode_frames("input.mp4", max_frames=N)):
    frames.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))

print("loading VDA small...", flush=True)
backend = VideoDepthAnythingBackend("small")
t0 = time.time()
depths = backend.estimate_chunk(frames)
print(f"depth chunk: {time.time() - t0:.0f}s", flush=True)

# A: depth stability (chunk-consistent normalization like the pipeline)
sample = np.concatenate([d[::8, ::8].ravel() for d in depths])
lo, hi = np.percentile(sample, 1), np.percentile(sample, 99)
dn = [warp.normalize_inv_depth_with(d, lo, hi) for d in depths]
stack = np.stack(dn)
print(f"A. depth per-pixel std: mean={stack.std(axis=0).mean():.4f} "
      f"p99={np.percentile(stack.std(axis=0), 99):.4f}", flush=True)

# B vs C: warp stability with own vs fixed depth
from stereo360.temporal_fill import stabilize_depth

dn_stab = [d.copy() for d in dn]
stabilize_depth(dn_stab)
def warp_all(depth_list):
    rights = []
    for i, f in enumerate(frames):
        r, hole = warp.right_eye_from_disparity(
            f, depth_list[i].copy(), strength=0.8, fg_erode=3,
            inpaint=False, normalize=False)
        rights.append(r.astype(np.float32))
    s = np.stack(rights)
    return s.std(axis=0).mean(axis=-1)  # (H, W) std in gray levels

std_own = warp_all(dn)
std_stab = warp_all(dn_stab)
std_fix = warp_all([dn[0]] * N)
print(f"B. warp+own depth std:  mean={std_own.mean():.2f} "
      f"p99={np.percentile(std_own, 99):.2f}", flush=True)
print(f"B2. warp+stabilized depth std: mean={std_stab.mean():.2f} "
      f"p99={np.percentile(std_stab, 99):.2f}", flush=True)
print(f"C. warp+fixed depth std: mean={std_fix.mean():.2f} "
      f"p99={np.percentile(std_fix, 99):.2f}", flush=True)

# D: after fill (Telea, background-biased)
filled = []
for i, f in enumerate(frames):
    r, hole = warp.right_eye_from_disparity(
        f, dn[i].copy(), strength=0.8, fg_erode=3,
        inpaint=False, normalize=False)
    r = warp.fill_holes(r, hole, dn[i])
    filled.append(r.astype(np.float32))
std_fill = np.stack(filled).std(axis=0).mean(axis=-1)
print(f"D. after Telea fill std: mean={std_fill.mean():.2f} "
      f"p99={np.percentile(std_fill, 99):.2f}", flush=True)
