"""Render depth + right eye for a real frame with several flag combos."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from stereo360 import pipeline, warp
from stereo360.depth.depth_anything import DepthAnythingBackend

W, H = 3840, 1920
face = W // 4

frame = cv2.cvtColor(cv2.imread("frame0.png"), cv2.COLOR_BGR2RGB)
if frame is None:
    frame = cv2.cvtColor(cv2.imread("frame0.jpg"), cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
print("frame:", frame.shape, flush=True)

backend = DepthAnythingBackend()
t0 = time.time()
disp = pipeline.depth_map_for_frame(frame, face, backend)
print(f"depth: {time.time() - t0:.0f}s", flush=True)

# Depth visualization (log-ish stretch)
dv = disp.copy()
dv = (dv - dv.min()) / (dv.max() - dv.min())
cv2.imwrite("debug_depth.png", (dv * 255).astype(np.uint8))

warp.normalize_inv_depth(disp)  # in place, per-frame like M2 path

for name, kw in [
    ("default", dict(fg_erode=2)),
    ("noerode", dict(fg_erode=0)),
    ("nostrength", dict(fg_erode=2, strength=0.5)),
]:
    s = kw.pop("strength", 1.0)
    right, hole = warp.right_eye_from_disparity(
        frame, disp.copy(), strength=s, inpaint=False, normalize=False,
        **kw)
    filled = warp.fill_holes(right, hole, disp)
    cv2.imwrite(f"debug_right_{name}.png",
                cv2.cvtColor(filled, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"debug_hole_{name}.png", hole)
    print(f"wrote debug_right_{name}.png hole%={100 * (hole > 0).mean():.1f}",
          flush=True)
