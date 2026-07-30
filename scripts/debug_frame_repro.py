"""Single-frame repro: depth -> warp -> save right eye + hole mask + depth.

Usage: python scripts/debug_frame_repro.py [frame] [out_prefix]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from stereo360 import projection, warp
from stereo360.depth.onnx_backend import OnnxDepthBackend
from stereo360.pipeline import depth_map_for_frame

frame_path = sys.argv[1] if len(sys.argv) > 1 else "frame.png"
prefix = sys.argv[2] if len(sys.argv) > 2 else "repro"

frame = cv2.imread(frame_path)
assert frame is not None, frame_path
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
h, w = frame.shape[:2]
if h == w:  # stacked top/bottom stereo -> take the left-eye (top) equirect
    frame = frame[: h // 2]
    h, w = frame.shape[:2]
face_size = min(512, h)

backend = OnnxDepthBackend("models/depth_anything_v2_small.onnx")
disp = depth_map_for_frame(frame, face_size, backend)

# Depth visualization
dn = (disp - disp.min()) / (disp.max() - disp.min() + 1e-9)
cv2.imwrite(f"{prefix}_depth.png", (dn * 255).astype(np.uint8))

right, hole = warp.right_eye_from_disparity(frame, disp, strength=1.0)
cv2.imwrite(f"{prefix}_right.png",
            cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{prefix}_hole.png", hole)
print("hole pixels:", int((hole > 0).sum()), "of", hole.size)
