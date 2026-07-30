"""Smoke test: ONNX backend vs PyTorch backend on a real frame."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from stereo360.depth.depth_anything import DepthAnythingBackend
from stereo360.depth.onnx_backend import OnnxDepthBackend, pick_provider

frame = cv2.cvtColor(cv2.imread("frame0.png"), cv2.COLOR_BGR2RGB)
if frame is None:
    frame = cv2.cvtColor(cv2.imread("frame0.jpg"), cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (512, 256))

onnx = OnnxDepthBackend("models/depth_anything_v2_small.onnx")
print("provider:", onnx.provider)
t0 = time.time()
d_onnx = onnx.estimate(frame)
t1 = time.time()
d_onnx2 = onnx.estimate(frame)
print(f"onnx: first {t1 - t0:.2f}s, warm {time.time() - t1:.2f}s, "
      f"range [{d_onnx.min():.1f}, {d_onnx.max():.1f}]")

pt = DepthAnythingBackend()
d_pt = pt.estimate(frame)


def norm(d):
    return (d - d.min()) / (d.max() - d.min() + 1e-9)


corr = np.corrcoef(norm(d_onnx).ravel(), norm(d_pt).ravel())[0, 1]
print(f"correlation ONNX vs PyTorch: {corr:.4f}")
