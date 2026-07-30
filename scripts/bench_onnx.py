"""M4 benchmark: ONNX backend through the full pipeline at reduced resolution."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from stereo360 import pipeline
from stereo360.depth.onnx_backend import OnnxDepthBackend

H, W = 960, 1920
rng = np.random.default_rng(0)
frame = (rng.random((H, W, 3)) * 255).astype(np.uint8)

backend = OnnxDepthBackend("models/depth_anything_v2_small.onnx")
print(f"provider: {backend.provider}, frame: {W}x{H}")
for mode in ("simple", "learned"):
    for i in range(2):
        t0 = time.time()
        right = pipeline.right_eye_from_depth(frame, W // 4, backend, 1.0,
                                              inpaint_mode=mode)
        print(f"{mode} frame {i}: {time.time() - t0:.2f}s")
