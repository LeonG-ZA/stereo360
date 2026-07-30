"""Measure RSS of the 8K per-frame depth->warp pipeline with a dummy backend."""
import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil

from stereo360 import pipeline
from stereo360.depth.base import DepthBackend


class Dummy(DepthBackend):
    def estimate(self, f):
        rng = np.random.default_rng(0)
        return rng.random((f.shape[0], f.shape[1]), dtype=np.float32)


def main():
    proc = psutil.Process(os.getpid())
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (3840, 7680, 3), dtype=np.uint8)
    b = Dummy()
    for i in range(3):
        right = pipeline.right_eye_from_depth(frame, 1920, b, 1.0)
        del right
        gc.collect()
        print(f"iter {i}: RSS={proc.memory_info().rss / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
