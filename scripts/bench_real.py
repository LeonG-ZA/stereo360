"""Measure peak RSS of the real depth backend + 8K pipeline over a few frames."""
import gc
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil

from stereo360 import pipeline
from stereo360.depth.depth_anything import DepthAnythingBackend

_peak = 0
_stop = False


def _watch(proc):
    global _peak
    while not _stop:
        _peak = max(_peak, proc.memory_info().rss)
        time.sleep(0.2)


def main():
    global _stop
    proc = psutil.Process(os.getpid())
    t = threading.Thread(target=_watch, args=(proc,), daemon=True)
    t.start()

    print("loading backend...", flush=True)
    backend = DepthAnythingBackend()
    gc.collect()
    print(f"after model load: RSS={proc.memory_info().rss / 2**30:.2f} GiB", flush=True)

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (3840, 7680, 3), dtype=np.uint8)
    for i in range(2):
        t0 = time.time()
        right = pipeline.right_eye_from_depth(frame, 1920, backend, 1.0)
        del right
        gc.collect()
        print(f"frame {i}: {time.time() - t0:.1f}s  "
              f"RSS={proc.memory_info().rss / 2**30:.2f} GiB  "
              f"peak={_peak / 2**30:.2f} GiB", flush=True)

    _stop = True


if __name__ == "__main__":
    main()
