"""Keep the test suite pointed at the reference (numpy) warp.

`warp.right_eye_from_disparity` delegates to the torch/GPU implementation
whenever torch reports a usable device. On a CUDA machine that would silently
take every warp test off the numpy path — and the numpy path is the reference
the GPU one is checked against, as well as the only path AMD-on-Windows can
use. So the default here is the CPU path, and the GPU is exercised by the
dedicated equivalence test in test_gpu_warp.py.
"""

import pytest


@pytest.fixture(autouse=True)
def _force_cpu_warp(monkeypatch):
    monkeypatch.setenv("STEREO360_GPU_WARP", "0")
