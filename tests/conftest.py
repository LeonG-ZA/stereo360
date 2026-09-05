"""Test-wide defaults: the reference warp, and a windowless Qt.

Keep the test suite pointed at the reference (numpy) warp.

`warp.right_eye_from_disparity` delegates to the torch/GPU implementation
whenever torch reports a usable device. On a CUDA machine that would silently
take every warp test off the numpy path — and the numpy path is the reference
the GPU one is checked against, as well as the only path AMD-on-Windows can
use. So the default here is the CPU path, and the GPU is exercised by the
dedicated equivalence test in test_gpu_warp.py.
"""

import os

import pytest

# Qt draws nowhere unless someone asks otherwise.
#
# The interface tests each start a real QML window -- one per test, and there
# are a couple of hundred of them -- so a full run means windows appearing and
# vanishing over whatever else is on screen for twenty-odd minutes. The
# offscreen platform renders the same scene graph without ever creating one,
# and these tests assert on properties and row visibility rather than pixels,
# so nothing they check depends on a window existing.
#
# Set here rather than in a fixture because it has to be true before Qt is
# imported, and `setdefault` rather than assignment so it stays overridable:
# `QT_QPA_PLATFORM=windows pytest ...` puts the windows back when a failure
# needs watching. It reaches the QML subprocesses too, which inherit this
# environment.
#
# A test that ever screenshots the window would need the real platform and
# should say so itself.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _force_cpu_warp(monkeypatch):
    monkeypatch.setenv("STEREO360_GPU_WARP", "0")
