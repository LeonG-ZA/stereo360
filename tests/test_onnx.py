"""Tests for the ONNX Runtime depth backend (M4)."""

import os

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

from stereo360.depth.onnx_backend import OnnxDepthBackend, pick_provider

MODEL = "models/depth_anything_v2_small.onnx"


def test_pick_provider_defaults_to_available():
    assert pick_provider() in ort.get_available_providers()


def test_pick_provider_rejects_missing():
    with pytest.raises(RuntimeError):
        pick_provider("NotARealProvider")


@pytest.mark.skipif(not os.path.exists(MODEL),
                    reason="exported ONNX model not present")
def test_onnx_estimate_matches_shape_and_finiteness():
    backend = OnnxDepthBackend(MODEL, provider="CPUExecutionProvider")
    frame = np.random.default_rng(0).integers(
        0, 255, (128, 256, 3), dtype=np.uint8)
    d = backend.estimate(frame)
    assert d.shape == (128, 256)
    assert np.isfinite(d).all()
    # Depth Anything V2 emits inverse depth: non-negative values.
    assert d.min() >= 0
