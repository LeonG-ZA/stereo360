"""Video Depth Anything's output convention.

The backend used to invert this model's output, on the belief that it returned
metric depth. It returns *inverse* depth, the same convention as Depth
Anything V2, so inverting flipped near and far on every frame -- and outdoors
it destroyed the stereo entirely, because the model reports exactly 0 for sky
and 1/max(0, 1e-4) turned every sky pixel into 10000.

Measured on real footage before the fix: raw output correlated +0.99 with
Depth Anything V2, the inverted output -0.67; a landscape's sky read 0.019
against 6.05 for ground a metre away; and disparity on a nearby bench was
1.3px where the default backend gave 11.4px.
"""

import numpy as np
import pytest

from stereo360.depth.video_depth_anything import VideoDepthAnythingBackend


class _FakeModel:
    """Stands in for the real network: returns a known inverse-depth map."""

    def __init__(self, values):
        self.values = values

    def infer_video_depth(self, batch, **kw):
        n = len(batch)
        return [self.values.copy() for _ in range(n)], None


def _backend(values):
    """A backend wired to the fake model, without loading real weights."""
    b = object.__new__(VideoDepthAnythingBackend)
    b._model = _FakeModel(values)
    b._input_size = 518
    b._fp16 = False
    b.device = "cpu"
    return b


def test_output_is_not_inverted():
    """Larger in, larger out. Inverting would reverse the ordering."""
    values = np.array([[0.0, 1.0], [2.0, 8.0]], np.float32)
    out = _backend(values).estimate_chunk(
        [np.zeros((2, 2, 3), np.uint8)])
    got = np.asarray(out[0], np.float64)

    assert got[1, 1] > got[1, 0] > got[0, 1], "near/far ordering flipped"
    np.testing.assert_allclose(got, values, rtol=1e-6)


def test_zero_stays_zero_rather_than_becoming_enormous():
    """The model reports 0 for sky, meaning 'no estimate' -- which as inverse
    depth is infinitely far. Inverting turned it into 10000, four orders of
    magnitude nearer than anything real, and that one value set the
    normalisation range for the whole frame."""
    values = np.zeros((4, 4), np.float32)
    values[3, 3] = 5.0                       # one real, near pixel
    out = np.asarray(_backend(values).estimate_chunk(
        [np.zeros((4, 4, 3), np.uint8)])[0], np.float64)

    assert out[0, 0] == 0.0, "sky must read as infinitely far, not nearest"
    assert out[3, 3] == pytest.approx(5.0)
    assert out.max() == pytest.approx(5.0), "no pixel may exceed the scene"


def test_a_sky_heavy_frame_keeps_its_dynamic_range():
    """The failure mode that made outdoor footage flat: normalising over a
    range set by sky outliers squeezed the real scene to nothing."""
    values = np.zeros((100, 100), np.float32)
    values[60:, :] = np.linspace(1.0, 6.0, 40)[:, None]   # ground, 40% of frame
    out = np.asarray(_backend(values).estimate_chunk(
        [np.zeros((100, 100, 3), np.uint8)])[0], np.float64)

    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    norm = np.clip((out - lo) / max(hi - lo, 1e-9), 0, 1)
    ground = norm[60:]
    iqr = np.percentile(ground, 75) - np.percentile(ground, 25)
    # Before the fix this measured 0.0003 on real footage.
    assert iqr > 0.2, f"real scene collapsed after normalising (iqr={iqr:.4f})"


def test_negative_values_are_clipped_not_wrapped():
    values = np.array([[-1.0, 2.0]], np.float32)
    out = np.asarray(_backend(values).estimate_chunk(
        [np.zeros((1, 2, 3), np.uint8)])[0], np.float64)
    assert out[0, 0] == 0.0 and out[0, 1] == pytest.approx(2.0)
