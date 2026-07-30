"""Tests for M3 chunked temporal pipeline (overlap blending, frame order)."""

import numpy as np

from stereo360 import pipeline
from stereo360.depth.base import DepthBackend


class RampBackend(DepthBackend):
    """Depth = constant per frame index; records chunk calls."""

    def __init__(self):
        self.chunks = []

    def estimate(self, frame_rgb):
        return np.full(frame_rgb.shape[:2], 1.0, dtype=np.float32)

    def estimate_chunk(self, frames_rgb):
        self.chunks.append(len(frames_rgb))
        # Distinct depth level per frame position in the *sequence* is not
        # available here; use a ramp within the chunk for blend testing.
        return [np.full(f.shape[:2], float(i), dtype=np.float32)
                for i, f in enumerate(frames_rgb)]


class FakeEncoder:
    def __init__(self):
        self.frames = []

    def write(self, img):
        self.frames.append(img)


def _sink(encoder, cancel=None):
    """The chunked path writes through a _Sink, which owns stacking and the
    cancellation check as well as the encoder."""
    from stereo360.events import Reporter

    return pipeline._Sink(encoder, Reporter(), cancel)


def _frames(n, h=16, w=32):
    """SourceFrames, which is what the chunked loop consumes.

    `faces` is None for equirect input -- the loop then builds them itself.
    A cubemap source supplies its own; see test_projection.py.
    """
    return [pipeline.SourceFrame(np.full((h, w, 3), i, dtype=np.uint8), None)
            for i in range(n)]


def test_chunked_emits_all_frames_in_order():
    frames = _frames(10)
    enc = FakeEncoder()
    backend = RampBackend()
    pipeline._convert_chunked(iter(frames), _sink(enc), face_size=8,
                              backend=backend, strength=1.0, chunk_size=4,
                              chunk_overlap=2)
    assert len(enc.frames) == 10
    # Top half of each emitted frame must be the original frame, in order.
    for i, out in enumerate(enc.frames):
        assert out.shape == (32, 32, 3)
        assert (out[:16] == i).all()


def test_chunk_calls_have_expected_sizes():
    frames = _frames(10)
    enc = FakeEncoder()
    backend = RampBackend()
    pipeline._convert_chunked(iter(frames), _sink(enc), face_size=8,
                              backend=backend, strength=1.0, chunk_size=4,
                              chunk_overlap=2)
    # estimate_chunk is called once per cubemap face (6) per flush.
    # Flushes: 4 full chunks of 4 (advance 2) + final flush of the 2
    # held-back overlap frames -> [4]*4 + [2], times 6 faces.
    assert backend.chunks == [4] * 24 + [2] * 6


def test_blend_overlap_ramp():
    a = [np.zeros((2, 2), dtype=np.float32) for _ in range(3)]
    b = [np.full((2, 2), 10.0, dtype=np.float32) for _ in range(3)]
    out = pipeline._blend_overlap(a, b)
    # weights 1/4, 2/4, 3/4 favoring the later chunk
    np.testing.assert_allclose(out[0], 2.5, rtol=1e-6)
    np.testing.assert_allclose(out[1], 5.0, rtol=1e-6)
    np.testing.assert_allclose(out[2], 7.5, rtol=1e-6)


def test_overlap_must_be_smaller_than_chunk():
    import pytest

    with pytest.raises(ValueError):
        pipeline._convert_chunked(iter(_frames(4)), _sink(FakeEncoder()), 8,
                                  RampBackend(), 1.0, chunk_size=4,
                                  chunk_overlap=4)


def test_cancel_stops_before_the_next_chunk():
    """Cancelling must not have to wait out a chunk's depth estimation.

    The check runs at the top of flush(), so a cancel raised after the first
    chunk stops before the second chunk's six face inferences start -- the
    part that takes seconds and produces nothing visible.
    """
    import pytest

    from stereo360.events import Cancelled

    enc = FakeEncoder()
    backend = RampBackend()
    # Ask to stop once the first chunk has been written.
    sink = _sink(enc, cancel=lambda: len(enc.frames) >= 4)

    with pytest.raises(Cancelled):
        pipeline._convert_chunked(iter(_frames(12)), sink, face_size=8,
                                  backend=backend, strength=1.0,
                                  chunk_size=4, chunk_overlap=0)

    assert len(enc.frames) == 4
    # Six faces for the first chunk and nothing more: the second chunk's
    # inference never started.
    assert backend.chunks == [4] * 6


def test_chunk_normalize_shares_scale_across_frames():
    # Frame A: uniform 0.5. Frame B: 0.5 with one extreme near pixel.
    a = np.full((32, 32), 0.5, dtype=np.float32)
    b = np.full((32, 32), 0.5, dtype=np.float32)
    b[0, 0] = 100.0
    pipeline._chunk_normalize([a, b])
    # The shared depth value 0.5 must map to the SAME normalized value in
    # both frames; per-frame normalization would give different scales.
    np.testing.assert_allclose(a[16, 16], b[16, 16], rtol=1e-5)


def test_default_estimate_chunk_is_per_frame():
    class Counting(DepthBackend):
        n = 0

        def estimate(self, f):
            self.n += 1
            return np.zeros(f.shape[:2], dtype=np.float32)

    b = Counting()
    # Raw arrays here, not SourceFrames: this exercises the backend's own
    # fallback, which is below the pipeline and knows nothing about sources.
    raw = [np.zeros((4, 4, 3), np.uint8) for _ in range(3)]
    out = b.estimate_chunk(raw)
    assert b.n == 3 and len(out) == 3


def test_depth_range_smooths_instead_of_tracking_each_frame():
    """The single-frame path's wobble fix.

    Recomputed percentiles rescale the whole disparity field every frame,
    which moves stationary objects horizontally. Measured on 8K footage the
    per-frame span swung 20% across six frames, and smoothing it cut
    frame-to-frame disparity jitter from 0.399px to 0.215px.
    """
    rng = pipeline.DepthRange(alpha=0.05)

    # First frame defines the range outright -- no warm-up drift.
    flat = np.full((64, 64), 5.0, np.float32)
    flat[0, 0] = 0.0
    lo0, hi0 = rng.update(flat)

    # A frame whose own percentiles are far away must move the range only a
    # little, not jump to it.
    other = np.full((64, 64), 50.0, np.float32)
    lo1, hi1 = rng.update(other)
    assert lo0 <= lo1 < 50.0
    assert abs(hi1 - hi0) < 0.2 * abs(50.0 - hi0), "range jumped, not eased"


def test_depth_range_converges_on_a_sustained_change():
    """A real scene change must still be followed, just not instantly."""
    rng = pipeline.DepthRange(alpha=0.5)
    a = np.zeros((32, 32), np.float32)
    rng.update(a)
    b = np.full((32, 32), 10.0, np.float32)
    for _ in range(30):
        lo, hi = rng.update(b)
    assert abs(hi - 10.0) < 0.1 and abs(lo - 10.0) < 0.1


def test_depth_range_is_used_by_the_single_frame_path():
    """Guards the wiring: without a DepthRange the warp normalises per frame,
    which is the bug."""
    import inspect

    source = inspect.getsource(pipeline.convert)
    assert "DepthRange()" in source
    sig = inspect.signature(pipeline.right_eye_from_depth)
    assert "depth_range" in sig.parameters


def test_stabiliser_holds_static_depth_still():
    """The residual wobble: a stationary object's depth drifting slightly each
    frame moves it horizontally in the right eye."""
    st = pipeline.DepthStabiliser(tau=0.02)
    base = np.full((16, 16), 0.5, np.float32)
    st.apply(base.copy())

    noisy = base + 0.004        # inside the lock band
    out = st.apply(noisy.copy())
    # Pulled most of the way back toward the previous value, not accepted.
    assert np.abs(out - base).mean() < 0.4 * 0.004


def test_stabiliser_does_not_smear_motion():
    """A pixel that genuinely moves must keep its new depth, or moving objects
    would drag their old depth behind them."""
    st = pipeline.DepthStabiliser(tau=0.02)
    st.apply(np.full((16, 16), 0.2, np.float32))

    moved = np.full((16, 16), 0.9, np.float32)      # far outside the band
    out = st.apply(moved.copy())
    np.testing.assert_allclose(out, 0.9, atol=1e-6)


def test_stabiliser_never_locks_permanently():
    """Weight must stay below 1. At exactly 1 a pixel is replaced by its
    history outright and can never update, so anything drifting slower than
    tau per frame would freeze at its first value for the whole render."""
    st = pipeline.DepthStabiliser(tau=0.02)
    value = 0.0
    st.apply(np.full((8, 8), value, np.float32))
    for _ in range(60):
        value += 0.001                              # far inside the band
        out = st.apply(np.full((8, 8), value, np.float32))
    # It lags, but it has tracked most of the way rather than staying at 0.
    assert out.mean() > 0.6 * value, "stabiliser froze on a slow drift"


def test_stabiliser_can_be_disabled():
    st = pipeline.DepthStabiliser(tau=0.0)
    st.apply(np.zeros((8, 8), np.float32))
    out = st.apply(np.full((8, 8), 0.001, np.float32))
    np.testing.assert_allclose(out, 0.001)


def test_stabiliser_is_wired_into_the_single_frame_path():
    import inspect

    assert "DepthStabiliser(" in inspect.getsource(pipeline.convert)
    assert "temporal_depth" in inspect.signature(pipeline.convert).parameters
