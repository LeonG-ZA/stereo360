"""RIFE: the interpolator that needs nothing installed.

Everything here runs without a GPU. The two tests that need the graph skip
when it has not been fetched, so a clone without models/ still goes green.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stereo360 import interpolate as fi                       # noqa: E402


def _have_model():
    return os.path.exists(
        Path(__file__).resolve().parent.parent / fi.DEFAULT_MODEL)


needs_model = pytest.mark.skipif(
    not _have_model(),
    reason="fetch the graph first: python scripts/fetch_rife.py")


# ------------------------------------------------------------------- offered


@pytest.mark.parametrize("fps,offered", [
    (24, True), (25, True), (29.97, True), (30, True),
    (30.1, False), (50, False), (60, False), (0, False), (-1, False),
])
def test_only_slow_sources_are_offered_interpolation(fps, offered):
    """Judder is a low-frame-rate artifact. Above 30 there is nothing to fix,
    and the frames would still have to go through the stereo pass."""
    assert fi.offered_for(fps) is offered


def test_describe_says_no_without_a_graph(tmp_path):
    """A GUI calls this on every machine, so absence has to be an answer
    rather than an exception."""
    found = fi.describe(30, str(tmp_path / "nothing.onnx"))
    assert found["available"] is False
    assert "nothing.onnx" in found["reason"]


@needs_model
def test_describe_finds_the_installed_graph():
    found = fi.describe(30)
    assert found["available"] is True
    assert found["offered"] is True
    assert fi.describe(60)["offered"] is False


def _pairs(n, first=0):
    """(frame, payload) as the decoder produces them, payload marking real."""
    for i in range(first, first + n):
        yield np.full((32, 64, 3), (i * 7) % 255, np.uint8), i


# --------------------------------------------------------------------- tiles


def test_tiles_stay_under_the_directml_limit():
    """DirectML returns wrong pixels rather than an error above 2^22 pixels --
    correct at 2816x1408, corrupt at 3072x1536 -- so the tile plus the context
    around it has to stay below that. This is the check that a later tweak to
    the tile size cannot quietly cross the line."""
    tw, th = fi._tile_for("DmlExecutionProvider")
    assert (tw + 2 * fi._OVERLAP) * (th + 2 * fi._OVERLAP) <= fi._DML_LIMIT


def test_the_window_wraps_sideways_and_clamps_at_the_poles():
    """The whole seam fix in one function: on an equirect frame the column
    before the first is the last one, so a tile at +/-180 can see across it.
    Top and bottom do not wrap -- there is nothing above the north pole."""
    img = np.arange(5 * 8).reshape(5, 8, 1).astype(np.uint8).repeat(3, axis=2)

    left = fi._window(img, 0, 1, -2, 2)[0, :, 0]
    assert list(left) == [6, 7, 0, 1], "the frame wraps round on itself"

    right = fi._window(img, 0, 1, 6, 10)[0, :, 0]
    assert list(right) == [6, 7, 0, 1]

    top = fi._window(img, -2, 2, 0, 1)[:, 0, 0]
    assert list(top) == [0, 0, 0, 8], "the pole repeats rather than wrapping"


def test_an_inside_tile_is_a_view_and_not_a_copy():
    """The tiles that touch neither the seam nor a pole are most of them, and
    copying those is what made interpolation seven times slower inside a real
    render: the copies compete for CPU with the encoder, which at 8K wants
    every core it can get. Index arrays copy unconditionally; slices do not.
    """
    img = np.zeros((64, 128, 3), np.uint8)

    inside = fi._window(img, 8, 40, 8, 40)
    assert inside.base is img, "an interior tile must not be copied"

    # The edges still cost a copy, which is the price of wrapping the sphere.
    assert fi._window(img, 8, 40, -4, 28).base is not img
    assert fi._window(img, -4, 28, 8, 40).base is not img


def test_the_buffer_is_reused_across_a_frames_tiles():
    """One allocation per frame rather than one per tile -- and one input
    shape, so the runtime does not recompile its kernels for a second."""
    seen = []

    class _Sess:
        def get_inputs(self):
            class _In:
                name = "input"
            return [_In()]

        def run(self, _out, feed):
            x = feed["input"]
            seen.append(id(x))
            return [np.zeros((1, 3) + x.shape[2:], np.float32)]

    frame = np.zeros((128, 256, 3), np.uint8)
    fi.between(_Sess(), frame, frame, 0.5, tile=(64, 64))
    assert len(seen) == 8, "a 256x128 frame in 64x64 tiles is eight of them"
    assert len(set(seen)) == 1, "every tile should reuse one buffer"


# --------------------------------------------------------------------- rules


def test_a_fast_source_is_refused():
    with pytest.raises(fi.InterpolateError) as e:
        fi.Streamer(60)
    assert "30" in str(e.value)


def test_a_target_below_the_source_is_refused():
    """Asking for 24 from 30 would drop frames, which is not what this is."""
    with pytest.raises(fi.InterpolateError) as e:
        fi.Streamer(30, 24)
    assert "drop frames" in str(e.value)


def test_a_missing_graph_says_how_to_get_one(tmp_path):
    with pytest.raises(fi.InterpolateError) as e:
        fi.Streamer(30, model=str(tmp_path / "nothing.onnx"))
    assert "fetch_rife" in str(e.value)


# ------------------------------------------------------------- the streamer
#
# Interpolation is a filter in the frame stream the renderer pulls from, so
# what it has to get right is arithmetic: which source frames a run of output
# frames needs, and which instant each of those output frames sits at.


@needs_model
def test_every_gap_gets_a_frame():
    s = fi.Streamer(30, 60)
    got = list(s.stream(_pairs(5)))
    assert len(got) == 9, "5 real frames leave 4 gaps"
    # A payload rides with a frame the camera really shot; an invented one
    # carries None, and the depth stage rebuilds what it needs.
    assert [p for _, p in got] == [0, None, 1, None, 2, None, 3, None, 4]


@needs_model
def test_the_frame_range_counts_output_frames():
    """What someone means by "render frames 101 to 109" is output frames --
    they are what comes out and what the progress bar counts."""
    s = fi.Streamer(30, 60)
    skip, take = s.window(101, 6)
    assert skip == 50, "output 101 sits between source 50 and 51"

    got = list(s.stream(_pairs(take, first=skip), first_output=101,
                        first_source=skip, count=6))
    assert len(got) == 6
    # 101 is odd, so it is an invented instant; the real frames land between.
    assert [p for _, p in got] == [None, 51, None, 52, None, 53]


@needs_model
def test_a_rate_that_is_not_a_multiple():
    """24 to 60 is 2.5x, so most output instants fall between source frames
    and only every fifth lands on one."""
    s = fi.Streamer(24, 60)
    got = list(s.stream(_pairs(6)))
    real = [p for _, p in got if p is not None]
    assert real == [0, 2, 4], "only the instants that coincide are passed on"
    assert len(got) == 13


def test_the_total_is_output_frames():
    """What the progress bar is counting up to."""
    s = fi.Streamer.__new__(fi.Streamer)      # no model needed for arithmetic
    s.src_fps, s.target_fps = 30.0, 60.0
    assert s.total(100) == 199                # 99 gaps, each gaining one
    assert s.total(1) == 1
    assert s.total(None) is None


# ----------------------------------------------------------------- the model


def _pattern(w=512, h=256):
    """Something with structure a flow can lock onto -- not noise, which has
    no motion to find, and not a flat field, which has no error to measure."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    a = 128 + 100 * np.sin(x / 23.0) * np.cos(y / 17.0)
    b = 128 + 90 * np.sin((x + y) / 31.0)
    return np.stack([a, b, (a + b) / 2], -1).clip(0, 255).astype(np.uint8)


@needs_model
def test_the_middle_frame_lands_in_the_middle():
    """The claim being tested is that this interpolates rather than blends: a
    known translation has a known middle, and the answer has to be nearer to
    it than to either of the frames it was given."""
    sess, _ = fi._session(fi.model_path())
    a = _pattern()
    b = np.roll(a, -16, axis=1)
    truth = np.roll(a, -8, axis=1)

    mid = fi.between(sess, a, b, 0.5, tile=(256, 128))

    def err(x, y):
        return np.abs(x.astype(np.int16) - y.astype(np.int16)).mean()

    assert err(mid, truth) < err(mid, a), "it is not just copying the first"
    assert err(mid, truth) < err(mid, b), "or the second"
    assert err(mid, truth) < err(a, truth) / 2, "or blending the two"


@needs_model
def test_the_timestep_reaches_the_model():
    """t=0 has to give back the first frame and t=1 the second. The channel
    order and the meaning of that seventh plane are conventions of whoever
    exported the graph, and having them backwards would still produce a
    plausible-looking picture."""
    sess, _ = fi._session(fi.model_path())
    a = _pattern()
    b = np.roll(a, -16, axis=1)

    def err(x, y):
        return np.abs(x.astype(np.int16) - y.astype(np.int16)).mean()

    assert err(fi.between(sess, a, b, 0.0, tile=(256, 128)), a) < 2
    assert err(fi.between(sess, a, b, 1.0, tile=(256, 128)), b) < 2


@needs_model
def test_the_model_is_released_when_the_pass_is_done():
    """It has to go before the renderer loads its own.

    Left resident it holds a working set on the GPU for the whole of an 8K
    render it will never take part in again -- which is not merely wasteful:
    the render that followed deadlocked, both ffmpeg processes and the parent
    idle at zero CPU, until the session was dropped.
    """
    s = fi.Streamer(30, 60)
    assert s._sess is not None
    s.close()
    assert s._sess is None
    s.close()          # twice is safe; `run` closes in a finally
