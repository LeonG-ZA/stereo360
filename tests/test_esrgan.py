"""Real-ESRGAN: the photo upscaler that needs nothing installed.

Everything here runs without a GPU. The tests that need the graph skip when it
has not been fetched, so a clone without models/ still goes green.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stereo360 import esrgan                                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
needs_model = pytest.mark.skipif(
    not (ROOT / esrgan.DEFAULT_MODEL).exists(),
    reason="fetch the graph first: python scripts/fetch_esrgan.py")


def test_describe_says_no_without_a_graph(tmp_path):
    """A GUI asks on every machine, so absence has to be an answer."""
    found = esrgan.describe(str(tmp_path / "nothing.onnx"))
    assert found["available"] is False
    assert "nothing.onnx" in found["reason"]


@needs_model
def test_describe_marks_it_as_stills_only():
    """The interface needs to know *why* it is not offered for a video, so
    the reason travels with the answer rather than being inferred."""
    found = esrgan.describe()
    assert found["available"] is True
    assert found["stills_only"] is True


def test_a_missing_graph_says_how_to_get_one(tmp_path):
    with pytest.raises(esrgan.EsrganError) as e:
        esrgan.run_still("in.jpg", str(tmp_path / "out.jpg"),
                         model=str(tmp_path / "nothing.onnx"))
    assert "fetch_esrgan" in str(e.value)


def test_the_window_wraps_sideways_and_clamps_at_the_poles():
    """A 360 photo has the same seam a frame does."""
    img = np.arange(5 * 8).reshape(5, 8, 1).astype(np.uint8).repeat(3, axis=2)

    assert list(esrgan._window(img, 0, 1, -2, 2)[0, :, 0]) == [6, 7, 0, 1]
    assert list(esrgan._window(img, 0, 1, 6, 10)[0, :, 0]) == [6, 7, 0, 1]
    assert list(esrgan._window(img, -2, 2, 0, 1)[:, 0, 0]) == [0, 0, 0, 8]


def test_an_inside_tile_is_a_view_and_not_a_copy():
    img = np.zeros((64, 128, 3), np.uint8)
    assert esrgan._window(img, 8, 40, 8, 40).base is img
    assert esrgan._window(img, 8, 40, -4, 28).base is not img


class _Sess:
    """A stand-in that upscales by nearest neighbour, so the tiling can be
    checked without a GPU or a 5 MB graph."""

    def get_inputs(self):
        class _In:
            name = "input"
            #: Three channels: this stands in for an RGB graph, and
            #: `esrgan.upscale` reads the shape to decide whether a model is
            #: one of ArtCNN's luma doublers instead.
            shape = [1, 3, "h", "w"]
        return [_In()]

    def run(self, _out, feed):
        x = feed["input"]
        return [np.repeat(np.repeat(x, esrgan.NATIVE_SCALE, axis=2),
                          esrgan.NATIVE_SCALE, axis=3)]


@pytest.mark.parametrize("scale", [1.0, 2.0, 4.0])
def test_the_output_is_the_size_that_was_asked_for(scale):
    """The graph is 4x whatever the job wants, so every tile comes back down
    to the wanted size -- and the pieces have to tile the result exactly."""
    frame = np.zeros((96, 192, 3), np.uint8)
    got = esrgan.upscale(_Sess(), frame, scale, tile=64, overlap=8)
    assert got.shape == (int(96 * scale), int(192 * scale), 3)


def test_the_tiles_land_where_they_belong():
    """A tile written to the wrong offset shows as a grid of seams, which a
    size check alone would not catch."""
    frame = np.zeros((64, 128, 3), np.uint8)
    frame[:32, :64] = 200            # one bright quadrant to follow

    got = esrgan.upscale(_Sess(), frame, 2.0, tile=32, overlap=8)

    assert got[:64, :128].min() > 150, "the bright quadrant moved"
    assert got[64:, 128:].max() < 50, "something bled into the dark one"


def test_a_scale_of_zero_is_refused():
    with pytest.raises(esrgan.EsrganError):
        esrgan.upscale(_Sess(), np.zeros((8, 8, 3), np.uint8), 0)


@needs_model
def test_video_is_no_longer_refused(tmp_path):
    """Siax on video used to exit non-zero. It now renders.

    The bar came from an amplification figure read against the Lanczos
    floor -- Siax turns a one-level wobble into four. What that missed is
    that the floor is not the target: watched over 120 frames, the
    *untouched* 8K itself changed 1.20x as much as Lanczos did, from sensor
    noise and wind. A model sitting on the floor is too smooth rather than
    admirably steady, so a number measured against it was answering a
    different question.

    Every model in the table was then watched at 1:1 over 120 frames and
    none of them crawled. The figures stay in AMPLIFY as measurement; the
    choice went to whoever is looking at the footage.
    """
    src = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=256x128:rate=30", "-frames:v", "2", str(src)],
        check=True, capture_output=True)
    out = tmp_path / "out.mp4"

    done = subprocess.run(
        [sys.executable, "-m", "stereo360", str(src),
         "-o", str(out), "--upscale", "siax"],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT))

    said = done.stdout + done.stderr
    assert done.returncode == 0, said
    assert out.exists() and out.stat().st_size > 0
    assert "photos" not in said, "no model is for photos only any more"


class _LumaSess:
    """A stand-in ArtCNN: one channel in, doubled by nearest, one channel out."""

    def get_inputs(self):
        class _In:
            name = "input"
            shape = [1, 1, "h", "w"]
        return [_In()]

    def run(self, _out, feed):
        x = feed["input"]
        return [np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)]


@pytest.mark.parametrize("scale", [2.0, 3.0, 4.0])
def test_a_luma_graph_fills_the_frame_at_any_scale(scale):
    """The graph doubles; the job may ask for more. Slicing the tile at the
    graph's scale and resizing afterwards cut a window twice too large out of
    a tile half the size, which tiled garbage across the frame -- 17.6 dB on
    a real 2K source, where plain Lanczos gets 33.

    A horizontal ramp catches it: any correct enlargement of a ramp still
    climbs left to right, and the broken one stepped backwards at every tile
    edge.
    """
    from stereo360 import artcnn

    w, h = 900, 64                       # wider than one 768px tile
    ramp = np.tile(np.linspace(0, 255, w, dtype=np.float32), (h, 1))
    frame = np.dstack([ramp] * 3).astype(np.uint8)

    got = artcnn.upscale(_LumaSess(), frame, scale)

    assert got.shape == (int(round(h * scale)), int(round(w * scale)), 3)
    row = got[got.shape[0] // 2, :, 0].astype(np.int16)
    assert row.min() >= 0 and row.max() <= 255
    backwards = np.diff(row) < -2
    assert not backwards.any(), (
        f"the ramp steps backwards at {int(backwards.sum())} places -- "
        f"tiles are not landing where they belong")
