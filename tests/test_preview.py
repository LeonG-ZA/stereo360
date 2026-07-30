"""Single-frame preview rendering.

The preview only earns its place if it predicts the render, so most of these
check that it agrees with `convert` rather than merely producing a file.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from stereo360 import ffmpeg_io, pipeline
from test_end_to_end import make_test_video


def _read(path):
    """RGB image from disk, via imdecode for the same reason we encode that
    way -- imread goes through the ANSI API on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def test_preview_writes_a_top_bottom_image(tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    out = tmp_path / "preview.png"
    make_test_video(src, w=256, h=128, frames=8, with_audio=False)

    res = pipeline.preview_frame(src, str(out), frame_index=3, face_size=64,
                                 width=0)
    assert out.exists()
    assert res.frame_index == 3
    img = _read(out)
    # Top-bottom stack of the source geometry, full resolution.
    assert img.shape == (256, 256, 3)
    assert (res.width, res.height) == (256, 256)


def test_preview_matches_what_convert_would_write(tmp_path: Path):
    """The whole point: what you judge is what you get.

    Renders frame 4 both ways and requires the preview to be the same pixels
    the encoder was handed, so tuning against a preview is meaningful.
    """
    src = str(tmp_path / "in.mp4")
    out = tmp_path / "preview.png"
    make_test_video(src, w=256, h=128, frames=8, with_audio=False)

    pipeline.preview_frame(src, str(out), frame_index=4, face_size=64, width=0)
    preview = _read(out)

    # The same frame through the conversion path, captured before encoding.
    written = []

    class Capture:
        def write(self, img):
            written.append(img)

    frames = list(ffmpeg_io.decode_frames(src))
    sink = pipeline._Sink(Capture(), pipeline.Reporter(), None)
    left = frames[4]
    right = pipeline.right_eye_passthrough(left, 64)
    sink.write(left, right)

    np.testing.assert_array_equal(preview, written[0])


def test_preview_downscales_by_default(tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    out = tmp_path / "preview.png"
    make_test_video(src, w=512, h=256, frames=4, with_audio=False)

    res = pipeline.preview_frame(src, str(out), frame_index=0, face_size=128,
                                 width=128)
    assert (res.width, res.height) == (128, 128)
    assert _read(out).shape == (128, 128, 3)


def test_preview_never_upscales(tmp_path: Path):
    """--preview-width is a cap, not a target; a small source stays small."""
    src = str(tmp_path / "in.mp4")
    out = tmp_path / "preview.png"
    make_test_video(src, w=128, h=64, frames=2, with_audio=False)

    res = pipeline.preview_frame(src, str(out), frame_index=0, face_size=32,
                                 width=4096)
    assert (res.width, res.height) == (128, 128)


def test_preview_rejects_a_video_output_path(tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=128, h=64, frames=2, with_audio=False)
    with pytest.raises(ValueError, match="no image extension"):
        pipeline.preview_frame(src, str(tmp_path / "out.mp4"), face_size=32)


def test_preview_rejects_a_frame_past_the_end(tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=128, h=64, frames=5, with_audio=False)
    with pytest.raises(ValueError, match="past the end"):
        pipeline.preview_frame(src, str(tmp_path / "p.png"), frame_index=99,
                               face_size=32)


def test_preview_rejects_a_negative_frame(tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=128, h=64, frames=5, with_audio=False)
    with pytest.raises(ValueError, match="must not be negative"):
        pipeline.preview_frame(src, str(tmp_path / "p.png"), frame_index=-1,
                               face_size=32)


def test_preview_frames_differ_from_each_other(tmp_path: Path):
    """Guards the seek: previewing frame N must actually give frame N."""
    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=256, h=128, frames=10, with_audio=False)

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    pipeline.preview_frame(src, str(a), frame_index=0, face_size=64, width=0)
    pipeline.preview_frame(src, str(b), frame_index=8, face_size=64, width=0)

    frames = list(ffmpeg_io.decode_frames(src))
    np.testing.assert_array_equal(_read(a)[:128], frames[0])
    np.testing.assert_array_equal(_read(b)[:128], frames[8])


def test_cli_preview_exits_without_writing_a_video(tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    out = tmp_path / "preview.png"
    make_test_video(src, w=256, h=128, frames=6, with_audio=False)

    proc = subprocess.run(
        [sys.executable, "-m", "stereo360", src, "-o", str(out),
         "--passthrough", "--face-size", "64", "--preview-frame", "2",
         "--preview-width", "128", "--progress-json"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=180)
    assert proc.returncode == 0, proc.stderr

    ev = [json.loads(x) for x in proc.stdout.splitlines() if x.strip()]
    start = next(e for e in ev if e["type"] == "start")
    done = next(e for e in ev if e["type"] == "done")
    assert start["preview"] is True and start["frame_index"] == 2
    assert done["preview"] is True and done["frames"] == 1
    assert done["width"] == 128 and done["height"] == 128

    assert out.exists()
    assert _read(out).shape == (128, 128, 3)
    assert not (tmp_path / "preview.mp4").exists()


def test_cli_preview_reports_a_bad_path_as_an_error_event(tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=128, h=64, frames=3, with_audio=False)

    proc = subprocess.run(
        [sys.executable, "-m", "stereo360", src, "-o",
         str(tmp_path / "out.mp4"), "--passthrough", "--face-size", "32",
         "--preview-frame", "1", "--progress-json"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=180)
    assert proc.returncode == 1
    ev = [json.loads(x) for x in proc.stdout.splitlines() if x.strip()]
    err = next(e for e in ev if e["type"] == "error")
    assert "image extension" in err["message"]
