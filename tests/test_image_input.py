"""A single 360 photo in, a stereoscopic photo out.

Almost none of this is new machinery. ffmpeg reads a still as a one-frame
video and the preview path already writes an image, so depth, warp and stack
were converting photos before anyone asked them to. What is new is the front
door: recognising a still, defaulting to full resolution instead of a 2048-wide
preview, and refusing the flags that only mean something for a video.

The video paths must be untouched. Everything here that renders also checks
that the equivalent video invocation still behaves.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from stereo360 import ffmpeg_io, pipeline

ROOT = Path(__file__).resolve().parent.parent


def equirect(tmp_path: Path, name="src.jpg", w=512, h=256) -> str:
    """A small 2:1 still, which is what a 360 photo looks like."""
    out = str(tmp_path / name)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc2=size={w}x{h}", "-frames:v", "1", "-y", out],
        check=True, capture_output=True)
    return out


def run(*args, expect=0):
    proc = subprocess.run([sys.executable, "-m", "stereo360", *args],
                          capture_output=True, text=True, cwd=str(ROOT),
                          timeout=900)
    assert proc.returncode == expect, (
        f"exit {proc.returncode}, wanted {expect}\n{proc.stdout}\n{proc.stderr}")
    return proc


# ----------------------------------------------------------- what is a still

@pytest.mark.parametrize("path,expected", [
    ("a.jpg", True), ("a.JPG", True), ("a.jpeg", True), ("a.png", True),
    ("a.tif", True), ("a.webp", True),
    ("a.mp4", False), ("a.mov", False), ("a", False), ("a.jpeg.mp4", False),
])
def test_stills_are_recognised_by_extension(path, expected):
    """By extension, because probing cannot tell: ffmpeg reads a JPEG as a
    one-frame 25 fps video, which is indistinguishable from a short clip."""
    assert ffmpeg_io.is_image_path(path) is expected


def test_a_still_probes_as_a_video_which_is_why_extension_is_used(tmp_path):
    """The evidence for that decision, so it is not taken on trust."""
    info = ffmpeg_io.probe(equirect(tmp_path))
    assert info.fps > 0, "a still reports a frame rate"
    assert not info.has_audio


def test_the_reader_and_the_writer_agree_on_the_set():
    """A format the pipeline will write must be one it recognises as a still,
    or `-o out.png` from an image input would be refused by one and accepted
    by the other."""
    assert pipeline._PREVIEW_SUFFIXES is ffmpeg_io.IMAGE_SUFFIXES


# ------------------------------------------------------------- the front door

def test_a_photo_converts_with_no_flags_at_all(tmp_path):
    """The whole point of step 1. This used to need
    `--preview-frame 0 --preview-width 0`, which describes the mechanism
    rather than the intent and silently ships a 2048-wide photo if the second
    half is forgotten."""
    src = equirect(tmp_path)
    dst = str(tmp_path / "out.jpg")
    proc = run(src, "-o", dst)

    assert Path(dst).exists()
    info = ffmpeg_io.probe(dst)
    assert (info.width, info.height) == (512, 512), "360 stacks top-bottom"
    assert "Wrote" in proc.stdout, "a photo is not a 'preview'"


def test_the_photo_comes_out_at_full_resolution(tmp_path):
    """Not capped at the 2048 a preview uses. A 7680x7680 stereo photo
    displays on a Quest 3 -- the 35.6 Mpx cap is the video decoder's, and a
    JPEG is a texture -- so there is nothing to gain by shrinking it."""
    src = equirect(tmp_path, w=4096, h=2048)
    dst = str(tmp_path / "big.jpg")
    run(src, "-o", dst)
    assert ffmpeg_io.probe(dst).width == 4096


def test_vr180_works_for_photos_too(tmp_path):
    src = equirect(tmp_path)
    dst = str(tmp_path / "half.jpg")
    run(src, "-o", dst, "--output-mode", "vr180")
    info = ffmpeg_io.probe(dst)
    assert (info.width, info.height) == (512, 256), "side by side, half sphere"


def test_the_exit_code_is_zero(tmp_path):
    """Caught a real bug: the image branch fell through to the video summary,
    which asks a PreviewResult for `cancelled`. The file was written and the
    process then died, so only the exit code told the truth."""
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "ok.jpg"))
    assert "Traceback" not in proc.stderr, proc.stderr


# ------------------------------------------------------------------ refusals

def test_an_image_will_not_write_a_video(tmp_path):
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "out.mp4"), expect=1)
    assert "output must be one too" in proc.stderr


@pytest.mark.parametrize("flag,value", [
    ("--max-frames", "10"),
    ("--start-frame", "5"),
    ("--preview-frame", "0"),
])
def test_video_only_flags_are_named_not_ignored(tmp_path, flag, value):
    """Silently ignoring a flag someone typed teaches them that the flags are
    decorative. Each refusal says why it does not apply."""
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "out.jpg"), flag, value, expect=2)
    assert flag in proc.stderr
    assert "the input is an image" in proc.stderr


def test_spatial_audio_is_refused_for_a_photo(tmp_path):
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "out.jpg"), "--spatial-audio",
               expect=2)
    assert "no audio track" in proc.stderr


def test_a_square_image_is_still_refused_as_a_half_equirect(tmp_path):
    """The input checks apply to photos exactly as they do to video -- a 1:1
    still is a 180 crop, and the tool wants the original sphere."""
    src = equirect(tmp_path, name="square.jpg", w=256, h=256)
    proc = run(src, "-o", str(tmp_path / "out.jpg"), expect=1)
    assert "180-degree footage" in proc.stderr


# ------------------------------------------------- the video paths still work

def test_video_conversion_is_unaffected(tmp_path):
    """The rule from vr180.md, kept here: nothing about adding photos may
    change what a video does."""
    from test_end_to_end import make_test_video

    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=4, with_audio=False)
    proc = run(src, "-o", dst, "--max-frames", "2", "--passthrough",
               "--face-size", "32")
    assert "Done:" in proc.stdout
    assert ffmpeg_io.probe(dst).height == 128


def test_a_video_preview_is_still_called_a_preview(tmp_path):
    """The two single-image paths report differently on purpose: one is the
    deliverable, the other is a look at what the deliverable will be."""
    from test_end_to_end import make_test_video

    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=128, h=64, frames=4, with_audio=False)
    proc = run(src, "-o", str(tmp_path / "p.png"), "--preview-frame", "1",
               "--preview-width", "64", "--face-size", "32", "--passthrough")
    assert "Preview of frame 1" in proc.stdout


def test_video_only_flags_are_fine_on_a_video(tmp_path):
    """The refusal must be scoped to image input, not applied everywhere."""
    from test_end_to_end import make_test_video

    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=128, h=64, frames=6, with_audio=False)
    run(src, "-o", str(tmp_path / "out.mp4"), "--max-frames", "2",
        "--start-frame", "1", "--passthrough", "--face-size", "32")


# ---------------------------------------------------------------- the wrapper

def test_convert_image_defaults_to_full_size(tmp_path):
    """`width=0` means "do not downscale". Getting this wrong is invisible --
    the photo just quietly comes out at 2048."""
    import inspect

    src = equirect(tmp_path, w=1024, h=512)
    dst = str(tmp_path / "full.jpg")
    pipeline.convert_image(src, dst, depth_backend=None, face_size=128)
    assert ffmpeg_io.probe(dst).width == 1024
    assert "width" in inspect.getsource(pipeline.convert_image)


def test_convert_image_passes_settings_through(tmp_path):
    """It is a thin wrapper, so anything preview_frame understands must still
    arrive -- otherwise photos would silently ignore half the options."""
    src = equirect(tmp_path)
    dst = str(tmp_path / "yaw.jpg")
    pipeline.convert_image(src, dst, depth_backend=None, face_size=64,
                           output_mode="vr180", yaw=90.0)
    info = ffmpeg_io.probe(dst)
    assert (info.width, info.height) == (512, 256)
