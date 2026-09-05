"""FSRCNNX: the upscaler that runs as a shader inside ffmpeg.

Nothing here needs a GPU except the tests marked otherwise, which skip when
libplacebo or a Vulkan device is missing.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stereo360 import fsrcnnx                                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
needs_shader = pytest.mark.skipif(
    not (ROOT / fsrcnnx.DEFAULT_SHADER).exists(),
    reason="fetch it first: python scripts/fetch_fsrcnnx.py")


def test_describe_says_no_without_the_shader(tmp_path):
    found = fsrcnnx.describe(str(tmp_path / "nothing.glsl"))
    assert found["available"] is False
    assert "nothing.glsl" in found["reason"]


def test_a_missing_shader_says_how_to_get_one(tmp_path):
    with pytest.raises(fsrcnnx.ShaderError) as e:
        fsrcnnx.run("in.mp4", str(tmp_path / "out.mkv"),
                    width=64, height=32,
                    shader=str(tmp_path / "nothing.glsl"))
    assert "fetch_fsrcnnx" in str(e.value)


def test_the_chain_asks_for_the_size_wanted():
    """The shader doubles; libplacebo resamples to whatever was asked for, so
    a scale other than 2 still has to land on the right size."""
    assert "w=7680:h=3840" in fsrcnnx.chain(3840, 1920, 2.0)
    assert "w=5760:h=2880" in fsrcnnx.chain(3840, 1920, 1.5)
    assert "custom_shader_path=" in fsrcnnx.chain(3840, 1920)


def test_a_windows_path_survives_the_filter_parser():
    """The filtergraph parser splits options on ':' and filters on ',', so an
    absolute Windows path needs quoting *and* the drive colon escaped. Either
    alone fails, and it fails as `Invalid argument` with nothing to say a path
    was the problem -- which is exactly the sort of thing worth pinning."""
    got = fsrcnnx.chain(64, 32, 2.0, shader=r"C:\shaders\FSRCNNX.glsl")
    assert r"custom_shader_path='C\:/shaders/FSRCNNX.glsl'" in got


@needs_shader
@pytest.mark.skipif(not fsrcnnx.usable(), reason="no libplacebo or no Vulkan")
def test_ffmpeg_accepts_an_absolute_path(tmp_path):
    """The escaping above, checked against the thing that has to accept it."""
    shader = str((ROOT / fsrcnnx.DEFAULT_SHADER).resolve())
    assert ":" in shader, "this test is about drive letters"

    src = tmp_path / "in.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=160x80:rate=30", "-frames:v", "1", str(src)],
        check=True, capture_output=True)
    fsrcnnx.run(str(src), str(tmp_path / "out.mkv"), width=160, height=80,
                shader=shader)


@needs_shader
def test_it_reports_whether_this_machine_can_run_it():
    """Asked by running it, not by reading build flags: a build can carry
    libplacebo and still have no Vulkan device to run it on, and that failure
    belongs before a render rather than inside one."""
    answer = fsrcnnx.usable(recheck=True)
    assert answer in (True, False)
    assert fsrcnnx.describe()["available"] is answer


@needs_shader
@pytest.mark.skipif(not fsrcnnx.usable(), reason="no libplacebo or no Vulkan")
def test_it_doubles_a_real_clip(tmp_path):
    src = tmp_path / "in.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=320x160:rate=30", "-frames:v", "3", str(src)],
        check=True, capture_output=True)

    dst = tmp_path / "out.mkv"
    fsrcnnx.run(str(src), str(dst), width=320, height=160, scale=2.0,
                shader=str(ROOT / fsrcnnx.DEFAULT_SHADER))

    size = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(dst)],
        capture_output=True, text=True, check=True).stdout.strip()
    assert size.startswith("640,320")
