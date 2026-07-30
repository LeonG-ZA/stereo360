"""A source deeper than 8 bits gets flattened, and has to say so.

`decode_frames` asks ffmpeg for rgb24 and the encoder is fed rgb24, so a 10-bit
source loses its precision before any pipeline stage sees it. Measured on a
10-bit gradient: 161 distinct levels in, 35 out. `--bitdepth 10` gives a real
Main10 file but cannot recover it -- the same test gave 41 levels, a minimum
step of one whole 8-bit code.

That is a reasonable thing for the pipeline to do and an unreasonable thing for
it to do silently.
"""

import numpy as np
import pytest

from stereo360 import ffmpeg_io, pipeline
from stereo360.events import Reporter


class Recorder(Reporter):
    """Reporter that keeps what it was told."""

    def __init__(self) -> None:
        super().__init__()
        self.warnings: list = []
        self.fields: list = []

    def warning(self, message, **fields):
        self.warnings.append(message)
        self.fields.append(fields)

    def info(self, message, **fields):
        pass


def info_with(pix_fmt):
    return ffmpeg_io.VideoInfo(width=64, height=32, fps=30.0, frame_count=1,
                               duration=1.0, has_audio=False, pix_fmt=pix_fmt)


@pytest.mark.parametrize("pix_fmt,expected", [
    ("yuv420p", 8),
    ("yuvj420p", 8),
    ("yuv422p", 8),
    ("yuv444p", 8),
    ("gray", 8),
    ("yuv420p10le", 10),
    ("yuv422p10le", 10),
    ("yuv444p12le", 12),
    ("yuv420p9be", 9),
    ("gray10le", 10),
    ("p010le", 10),
    ("p016le", 16),
])
def test_bit_depth_is_read_from_the_pixel_format(pix_fmt, expected):
    assert ffmpeg_io.bit_depth_from_pix_fmt(pix_fmt) == expected


@pytest.mark.parametrize("pix_fmt", ["rgb24", "rgb48le", "bgra", "gbrp10le",
                                     None, "", "something_new"])
def test_unparseable_formats_say_so_rather_than_guessing(pix_fmt):
    """Packed RGB digits are total bits per pixel, not per component -- rgb24
    is 8-bit and rgb48 is 16-bit. A wrong guess here would produce a warning
    that is worse than no warning, so these return None."""
    assert ffmpeg_io.bit_depth_from_pix_fmt(pix_fmt) is None


def test_video_info_exposes_bit_depth_like_it_exposes_chroma():
    assert info_with("yuv420p10le").bit_depth == 10
    assert info_with("yuv420p10le").chroma == "4:2:0"
    assert info_with("yuv420p").bit_depth == 8


def test_an_8_bit_source_is_not_warned_about():
    rec = Recorder()
    assert not pipeline.warn_if_source_is_deeper_than_8_bit(
        info_with("yuv420p"), 8, rec)
    assert rec.warnings == []


def test_an_unknown_format_is_not_warned_about():
    """Silence beats a warning built on a guess."""
    rec = Recorder()
    assert not pipeline.warn_if_source_is_deeper_than_8_bit(
        info_with("rgb24"), 8, rec)
    assert rec.warnings == []


def test_a_10_bit_source_is_warned_about():
    rec = Recorder()
    assert pipeline.warn_if_source_is_deeper_than_8_bit(
        info_with("yuv420p10le"), 8, rec)
    assert len(rec.warnings) == 1
    text = rec.warnings[0]
    assert "10-bit" in text and "8-bit" in text
    assert rec.fields[0]["source_bit_depth"] == 10
    assert rec.fields[0]["precision_preserved"] is False


def test_asking_for_10_bit_output_still_warns():
    """The more misleading case, not the less: the file really is Main10, so it
    looks like the precision survived when it did not."""
    rec = Recorder()
    assert pipeline.warn_if_source_is_deeper_than_8_bit(
        info_with("yuv420p10le"), 10, rec)
    assert len(rec.warnings) == 1
    assert "cannot bring back" in rec.warnings[0]
    assert rec.fields[0]["output_bit_depth"] == 10
    assert rec.fields[0]["precision_preserved"] is False


def test_a_12_bit_source_is_warned_about_as_12_bit():
    rec = Recorder()
    pipeline.warn_if_source_is_deeper_than_8_bit(
        info_with("yuv444p12le"), 8, rec)
    assert "12-bit" in rec.warnings[0]
    assert rec.fields[0]["source_bit_depth"] == 12


def test_both_entry_points_check():
    """A preview is what settings get judged from, so it carries the same
    caveat as a render."""
    import inspect

    for fn in (pipeline.convert, pipeline.preview_frame):
        assert "warn_if_source_is_deeper_than_8_bit" in inspect.getsource(fn), \
            fn.__name__


def test_the_decoder_really_does_flatten_to_8_bit():
    """Guards the premise. If the decode ever gains a wider path, this warning
    becomes wrong and should be revisited rather than left in place."""
    import inspect

    src = inspect.getsource(ffmpeg_io.decode_frames)
    assert '"rgb24"' in src
    assert '"rgb48le"' not in src
