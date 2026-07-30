"""Colour handling across the decode -> RGB -> encode round trip.

Two things are easy to get wrong here and invisible when they are:

  * the output carrying no colour tags, so a player must guess the matrix
    (and for a 7680-wide frame it guesses BT.709);
  * swscale's default rounding, which makes a 10-bit output no better than an
    8-bit one.

Both were true before these tests existed.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from stereo360 import ffmpeg_io, pipeline
from stereo360.events import Reporter
from stereo360.ffmpeg_io import ColorTags
from test_end_to_end import make_test_video


def _tagged_video(path: str, space: str, rng: str, w=128, h=64,
                  frames=4) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc2=size={w}x{h}:rate=10:duration={frames / 10}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-colorspace", space, "-color_primaries", space,
         "-color_trc", space, "-color_range", rng, path], check=True)


def _probe_color(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=color_space,color_range,color_transfer,color_primaries",
         "-of", "json", path], capture_output=True, text=True,
        check=True).stdout
    import json
    return json.loads(out)["streams"][0]


# ------------------------------------------------------------------ probing


def test_probe_reads_colour_tags(tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    _tagged_video(src, "smpte170m", "tv")
    color = ffmpeg_io.probe(src).color
    assert color.declared
    assert color.space == "smpte170m" and color.range == "tv"


def test_probe_reports_undeclared_tags_as_none(tmp_path: Path):
    """An untagged source must stay untagged: guessing a matrix on the
    output would introduce exactly the shift this is meant to avoid."""
    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=128, h=64, frames=3, with_audio=False)
    color = ffmpeg_io.probe(src).color
    assert color.space is None and color.primaries is None


def test_color_tags_build_encoder_flags():
    tags = ColorTags(range="tv", space="bt709", transfer="bt709",
                     primaries="bt709")
    args = tags.encoder_args()
    assert args[args.index("-colorspace") + 1] == "bt709"
    assert args[args.index("-color_range") + 1] == "tv"
    assert not ColorTags().declared
    assert ColorTags().encoder_args() == []


# ------------------------------------------------------- end-to-end tagging


@pytest.mark.parametrize("space,rng", [("smpte170m", "tv"), ("bt709", "tv")])
def test_conversion_carries_colour_tags_through(tmp_path: Path, space, rng):
    """The output must declare the source's matrix and range.

    Only those two are asserted, deliberately. x264 and x265 do not write
    transfer or primaries into the VUI whatever they are asked for, so a
    stricter check would either fail on a fully tagged source or pass only
    because the fixture happens to lose them too. Matrix and range are the
    pair that decides how the image decodes.
    """
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    _tagged_video(src, space, rng)
    assert ffmpeg_io.probe(src).color.space == space

    pipeline.convert(src, dst, face_size=32, use_cubemap=False)

    after = ffmpeg_io.probe(dst).color
    assert after.space == space, after
    assert after.range == rng, after


def test_untagged_source_produces_untagged_output(tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=3, with_audio=False)

    pipeline.convert(src, dst, face_size=32, use_cubemap=False)

    got = _probe_color(dst)
    assert got.get("color_space") in (None, "unknown")


# --------------------------------------------------------- rounding quality


def _ramp(size=256) -> np.ndarray:
    """A near-flat gradient with real chroma variation.

    The sky/wall case: smooth enough that conversion rounding dominates, and
    the only content where these differences are measurable rather than lost
    under texture.
    """
    x = np.linspace(0, 30, size, dtype=np.float32)
    img = np.zeros((size, size, 3), np.uint8)
    img[:, :, 0] = (100 + x[None, :]).round().clip(0, 255)
    img[:, :, 1] = (110 + x[None, :] * 0.8).round().clip(0, 255)
    img[:, :, 2] = (130 + x[None, :] * 0.6).round().clip(0, 255)
    return img


def _encode(img: np.ndarray, dst: str, bitdepth: int) -> None:
    h, w = img.shape[:2]
    with ffmpeg_io.VideoEncoder(dst, w, h, 10.0, codec="libx265", crf=12,
                               bitdepth=bitdepth,
                               color=ColorTags(range="tv", space="bt709",
                                               transfer="bt709",
                                               primaries="bt709")) as enc:
        enc.write(img)


def _rmse_8bit(img: np.ndarray, dst: str) -> float:
    back = next(iter(ffmpeg_io.decode_frames(dst)))
    return float(np.sqrt(((back.astype(np.int32)
                           - img.astype(np.int32)) ** 2).mean()))


def _rmse_16bit(img: np.ndarray, dst: str) -> float:
    """Error as seen by a consumer that keeps more than 8 bits."""
    h, w = img.shape[:2]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", dst, "-frames:v", "1",
         "-pix_fmt", "rgb48le", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    got = np.frombuffer(raw[:w * h * 6], "<u2").reshape(h, w, 3) / 257.0
    return float(np.sqrt(((got - img.astype(np.float64)) ** 2).mean()))


def test_accurate_scaling_improves_the_eight_bit_output(tmp_path: Path,
                                                        monkeypatch):
    """Guards the sws flags by turning them off and showing it gets worse.

    This is the change's real justification: it improves the *default* 8-bit
    output that every user gets, at no measurable cost.
    """
    img = _ramp()

    monkeypatch.setattr(ffmpeg_io, "_SWS_FLAGS", "")
    plain = str(tmp_path / "plain.mp4")
    _encode(img, plain, 8)
    without = _rmse_8bit(img, plain)

    monkeypatch.setattr(ffmpeg_io, "_SWS_FLAGS", "+accurate_rnd+full_chroma_int")
    accurate = str(tmp_path / "accurate.mp4")
    _encode(img, accurate, 8)
    with_flags = _rmse_8bit(img, accurate)

    assert with_flags < without, (
        f"accurate rounding should reduce error: {with_flags:.3f} "
        f"vs {without:.3f}")


def test_ten_bit_keeps_more_precision_than_eight(tmp_path: Path):
    """Why 10-bit is worth offering, measured where the benefit is real.

    Deliberately compared at 16-bit output. Decoded back down to 8-bit RGB
    10-bit measures *worse* than 8-bit (2.10 vs 1.29), because the 10->8
    crush adds error the 8-bit path never pays -- so the benefit is only
    realised by a playback chain that keeps the extra bits. Asserting it
    through an 8-bit decode would be asserting something false.
    """
    img = _ramp()
    eight = str(tmp_path / "d8.mp4")
    ten = str(tmp_path / "d10.mp4")
    _encode(img, eight, 8)
    _encode(img, ten, 10)

    assert _rmse_16bit(img, ten) < _rmse_16bit(img, eight)


def test_ten_bit_output_is_tagged_main10(tmp_path: Path):
    img = _ramp(128)
    dst = str(tmp_path / "d10.mp4")
    _encode(img, dst, 10)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=pix_fmt", "-of", "csv=p=0", dst],
        capture_output=True, text=True, check=True).stdout.strip()
    assert out == "yuv420p10le"


# ------------------------------------------------------ chroma subsampling


@pytest.mark.parametrize("pix_fmt,expected", [
    ("yuv420p", "4:2:0"), ("yuvj420p", "4:2:0"), ("yuva420p", "4:2:0"),
    ("yuv422p10le", "4:2:2"), ("yuv444p", "4:4:4"), ("yuv444p12le", "4:4:4"),
    ("nv12", "4:2:0"), ("nv16", "4:2:2"),
    ("gbrp", "4:4:4"), ("rgb24", "4:4:4"), ("gray", "4:0:0"),
    ("yuv411p", "4:1:1"), (None, None), ("something_odd", None),
])
def test_chroma_from_pix_fmt(pix_fmt, expected):
    assert ffmpeg_io.chroma_from_pix_fmt(pix_fmt) == expected


def test_supported_chroma_keeps_hardware_encoders_on_420():
    assert ffmpeg_io.supported_chroma("libx265") == ("4:2:0", "4:2:2", "4:4:4")
    assert ffmpeg_io.supported_chroma("hevc_nvenc") == ("4:2:0",)


def _info(pix_fmt):
    return ffmpeg_io.VideoInfo(width=8, height=4, fps=10.0, frame_count=1,
                               duration=0.1, has_audio=False, pix_fmt=pix_fmt)


class _Rec(Reporter):
    def __init__(self):
        self.info_msgs, self.warnings = [], []

    def info(self, message, **f):
        self.info_msgs.append(message)

    def warning(self, message, **f):
        self.warnings.append(message)


def test_resolve_chroma_defaults_to_420_without_the_flag():
    rec = _Rec()
    assert pipeline._resolve_chroma(_info("yuv444p"), "libx265", False,
                                   rec) == "4:2:0"
    assert not rec.info_msgs and not rec.warnings, "should say nothing"


def test_resolve_chroma_follows_the_source_when_asked():
    rec = _Rec()
    assert pipeline._resolve_chroma(_info("yuv444p"), "libx265", True,
                                   rec) == "4:4:4"
    assert any("4:4:4" in m for m in rec.info_msgs)


def test_resolve_chroma_degrades_rather_than_failing(recwarn):
    """A ticked box is a preference. Failing a two-hour render because a
    hardware encoder cannot oblige would be the wrong trade."""
    rec = _Rec()
    assert pipeline._resolve_chroma(_info("yuv444p"), "hevc_nvenc", True,
                                   rec) == "4:2:0"
    assert any("cannot produce" in w for w in rec.warnings)

    rec = _Rec()
    assert pipeline._resolve_chroma(_info("yuv411p"), "libx265", True,
                                   rec) == "4:2:0"
    assert rec.warnings

    rec = _Rec()
    assert pipeline._resolve_chroma(_info(None), "libx265", True,
                                   rec) == "4:2:0"
    assert any("Could not determine" in w for w in rec.warnings)


def _make_444(path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=128x64:rate=10:duration=0.3", "-c:v", "libx264",
         "-pix_fmt", "yuv444p", path], check=True)


def test_source_subsampling_is_honoured_end_to_end(tmp_path: Path):
    src = str(tmp_path / "in444.mp4")
    _make_444(src)
    assert ffmpeg_io.probe(src).chroma == "4:4:4"

    followed = str(tmp_path / "followed.mp4")
    pipeline.convert(src, followed, face_size=32, use_cubemap=False,
                     codec="libx265", source_subsampling=True)
    assert ffmpeg_io.probe(followed).pix_fmt == "yuv444p"


def test_420_is_the_default_even_for_a_444_source(tmp_path: Path):
    """4:2:0 is the only layout headsets decode in hardware at 8K, so it must
    not be something a user gives up by accident."""
    src = str(tmp_path / "in444.mp4")
    _make_444(src)

    dst = str(tmp_path / "default.mp4")
    pipeline.convert(src, dst, face_size=32, use_cubemap=False,
                     codec="libx265")
    assert ffmpeg_io.probe(dst).pix_fmt == "yuv420p"


def test_encoder_rejects_a_chroma_the_codec_cannot_do(tmp_path: Path):
    with pytest.raises(ValueError, match="cannot encode"):
        ffmpeg_io.VideoEncoder(str(tmp_path / "x.mp4"), 64, 64, 10.0,
                               codec="hevc_nvenc", chroma="4:4:4")


def test_probe_json_reports_chroma(tmp_path: Path):
    import json as _json
    import sys

    src = str(tmp_path / "in444.mp4")
    _make_444(src)
    out = subprocess.run(
        [sys.executable, "-m", "stereo360", src, "--probe-json"],
        capture_output=True, text=True, check=True,
        cwd=str(Path(__file__).resolve().parent.parent)).stdout
    data = _json.loads(out)
    assert data["chroma"] == "4:4:4" and data["pix_fmt"] == "yuv444p"
    assert data["width"] == 128 and data["frame_count"] == 3
