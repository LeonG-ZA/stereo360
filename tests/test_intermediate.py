"""What a pre-pass writes its working file with.

The file exists for minutes and is deleted, so it is tempting to treat it as
throwaway -- but the converter estimates depth from it, so what it loses is
lost underneath the depth as well as in the picture.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stereo360 import intermediate                             # noqa: E402


@pytest.fixture
def formats(monkeypatch):
    """Pretend an ffmpeg whose encoders take what the real ones take here."""
    table = {
        "hevc_nvenc": {"yuv420p", "p010le", "yuv444p", "gbrp"},
        "ffv1": {"yuv420p", "yuv422p", "yuv444p", "yuv420p10le",
                 "yuv422p10le", "yuv444p10le"},
        "libx264": {"yuv420p", "yuv422p", "yuv444p", "yuv420p10le"},
    }
    monkeypatch.setattr(intermediate, "supported",
                        lambda ffmpeg, name: frozenset(table.get(name, ())))
    return table


# ------------------------------------------------------------ what is kept


@pytest.mark.parametrize("pix_fmt", [
    "yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv444p10le",
])
def test_a_sources_format_is_carried_through(formats, pix_fmt):
    """A 10-bit source became 8-bit and a 4:4:4 one became 4:2:0, before the
    depth model ever saw them -- while the pipeline separately offers
    --bitdepth 10 and warns about deep sources."""
    # p010le is yuv420p10le packed the other way up -- the same ten bits, so
    # it counts as keeping the format rather than losing it.
    same = {"yuv420p10le": {"yuv420p10le", "p010le"}}.get(pix_fmt, {pix_fmt})
    args, note = intermediate.encoder_args("ffmpeg", pix_fmt=pix_fmt)
    assert args[args.index("-pix_fmt") + 1] in same
    assert "wanted" not in note, "nothing should have been given up"


def test_ten_bit_420_may_be_repacked_but_not_reduced(formats):
    """hevc_nvenc spells it p010le -- the same ten bits the other way up, so
    taking that is keeping the format, not losing it."""
    formats["ffv1"] = set()          # force the NVENC branch
    args, note = intermediate.encoder_args("ffmpeg", pix_fmt="yuv420p10le")
    assert args[args.index("-pix_fmt") + 1] == "p010le"


def test_keeping_the_format_outranks_the_encoder_order(formats):
    """hevc_nvenc leads the list for speed and cannot do 4:2:2. Dropping a
    source's chroma to keep the faster encoder is the wrong way round."""
    args, note = intermediate.encoder_args("ffmpeg", pix_fmt="yuv422p")
    assert "ffv1" in args
    assert args[args.index("-pix_fmt") + 1] == "yuv422p"


@pytest.mark.parametrize("pix_fmt", ["gbrp", "rgb24", "bgr0", None, "nonsense"])
def test_rgb_never_survives(formats, pix_fmt):
    """Topaz hands back RGB and the converter's encoder then refuses the gbr
    colour tags. That is what the old blanket yuv420p was really for."""
    args, _ = intermediate.encoder_args("ffmpeg", pix_fmt=pix_fmt)
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"


def test_giving_up_a_format_is_said_out_loud(formats):
    """When nothing can carry it, the note has to admit what was dropped."""
    for name in formats:
        formats[name] = {"yuv420p"}
    _, note = intermediate.encoder_args("ffmpeg", pix_fmt="yuv444p10le")
    assert "wanted yuv444p10le" in note


# --------------------------------------------------------- lossless or not


def test_lossless_is_the_default_where_it_fits(formats):
    args, note = intermediate.encoder_args("ffmpeg", pix_fmt="yuv420p",
                                           lossless=True)
    assert "lossless" in note and "near" not in note
    assert "-tune" in args and "lossless" in args


def test_near_lossless_is_what_happens_when_it_does_not(formats):
    args, note = intermediate.encoder_args("ffmpeg", pix_fmt="yuv420p",
                                           lossless=False)
    assert "near-lossless" in note
    assert "-qp" in args


def test_a_short_job_gets_lossless(monkeypatch, tmp_path):
    monkeypatch.setattr(intermediate.shutil, "disk_usage",
                        lambda p: type("U", (), {"free": 500 * 1024 ** 3})())
    assert intermediate.wants_lossless(7680, 3840, 120, str(tmp_path)) is True


def test_an_hour_of_8k_does_not(monkeypatch, tmp_path):
    """About 10 MB a frame at 8K: an hour is a terabyte, and a working file is
    not worth filling someone's disk over."""
    monkeypatch.setattr(intermediate.shutil, "disk_usage",
                        lambda p: type("U", (), {"free": 500 * 1024 ** 3})())
    assert intermediate.wants_lossless(7680, 3840, 108000, str(tmp_path)) is False


def test_an_unknown_length_asks_for_room(monkeypatch, tmp_path):
    """Our own intermediates carry no frame count, so this case is real."""
    for free, expected in ((500 * 1024 ** 3, True), (10 * 1024 ** 3, False)):
        monkeypatch.setattr(intermediate.shutil, "disk_usage",
                            lambda p, f=free: type("U", (), {"free": f})())
        assert intermediate.wants_lossless(7680, 3840, None,
                                           str(tmp_path)) is expected


def test_an_unreadable_drive_does_not_stop_the_run(monkeypatch, tmp_path):
    def boom(path):
        raise OSError("no such drive")

    monkeypatch.setattr(intermediate.shutil, "disk_usage", boom)
    assert intermediate.wants_lossless(7680, 3840, 120, str(tmp_path)) is True


def test_an_ffmpeg_with_nothing_still_gets_an_answer(monkeypatch):
    """Better a working file than a crash: ffv1 is in every build."""
    monkeypatch.setattr(intermediate, "supported",
                        lambda ffmpeg, name: frozenset())
    args, note = intermediate.encoder_args("ffmpeg", pix_fmt="yuv420p")
    assert "ffv1" in args and "yuv420p" in args
