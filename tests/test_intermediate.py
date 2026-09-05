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
    # Every encoder in the table opens on this pretend machine. Faked because
    # `opens` really runs ffmpeg, so without this these tests assert a
    # different encoder depending on whether the machine running them owns an
    # NVIDIA card -- which is the very thing `opens` was added to stop the
    # *pipeline* assuming.
    monkeypatch.setattr(intermediate, "opens",
                        lambda ffmpeg, name: name in table)
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
    source's chroma to keep the faster encoder is the wrong way round.

    Asserts the rule and not the winner. This named ffv1 while ffv1 happened
    to be second, and would have failed when libx264 -- which also carries
    4:2:2 -- moved ahead of it, for a reordering that changed nothing this
    test is about."""
    args, note = intermediate.encoder_args("ffmpeg", pix_fmt="yuv422p")
    assert "hevc_nvenc" not in args
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


# ------------------------------------------- an encoder that cannot open

def test_an_encoder_that_cannot_open_here_is_not_chosen(monkeypatch, formats):
    """`supported` reads what an encoder was compiled to accept, which says
    nothing about whether it can run. Every full ffmpeg build lists
    hevc_nvenc, NVIDIA card or not, and it leads both tables for speed -- so
    on an AMD or Intel machine it was picked and then failed at the first
    frame with "Could not open encoder before EOF". That is every pre-pass at
    once: Topaz, RIFE and the shader all write their working file here."""
    monkeypatch.setattr(intermediate, "opens",
                        lambda ffmpeg, name: name != "hevc_nvenc")
    args, note = intermediate.encoder_args("ffmpeg", pix_fmt="yuv420p",
                                           lossless=True)
    assert "hevc_nvenc" not in args
    assert "hevc_nvenc" not in note
    assert "-c:v" in args


def test_the_fallback_also_skips_an_encoder_that_cannot_open(monkeypatch):
    """The give-up path picks whatever can carry yuv420p, and had the same
    hole: it would have named the encoder that cannot start."""
    monkeypatch.setattr(intermediate, "supported",
                        lambda ffmpeg, name: frozenset({"yuv420p"}))
    monkeypatch.setattr(intermediate, "opens",
                        lambda ffmpeg, name: name != "hevc_nvenc")
    args, _ = intermediate.encoder_args("ffmpeg", pix_fmt="yuv422p",
                                        lossless=True)
    assert "hevc_nvenc" not in args


def test_opens_is_asked_once_per_encoder(monkeypatch):
    """It starts an ffmpeg to find out, so a per-frame or per-file re-ask
    would be paid over and over for an answer that cannot change."""
    calls = []

    class Done:
        returncode = 0

    def fake(cmd, **kw):
        calls.append(cmd)
        return Done()

    monkeypatch.setattr(intermediate.subprocess, "run", fake)
    monkeypatch.setattr(intermediate, "_opens", {})
    for _ in range(4):
        intermediate.opens("ffmpeg", "libx264")
    assert len(calls) == 1


def test_the_probe_frame_clears_hardware_encoder_minimums():
    """A probe frame below a hardware encoder's minimum size rejects an
    encoder that works perfectly at real sizes -- worse than not checking,
    because it pushes a machine with working silicon onto a CPU encoder.

    Measured on a Radeon 780M: h264_amf and hevc_amf fail at 64x64 with
    `encoder->Init() failed with error 5` and succeed from 128x128. The
    binding minimums cannot be measured on that machine, so they are asserted
    from the vendors' documented floors: NVENC needs 145 wide for H.264 and
    160 for AV1, QSV about 176x144."""
    from stereo360 import intermediate

    w, h = (int(n) for n in intermediate._PROBE_SIZE.split("x"))
    assert w >= 160 and h >= 144, "below a documented encoder minimum"
    assert w % 2 == 0 and h % 2 == 0, "yuv420p needs even dimensions"


def test_the_probe_asks_at_the_declared_size(monkeypatch):
    """The constant is only worth having if the command actually uses it."""
    from stereo360 import intermediate

    seen = []

    class Done:
        returncode = 0

    monkeypatch.setattr(intermediate, "_opens", {})
    monkeypatch.setattr(intermediate.subprocess, "run",
                        lambda cmd, **kw: (seen.append(cmd), Done())[1])
    intermediate.opens("ffmpeg", "libx264")
    assert any(intermediate._PROBE_SIZE in str(a) for a in seen[0])


def test_the_faster_lossless_encoder_comes_first(monkeypatch, formats):
    """Both are bit-exact; x264 is the faster and smaller of the two, measured
    on 8K frames at 0.074 s/frame against ffv1's 0.126, and 27 MB against 40.
    ffv1 led the list anyway, so every machine without an NVIDIA card -- the
    machines this fallback exists for -- took the slower one."""
    # No NVIDIA card, which is the case this ordering is about: with one,
    # hevc_nvenc leads and neither of these two is reached.
    monkeypatch.setattr(intermediate, "opens",
                        lambda ffmpeg, name: name != "hevc_nvenc")
    args, _ = intermediate.encoder_args("ffmpeg", pix_fmt="yuv420p",
                                        lossless=True)
    assert "libx264" in args
    assert "ffv1" not in args


def test_ffv1_still_takes_what_x264_cannot(monkeypatch, formats):
    """Reordering must not cost coverage. ffv1 carries 12-, 14- and 16-bit,
    which x264 does not, and the format check runs before the order -- so a
    deep source lands on ffv1 rather than being reduced to fit the faster
    encoder."""
    formats["ffv1"] = formats["ffv1"] | {"yuv420p12le"}
    monkeypatch.setattr(intermediate, "opens",
                        lambda ffmpeg, name: name != "hevc_nvenc")
    args, _ = intermediate.encoder_args("ffmpeg", pix_fmt="yuv420p12le",
                                        lossless=True)
    assert "ffv1" in args
    assert args[args.index("-pix_fmt") + 1] == "yuv420p12le"
