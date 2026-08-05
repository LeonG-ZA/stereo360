"""Encoder discovery and per-vendor flag dialects.

Availability is resolution-dependent in a way that matters here: measured on
this hardware, hevc_amf encodes 3840x3840 and refuses 7680x7680, and
h264_nvenc stops at 4096x4096. A top-bottom stereo frame is twice the height
of its source, so it lands on the wrong side of those limits far more often
than ordinary video does.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stereo360 import encoders, ffmpeg_io

ROOT = Path(__file__).resolve().parent.parent
SMALL = (320, 240)          # every encoder that exists at all manages this


# ---------------------------------------------------------------- families


@pytest.mark.parametrize("codec,family", [
    ("libx264", "sw"), ("libx265", "sw"),
    ("hevc_nvenc", "nvenc"), ("h264_nvenc", "nvenc"),
    ("hevc_qsv", "qsv"), ("h264_qsv", "qsv"),
    ("hevc_amf", "amf"), ("h264_amf", "amf"),
])
def test_encoder_family(codec, family):
    assert ffmpeg_io.encoder_family(codec) == family


def test_each_family_speaks_its_own_quality_dialect():
    """Only x264/x265 have -crf; passing it to the others is a hard error, so
    the translation has to be per family rather than a shared default."""
    def args(codec):
        return ffmpeg_io._quality_args(
            ffmpeg_io.encoder_family(codec), codec, 18, "slow")

    assert args("libx265") == ["-crf", "18", "-preset", "slow"]
    assert args("hevc_nvenc") == ["-rc", "vbr", "-cq", "18",
                                  "-preset", "p5"]
    assert args("hevc_qsv") == ["-global_quality", "18", "-preset", "slow"]
    assert args("hevc_amf") == ["-rc", "cqp", "-qp_i", "18", "-qp_p", "18",
                                "-quality", "quality"]
    # No family may emit -crf except the software one.
    for codec in ("hevc_nvenc", "hevc_qsv", "hevc_amf"):
        assert "-crf" not in args(codec)


def test_hardware_encoders_are_kept_on_420():
    for codec in ("hevc_nvenc", "hevc_qsv", "hevc_amf"):
        assert ffmpeg_io.supported_chroma(codec) == ("4:2:0",)
    assert ffmpeg_io.supported_chroma("libx265") == ("4:2:0", "4:2:2", "4:4:4")


# ------------------------------------------------------------------ probing


def test_probe_reports_every_candidate_this_ffmpeg_has():
    """Candidate order is preserved, minus anything not built in -- a Windows
    build has no VideoToolbox and a macOS one has no AMF."""
    infos = encoders.probe(*SMALL)
    compiled = encoders._compiled_in()
    assert [i.name for i in infos] == [c[0] for c in encoders.CANDIDATES
                                       if c[0] in compiled]
    assert all(i.detail for i in infos)
    # Software encoders are always there if ffmpeg is.
    by = {i.name: i for i in infos}
    assert by["libx264"].available and by["libx265"].available
    assert not by["libx264"].hardware and by["hevc_nvenc"].hardware


def test_available_hardware_is_marked_not_recommended():
    """They are the speed option, not the quality one, and the list has to say
    so where the choice is made."""
    for info in encoders.probe(*SMALL):
        if info.available and info.hardware:
            assert "not recommended" in info.detail


def test_recommended_prefers_a_software_encoder():
    infos = encoders.probe(*SMALL)
    assert encoders.recommended(infos) == "libx264"
    # Even when hardware is available and listed first.
    reordered = [i for i in infos if i.hardware] + \
                [i for i in infos if not i.hardware]
    assert encoders.recommended(reordered) == "libx264"


def test_the_default_does_not_follow_the_display_order():
    """The list is ordered best-quality-first for the dropdown, which puts
    libx265 above libx264. `recommended` must not drift along with it: it
    answers "what is safe to encode with when nobody chose", and the answer
    is the most compatible one, not the best-looking one."""
    assert encoders.CANDIDATES[0][0] == "libx265", "display order"
    assert encoders.recommended(encoders.probe(*SMALL)) == "libx264"


def test_the_list_reads_as_a_quality_ranking():
    """Ordered for the dropdown: software first, then H.265 hardware, then
    H.264 hardware. Grouped by codec rather than by vendor, since only one
    vendor's entries can work on a given machine and the codec is the part
    that is actually a choice."""
    names = [c[0] for c in encoders.CANDIDATES]
    hardware = {c[0]: c[1] for c in encoders.CANDIDATES}

    groups = [[], [], []]
    for name in names:
        if not hardware[name]:
            groups[0].append(name)
        elif name.startswith("hevc_"):
            groups[1].append(name)
        else:
            groups[2].append(name)
    assert names == groups[0] + groups[1] + groups[2], \
        "software, then HEVC hardware, then H.264 hardware"
    assert groups[0] == ["libx265", "libx264"], "better compression first"

    # Every hardware backend offered for one codec is offered for the other,
    # so a machine does not lose its accelerator by picking H.264.
    assert ([n[len("hevc_"):] for n in groups[1]]
            == [n[len("h264_"):] for n in groups[2]])


def test_probe_is_resolution_dependent():
    """The whole reason for probing per project rather than once."""
    small = {i.name: i.available for i in encoders.probe(*SMALL)}
    huge = {i.name: i.available for i in encoders.probe(7680, 7680)}
    # Nothing may gain availability by being asked for a bigger frame.
    for name in small:
        if huge[name]:
            assert small[name], f"{name} works at 8K but not at {SMALL}"


def test_failure_reason_is_the_encoders_own_message():
    """ffmpeg ends every one of these failures with the same generic wrapper;
    the specific cause is the first line the encoder itself logged."""
    stderr = (
        "[h264_nvenc @ 0000020345df7c00] No capable devices found\n"
        "[vost#0:0/h264_nvenc @ 1] Error while opening encoder\n"
        "[vf#0:0 @ 2] Terminating thread with return code -22 "
        "(Invalid argument)\n"
    )
    assert encoders._reason(stderr, "h264_nvenc") == \
        "no GPU here can encode this size"

    qsv = "[hevc_qsv @ 1] Error creating a MFX session: -9.\n" \
          "[vf#0:0 @ 2] Terminating thread with return code -22\n"
    assert encoders._reason(qsv, "hevc_qsv") == "no Intel Quick Sync device"

    amf = "[hevc_amf @ 1] encoder->Init() failed with error 5\n" \
          "[vf#0:0 @ 2] Terminating thread with return code -22\n"
    assert "AMD" in encoders._reason(amf, "hevc_amf")

    # Anything unrecognised still yields the encoder's own words, not ffmpeg's
    # generic tail.
    odd = "[hevc_amf @ 1] something new and specific\n" \
          "[vf#0:0 @ 2] Terminating thread with return code -22\n"
    assert encoders._reason(odd, "hevc_amf") == "something new and specific"


def test_probe_encoders_cli_emits_json():
    proc = subprocess.run(
        [sys.executable, "-m", "stereo360", "-", "--probe-encoders",
         f"{SMALL[0]}x{SMALL[1]}"],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["width"] == SMALL[0] and data["height"] == SMALL[1]
    assert data["recommended"] == "libx264"
    compiled = encoders._compiled_in()
    assert {e["name"] for e in data["encoders"]} == \
        {c[0] for c in encoders.CANDIDATES if c[0] in compiled}


def test_probe_encoders_rejects_a_bad_size():
    proc = subprocess.run(
        [sys.executable, "-m", "stereo360", "-", "--probe-encoders", "huge"],
        capture_output=True, text=True, timeout=180, cwd=str(ROOT))
    assert proc.returncode != 0
    assert "WxH" in proc.stderr


# ------------------------------------------------------------- real encodes


def test_every_available_encoder_actually_encodes(tmp_path: Path):
    """The probe's promise, kept: anything it calls available must survive a
    real VideoEncoder round trip with our flag translation, not just ffmpeg's
    defaults."""
    frame = np.random.randint(0, 255, (256, 256, 3), np.uint8)
    for info in encoders.probe(256, 256):
        if not info.available:
            continue
        dst = str(tmp_path / f"{info.name}.mp4")
        with ffmpeg_io.VideoEncoder(dst, 256, 256, 10.0, codec=info.name,
                                    crf=20, preset="medium") as enc:
            enc.write(frame)
        assert ffmpeg_io.probe(dst).width == 256, info.name


# --------------------------------------------------- vaapi / videotoolbox


@pytest.mark.parametrize("codec,family", [
    ("hevc_vaapi", "vaapi"), ("h264_vaapi", "vaapi"),
    ("hevc_videotoolbox", "videotoolbox"),
    ("h264_videotoolbox", "videotoolbox"),
])
def test_platform_families_are_recognised(codec, family):
    assert ffmpeg_io.encoder_family(codec) == family


def test_vaapi_needs_a_device_and_an_upload_filter():
    """VAAPI's only supported input pixel format is `vaapi`, so it is not just
    a different quality flag: frames must be uploaded to the GPU first, and a
    device opened before the graph is built."""
    assert ffmpeg_io.hw_input_args("vaapi") == [
        "-vaapi_device", ffmpeg_io.VAAPI_DEVICE]
    assert ffmpeg_io.hw_filter_args("vaapi") == [
        "-vf", "format=nv12,hwupload"]
    # The filter chain decides the format, so naming one would fight it.
    assert ffmpeg_io.pix_fmt_for("vaapi", "4:2:0", 8) is None

    # Every other family takes software frames directly.
    for family in ("sw", "nvenc", "qsv", "amf", "videotoolbox"):
        assert ffmpeg_io.hw_input_args(family) == []
        assert ffmpeg_io.hw_filter_args(family) == []


def test_vaapi_and_videotoolbox_quality_dialects():
    def args(codec):
        return ffmpeg_io._quality_args(
            ffmpeg_io.encoder_family(codec), codec, 18, "slow")

    assert args("hevc_vaapi") == ["-rc_mode", "CQP", "-qp", "18"]
    # VideoToolbox counts the other way: higher is better.
    assert args("hevc_videotoolbox") == ["-q:v", "64"]
    assert ffmpeg_io._quality_args("videotoolbox", "x", 13, "slow") == \
        ["-q:v", "74"]
    for codec in ("hevc_vaapi", "hevc_videotoolbox"):
        assert "-crf" not in args(codec)


def test_vaapi_qp_stays_in_range():
    """VAAPI's QP is H.264/HEVC's own 0-51; a CRF outside that would be
    rejected outright."""
    for crf, expected in ((0, "0"), (18, "18"), (51, "51"), (99, "51")):
        args = ffmpeg_io._quality_args("vaapi", "hevc_vaapi", crf, "medium")
        assert args[args.index("-qp") + 1] == expected


def test_videotoolbox_quality_stays_in_range():
    for crf in (0, 13, 18, 51, 99):
        args = ffmpeg_io._quality_args("videotoolbox", "x", crf, "medium")
        value = int(args[args.index("-q:v") + 1])
        assert 1 <= value <= 100


def test_probe_omits_encoders_this_ffmpeg_lacks():
    """A dropdown listing VideoToolbox on Windows is noise; the platform that
    cannot have it should simply not offer it."""
    names = {i.name for i in encoders.probe(*SMALL)}
    compiled = encoders._compiled_in()
    assert names <= compiled
    # But they can be asked for explicitly, for diagnostics.
    verbose = {i.name: i for i in
               encoders.probe(*SMALL, include_missing=True)}
    for name, _, _ in encoders.CANDIDATES:
        if name not in compiled:
            assert not verbose[name].available
            assert "not built into this ffmpeg" in verbose[name].detail


def test_probe_and_encoder_agree_on_the_invocation():
    """A probe that tested different flags from the render would be worse than
    no probe: it would call an encoder usable and then fail on frame one."""
    import inspect

    source = inspect.getsource(encoders._try_encode)
    assert "hw_input_args" in source and "hw_filter_args" in source
