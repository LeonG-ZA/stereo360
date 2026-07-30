"""Spatial-media metadata: the boxes, and where they go.

Placement is what makes this work. A player looks for the Spherical V1 blob
inside the video `trak` and the V2 boxes inside the video *sample entry*; put
them anywhere else and a conforming parser sees flat 2D video, which is
exactly the symptom of having to re-run Google's injector by hand.
"""

import shutil
import struct
import subprocess

import pytest

from stereo360 import spherical


def _mp4(tmp_path, name, audio_channels=0):
    """A tiny real MP4, moov last (no faststart), optionally with audio."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    out = tmp_path / name
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc=size=160x160:rate=10:duration=1"]
    if audio_channels:
        cmd += ["-f", "lavfi", "-i", "anoisesrc=d=1:r=48000",
                "-map", "0:v", "-map", "1:a", "-c:a", "aac",
                "-ac", str(audio_channels)]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-movflags", "-faststart", "-y", str(out)]
    if subprocess.run(cmd, capture_output=True).returncode:
        pytest.skip("ffmpeg could not build the fixture")
    return str(out)


def _boxes_in_trak(path):
    """Types of the boxes directly inside each trak."""
    data = open(path, "rb").read()
    start, size, header = spherical._find_moov(data)
    found = []
    for pos, btype, _, bheader, path_ in spherical._walk(
            data, start + header, start + size):
        if path_ and path_[-1] == "trak":
            found.append(btype)
    return found, data


def test_v1_uuid_goes_inside_the_video_trak(tmp_path):
    """Not at moov level, which is where it used to land.

    The spec puts it in the track; a parser that follows the spec will not
    look at moov's own children, so a file with it there reads as 2D.
    """
    path = _mp4(tmp_path, "v1.mp4")
    spherical.inject_spherical_metadata(path)
    boxes, data = _boxes_in_trak(path)
    assert "uuid" in boxes, f"uuid not a child of trak; found {boxes}"
    assert b"GSpherical:Spherical" in data
    assert b"<GSpherical:StereoMode>top-bottom" in data


def test_v2_boxes_are_written(tmp_path):
    path = _mp4(tmp_path, "v2.mp4")
    spherical.inject_spherical_metadata(path)
    data = open(path, "rb").read()
    for tag in (b"st3d", b"sv3d", b"svhd", b"proj", b"prhd", b"equi"):
        assert tag in data, f"{tag.decode()} missing"
    # st3d payload: full box (4 bytes) then the stereo mode. 1 = top-bottom.
    i = data.find(b"st3d")
    assert data[i + 4 + 4] == 1


def test_ffmpeg_recognises_the_result(tmp_path):
    """The real test of placement: does a parser other than ours find it?"""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("ffprobe not on PATH")
    path = _mp4(tmp_path, "probe.mp4")
    spherical.inject_spherical_metadata(path)
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream_side_data=side_data_type", "-of", "default=nw=1", path],
        capture_output=True, text=True).stdout
    assert "Spherical Mapping" in out, out
    assert "Stereo 3D" in out, out


def test_injection_is_idempotent(tmp_path):
    path = _mp4(tmp_path, "twice.mp4")
    spherical.inject_spherical_metadata(path)
    once = open(path, "rb").read()
    spherical.inject_spherical_metadata(path)
    assert open(path, "rb").read() == once


def test_spatial_audio_writes_sa3d_for_ambix(tmp_path):
    path = _mp4(tmp_path, "ambix.mp4", audio_channels=4)
    spherical.inject_spherical_metadata(path, spatial_audio=True)
    data = open(path, "rb").read()
    i = data.find(b"SA3D")
    assert i > 0, "SA3D not written"
    version, atype, order, ordering, norm, channels = struct.unpack(
        ">BBIBBI", data[i + 4:i + 4 + 12])
    assert (version, atype) == (0, 0)          # periphonic
    assert ordering == 0 and norm == 0         # ACN / SN3D, i.e. ambiX
    assert (order, channels) == (1, 4)         # first order


def test_spatial_audio_refuses_non_ambisonic_audio(tmp_path):
    """Silently mislabelling stereo as ambisonics would break playback."""
    path = _mp4(tmp_path, "stereo.mp4", audio_channels=2)
    with pytest.raises(ValueError, match="ambiX"):
        spherical.inject_spherical_metadata(path, spatial_audio=True)


def test_spatial_audio_refuses_when_there_is_no_audio(tmp_path):
    path = _mp4(tmp_path, "silent.mp4")
    with pytest.raises(ValueError, match="no audio track"):
        spherical.inject_spherical_metadata(path, spatial_audio=True)


def test_sa3d_is_not_written_unless_asked(tmp_path):
    path = _mp4(tmp_path, "quiet.mp4", audio_channels=4)
    spherical.inject_spherical_metadata(path)
    assert b"SA3D" not in open(path, "rb").read()
