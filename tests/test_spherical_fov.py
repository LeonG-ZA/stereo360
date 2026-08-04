"""Partial-sphere projection bounds, for VR180 output.

The `equi` box says how much of a full sphere the frame covers. It was
hardcoded to all-zero — "the whole thing" — which is right for 360 and a lie
for 180. These tests pin the arithmetic, the round trip through a real reader,
and the decision to drop V1 when the sphere is partial.

Nothing in the pipeline passes a partial field yet; this is the metadata layer
on its own, which is why it can be tested without rendering anything.
"""

import json
import shutil
import struct
import subprocess

import pytest

from stereo360 import spherical


def _mp4(tmp_path, name, width=320, height=160):
    """A tiny real MP4 with moov last, so injection does not move mdat."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    out = tmp_path / name
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
           "-i", f"testsrc=size={width}x{height}:rate=10:duration=1", "-an",
           "-c:v", "libx264", "-movflags", "-faststart", "-y", str(out)]
    if subprocess.run(cmd, capture_output=True).returncode:
        pytest.skip("ffmpeg could not build the fixture")
    return str(out)


def _equi_from_file(path):
    """The four bounds as fractions, read back out of the written box."""
    data = open(path, "rb").read()
    i = data.find(b"equi")
    assert i != -1, "no equi box was written"
    top, bottom, left, right = struct.unpack(">IIII", data[i + 8:i + 24])
    return tuple(v / 2 ** 32 for v in (top, bottom, left, right))


# --------------------------------------------------------------- arithmetic

@pytest.mark.parametrize("fov,side", [
    (360.0, 0.0),
    (270.0, 0.125),
    (180.0, 0.25),
    (90.0, 0.375),
])
def test_bounds_crop_the_right_fraction(fov, side):
    top, bottom, left, right = spherical._equi_bounds(fov)
    assert left == right == pytest.approx(side * 2 ** 32, rel=1e-9)
    assert top == bottom == 0, "vertical is never cropped"


@pytest.mark.parametrize("bad", [0.0, -1.0, 361.0, float("nan")])
def test_an_impossible_field_of_view_is_refused(bad):
    with pytest.raises(ValueError, match="horizontal_fov"):
        spherical._equi_bounds(bad)


def test_the_field_of_view_is_validated_before_the_file_is_touched(tmp_path):
    """A bad value must not leave a half-written file behind."""
    path = _mp4(tmp_path, "guard.mp4")
    before = open(path, "rb").read()
    with pytest.raises(ValueError, match="horizontal_fov"):
        spherical.inject_spherical_metadata(path, horizontal_fov=0.0)
    assert open(path, "rb").read() == before, "the file was modified anyway"


# ------------------------------------------------------------- what is written

def test_the_default_is_still_a_full_sphere(tmp_path):
    """The 360 path must be untouched by this change."""
    path = _mp4(tmp_path, "full.mp4")
    spherical.inject_spherical_metadata(path)
    assert _equi_from_file(path) == (0.0, 0.0, 0.0, 0.0)


def test_180_crops_a_quarter_from_each_side(tmp_path):
    path = _mp4(tmp_path, "half.mp4")
    spherical.inject_spherical_metadata(path, stereo_mode="left-right",
                                        horizontal_fov=180.0)
    top, bottom, left, right = _equi_from_file(path)
    assert (left, right) == (0.25, 0.25)
    assert (top, bottom) == (0.0, 0.0)


def test_v1_is_written_for_a_full_sphere_and_omitted_for_a_partial_one(tmp_path):
    """V1 cannot say "half a sphere" without using fields whose meaning for a
    side-by-side frame is ambiguous. Claiming a full sphere would make a
    V1-only reader stretch 180 degrees across 360 — exactly what VLC 3.0.21
    does. Omitting it leaves such a reader showing flat 2D: obviously wrong
    rather than subtly wrong."""
    full = _mp4(tmp_path, "v1full.mp4")
    spherical.inject_spherical_metadata(full)
    assert b"GSpherical:Spherical" in open(full, "rb").read()

    half = _mp4(tmp_path, "v1half.mp4")
    spherical.inject_spherical_metadata(half, stereo_mode="left-right",
                                        horizontal_fov=180.0)
    data = open(half, "rb").read()
    assert b"GSpherical:Spherical" not in data, "V1 claims a full sphere"
    assert spherical.SPHERICAL_UUID not in data
    # V2 still has to be there, or the file describes nothing at all.
    assert b"st3d" in data and b"sv3d" in data and b"equi" in data


def test_stereo_mode_and_bounds_are_independent(tmp_path):
    """A 360 file can be left-right, and a 180 file could in principle be mono.
    Nothing should couple them."""
    path = _mp4(tmp_path, "mono180.mp4")
    spherical.inject_spherical_metadata(path, stereo_mode="mono",
                                        horizontal_fov=180.0)
    data = open(path, "rb").read()
    # Compare against the module's own box rather than hand-computing an
    # offset into it, which is easy to get wrong and says less when it breaks.
    assert spherical._st3d("mono") in data
    assert spherical._st3d("left-right") not in data
    assert _equi_from_file(path)[2] == 0.25


# ----------------------------------------------------------- round trip

def _ffprobe_projection(path):
    exe = shutil.which("ffprobe")
    if exe is None:
        pytest.skip("ffprobe not on PATH")
    out = subprocess.run(
        [exe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream_side_data", "-of", "json", path],
        capture_output=True, text=True).stdout
    for sd in json.loads(out)["streams"][0].get("side_data_list", []):
        if sd.get("side_data_type") == "Spherical Mapping":
            return sd
    return None


def test_a_real_reader_agrees_it_is_a_180_field(tmp_path):
    """The arithmetic being self-consistent proves nothing on its own. This
    checks an independent implementation reads the same field of view back."""
    width, height = 320, 160
    path = _mp4(tmp_path, "probe180.mp4", width, height)
    spherical.inject_spherical_metadata(path, stereo_mode="left-right",
                                        horizontal_fov=180.0)
    sd = _ffprobe_projection(path)
    assert sd is not None, "ffprobe saw no spherical mapping at all"
    assert sd["projection"] == "tiled equirectangular", sd

    # ffprobe reports the bounds as pixels *outside* the frame, so the field of
    # view comes back out of them.
    total = width + sd["bound_left"] + sd["bound_right"]
    assert 360.0 * width / total == pytest.approx(180.0, abs=1.0), sd
    assert sd["bound_top"] == sd["bound_bottom"] == 0


def test_a_real_reader_still_sees_a_full_sphere_by_default(tmp_path):
    path = _mp4(tmp_path, "probe360.mp4")
    spherical.inject_spherical_metadata(path)
    sd = _ffprobe_projection(path)
    assert sd is not None
    assert sd["projection"] == "equirectangular", sd
    assert "bound_left" not in sd or sd["bound_left"] == 0


# ------------------------------------------- detecting what is already there

def test_v2_metadata_is_detected_without_v1(tmp_path):
    """The bug this closes: `has_spherical_metadata` walked only as deep as
    `stbl`, and `st3d`/`sv3d` live one level further in, inside the video
    sample entry. So it found V1 and nothing else.

    It went unnoticed while every file carried V1 too. VR180 omits V1 -- it
    cannot express a partial sphere -- and the function then said "no
    metadata" about a file full of it.
    """
    from stereo360 import spherical

    out = _tagged(tmp_path, "v2.mp4", stereo_mode="left-right",
                  horizontal_fov=180.0)
    data = out.read_bytes()
    assert spherical.SPHERICAL_UUID not in data, "V1 should be absent here"
    assert b"st3d" in data and b"sv3d" in data
    assert spherical.has_spherical_metadata(str(out))


def test_v1_metadata_is_still_detected(tmp_path):
    """360 output carries both; the V1 path must keep working."""
    from stereo360 import spherical

    out = _tagged(tmp_path, "v1.mp4", stereo_mode="top-bottom",
                  horizontal_fov=360.0)
    assert spherical.SPHERICAL_UUID in out.read_bytes()
    assert spherical.has_spherical_metadata(str(out))


def test_an_untagged_file_is_still_reported_as_untagged(tmp_path):
    """The predicate has to be able to say no, or injection never happens."""
    from stereo360 import spherical

    plain = _plain_mp4(tmp_path, "plain.mp4")
    assert not spherical.has_spherical_metadata(str(plain))


@pytest.mark.parametrize("mode,fov", [("top-bottom", 360.0),
                                      ("left-right", 180.0)])
def test_injecting_twice_changes_nothing(tmp_path, mode, fov):
    """What the guard is for. Before the fix it never fired for VR180, so a
    second call would have written a second set of boxes into the file."""
    from stereo360 import spherical

    out = _tagged(tmp_path, "twice.mp4", stereo_mode=mode, horizontal_fov=fov)
    first = out.read_bytes()
    spherical.inject_spherical_metadata(str(out), stereo_mode=mode,
                                        horizontal_fov=fov)
    again = out.read_bytes()
    assert again == first, "a second injection must be a no-op"
    assert again.count(b"st3d") == 1 and again.count(b"sv3d") == 1


def _plain_mp4(tmp_path, name):
    """A small MP4 with no spherical metadata at all."""
    out = tmp_path / name
    if subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc=size=64x32:rate=10:duration=1", "-an", "-c:v",
             "libx264", "-movflags", "-faststart", "-y", str(out)],
            capture_output=True).returncode:
        pytest.skip("ffmpeg could not build the fixture")
    return out


def _tagged(tmp_path, name, *, stereo_mode, horizontal_fov):
    from stereo360 import spherical

    out = _plain_mp4(tmp_path, name)
    spherical.inject_spherical_metadata(str(out), stereo_mode=stereo_mode,
                                        horizontal_fov=horizontal_fov)
    return out
