"""Reading a cubemap-projected input instead of assuming equirectangular.

Before this, a cubemap file was silently treated as equirectangular and the
output was geometric nonsense from the first frame, with no warning anywhere.
"""

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stereo360 import ffmpeg_io, pipeline, projection, spherical
from stereo360.events import Reporter

ROOT = Path(__file__).resolve().parent.parent
F = 64                                  # face size for the fixtures


def _equirect(path: str, frames: int = 3, face: int = F) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc2=size={face * 4}x{face * 2}:rate=10:"
         f"duration={frames / 10}", "-c:v", "libx264", "-crf", "12",
         "-pix_fmt", "yuv420p", "-movflags", "-faststart", path], check=True)


def _cubemap(src: str, dst: str, tag: bool = True, face: int = F) -> None:
    """A real 3x2 cubemap made by ffmpeg's v360, optionally tagged cbmp."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src,
         "-vf", f"v360=e:c3x2:w={face * 3}:h={face * 2}",
         "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
         "-movflags", "-faststart", dst], check=True)
    if not tag:
        return
    real = spherical._sv3d
    spherical._sv3d = lambda: spherical._box(
        "sv3d",
        spherical._full_box("svhd", b"stereo360\x00")
        + spherical._box("proj",
                         spherical._full_box("prhd", struct.pack(">iii", 0, 0, 0))
                         + spherical._full_box("cbmp", struct.pack(">II", 0, 0))))
    try:
        spherical.inject_spherical_metadata(dst, stereo_mode="mono")
    finally:
        spherical._sv3d = real


class _Rec(Reporter):
    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, message, **f):
        self.infos.append(message)

    def warning(self, message, **f):
        self.warnings.append(message)


# ------------------------------------------------------------------ probing


def test_probe_reads_the_declared_projection(tmp_path: Path):
    src = str(tmp_path / "e.mp4")
    cube = str(tmp_path / "c.mp4")
    _equirect(src)
    _cubemap(src, cube)

    assert ffmpeg_io.probe(cube).projection == "cubemap"
    assert ffmpeg_io.probe(cube).cubemap_padding == 0
    # An ordinary file declares nothing, which is not the same as "not 360".
    assert ffmpeg_io.probe(src).projection is None


# ------------------------------------------------------------------ slicing


def test_cubemap_slices_into_our_faces(tmp_path: Path):
    src = str(tmp_path / "e.mp4")
    cube = str(tmp_path / "c.mp4")
    _equirect(src)
    _cubemap(src, cube, tag=False)

    frame = next(iter(ffmpeg_io.decode_frames(cube)))
    faces = projection.cubemap_3x2_to_faces(frame)
    assert set(faces) == set(projection.FACES)
    assert all(f.shape == (F, F, 3) for f in faces.values())


def test_face_layout_matches_our_own_projection(tmp_path: Path):
    """The mapping that must not be guessed.

    Each tile of ffmpeg's c3x2 is compared against the face our own
    `equirect_to_cubemap` produces from the same frame. A wrong face order or
    rotation still yields a plausible-looking picture, so this checks the
    right assignment wins by a wide margin rather than merely that it works.
    """
    src = str(tmp_path / "e.mp4")
    cube = str(tmp_path / "c.mp4")
    _equirect(src, face=128)
    _cubemap(src, cube, tag=False, face=128)

    equi = next(iter(ffmpeg_io.decode_frames(src)))
    ours = projection.equirect_to_cubemap(equi, 128)
    theirs = projection.cubemap_3x2_to_faces(
        next(iter(ffmpeg_io.decode_frames(cube))))

    for name, mine in ours.items():
        errs = {other: float(np.abs(mine.astype(np.int16)
                                    - theirs[other].astype(np.int16)).mean())
                for other in theirs}
        best = min(errs, key=errs.get)
        runner_up = min((v for k, v in errs.items() if k != best))
        assert best == name, f"{name} matched {best} instead"
        assert errs[name] * 4 < runner_up, (
            f"{name}: match {errs[name]:.1f} not clearly better than "
            f"{runner_up:.1f}")


def test_padding_is_cropped_away():
    """cbmp padding is duplicated edge samples; using them would smear the
    face borders."""
    pad = 4
    tile = F + 2 * pad
    frame = np.zeros((tile * 2, tile * 3, 3), np.uint8)
    for row in range(2):
        for col in range(3):
            frame[row * tile:(row + 1) * tile,
                  col * tile:(col + 1) * tile] = 200        # padding value
            frame[row * tile + pad:(row + 1) * tile - pad,
                  col * tile + pad:(col + 1) * tile - pad] = 50   # real face
    faces = projection.cubemap_3x2_to_faces(frame, padding=pad)
    for name, face in faces.items():
        assert face.shape == (F, F, 3), name
        assert (face == 50).all(), f"{name} kept padding pixels"


def test_slicing_rejects_a_layout_that_cannot_be_a_cubemap():
    with pytest.raises(ValueError, match="divisible"):
        projection.cubemap_3x2_to_faces(np.zeros((100, 100, 3), np.uint8))
    # 2:1 is equirect's aspect, and gives non-square tiles.
    with pytest.raises(ValueError, match="square"):
        projection.cubemap_3x2_to_faces(np.zeros((60, 120, 3), np.uint8))


# --------------------------------------------------------------- resolution


def test_untagged_input_is_assumed_equirectangular():
    """Most files declare nothing, and equirect is the only projection V1 can
    express or YouTube accepts on upload."""
    info = ffmpeg_io.VideoInfo(1024, 512, 30.0, 10, 1.0, False)
    assert pipeline.resolve_projection(info, "auto", Reporter()) == \
        "equirectangular"


def test_declared_cubemap_is_used_and_announced():
    info = ffmpeg_io.VideoInfo(768, 512, 30.0, 10, 1.0, False,
                               projection="cubemap")
    rec = _Rec()
    assert pipeline.resolve_projection(info, "auto", rec) == "cubemap"
    assert any("cubemap" in m for m in rec.infos)


def test_unreadable_projection_stops_rather_than_guessing():
    """Treating a mesh projection as equirect would produce nonsense with no
    hint as to why."""
    info = ffmpeg_io.VideoInfo(1024, 512, 30.0, 10, 1.0, False,
                               projection="mesh")
    with pytest.raises(ValueError, match="cannot read"):
        pipeline.resolve_projection(info, "auto", Reporter())


def test_explicit_override_wins_and_says_so():
    info = ffmpeg_io.VideoInfo(768, 512, 30.0, 10, 1.0, False,
                               projection="cubemap")
    rec = _Rec()
    assert pipeline.resolve_projection(info, "equirectangular", rec) == \
        "equirectangular"
    assert any("declares" in m for m in rec.warnings)


def test_cubemap_geometry_derives_the_output_size():
    """A cubemap of F-px faces carries the angular detail of a 4F-wide
    equirect, the same rule the equirect path uses in reverse."""
    info = ffmpeg_io.VideoInfo(768, 512, 30.0, 10, 1.0, False,
                               projection="cubemap")
    face, w, h = pipeline.source_geometry(info, "cubemap", None, Reporter())
    assert (face, w, h) == (256, 1024, 512)

    # Padding shrinks the usable face.
    info.cubemap_padding = 8
    face, w, h = pipeline.source_geometry(info, "cubemap", None, Reporter())
    assert (face, w, h) == (240, 960, 480)


def test_face_size_override_is_refused_for_cubemap_input():
    info = ffmpeg_io.VideoInfo(768, 512, 30.0, 10, 1.0, False,
                               projection="cubemap")
    rec = _Rec()
    face, _, _ = pipeline.source_geometry(info, "cubemap", 512, rec)
    assert face == 256, "must use the source's own faces"
    assert any("ignored" in m for m in rec.warnings)


# ------------------------------------------------------------- end to end


def test_cubemap_input_converts_and_matches_the_equirect_path(tmp_path: Path):
    """The point of the feature: the same scene through either projection
    should land in the same place."""
    src = str(tmp_path / "e.mp4")
    cube = str(tmp_path / "c.mp4")
    _equirect(src, frames=2, face=128)
    _cubemap(src, cube, face=128)

    from_equi = str(tmp_path / "out_e.mp4")
    from_cube = str(tmp_path / "out_c.mp4")
    pipeline.convert(src, from_equi, max_frames=1, use_cubemap=False)
    pipeline.convert(cube, from_cube, max_frames=1, use_cubemap=False)

    a = ffmpeg_io.probe(from_equi)
    b = ffmpeg_io.probe(from_cube)
    assert (a.width, a.height) == (b.width, b.height)

    left_a = next(iter(ffmpeg_io.decode_frames(from_equi)))[:a.height // 2]
    left_b = next(iter(ffmpeg_io.decode_frames(from_cube)))[:b.height // 2]
    err = float(np.abs(left_a.astype(np.int16)
                       - left_b.astype(np.int16)).mean())
    # Two resamples and two compression passes separate them; anything near
    # this means the geometry agrees. A wrong face order lands far higher.
    assert err < 12, f"left eyes disagree: mean|err|={err:.1f}"


def test_cubemap_faces_reach_the_depth_stage(tmp_path: Path):
    """The efficiency claim: the source's own faces are used, not rebuilt."""
    src = str(tmp_path / "e.mp4")
    cube = str(tmp_path / "c.mp4")
    _equirect(src, frames=1)
    _cubemap(src, cube)

    info = ffmpeg_io.probe(cube)
    frames = list(pipeline.read_source(cube, info, "cubemap", F * 4, F * 2,
                                       1, 0))
    assert len(frames) == 1
    assert frames[0].faces is not None
    assert set(frames[0].faces) == set(projection.FACES)
    assert frames[0].equirect.shape == (F * 2, F * 4, 3)

    # Equirect input carries no faces; the depth stage builds them itself.
    plain = list(pipeline.read_source(src, ffmpeg_io.probe(src),
                                      "equirectangular", F * 4, F * 2, 1, 0))
    assert plain[0].faces is None


def test_cli_rejects_an_unreadable_projection(tmp_path: Path):
    src = str(tmp_path / "e.mp4")
    _equirect(src, frames=1)
    proc = subprocess.run(
        [sys.executable, "-m", "stereo360", src, "-o",
         str(tmp_path / "o.mp4"), "--passthrough", "--max-frames", "1",
         "--input-projection", "cubemap", "--progress-json"],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    # A 4:2 equirect cannot be sliced as a 3x2 cubemap; it must say so.
    assert proc.returncode == 1
    events = [json.loads(x) for x in proc.stdout.splitlines() if x.strip()]
    assert any(e["type"] == "error" for e in events), proc.stdout
