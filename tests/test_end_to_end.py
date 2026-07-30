"""End-to-end M1 test: synthetic video -> convert -> verify output properties."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from stereo360 import ffmpeg_io, pipeline, spherical


def make_test_video(path: str, w=512, h=256, fps=10, frames=20, with_audio=True) -> None:
    """Generate a small synthetic 360-ish test video with a tone audio track."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}:duration={frames / fps}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={frames / fps}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [path]
    subprocess.run(cmd, check=True)


def test_full_pipeline(tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src)

    pipeline.convert(src, dst, face_size=128, max_frames=10)

    info = ffmpeg_io.probe(dst)
    assert info.height == 512 and info.width == 512  # top-bottom: doubled height
    assert info.has_audio

    # Spherical metadata present and parseable
    assert spherical.has_spherical_metadata(dst)

    # ffprobe should see a valid file with the expected number of frames
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-count_frames", "-show_streams", dst],
        capture_output=True, text=True, check=True,
    ).stdout
    vstream = next(s for s in json.loads(out)["streams"] if s["codec_type"] == "video")
    assert int(vstream["nb_read_frames"]) == 10


def test_ten_bit_encode(tmp_path: Path):
    """10-bit encoding paths (x264 high10 + x265) produce valid 10-bit output."""
    import json as _json

    src = str(tmp_path / "in.mp4")
    make_test_video(src, frames=3)
    frames = list(ffmpeg_io.decode_frames(src))
    info = ffmpeg_io.probe(src)

    for codec in ("libx264", "libx265"):
        dst = str(tmp_path / f"ten_{codec}.mp4")
        with ffmpeg_io.VideoEncoder(
            dst, info.width, info.height, info.fps, codec=codec, bitdepth=10
        ) as enc:
            for f in frames:
                enc.write(f)
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", dst],
            capture_output=True, text=True, check=True,
        ).stdout
        vstream = next(s for s in _json.loads(out)["streams"]
                       if s["codec_type"] == "video")
        assert vstream["pix_fmt"] == "yuv420p10le", f"{codec}: {vstream['pix_fmt']}"


def test_encoder_write_frame_forms_are_equivalent(tmp_path: Path):
    """uint8, strided and float32 frames all reach ffmpeg as identical bytes.

    write() hands an already-uint8 C-contiguous array's own memory straight to
    the pipe and converts anything else, so the three forms have to stay
    indistinguishable downstream.
    """
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (64, 128, 3), dtype=np.uint8)
    variants = {
        "contiguous": frame,
        # Same values, but strided -- the fast path must not take this one.
        "strided": np.repeat(frame, 2, axis=1)[:, ::2],
        "float32": frame.astype(np.float32),
    }
    assert np.array_equal(variants["strided"], frame)
    assert not variants["strided"].flags.c_contiguous

    digests = {}
    for name, arr in variants.items():
        dst = str(tmp_path / f"{name}.mp4")
        with ffmpeg_io.VideoEncoder(dst, 128, 64, 10.0) as enc:
            for _ in range(3):
                enc.write(arr)
        digests[name] = hashlib.md5(Path(dst).read_bytes()).hexdigest()

    assert len(set(digests.values())) == 1, digests


def test_decode_encode_roundtrip(tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "rt.mp4")
    make_test_video(src, frames=5)

    info = ffmpeg_io.probe(src)
    frames = list(ffmpeg_io.decode_frames(src))
    assert len(frames) == 5
    assert frames[0].shape == (info.height, info.width, 3)

    with ffmpeg_io.VideoEncoder(dst, info.width, info.height, info.fps) as enc:
        for f in frames:
            enc.write(f)
    info2 = ffmpeg_io.probe(dst)
    assert info2.width == info.width and info2.height == info.height
