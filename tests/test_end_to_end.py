"""End-to-end M1 test: synthetic video -> convert -> verify output properties."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from stereo360 import ffmpeg_io, pipeline, spherical
from stereo360.events import Reporter


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


# ------------------------------------------------- ambisonic rotation


def make_ambisonic_video(path, azimuth_deg, w=256, h=128, frames=10, fps=10):
    """A test clip whose 4-channel track holds one source at a known bearing.

    The audio is built as raw ambiX rather than by any ffmpeg helper, so what
    goes in is known exactly and what comes out can be compared against it.
    """
    import math
    import wave

    seconds = frames / fps
    n = int(48000 * seconds)
    t = np.arange(n) / 48000.0
    s = 0.5 * np.sin(2 * math.pi * 440 * t)
    phi = math.radians(azimuth_deg)
    data = np.zeros((n, 4))
    data[:, 0] = s                      # W
    data[:, 1] = s * math.sin(phi)      # Y
    data[:, 3] = s * math.cos(phi)      # X

    wav = str(Path(path).with_suffix(".wav"))
    with wave.open(wav, "wb") as f:
        f.setnchannels(4)
        f.setsampwidth(4)
        f.setframerate(48000)
        f.writeframes((data * (2 ** 31 - 1)).astype("<i4").tobytes())

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}:d={seconds}",
         "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "384k", "-shortest", path],
        check=True)
    return path


def measure_bearing(path):
    """Where the rendered file's soundfield puts the source, in degrees."""
    import math

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0",
         "-f", "s32le", "-c:a", "pcm_s32le", "-"],
        check=True, capture_output=True).stdout
    a = np.frombuffer(raw, dtype="<i4").reshape(-1, 4) / (2 ** 31 - 1)
    ref = a[:, 0]
    return math.degrees(math.atan2(float(a[:, 1] @ ref), float(a[:, 3] @ ref)))


def test_a_yaw_turns_the_soundfield_with_the_picture(tmp_path: Path):
    """The whole point of the feature. A source 90 degrees to the right is at
    ambisonic azimuth -90; pointing the VR180 field at it must bring it round
    to the front, or the viewer looks straight at something they hear beside
    them."""
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_ambisonic_video(src, azimuth_deg=-90)

    pipeline.convert(src, dst, face_size=64, max_frames=6,
                     output_mode="vr180", yaw=90.0, spatial_audio=True)

    got = measure_bearing(dst)
    assert abs((got + 180) % 360 - 180) < 3.0, (
        f"source ended up at {got:.1f} deg, should be straight ahead")
    # The re-encode must not cost the track its description: SA3D is written
    # by reading the channel count back out of the sample entry, so a codec
    # that wrote the wrong number would leave the file claiming a different
    # order than it has -- or refuse to be tagged at all.
    assert ffmpeg_io.probe(dst).audio_channels == 4
    # Not has_spherical_metadata(), which only looks for the V1 uuid box --
    # deliberately absent in VR180, since V1 cannot express a partial sphere.
    assert b"SA3D" in Path(dst).read_bytes()


def test_without_a_yaw_the_audio_is_copied_untouched(tmp_path: Path):
    """No rotation means no re-encode, so the default path keeps its audio
    bit-exact. A feature that silently costs everyone a generation of AAC
    would not be worth having."""
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_ambisonic_video(src, azimuth_deg=40)

    pipeline.convert(src, dst, face_size=64, max_frames=6,
                     output_mode="vr180", yaw=0.0, spatial_audio=True)

    assert abs(measure_bearing(dst) - 40.0) < 3.0
    # Bitrate would be the obvious check and is not a valid one: the render is
    # truncated to 6 frames, so the average changes even on a pure copy. Ask
    # the planner directly instead -- returning no filter is exactly what
    # "copied" means to the encoder.
    info = ffmpeg_io.probe(src)
    assert pipeline.plan_audio_rotation(
        info, yaw=0.0, spatial_audio=True, reporter=Reporter()) == (None, None)


def test_the_rotation_is_reported(tmp_path: Path):
    """It changes the audio and costs a generation, so it is not something to
    do quietly."""
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_ambisonic_video(src, azimuth_deg=0)

    said = []

    class Say(Reporter):
        def info(self, message, **kw):
            said.append(("info", message))

        def warning(self, message, **kw):
            said.append(("warning", message))

    pipeline.convert(src, dst, face_size=64, max_frames=6,
                     output_mode="vr180", yaw=45.0, spatial_audio=True,
                     reporter=Say())
    assert any("Rotating the order-1 soundfield" in m for _, m in said)


def test_ambisonic_looking_audio_without_the_flag_is_flagged(tmp_path: Path):
    """Four channels and a yaw is almost certainly a forgotten flag, and the
    consequence -- every sound at the wrong bearing -- is inaudible as an
    error."""
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_ambisonic_video(src, azimuth_deg=0)

    warned = []

    class Say(Reporter):
        def warning(self, message, **kw):
            warned.append(message)

        def info(self, message, **kw):
            pass

    pipeline.convert(src, dst, face_size=64, max_frames=6,
                     output_mode="vr180", yaw=45.0, spatial_audio=False,
                     reporter=Say())
    assert any("--spatial-audio" in m for m in warned)
    # ... and it is only a hint: the render still happens, untouched.
    assert abs(measure_bearing(dst)) < 3.0


def test_plain_stereo_and_a_yaw_says_nothing(tmp_path: Path):
    """A two-channel bed has no soundfield to turn and is usually head-locked
    on purpose. Warning about it would be noise on every ordinary render."""
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=256, h=128, frames=6, fps=10)

    warned = []

    class Say(Reporter):
        def warning(self, message, **kw):
            warned.append(message)

        def info(self, message, **kw):
            pass

    pipeline.convert(src, dst, face_size=64, max_frames=4,
                     output_mode="vr180", yaw=45.0, reporter=Say())
    assert not [m for m in warned if "ambiX" in m or "soundfield" in m]


def test_the_probe_reports_the_audio_channel_count(tmp_path: Path):
    """What the decision rests on."""
    src = str(tmp_path / "in.mp4")
    make_ambisonic_video(src, azimuth_deg=0)
    assert ffmpeg_io.probe(src).audio_channels == 4

    plain = str(tmp_path / "plain.mp4")
    make_test_video(plain, w=128, h=64, frames=4, fps=10)
    assert ffmpeg_io.probe(plain).audio_channels == 1

    silent = str(tmp_path / "silent.mp4")
    make_test_video(silent, w=128, h=64, frames=4, fps=10, with_audio=False)
    assert ffmpeg_io.probe(silent).audio_channels is None
