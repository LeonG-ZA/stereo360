"""Turning the soundfield with the view.

A yaw moves the picture and leaves the sound behind. The failure is peculiarly
hard to notice -- nothing is missing, nothing distorts, everything is simply at
the wrong bearing -- and a sign error makes it exactly twice as wrong rather
than obviously broken. So the bearing is *measured*: a source is encoded at a
known azimuth, pushed through the real ffmpeg filter, and decoded back to see
where it landed.

Conventions are the whole difficulty. ambiX is ACN order (W, Y, Z, X -- not
W, X, Y, Z) with SN3D normalisation, and ambisonic azimuth runs anticlockwise
so a sound on your right is at a negative angle, while this tool's yaw is
longitude and runs the other way. See stereo360/ambisonics.py.
"""

import math
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from stereo360 import ambisonics

RATE = 48000


# ------------------------------------------------------------------ helpers

def encode_source(azimuth_deg: float, order: int, n: int = 4800) -> np.ndarray:
    """ambiX for one 440 Hz source at `azimuth_deg`, on the horizon.

    Only the m = +-1 pair of each degree is needed to read a bearing back, so
    the other harmonics are left at zero rather than given real SN3D gains --
    a rotation cannot move energy between degrees, so nothing here depends on
    them being right.
    """
    t = np.arange(n) / RATE
    s = 0.5 * np.sin(2 * math.pi * 440 * t)
    phi = math.radians(azimuth_deg)
    out = np.zeros((n, ambisonics.channels_for_order(order)))
    out[:, 0] = s                                   # W
    for degree in range(1, order + 1):
        base = degree * degree + degree
        out[:, base + 1] = s * math.cos(phi)        # (l, +1)
        out[:, base - 1] = s * math.sin(phi)        # (l, -1)
    return out


def write_wav(path: str, data: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(4)
        w.setframerate(RATE)
        w.writeframes((data * (2 ** 31 - 1)).astype("<i4").tobytes())


def apply_filter(tmp_path: Path, data: np.ndarray, expr: str) -> np.ndarray:
    """Run the real filter through the real ffmpeg, in and out as PCM."""
    src, dst = str(tmp_path / "in.wav"), str(tmp_path / "out.raw")
    write_wav(src, data)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", src, "-af", expr,
         "-f", "s32le", "-c:a", "pcm_s32le", dst],
        check=True, capture_output=True)
    raw = Path(dst).read_bytes()
    return np.frombuffer(raw, dtype="<i4").reshape(
        -1, data.shape[1]) / (2 ** 31 - 1)


def bearing(data: np.ndarray, degree: int = 1) -> float:
    """Where the (l, +1)/(l, -1) pair says the source is, in degrees."""
    base = degree * degree + degree
    ref = data[:, 0]
    return math.degrees(math.atan2(float(data[:, base - 1] @ ref),
                                   float(data[:, base + 1] @ ref)))


def wrap(deg: float) -> float:
    return (deg + 180) % 360 - 180


def _through_encoder(tmp_path: Path, data: np.ndarray, chain: str,
                     args: list) -> np.ndarray:
    """Filter, encode into an MP4, decode back. The whole real path."""
    src, dst = str(tmp_path / "e_in.wav"), str(tmp_path / "e_out.mp4")
    write_wav(src, data)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", chain,
                    *args, dst], check=True, capture_output=True)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", dst, "-f", "s32le", "-c:a",
         "pcm_s32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i4").reshape(
        -1, data.shape[1]) / (2 ** 31 - 1)


def _channel_mapping(sent: np.ndarray, got: np.ndarray) -> list:
    """Which input channel each output channel actually carries.

    Cross-correlation rather than picking a spectrum peak. The peak method
    looked convincing and was wrong -- it reported Opus permuting 9 and 16
    channels when the mapping is in fact the identity, which nearly cost the
    higher orders their support.
    """
    n = min(len(sent), len(got))
    return [int(np.argmax([abs(np.corrcoef(got[:n, j], sent[:n, i])[0, 1])
                           for i in range(sent.shape[1])]))
            for j in range(got.shape[1])]


# ----------------------------------------------------------- channel counts

@pytest.mark.parametrize("channels,order", [(4, 1), (9, 2), (16, 3)])
def test_ambix_channel_counts_map_to_orders(channels, order):
    assert ambisonics.order_for_channels(channels) == order
    assert ambisonics.channels_for_order(order) == channels


@pytest.mark.parametrize("channels", [None, 0, 1, 2, 3, 5, 6, 8, 10, 25, 36])
def test_anything_else_is_not_ambix(channels):
    """25 and 36 are fourth and fifth order, real but beyond what SA3D is
    written for here; 6 is 5.1, which is emphatically not a soundfield."""
    assert ambisonics.order_for_channels(channels) is None


# ------------------------------------------------------------- the matrix

def test_a_zero_yaw_is_the_identity():
    for order in (1, 2, 3):
        m = np.array(ambisonics.yaw_matrix(order, 0.0))
        assert m == pytest.approx(np.eye(len(m)), abs=1e-12)


def test_w_is_never_touched():
    """The omnidirectional channel has no direction to turn."""
    for order in (1, 2, 3):
        row = ambisonics.yaw_matrix(order, 137.0)[0]
        assert row[0] == 1.0
        assert not any(row[1:])


@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_m_zero_channels_are_never_touched(order):
    """Z and its higher-degree equivalents point along the rotation axis."""
    m = ambisonics.yaw_matrix(order, 42.0)
    for degree in range(order + 1):
        base = degree * degree + degree
        assert m[base][base] == 1.0
        assert sum(abs(v) for v in m[base]) == 1.0


@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_matrix_is_orthogonal(order):
    """A rotation preserves energy. If it did not, turning the view would
    change the loudness of the mix."""
    m = np.array(ambisonics.yaw_matrix(order, 61.0))
    assert (m @ m.T) == pytest.approx(np.eye(len(m)), abs=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_opposite_yaws_undo_each_other(order):
    a = np.array(ambisonics.yaw_matrix(order, 73.0))
    b = np.array(ambisonics.yaw_matrix(order, -73.0))
    assert (a @ b) == pytest.approx(np.eye(len(a)), abs=1e-12)


def test_first_order_matches_the_textbook_form():
    """X and Y turn by the angle, W and Z stay. Spelled out because ACN order
    is W, Y, Z, X and it is easy to write the matrix into the wrong slots."""
    m = ambisonics.yaw_matrix(1, 90.0)
    w, y, z, x = 0, 1, 2, 3
    assert m[w][w] == 1.0 and m[z][z] == 1.0
    assert m[x][x] == pytest.approx(0.0, abs=1e-12)
    assert m[x][y] == pytest.approx(-1.0)
    assert m[y][x] == pytest.approx(1.0)


# ------------------------------------------------------------ the filter

def test_the_layout_is_an_unknown_one_not_a_default():
    """`9c` does not parse at all -- no default layout has 9 channels -- so a
    lowercase spelling would mean second order simply never runs. And an
    unknown layout is what ambisonics honestly is: W/Y/Z/X are not speaker
    positions, and naming them invites a remap."""
    assert ambisonics.pan_filter(1, 30).startswith("pan=4C|")
    assert ambisonics.pan_filter(2, 30).startswith("pan=9C|")
    assert ambisonics.pan_filter(3, 30).startswith("pan=16C|")


def test_a_named_layout_would_not_be_harmless(tmp_path: Path):
    """The reason the line above matters, demonstrated rather than asserted.
    `quad` is a perfectly ordinary 4-channel layout and it does not survive:
    one channel is lost and another duplicated on the way to the file."""
    data = np.stack([
        0.2 * np.sin(2 * np.pi * (400 + 37 * i) * np.arange(19200) / RATE
                     + i) * (1 + 0.5 * np.sin(2 * np.pi * (3 + i)
                                              * np.arange(19200) / RATE))
        for i in range(4)], axis=1)
    identity = "|".join(f"c{i}=c{i}" for i in range(4))
    out = _through_encoder(tmp_path, data, f"pan=quad|{identity}",
                           ["-c:a", "aac", "-b:a", "384k"])
    assert _channel_mapping(data, out) != [0, 1, 2, 3]


def test_aac_will_not_take_the_unknown_layout_on_its_own(tmp_path: Path):
    """Why `audio_filter` is not just `pan_filter`. Caught the hard way: the
    unit tests passed on raw PCM and the first real render died with
    "Unsupported channel layout '4 channels'"."""
    data = encode_source(10.0, 1, n=4800)
    with pytest.raises(subprocess.CalledProcessError):
        _through_encoder(tmp_path, data, ambisonics.pan_filter(1, 30),
                         ["-c:a", "aac", "-b:a", "384k"])


def test_the_filter_never_renormalises():
    """`c0<...` would rescale each row to sum to 1 and destroy the rotation.
    The difference is one character."""
    expr = ambisonics.pan_filter(2, 45)
    assert "<" not in expr
    # One '=' per channel spec, plus the one in "pan=".
    assert expr.count("=") == expr.count("|") + 1


def test_ffmpeg_accepts_the_filter_at_every_order(tmp_path: Path):
    for order in (1, 2, 3):
        data = encode_source(20.0, order, n=480)
        out = apply_filter(tmp_path, data, ambisonics.pan_filter(order, 33.0))
        assert out.shape[1] == ambisonics.channels_for_order(order)


# ------------------------------------------------- what it sounds like

@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("azimuth,yaw", [
    (-90, 90),      # a source on the right, brought to the front
    (-150, 150),    # ... from behind the right shoulder
    (0, 0),         # nothing asked, nothing moved
    (0, 90),
    (30, -30),
    (120, 90),      # wraps past the back
])
def test_a_source_lands_where_the_view_points(tmp_path, order, azimuth, yaw):
    """The one that matters. Pointing the field `yaw` to the right must bring
    whatever was `yaw` to the right round to the front -- and in ambisonic
    azimuth, which runs the other way, that is the source at `-yaw`."""
    data = encode_source(azimuth, order)
    out = apply_filter(tmp_path, data, ambisonics.pan_filter(order, yaw))
    assert wrap(bearing(out) - (azimuth + yaw)) == pytest.approx(0.0, abs=0.5)


@pytest.mark.parametrize("order", [2, 3])
def test_every_degree_agrees_about_the_bearing(order):
    """Each degree turns by m times the angle, so a factor dropped from one of
    them would still leave the first-order pair correct. Checked on the matrix
    rather than through ffmpeg so the higher degrees can carry real signal."""
    azimuth, yaw = 25.0, 47.0
    data = encode_source(azimuth, order)
    m = np.array(ambisonics.yaw_matrix(order, yaw))
    out = data @ m.T
    for degree in range(1, order + 1):
        assert wrap(bearing(out, degree) - (azimuth + yaw)) == \
            pytest.approx(0.0, abs=0.01), f"degree {degree}"


def test_rotation_does_not_change_the_loudness(tmp_path: Path):
    data = encode_source(37.0, 1)
    out = apply_filter(tmp_path, data, ambisonics.pan_filter(1, 63.0))
    before = float(np.sqrt((data ** 2).sum()))
    after = float(np.sqrt((out ** 2).sum()))
    assert after == pytest.approx(before, rel=1e-4)


# ------------------------------------------------------------- encoding

def test_first_order_stays_aac():
    """What Google's spatial media spec uses, and what anything reading SA3D
    already expects."""
    assert ambisonics.encode_args(4)[:2] == ["-c:a", "aac"]
    assert not ambisonics.needs_opus(4)


@pytest.mark.parametrize("channels", [9, 16])
def test_higher_orders_need_opus(channels):
    """ffmpeg's native AAC encoder refuses 9 channels outright ("Unsupported
    channel layout") and mistakes 16 for 9.1.6, refusing that too."""
    assert ambisonics.needs_opus(channels)
    args = ambisonics.encode_args(channels)
    assert args[:2] == ["-c:a", "libopus"]
    assert "-mapping_family" in args


@pytest.mark.parametrize("channels", [4, 9, 16])
def test_the_chosen_encoder_can_actually_write_that_many_channels(channels,
                                                                  tmp_path):
    """The claim above, checked against ffmpeg rather than believed."""
    out = str(tmp_path / "a.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"anullsrc=cl={channels}C:r=48000:d=0.3",
         *ambisonics.encode_args(channels), out],
        check=True, capture_output=True)
    probed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0", out],
        capture_output=True, text=True, check=True)
    assert int(probed.stdout.strip()) == channels


@pytest.mark.parametrize("order", [1, 2, 3])
def test_channel_order_survives_the_whole_encode(order, tmp_path: Path):
    """The end of the chain, and the check that matters most: ACN order in,
    ACN order out. A remap here would put every sound somewhere plausible and
    wrong, which is the hardest kind of audio bug to notice."""
    n = ambisonics.channels_for_order(order)
    t = np.arange(19200) / RATE
    data = np.stack([0.2 * np.sin(2 * np.pi * (400 + 37 * i) * t + i)
                     * (1 + 0.5 * np.sin(2 * np.pi * (3 + i) * t))
                     for i in range(n)], axis=1)
    identity = "|".join(f"c{i}=c{i}" for i in range(n))
    chain = f"pan={n}C|{identity}"
    if not ambisonics.needs_opus(n):
        chain += ",aformat=channel_layouts=4.0"
    out = _through_encoder(tmp_path, data, chain, ambisonics.encode_args(n))
    assert _channel_mapping(data, out) == list(range(n))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_real_filter_chain_encodes_and_keeps_the_bearing(order, tmp_path):
    """`audio_filter` plus `encode_args`, exactly as the pipeline uses them,
    through a lossy codec and back."""
    data = encode_source(-60.0, order)
    out = _through_encoder(
        tmp_path, data, ambisonics.audio_filter(order, 60.0),
        ambisonics.encode_args(ambisonics.channels_for_order(order)))
    assert wrap(bearing(out)) == pytest.approx(0.0, abs=1.0)


@pytest.mark.parametrize("channels", [4, 9, 16])
def test_the_channel_count_survives_into_the_sample_entry(channels, tmp_path):
    """SA3D describes the track by reading `channelcount` back out of the
    audio sample entry, so a codec that wrote the wrong number there would
    produce a file claiming an order it does not have."""
    from stereo360 import spherical

    out = str(tmp_path / "a.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"anullsrc=cl={channels}C:r=48000:d=0.3",
         *ambisonics.encode_args(channels), out],
        check=True, capture_output=True)

    data = Path(out).read_bytes()
    tag = b"Opus" if ambisonics.needs_opus(channels) else b"mp4a"
    entry = data.find(tag) - 4
    assert entry > 0, f"no {tag.decode()} sample entry"
    assert spherical._audio_channels(data, entry) == channels
    assert struct.calcsize(">H") == 2        # the field the reader assumes
