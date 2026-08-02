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


def _candidate(name: str, channels: int):
    """The Encoder for `name`, whether or not this build can run it."""
    return next(e for e in ambisonics._candidates(channels) if e.name == name)


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


# ------------------------------------------------------- choosing an encoder

def test_the_native_aac_encoder_is_never_offered():
    """ffmpeg's own `aac` will encode four channels quite happily and sounds
    materially worse than libfdk_aac at the same bitrate. Reaching for it
    because it is the one spelled "aac" is a well-worn way to degrade a
    master, so it is not in the list at any priority."""
    for channels in (4, 9, 16):
        names = [e.name for e in ambisonics._candidates(channels)]
        assert "aac" not in names, names
        assert "aac_mf" not in names


def test_libfdk_aac_is_first_in_line():
    """Best AAC there is, and AAC is what SA3D players expect. Absent from
    most Windows builds only because its licence is not GPL-compatible."""
    assert ambisonics._candidates(4)[0].name == "libfdk_aac"


def test_the_order_of_preference_is_quality_then_compatibility():
    assert [e.name for e in ambisonics._candidates(4)] == [
        "libfdk_aac", "libopus", "pcm_s24le"]


def test_auto_skips_what_this_ffmpeg_does_not_have(monkeypatch):
    """The probe runs a real encode rather than trusting `-encoders`, because
    a build can list a codec it cannot actually run."""
    monkeypatch.setattr(ambisonics, "_probe_cache", {})
    monkeypatch.setattr(ambisonics, "_can_encode",
                        lambda e, ch: e.name == "pcm_s24le")
    assert ambisonics.choose_encoder(4).name == "pcm_s24le"


def test_auto_takes_the_best_available(monkeypatch):
    monkeypatch.setattr(ambisonics, "_probe_cache", {})
    monkeypatch.setattr(ambisonics, "_can_encode", lambda e, ch: True)
    assert ambisonics.choose_encoder(4).name == "libfdk_aac"
    assert ambisonics.choose_encoder(4).note == "", \
        "the preferred choice needs no explaining"


def test_nothing_available_is_reported_rather_than_guessed(monkeypatch):
    monkeypatch.setattr(ambisonics, "_probe_cache", {})
    monkeypatch.setattr(ambisonics, "_can_encode", lambda e, ch: False)
    assert ambisonics.choose_encoder(4) is None


def test_an_explicit_choice_is_honoured():
    """"Can my headset play Opus" is not a question a probe can settle, so
    the user gets to decide."""
    assert ambisonics.choose_encoder(4, "pcm_s24le").name == "pcm_s24le"
    assert ambisonics.choose_encoder(16, "libopus").name == "libopus"


def test_an_impossible_explicit_choice_fails_before_the_render():
    """Honouring it blindly would kill the run at frame one, an hour of
    rendering after the mistake was made. The probe exercises the real chain,
    so a failure here is a failure there."""
    with pytest.raises(ValueError, match="cannot write 16 channels"):
        ambisonics.choose_encoder(16, "pcm_s24le")


def test_that_refusal_says_which_way_out():
    with pytest.raises(ValueError) as e:
        ambisonics.choose_encoder(16, "pcm_s24le")
    assert "libopus" in str(e.value)
    assert "--ambisonic-codec auto" in str(e.value)


def test_an_unknown_codec_is_refused_by_name():
    with pytest.raises(ValueError, match="Unknown ambisonic codec"):
        ambisonics.choose_encoder(4, "mp3")
    with pytest.raises(ValueError, match="libfdk_aac"):
        ambisonics.choose_encoder(4, "aac")     # names what is on offer


def test_the_fallbacks_say_why_they_are_not_the_first_choice():
    """A silent codec substitution in a master is not acceptable."""
    for name in ("libopus", "pcm_s24le"):
        assert _candidate(name, 4).note, name
    assert "untested" in _candidate("libopus", 4).note
    assert _candidate("pcm_s24le", 4).lossless


# --------------------------------------------- what actually reaches the file

@pytest.mark.parametrize("codec", ["libopus", "pcm_s24le"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_channel_order_survives_the_whole_encode(codec, order, tmp_path: Path):
    """ACN order in, ACN order out. A remap here would put every sound
    somewhere plausible and wrong, which is the hardest audio bug to notice.

    Only the codecs this build has; libfdk_aac is covered by the priority
    tests above and cannot be exercised where it is not installed.
    """
    n = ambisonics.channels_for_order(order)
    encoder = _candidate(codec, n)
    if not ambisonics._can_encode(encoder, n):
        pytest.skip(f"{codec} cannot write {n} channels in this build")

    t = np.arange(19200) / RATE
    data = np.stack([0.2 * np.sin(2 * np.pi * (400 + 37 * i) * t + i)
                     * (1 + 0.5 * np.sin(2 * np.pi * (3 + i) * t))
                     for i in range(n)], axis=1)
    identity = "|".join(f"c{i}=c{i}" for i in range(n))
    chain = f"pan={n}C|{identity}"
    layout = ambisonics._AAC_LAYOUT.get(n)
    if layout:
        chain += f",aformat=channel_layouts={layout}"
    out = _through_encoder(tmp_path, data, chain, list(encoder.args))
    assert _channel_mapping(data, out) == list(range(n))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_real_chain_encodes_and_keeps_the_bearing(order, tmp_path: Path):
    """`audio_filter` plus whatever `choose_encoder` picked, exactly as the
    pipeline uses them, through a real codec and back."""
    n = ambisonics.channels_for_order(order)
    encoder = ambisonics.choose_encoder(n)
    assert encoder is not None, "no usable ambisonic encoder in this build"
    out = _through_encoder(tmp_path, encode_source(-60.0, order),
                           ambisonics.audio_filter(order, 60.0),
                           list(encoder.args))
    assert wrap(bearing(out)) == pytest.approx(0.0, abs=1.0)


def test_pcm_really_is_lossless(tmp_path: Path):
    """The reason it is worth offering at all: rotating costs nothing. The
    audio is a rounding error beside an 8K picture, so paying in bytes to pay
    nothing in quality is a reasonable trade for a master."""
    data = encode_source(25.0, 1)
    encoder = ambisonics.choose_encoder(4, "pcm_s24le")
    out = _through_encoder(tmp_path, data,
                           ambisonics.audio_filter(1, 0.0), list(encoder.args))
    n = min(len(data), len(out))
    # 24-bit: a sample is within one part in 2^23 of where it started.
    assert np.abs(out[:n] - data[:n]).max() < 2 ** -22


@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_channel_count_survives_into_the_sample_entry(order, tmp_path):
    """SA3D describes the track by reading `channelcount` back out of the
    audio sample entry, so a codec that wrote the wrong number there would
    produce a file claiming an order it does not have."""
    from stereo360 import spherical

    channels = ambisonics.channels_for_order(order)
    encoder = ambisonics.choose_encoder(channels)
    out = str(tmp_path / "a.mp4")
    layout = ambisonics._AAC_LAYOUT.get(channels, f"{channels}C")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"anullsrc=cl={layout}:r=48000:d=0.3",
         *encoder.args, out], check=True, capture_output=True)

    data = Path(out).read_bytes()
    entry = max(data.find(tag) for tag in (b"mp4a", b"Opus", b"ipcm"))
    assert entry > 0, "no recognised audio sample entry"
    assert spherical._audio_channels(data, entry - 4) == channels
    assert struct.calcsize(">H") == 2        # the field the reader assumes


def test_the_probe_runs_the_rotation_too_not_just_the_codec():
    """`pcm_s24le` encodes 16 channels of silence from anullsrc without
    complaint and then refuses the identical silence with the pan filter in
    front of it, because the MP4 muxer will not take PCM with an unnamed
    layout. A probe that skips the filter answers a question nobody asked."""
    assert ambisonics._can_encode(_candidate("pcm_s24le", 4), 4), \
        "four channels have a named layout, so PCM is fine"
    assert not ambisonics._can_encode(_candidate("pcm_s24le", 16), 16), \
        "sixteen do not, and the muxer refuses"


def test_auto_therefore_avoids_pcm_at_higher_orders(monkeypatch):
    """Which is the point of probing rather than tabulating: the same codec
    is usable at one order and not at another."""
    monkeypatch.setattr(ambisonics, "_probe_cache", {})
    chosen = ambisonics.choose_encoder(16)
    assert chosen is None or chosen.name != "pcm_s24le"


def test_the_notes_name_flag_values_that_exist():
    """A note telling someone to pass `--ambisonic-codec pcm` when the choice
    is spelled `pcm_s24le` sends them to an argparse error."""
    valid = {e.name for e in ambisonics._candidates(4)} | {"auto"}
    for name in valid - {"auto"}:
        note = _candidate(name, 4).note
        for word in note.replace(".", " ").replace(";", " ").split():
            if word.startswith("--ambisonic-codec"):
                continue
        if "--ambisonic-codec" in note:
            after = note.split("--ambisonic-codec", 1)[1].split()[0]
            assert after.strip(".,;") in valid, f"{name}: {after}"


def test_the_opus_note_only_offers_pcm_where_pcm_works():
    """Telling someone at third order to use PCM instead sends them to an
    error, since MP4 will not carry 16 channels of it."""
    assert "pcm_s24le" in _candidate("libopus", 4).note
    for channels in (9, 16):
        note = _candidate("libopus", channels).note
        assert "pcm_s24le" not in note, note
        assert "only choice" in note


# ------------------------------------------------- how Opus labels the track

def _dops(path: Path) -> dict:
    """The OpusSpecificBox, which is MP4's stand-in for Ogg's OpusHead."""
    data = path.read_bytes()
    i = data.find(b"dOps")
    assert i > 0, "no dOps box"
    p = i + 4
    box = {
        "version": data[p],
        "channels": data[p + 1],
        "pre_skip": struct.unpack(">H", data[p + 2:p + 4])[0],
        "rate": struct.unpack(">I", data[p + 4:p + 8])[0],
        "gain": struct.unpack(">h", data[p + 8:p + 10])[0],
        "family": data[p + 10],
    }
    if box["family"] != 0:
        box["streams"] = data[p + 11]
        box["coupled"] = data[p + 12]
        box["mapping"] = list(data[p + 13:p + 13 + box["channels"]])
    return box


@pytest.mark.parametrize("order", [1, 2, 3])
def test_opus_declares_itself_ambisonic(order, tmp_path: Path):
    """Family 2 is RFC 8486's ambisonic mapping: ACN order, SN3D, (n+1)^2
    channels -- a description of ambiX so exact it could have been written
    for it.

    RFC 8486 signals this in Ogg's OpusHead and this writes MP4, so the
    obvious worry is that it does not carry. It does: the family is a
    property of the Opus stream, and MP4's dOps box mirrors OpusHead field
    for field. This reads the box back to prove it.
    """
    channels = ambisonics.channels_for_order(order)
    out = tmp_path / "a.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"anullsrc=cl={channels}C:r=48000:d=0.3",
         "-filter:a", ambisonics.audio_filter(order, 30.0),
         *_candidate("libopus", channels).args, str(out)],
        check=True, capture_output=True)

    box = _dops(out)
    assert box["family"] == 2, "not labelled as ambisonics"
    assert box["channels"] == channels
    assert box["mapping"] == list(range(channels)), \
        "the mapping table must not permute ACN order"
    assert box["coupled"] == 0, "ambisonic channels are not a stereo pair"
    assert box["streams"] == channels


def test_family_255_would_say_nothing_about_what_the_channels_are():
    """Why 2 and not the obvious 255. Both encode identically -- same streams,
    same coupling, same order, same size -- but 255 means "discrete channels,
    no meaning", leaving Google's SA3D box as the only thing in the file that
    calls this a soundfield."""
    assert ambisonics._OPUS_MAPPING_FAMILY == 2
    args = _candidate("libopus", 4).args
    assert "255" not in args
    assert args[args.index("-mapping_family") + 1] == "2"


# ---------------------------------- why the rotation cannot be metadata alone

def test_sa3d_has_nowhere_to_put_a_rotation():
    """The recurring hope is that a soundfield could be turned by a tag rather
    than by touching the samples. SA3D is the box that describes an ambisonic
    track and it carries no such field: version, type, order, ordering,
    normalisation, channel count, and a channel *map*.

    The map is an index permutation, and that is not enough even for a right
    angle -- a 90-degree yaw needs entries of -1, and 45 needs +-0.707. Only a
    yaw of zero is a permutation.
    """
    from stereo360 import spherical

    box = spherical._sa3d(4, 1)
    # 8 header + 12 fixed fields + 4 channels x 4 bytes. Nothing spare.
    assert len(box) == 8 + 12 + 4 * 4

    for yaw in (45, 90, 180):
        entries = {round(v, 6)
                   for row in ambisonics.yaw_matrix(1, yaw) for v in row}
        assert not entries <= {0.0, 1.0}, \
            f"yaw {yaw} would be expressible as a permutation"
    identity = {round(v, 6)
                for row in ambisonics.yaw_matrix(1, 0) for v in row}
    assert identity <= {0.0, 1.0}, "the do-nothing case, for contrast"


def test_the_projection_pose_is_left_at_zero():
    """`prhd` is the one rotation field in the metadata, and it turns the
    *picture*: it is in the video sample entry and describes where the
    projection sits. Declaring a pose instead of rotating the audio would put
    a VR180 file's content beside the viewer rather than in front.

    Zero is also the honest value here -- the crop has already put the chosen
    direction at the front and the soundfield has been turned to match.
    """
    from stereo360 import spherical

    body = spherical._sv3d(180.0)
    at = body.find(b"prhd")
    yaw, pitch, roll = struct.unpack(">iii", body[at + 8:at + 20])
    assert (yaw, pitch, roll) == (0, 0, 0)
