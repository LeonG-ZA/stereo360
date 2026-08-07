"""Turning an ambiX soundfield to follow where the VR180 field points.

The yaw control is what makes this necessary. Crop the sphere somewhere other
than the source's forward direction and every sound sits at the old bearing
while the picture has moved -- and in a 180 field that is worse than in 360,
because a source that ought to be visible in front can land behind the viewer,
in the hemisphere that no longer exists.

## Why a yaw is the easy rotation

A general re-orientation of an ambisonic signal needs Wigner-D matrices, and
they are a nuisance above first order. A rotation about the *vertical* axis is
not: in real spherical harmonics it never mixes one degree with another and
never touches the m = 0 channels. Each (+m, -m) pair simply turns by m times
the angle.

The encoding of a source at azimuth `phi` puts `cos(m.phi)` in channel (l, +m)
and `sin(m.phi)` in channel (l, -m), sharing everything else. Moving the source
to `phi + a` is then the angle-sum identity and nothing more:

    out[l, +m] = cos(m.a) . in[l, +m] - sin(m.a) . in[l, -m]
    out[l, -m] = sin(m.a) . in[l, +m] + cos(m.a) . in[l, -m]

Two consequences worth stating, because both remove a class of bug:

* It is **exact**. No approximation, no interpolation, no order limit -- the
  same two lines give third order as give first.
* It is **independent of normalisation**. Both channels of a pair carry the
  same factor, so SN3D and N3D rotate identically and mistaking one for the
  other cannot produce a wrong bearing here.

## Conventions, which are the part that bites

ambiX is ACN ordering with SN3D normalisation. ACN packs (l, m) at index
l^2 + l + m, so first order runs W, Y, Z, X -- *not* W, X, Y, Z.

Ambisonic azimuth is measured anticlockwise from the front, so a sound to the
listener's right is at a *negative* azimuth. `yaw` in this tool is longitude,
positive to the right. The two run in opposite directions, and the rotation
that reconciles them turns out to be `+yaw` regardless: pointing the field
`yaw` to the right means the sound that was at azimuth `-yaw` must arrive at
the front, and `-yaw + yaw = 0`.

Getting that sign backwards puts every source at twice the error rather than
none, and it is inaudible as anything but "the mix is wrong", so
`tests/test_ambisonics.py` pins it against a synthesised source.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from typing import List, NamedTuple, Optional

from .ffmpeg_io import NO_CONSOLE_WINDOW

#: Above third order the channel count outruns anything a consumer file
#: carries, and SA3D would have to describe it too. Not a maths limit.
MAX_ORDER = 3

#: What ffmpeg is told the pan produces. The capital is load-bearing: `Nc`
#: asks for the *default* layout of N channels, `NC` for an unknown one.
#:
#: `NC` for two reasons. There is no default layout for 9 channels, so `9c`
#: does not even parse and second order would simply never run. And an unknown
#: layout is what ambisonics honestly is -- W/Y/Z/X are not speaker positions,
#: and a name is an invitation to remap. That is not theoretical: with
#: `pan=quad` the round trip comes back carrying inputs 0, 1, 3, 3 -- one
#: channel lost, another duplicated -- somewhere between the filter and the
#: file.
_LAYOUT = "{n}C"

#: AAC will not take an unknown layout -- "Unsupported channel layout
#: '4 channels'" -- so the rotation is relabelled on the way out. Measured
#: identity end to end: channel i in is channel i out. Naming it 4.0 also
#: matches what Google's own spatial-media files report.
_AAC_LAYOUT = {4: "4.0"}


def order_for_channels(channels: Optional[int]) -> Optional[int]:
    """The ambisonic order a channel count describes, or None if it is not one.

    ambiX is (order + 1)^2 channels: 4, 9, 16 for first, second, third. A
    count that is not a perfect square is not ambiX, whatever it was tagged.
    """
    if not channels or channels < 4:
        return None
    order = int(round(math.sqrt(channels))) - 1
    if (order + 1) ** 2 != channels or not 1 <= order <= MAX_ORDER:
        return None
    return order


def channels_for_order(order: int) -> int:
    return (order + 1) ** 2


def yaw_matrix(order: int, yaw_degrees: float) -> List[List[float]]:
    """The channel mixing matrix for turning the field `yaw_degrees` right.

    Row i is how output channel i is made from the inputs, so
    `out[i] = sum(matrix[i][j] * in[j])`.
    """
    if not 1 <= order <= MAX_ORDER:
        raise ValueError(f"Ambisonic order {order} is outside 1..{MAX_ORDER}")

    n = channels_for_order(order)
    matrix = [[0.0] * n for _ in range(n)]
    alpha = math.radians(yaw_degrees)

    for degree in range(order + 1):
        base = degree * degree + degree            # where m = 0 sits
        matrix[base][base] = 1.0                   # never moves
        for m in range(1, degree + 1):
            plus, minus = base + m, base - m
            c, s = math.cos(m * alpha), math.sin(m * alpha)
            matrix[plus][plus] = c
            matrix[plus][minus] = -s
            matrix[minus][plus] = s
            matrix[minus][minus] = c
    return matrix


#: The most ffmpeg's native AAC encoder accepts: it caps a frame at 24576
#: bits for 4 channels, which is 6144 per channel, which at 48 kHz and 1024
#: samples per frame is exactly 288 kbit/s. Ask for 290 and it says "Too many
#: bits" and clamps. Used because it is the ceiling, not because the ceiling
#: is needed -- the encoder is the weak link and this is what limits the
#: damage. libfdk_aac is comfortably within its own limits here too.
_AAC_KBPS_PER_CHANNEL = 288

#: Opus is efficient enough that this is already far beyond transparent, and
#: the spatial information in ambisonics lives in the *differences* between
#: channels, which is what a codec discards first.
_OPUS_KBPS_PER_CHANNEL = 128


class Encoder(NamedTuple):
    """One way of writing rotated ambiX back into the file."""

    name: str
    args: List[str]
    lossless: bool
    #: What the user should know about this choice, or "" when it is the
    #: uncontroversial one.
    note: str
    #: True when the note is bad news rather than context -- a quality loss,
    #: or a codec measured not to play. Those are reported as warnings, not
    #: buried in an info line nobody reads.
    warn: bool = False


#: Opus channel mapping family. 2 is the ambisonic one from RFC 8486: ACN
#: order, SN3D normalisation, (n+1)^2 channels -- a description of ambiX so
#: exact it could have been written for this.
#:
#: RFC 8486 signals it in Ogg's `OpusHead`, and this writes MP4, so the
#: obvious worry is that it does not carry over. It does. The family is a
#: property of the Opus stream, not of Ogg, and MP4's `dOps` box
#: (OpusSpecificBox) mirrors OpusHead field for field -- version, channel
#: count, pre-skip, sample rate, gain, **ChannelMappingFamily**, then the
#: mapping table. Verified by reading the box back: family 2, N streams,
#: 0 coupled, identity mapping, at every order.
#:
#: Family 2 and family 255 encode to byte-identical structure here -- same
#: stream count, same coupling, same size, same channel order -- so the
#: choice costs nothing and buys self-description. With 255 ("discrete
#: channels, no meaning") the only thing in the file calling this a
#: soundfield is Google's SA3D box; with 2 the audio stream says so itself.
#:
#: The one caveat worth knowing: the Opus-in-ISOBMFF encapsulation spec
#: predates RFC 8486 and normatively references RFC 7845, which defines only
#: families 0 and 1. The field is a passthrough byte and ffmpeg writes it
#: without complaint, but a player is within its rights to reject a family it
#: does not recognise, where 255 invites it to hand over the channels
#: regardless. Untested on a headset either way -- see plans/vr180.md.
_OPUS_MAPPING_FAMILY = 2


def _lossy(name: str, channels: int, kbps_per_channel: int,
           extra: List[str] = ()) -> List[str]:
    return ["-c:a", name, *extra,
            "-b:a", f"{channels * kbps_per_channel}k"]


#: Tried in order. Which one plays is not a matter of opinion -- measured on
#: a Quest 3's native player and on VLC 3.0.21, same 20 s clip, only the audio
#: codec differing:
#:
#:                        Quest 3      VLC
#:   AAC 4ch              plays        plays
#:   PCM 24-bit 4ch       silent       plays
#:   Opus family 2        silent       silent
#:   Opus family 255      silent       silent
#:
#: So AAC is not merely the conventional choice, it is the only one a headset
#: will play, and Opus fails whichever mapping family it declares -- the
#: family was not the problem, Opus-in-MP4 was.
#:
#: That forces an uncomfortable order. ffmpeg's own `aac` is materially worse
#: than libfdk_aac at the same bitrate and would not be here on quality
#: grounds; it is here because a file nobody can hear is worse than one that
#: sounds a little worse. It is used at the highest bitrate it accepts, and it
#: says what it is.
#:
#: PCM and Opus stay available explicitly. PCM is lossless and plays on the
#: desktop, which is the right choice for an archival master that is not going
#: to a headset; Opus is the only thing that can carry 9 or 16 channels at all.
def _candidates(channels: int) -> List[Encoder]:
    fdk_note = ""
    aac_note = (
        "Using ffmpeg's built-in AAC encoder, which is audibly worse than "
        "libfdk_aac at the same bitrate -- this build of ffmpeg does not "
        "include libfdk_aac, and installing one that does is the only way to "
        "avoid the loss. Encoded at "
        f"{_AAC_KBPS_PER_CHANNEL} kbit/s per channel, the most it accepts, to "
        "limit the damage. Chosen anyway because it is the only codec a Quest "
        "3 plays: Opus is silent there in both mapping families, and PCM "
        "plays only on the desktop.")
    return [
        # Best AAC there is, and AAC is what plays. Absent from most Windows
        # builds -- its licence is not GPL-compatible, so gyan.dev and friends
        # leave it out.
        Encoder("libfdk_aac",
                _lossy("libfdk_aac", channels, _AAC_KBPS_PER_CHANNEL),
                False, fdk_note),
        # Only offered where AAC can carry the channel count at all: the
        # native encoder refuses 9 and mistakes 16 for 9.1.6.
        Encoder("aac", _lossy("aac", channels, _AAC_KBPS_PER_CHANNEL),
                False, aac_note, warn=True),
        # Lossless, and it does play -- just not on a headset.
        Encoder("pcm_s24le", ["-c:a", "pcm_s24le"], True,
                "24-bit PCM: nothing is re-compressed, so the rotation costs "
                "no quality at all. It plays on the desktop but NOT on a "
                "Quest 3, so this is for masters and for editing, not for "
                "sideloading."),
        # Takes any channel count, which above first order is the whole point,
        # since nothing else can. See _OPUS_MAPPING_FAMILY.
        Encoder("libopus",
                _lossy("libopus", channels, _OPUS_KBPS_PER_CHANNEL,
                       ["-mapping_family", str(_OPUS_MAPPING_FAMILY)]),
                False,
                "Opus was silent on a Quest 3 in testing, in both mapping "
                "families, and in VLC. At this order it is nonetheless the "
                "only encoder that can carry "
                f"{channels} channels into an MP4."
                if channels not in _AAC_LAYOUT else
                "Opus was silent on a Quest 3 in testing, in both mapping "
                "families, and in VLC. Kept only for players known to handle "
                "it; --ambisonic-codec auto will not choose it here.",
                warn=True),
    ]


def choose_encoder(channels: int,
                   requested: str = "auto") -> Optional[Encoder]:
    """The best available way to write `channels` back, or None if there is
    no way at all.

    `requested` names one explicitly; "auto" walks the list above and takes
    the first this ffmpeg actually has. Availability is measured by encoding
    real silence, not by reading `-encoders`: a build can list a codec it
    cannot run, and finding that out during a render wastes the render.
    """
    if channels <= 0:
        return None
    options = _candidates(channels)
    if requested != "auto":
        options = [e for e in options if e.name == requested]
        if not options:
            raise ValueError(
                f"Unknown ambisonic codec {requested!r}; expected 'auto' or "
                f"one of {', '.join(e.name for e in _candidates(4))}")
        # Probed like any other. The probe now runs the real chain, so there
        # is no case where it fails and the render would have worked -- and
        # finding out at frame one costs the whole render.
        if not _can_encode(options[0], channels):
            raise ValueError(
                f"--ambisonic-codec {requested} cannot write {channels} "
                f"channels here. "
                + (f"MP4 will not carry PCM with an unnamed layout, and there "
                   f"is no standard name for {channels} channels; libopus is "
                   f"the only option at this order."
                   if requested == "pcm_s24le" else
                   f"This build of ffmpeg cannot run it.")
                + " Use --ambisonic-codec auto to take the best available.")
        return options[0]
    for option in options:
        if _can_encode(option, channels):
            return option
    return None


#: Probing costs a subprocess, and the answer cannot change mid-run.
_probe_cache: dict = {}


def _can_encode(encoder: Encoder, channels: int) -> bool:
    """Whether this encoder can write the real chain, measured by doing it.

    The rotation filter is part of the probe, not just the codec. Without it
    the answer is wrong in the worst direction: `pcm_s24le` at 16 channels
    encodes silence from `anullsrc` quite happily and then refuses the same
    silence once the pan filter is in front of it, because the MP4 muxer will
    not take PCM with an unnamed layout. A probe that does not exercise the
    real path just produces a confident wrong answer later.
    """
    order = order_for_channels(channels)
    if order is None:
        return False
    key = (encoder.name, channels)
    if key not in _probe_cache:
        layout = _AAC_LAYOUT.get(channels, f"{channels}C")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "probe.mp4")
            result = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                 "-i", f"anullsrc=cl={layout}:r=48000:d=0.2",
                 # A real angle: an identity mix could be optimised away.
                 "-filter:a", audio_filter(order, 30.0),
                 *encoder.args, out],
                capture_output=True, **NO_CONSOLE_WINDOW)
            _probe_cache[key] = (result.returncode == 0
                                 and os.path.isfile(out)
                                 and os.path.getsize(out) > 0)
    return _probe_cache[key]


def _term(gain: float, channel: int) -> Optional[str]:
    # Rounded before the test, so a coefficient that prints as 0 is dropped
    # rather than written out as a term that does nothing.
    value = round(gain, 6)
    if value == 0.0:
        return None
    return f"{value:+.6f}*c{channel}"


def pan_filter(order: int, yaw_degrees: float) -> str:
    """`yaw_matrix` as an ffmpeg `pan` expression.

    `pan` is a plain linear mix of input channels, which is exactly what this
    rotation is, so no filter needs writing and nothing has to be decoded into
    Python. The `=` form is deliberate: `<` would renormalise each row to sum
    to 1 and quietly destroy the rotation.
    """
    matrix = yaw_matrix(order, yaw_degrees)
    n = len(matrix)
    parts = [_LAYOUT.format(n=n)]
    for out, row in enumerate(matrix):
        terms = [t for t in (_term(g, j) for j, g in enumerate(row)) if t]
        # A row cannot be empty -- every output takes at least one input --
        # but a silent channel would be a very quiet failure, so say so.
        if not terms:
            raise AssertionError(f"channel {out} would be silent")
        parts.append(f"c{out}=" + "".join(terms).lstrip("+"))
    return "pan=" + "|".join(parts)


def audio_filter(order: int, yaw_degrees: float) -> str:
    """The whole `-filter:a` chain: the rotation, plus any relabelling.

    Split from `pan_filter` because the relabelling is a fact about what
    follows, not about the rotation.

    Applied whenever a name exists for the channel count, rather than only for
    the encoders known to demand one. AAC refuses an unknown layout and so
    does the MP4 muxer for PCM; Opus, FLAC and ALAC do not care either way,
    and were each measured to pass the channels through in order with the
    label attached. One rule beats a table of exceptions.
    """
    chain = pan_filter(order, yaw_degrees)
    layout = _AAC_LAYOUT.get(channels_for_order(order))
    return chain + (f",aformat=channel_layouts={layout}" if layout else "")
