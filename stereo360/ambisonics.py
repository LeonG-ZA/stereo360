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
from typing import List, Optional

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


#: Per-channel bitrate for the re-encode a rotation forces. Generous on
#: purpose: the spatial information in ambisonics lives in the *differences*
#: between channels, which is exactly what a codec throws away first.
_KBPS_PER_CHANNEL = 96


def encode_args(channels: int) -> Optional[List[str]]:
    """ffmpeg arguments to write `channels` of rotated ambiX back, or None.

    First order is AAC, which is what Google's spatial media spec uses and
    what every player that reads SA3D already expects.

    Above that, ffmpeg's native AAC encoder simply refuses -- "Unsupported
    channel layout" for 9 channels, and for 16 it guesses 9.1.6 and refuses
    that too. libopus takes any count with `-mapping_family 255`; the channel
    order survives (measured by cross-correlating every output channel against
    every input one) and `channelcount` lands correctly in the sample entry,
    so SA3D still describes the track truthfully.

    What is *not* established is whether a headset plays Opus-in-MP4
    ambisonics -- Google's spec is AAC. So the caller announces the swap
    rather than making it quietly.
    """
    if channels <= 0:
        return None
    rate = f"{channels * _KBPS_PER_CHANNEL}k"
    if channels <= 8:
        return ["-c:a", "aac", "-b:a", rate]
    return ["-c:a", "libopus", "-mapping_family", "255", "-b:a", rate]


def needs_opus(channels: int) -> bool:
    """Whether writing this many channels means leaving AAC behind."""
    return channels > 8


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

    Split from `pan_filter` because the relabelling is a fact about the
    encoder that follows, not about the rotation.
    """
    chain = pan_filter(order, yaw_degrees)
    layout = _AAC_LAYOUT.get(channels_for_order(order))
    if layout and not needs_opus(channels_for_order(order)):
        chain += f",aformat=channel_layouts={layout}"
    return chain
