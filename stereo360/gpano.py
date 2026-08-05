"""Google Photo Sphere (GPano) XMP, so a photo is read as a sphere.

The still equivalent of `spherical.py`, and deliberately a separate module: a
photo carries its metadata as an **XMP packet in a JPEG APP1 segment**, which
shares no machinery at all with the `st3d`/`sv3d`/`SA3D` boxes an MP4 carries.

## What the spec has, and what it does not

GPano describes a *panorama*: its projection, and which part of a notional full
sphere the pixels cover. It defines **no stereo property of any kind** --
`GSpherical:StereoMode` belongs to the video spec and has no photo equivalent.
So this writer can say "equirectangular, and here is the field it covers", and
nothing about there being two eyes.

That turns out not to matter. Measured on a Quest 3, a stacked stereo frame is
read as stereo from **either** the presence of GPano **or** a filename carrying
the usual `_360_TB` tokens -- and a file with neither is not. Both signals are
worth having, and neither is required when the other is present.

## Why it describes one eye

GPano cannot describe a stacked pair, so it has to describe something else, and
the honest choice is the panorama one eye covers.

The device testing also showed the choice is free. Two files, identical but for
their GPano saying `7680x3840` and `7680x7680` about the same 7680x7680 image,
both displayed correctly -- so the reader is not doing arithmetic on those
fields to work out the layout. It marks the file as a panorama; the aspect
ratio does the rest.

Given that, describing one eye is simply the description that is true.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Optional

#: XMP's APP1 payload begins with this NUL-terminated namespace URI. A reader
#: uses it to tell an XMP APP1 from an Exif one, which shares the marker.
XMP_APP1_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"

_SOI = b"\xff\xd8"
_APP1 = b"\xff\xe1"

#: APP1 carries a 16-bit length including itself, so this is the ceiling on a
#: packet. Ours is about a kilobyte; the limit only matters to the format that
#: embeds a whole second JPEG, which this deliberately does not.
_MAX_APP1_PAYLOAD = 0xFFFF - 2

_PACKET = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"
    GPano:ProjectionType="equirectangular"
    GPano:UsePanoramaViewer="True"
    GPano:CroppedAreaImageWidthPixels="{crop_w}"
    GPano:CroppedAreaImageHeightPixels="{crop_h}"
    GPano:FullPanoWidthPixels="{full_w}"
    GPano:FullPanoHeightPixels="{full_h}"
    GPano:CroppedAreaLeftPixels="{left}"
    GPano:CroppedAreaTopPixels="{top}"
    GPano:StitchingSoftware="stereo360"/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def eye_geometry(width: int, height: int,
                 output_mode: str = "360") -> Dict[str, int]:
    """What one eye of a `width` x `height` stereo frame covers.

    360 stacks two full spheres, so an eye is the full width and half the
    height, cropping nothing.

    VR180 puts two half-spheres side by side, so an eye is half the width and
    covers 180 degrees of longitude. In GPano's terms that is a crop from the
    middle of a sphere twice as wide -- which is why `CroppedAreaLeftPixels`
    is a quarter of the frame rather than zero.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"a {width}x{height} image has no geometry")
    if output_mode == "vr180":
        eye_w, eye_h = width // 2, height
        return {"crop_w": eye_w, "crop_h": eye_h,
                "full_w": eye_w * 2, "full_h": eye_h,
                "left": eye_w // 2, "top": 0}
    eye_w, eye_h = width, height // 2
    return {"crop_w": eye_w, "crop_h": eye_h,
            "full_w": eye_w, "full_h": eye_h,
            "left": 0, "top": 0}


def build_packet(width: int, height: int,
                 output_mode: str = "360") -> bytes:
    """The XMP packet for a stereo frame of this size and layout."""
    packet = _PACKET.format(**eye_geometry(width, height, output_mode))
    data = packet.encode("utf-8")
    if len(data) + len(XMP_APP1_HEADER) > _MAX_APP1_PAYLOAD:
        raise ValueError("the XMP packet outgrew a single APP1 segment")
    return data


def _find_xmp_app1(data: bytes) -> Optional[tuple]:
    """(start, end) of an existing XMP APP1 segment, or None.

    Walks the marker chain rather than searching for the namespace string,
    which would also match the same bytes appearing inside compressed image
    data. That is not a hypothetical: the equivalent shortcut on MP4 found an
    `SA3D` 187 MB into a file, inside `mdat`, with nonsense in every field.
    """
    pos = 2                                     # past SOI
    while pos + 4 <= len(data):
        if data[pos:pos + 1] != b"\xff":
            return None                         # not a marker: give up
        marker = data[pos:pos + 2]
        if marker in (b"\xff\xd8", b"\xff\xd9") or marker == b"\xff\xda":
            return None                         # SOI/EOI/start of scan
        size = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        end = pos + 2 + size
        if marker == _APP1 and data[pos + 4:pos + 4 + len(XMP_APP1_HEADER)] \
                == XMP_APP1_HEADER:
            return pos, end
        pos = end
    return None


def inject_into_jpeg(path: str, width: int, height: int,
                     output_mode: str = "360") -> None:
    """Write GPano into a JPEG, replacing any packet already there.

    Replacing rather than appending, so running this twice leaves one packet.
    Nothing in this tool writes a JPEG that already has XMP, so it is not a
    live bug -- but the equivalent guard on the MP4 side was quietly broken
    for months, and the cost of getting it right here is ten lines.
    """
    data = Path(path).read_bytes()
    if data[:2] != _SOI:
        raise ValueError(f"{path!r} is not a JPEG, so it cannot carry XMP")

    payload = XMP_APP1_HEADER + build_packet(width, height, output_mode)
    segment = _APP1 + struct.pack(">H", len(payload) + 2) + payload

    existing = _find_xmp_app1(data)
    if existing is None:
        # Straight after SOI: readers are entitled to stop looking once the
        # image data starts, and some stop sooner than that.
        out = data[:2] + segment + data[2:]
    else:
        start, end = existing
        out = data[:start] + segment + data[end:]
    Path(path).write_bytes(out)


def read_projection(path: str) -> Optional[Dict[str, str]]:
    """GPano's fields as written in the file, or None if it carries none.

    Only for checking our own output -- it reads the attribute form this
    module writes, not XMP in general.
    """
    import re

    data = Path(path).read_bytes()
    if data[:2] != _SOI:
        return None
    found = _find_xmp_app1(data)
    if found is None:
        return None
    start, end = found
    body = data[start:end]
    return {m.group(1).decode(): m.group(2).decode()
            for m in re.finditer(rb'GPano:(\w+)="([^"]*)"', body)}
