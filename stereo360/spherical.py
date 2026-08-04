"""Inject Google spatial-media metadata so players auto-detect the output.

Writes what the Spatial Media Metadata Injector writes, in the places it puts
it (github.com/google/spatial-media):

  * Spherical Video **V1** -- an RDF/XML blob in a `uuid` box, inside the video
    `trak`. This is the legacy format most desktop players still look for.
  * Spherical Video **V2** -- `st3d` (stereo mode) and `sv3d` (projection)
    boxes inside the video *sample entry*. Newer players and headsets prefer
    this, and some ignore V1 entirely.
  * **SA3D** -- ambisonic audio description inside the audio sample entry,
    written only when the source actually carries ambiX audio.

Placement is the whole game here. An earlier version of this module wrote the
V1 uuid box at the top level of `moov` rather than inside the video `trak`,
which is why players kept needing the metadata re-injected by hand: a
conforming parser looks for it in the track, does not find it, and treats the
file as flat 2D video. It also claimed in its docstring to implement V2 while
writing only V1.

Requirement: `moov` must come AFTER `mdat` (we encode with `-movflags
-faststart` for exactly this reason). Every box this module grows lives inside
`moov`, so no `mdat` chunk data moves and no stco/co64 offsets need rewriting.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SPHERICAL_UUID = bytes.fromhex("ffcc8263f8554a938814587a02521fdd")

SPHERICAL_XML_TEMPLATE = """<rdf:SphericalVideo
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">
<GSpherical:Spherical>true</GSpherical:Spherical>
<GSpherical:Stitched>true</GSpherical:Stitched>
<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
<GSpherical:StereoMode>{stereo_mode}</GSpherical:StereoMode>
<GSpherical:StitchingSoftware>stereo360</GSpherical:StitchingSoftware>
<GSpherical:SourceCount>1</GSpherical:SourceCount>
</rdf:SphericalVideo>"""

# st3d stereo_mode values from the Spherical Video V2 spec.
_STEREO_MODES = {"mono": 0, "top-bottom": 1, "left-right": 2}

# Boxes we descend into when walking the tree. Everything else is a leaf.
#
# `stsd` is deliberately absent, and so is every sample entry. Both do hold
# child boxes, but neither starts them at its own header: `stsd` is a full box
# with an entry count in front, and a sample entry opens with a block of fixed
# codec fields whose length depends on whether the track is visual or audio. A
# walk that does not know which handler it is under cannot know where those
# children begin, so reaching them is `_video_sample_entry_children`'s job.
_CONTAINERS = {"moov", "trak", "mdia", "minf", "stbl"}


def _box(name: str, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + name.encode("latin-1") + payload


def _full_box(name: str, payload: bytes, version: int = 0,
              flags: int = 0) -> bytes:
    return _box(name, struct.pack(">BBBB", version, flags >> 16 & 0xFF,
                                  flags >> 8 & 0xFF, flags & 0xFF) + payload)


def _st3d(stereo_mode: str) -> bytes:
    return _full_box("st3d", struct.pack(">B", _STEREO_MODES[stereo_mode]))


def _equi_bounds(horizontal_fov: float) -> Tuple[int, int, int, int]:
    """(top, bottom, left, right) for `equi`, in 0.32 fixed point.

    The bounds say how much of a *full* sphere is cropped away, so a 180-degree
    field crops a quarter from each side and 360 crops nothing. Only the
    horizontal is parameterised: a VR180 frame covers the full 180 degrees
    vertically, and nothing here needs a vertical crop.
    """
    if not 0.0 < horizontal_fov <= 360.0:
        raise ValueError(f"horizontal_fov must be in (0, 360], got "
                         f"{horizontal_fov}")
    side = (1.0 - horizontal_fov / 360.0) / 2.0
    fixed = int(round(side * 2 ** 32))
    return 0, 0, fixed, fixed


def _sv3d(horizontal_fov: float = 360.0) -> bytes:
    """sv3d > (svhd, proj > (prhd, equi)) describing the projection.

    The `prhd` pose is deliberately zero, and it is the only rotation field
    anywhere in this metadata, so it is worth saying why it stays that way.

    It is 16.16 fixed-point yaw/pitch/roll and it does round-trip -- write 45
    and ffprobe reads back `yaw: 45` in the Spherical Mapping side data. But
    it lives in the *video* sample entry and describes where the projection
    sits in the global frame. It moves the picture, not the sound.

    So it cannot stand in for rotating a VR180 file's audio. Using it that way
    would be the opposite manoeuvre -- leave the soundfield alone and declare
    the video turned -- which places the content off to one side of the viewer
    at startup instead of in front of them, which is the whole point of VR180.

    Zero is also the truthful value for what this tool produces: the crop has
    already put the chosen direction at the front, and `ambisonics` has turned
    the soundfield to match, so picture and sound agree that front is front.
    """
    svhd = _full_box("svhd", b"stereo360\x00")
    prhd = _full_box("prhd", struct.pack(">iii", 0, 0, 0))     # 16.16 pose
    equi = _full_box("equi", struct.pack(">IIII",
                                         *_equi_bounds(horizontal_fov)))
    return _box("sv3d", svhd + _box("proj", prhd + equi))


def _sa3d(num_channels: int, ambisonic_order: int) -> bytes:
    """SA3D for ambiX: periphonic, ACN ordering, SN3D normalization."""
    payload = struct.pack(
        ">BBIBBI",
        0,                 # version
        0,                 # ambisonic_type: 0 = periphonic
        ambisonic_order,
        0,                 # channel ordering: 0 = ACN
        0,                 # normalization: 0 = SN3D
        num_channels,
    )
    payload += b"".join(struct.pack(">I", i) for i in range(num_channels))
    return _box("SA3D", payload)


def _parse_boxes(data: bytes, start: int, end: int):
    """Yield (box_start, box_type, box_size, header_size) for sibling boxes."""
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        btype = data[pos + 4:pos + 8].decode("latin-1", "replace")
        header = 8
        if size == 1:                                   # 64-bit largesize
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            header = 16
        elif size == 0:                                 # extends to EOF
            size = end - pos
        if size < header or pos + size > end:
            raise ValueError(f"Corrupt MP4 box at offset {pos}")
        yield pos, btype, size, header
        pos += size


def _walk(data: bytes, start: int, end: int, path: Tuple[str, ...] = ()):
    """Depth-first walk yielding (start, type, size, header, path)."""
    for pos, btype, size, header in _parse_boxes(data, start, end):
        here = path + (btype,)
        yield pos, btype, size, header, path
        if btype in _CONTAINERS:
            yield from _walk(data, pos + header, pos + size, here)


def _find_moov(data: bytes) -> Optional[Tuple[int, int, int]]:
    for pos, btype, size, header in _parse_boxes(data, 0, len(data)):
        if btype == "moov":
            return pos, size, header
    return None


def _track_handler(data: bytes, trak_start: int, trak_size: int,
                   trak_header: int) -> Optional[str]:
    """'vide' / 'soun' for a trak, read from its mdia > hdlr box."""
    for pos, btype, size, header, _ in _walk(
            data, trak_start + trak_header, trak_start + trak_size):
        if btype == "hdlr":
            # full box (4) + pre_defined (4), then the 4-char handler type
            return data[pos + header + 8:pos + header + 12].decode(
                "latin-1", "replace")
    return None


def _sample_entry(data: bytes, trak_start: int, trak_size: int,
                  trak_header: int) -> Optional[Tuple[int, int]]:
    """(start, size) of the first sample entry inside this trak's stsd."""
    for pos, btype, size, header, _ in _walk(
            data, trak_start + trak_header, trak_start + trak_size):
        if btype == "stsd":
            # stsd is a full box: version/flags (4) + entry_count (4)
            first = pos + header + 8
            if first + 8 > pos + size:
                return None
            entry_size = struct.unpack(">I", data[first:first + 4])[0]
            return first, entry_size
    return None


def _audio_channels(data: bytes, entry_start: int) -> int:
    """channelcount from an AudioSampleEntry."""
    # 6 reserved + 2 data_reference_index + 8 reserved, then channelcount.
    return struct.unpack(">H", data[entry_start + 8 + 16:
                                    entry_start + 8 + 18])[0]


#: Where a VisualSampleEntry's first child box starts, measured from the start
#: of the entry: the box header, the SampleEntry base (6 reserved + 2
#: data_reference_index), then 70 bytes of visual fields running from width and
#: height through compressorname and depth (ISO/IEC 14496-12 12.1.3). Confirmed
#: against a real file rather than trusted: parsing from here yields
#: ['avcC', 'btrt', 'st3d', 'sv3d'].
_VISUAL_SAMPLE_ENTRY_FIELDS = 8 + 8 + 70


def _video_sample_entry_children(data: bytes, trak_start: int, trak_size: int,
                                 trak_header: int):
    """Yield the boxes inside a video trak's first sample entry.

    That is `avcC`/`hvcC` and, once this module has been here, `st3d` and
    `sv3d`. Nothing is yielded for a track that is not video, since only a
    VisualSampleEntry has the fixed-field preamble measured above.

    Separate from `_walk` on purpose. Teaching that walk to descend into
    sample entries would change what `inject_spherical_metadata` sees while it
    is deciding where to splice, and that is not a side effect worth risking
    for a predicate.
    """
    if _track_handler(data, trak_start, trak_size, trak_header) != "vide":
        return
    entry = _sample_entry(data, trak_start, trak_size, trak_header)
    if entry is None:
        return
    estart, esize = entry
    if esize < _VISUAL_SAMPLE_ENTRY_FIELDS:
        return
    try:
        yield from _parse_boxes(data, estart + _VISUAL_SAMPLE_ENTRY_FIELDS,
                                estart + esize)
    except ValueError:
        # An entry laid out differently from the spec is not something to
        # guess at. Yielding nothing makes an injection go ahead and fail
        # loudly, which beats silently skipping it.
        return


def _has_v2_boxes(data: bytes, trak_start: int, trak_size: int,
                  trak_header: int) -> bool:
    """Whether this trak's sample entry carries `st3d` or `sv3d`."""
    return any(btype in ("st3d", "sv3d") for _, btype, _, _
               in _video_sample_entry_children(data, trak_start, trak_size,
                                               trak_header))


#: Where an AudioSampleEntry's child boxes start, measured from the start of
#: the entry: the box header, the SampleEntry base (6 reserved + 2
#: data_reference_index), then 20 bytes of audio fields -- 8 reserved,
#: channelcount, samplesize, pre_defined, reserved, samplerate.
_AUDIO_SAMPLE_ENTRY_FIELDS = 8 + 8 + 20

#: SA3D's own vocabulary. Anything else is ambisonic but not what this tool
#: assumes, and guessing would be worse than declining to.
_ACN, _SN3D, _PERIPHONIC = 0, 0, 0


def _read_moov(path: str) -> Optional[Tuple[bytes, int]]:
    """(moov bytes, its offset) without reading the rest of the file.

    Worth the seeking. Camera files put `moov` at the end -- an Insta360 X5
    clip has it 1.2 GB in -- so a probe that wants one small box should not
    pull a gigabyte through memory to reach it. Reading the front of the file
    and giving up is worse still: that is how the SA3D box in exactly such a
    file got reported as absent.
    """
    total = os.path.getsize(path)
    with open(path, "rb") as f:
        pos = 0
        while pos + 8 <= total:
            f.seek(pos)
            head = f.read(16)
            if len(head) < 8:
                return None
            size = struct.unpack(">I", head[0:4])[0]
            btype = head[4:8].decode("latin-1", "replace")
            if size == 1:
                if len(head) < 16:
                    return None
                size = struct.unpack(">Q", head[8:16])[0]
            elif size == 0:
                size = total - pos
            if size < 8 or pos + size > total:
                return None
            if btype == "moov":
                f.seek(pos)
                return f.read(size), pos
            pos += size
    return None


def read_ambisonic_description(path: str) -> Optional[Dict[str, int]]:
    """What the file's own `SA3D` box says its audio is, or None.

    This is the authoritative answer to "is this ambisonics", and it is the
    one VLC uses -- it reports "Channels: Ambisonics" for a file ffprobe
    describes as plain `4.0`, because ffprobe does not surface SA3D at all.

    None means the file does not say, which is not the same as "no". Plenty of
    ambiX is delivered untagged, and the channel count is the only hint left
    in that case.

    Returns the fields rather than a bool so a caller can refuse what it
    cannot handle: `channel_ordering` and `normalization` are ACN and SN3D
    here, and a file declaring FuMa is genuinely ambisonic and genuinely not
    what the rotation maths assumes.
    """
    found = _read_moov(path)
    if found is None:
        return None
    moov, offset = found
    size = struct.unpack(">I", moov[0:4])[0]
    header = 16 if size == 1 else 8
    try:
        traks = list(_walk(moov, header, len(moov)))
    except ValueError:
        return None

    for pos, btype, bsize, bheader, _ in traks:
        if btype != "trak":
            continue
        if _track_handler(moov, pos, bsize, bheader) != "soun":
            continue
        entry = _sample_entry(moov, pos, bsize, bheader)
        if entry is None:
            continue
        estart, esize = entry
        if esize < _AUDIO_SAMPLE_ENTRY_FIELDS:
            continue
        try:
            children = list(_parse_boxes(
                moov, estart + _AUDIO_SAMPLE_ENTRY_FIELDS, estart + esize))
        except ValueError:
            continue
        for kpos, ktype, _, kheader in children:
            if ktype != "SA3D":
                continue
            p = kpos + kheader
            if p + 12 > len(moov):
                return None
            return {
                "version": moov[p],
                "ambisonic_type": moov[p + 1],
                "order": struct.unpack(">I", moov[p + 2:p + 6])[0],
                "channel_ordering": moov[p + 6],
                "normalization": moov[p + 7],
                "channels": struct.unpack(">I", moov[p + 8:p + 12])[0],
            }
    return None


def declares_ambix(path: str) -> bool:
    """Whether the file declares ambiX this tool can actually work with.

    ACN ordering and SN3D normalisation, periphonic, and a channel count that
    matches the order it claims. A file failing any of those may still be
    ambisonic; it is just not the thing `stereo360.ambisonics` rotates.
    """
    sa3d = read_ambisonic_description(path)
    if sa3d is None:
        return False
    return (sa3d["ambisonic_type"] == _PERIPHONIC
            and sa3d["channel_ordering"] == _ACN
            and sa3d["normalization"] == _SN3D
            and 1 <= sa3d["order"] <= 3
            and sa3d["channels"] == (sa3d["order"] + 1) ** 2)


def has_spherical_metadata(path: str) -> bool:
    """Whether the file already carries spherical metadata, V1 **or** V2.

    Both, and that is the whole point. This used to find only V1 -- the `uuid`
    box, which sits directly under `moov`. It tested for `st3d` and `sv3d` too,
    but `_walk` stops at `stbl` and those live one level deeper still, inside
    the video sample entry, so that branch could never match.

    It went unnoticed while every file carried V1 as well. VR180 output omits
    V1, because V1 cannot express a partial sphere, and the function then
    reported "no metadata" about a file carrying `st3d`, `sv3d` and `equi`.

    That matters because `inject_spherical_metadata` uses this as its "already
    tagged, do nothing" guard. A guard that never fires would let a second
    injection write a second set of boxes into the same file.
    """
    data = Path(path).read_bytes()
    moov = _find_moov(data)
    if moov is None:
        return False
    start, size, header = moov
    for pos, btype, bsize, bheader, _ in _walk(data, start + header,
                                               start + size):
        if btype == "uuid" and data[pos + bheader:pos + bheader + 16] == \
                SPHERICAL_UUID:
            return True
        if btype == "trak" and _has_v2_boxes(data, pos, bsize, bheader):
            return True
    return False


def _apply(data: bytes, inserts: List[Tuple[int, bytes]],
           grow: Dict[int, int]) -> bytes:
    """Splice `inserts` in and add `grow[start]` to each named box's size."""
    out = bytearray()
    cursor = 0
    for offset, payload in sorted(inserts, key=lambda kv: kv[0]):
        out += data[cursor:offset]
        out += payload
        cursor = offset
    out += data[cursor:]

    # Patch sizes last, working on the spliced buffer: box starts before the
    # first insertion keep their offsets, and every box we grow encloses its
    # insertions, so its own start is never moved by them.
    shift = 0
    ordered = sorted(inserts, key=lambda kv: kv[0])
    for start, extra in sorted(grow.items()):
        shift = sum(len(p) for off, p in ordered if off <= start)
        pos = start + shift
        size = struct.unpack(">I", out[pos:pos + 4])[0]
        if size == 1:
            raise ValueError("64-bit box sizes are not supported here")
        struct.pack_into(">I", out, pos, size + extra)
    return bytes(out)


def inject_spherical_metadata(path: str, stereo_mode: str = "top-bottom",
                              spatial_audio: bool = False,
                              horizontal_fov: float = 360.0) -> None:
    """Insert spherical (V1 + V2) and optional SA3D metadata, in place.

    stereo_mode:    "top-bottom" | "left-right" | "mono"
    spatial_audio:  describe the audio track as ambiX ACN/SN3D. Only valid if
                    the source audio really is ambisonic; the channel count
                    must be a perfect square (4, 9, 16 -> order 1, 2, 3).
    horizontal_fov: degrees of longitude the frame covers. 360 (the default) is
                    a full sphere; 180 is VR180, and writes `equi` bounds that
                    crop a quarter from each side.

    **V1 is written only for a full sphere.** Its `ProjectionType` says
    "equirectangular" and means the whole thing, so on a 180 file it would be a
    lie -- a V1-only reader would stretch half a sphere across a full one,
    which is precisely what VLC 3.0.21 does when it ignores the V2 bounds.
    Omitting it means such a reader sees flat 2D video instead: obviously
    wrong, rather than subtly wrong.

    V1 does have `CroppedArea*` / `FullPano*` fields that could describe a
    partial panorama, so this is a choice rather than a limitation. They are
    not used because their meaning for a side-by-side stereo frame -- per eye,
    or per packed frame? -- is not something to guess at when V2 already works
    on the target devices.
    """
    if stereo_mode not in _STEREO_MODES:
        raise ValueError(f"Unknown stereo_mode {stereo_mode!r}; "
                         f"expected one of {sorted(_STEREO_MODES)}")
    _equi_bounds(horizontal_fov)        # validate before touching the file
    if has_spherical_metadata(path):
        return

    data = Path(path).read_bytes()
    moov = _find_moov(data)
    if moov is None:
        raise ValueError("No 'moov' box found — is this a valid MP4?")
    moov_start, moov_size, moov_header = moov

    for pos, btype, _, _ in _parse_boxes(data, 0, len(data)):
        if btype == "mdat" and pos > moov_start:
            raise ValueError(
                "mdat box located after moov; injection would shift chunk "
                "data. Re-encode without faststart.")

    inserts: List[Tuple[int, bytes]] = []
    grow: Dict[int, int] = {}

    def add(offset: int, payload: bytes, ancestors: List[int]) -> None:
        inserts.append((offset, payload))
        for anc in ancestors:
            grow[anc] = grow.get(anc, 0) + len(payload)

    # Ancestry of each sample entry, so every enclosing box can be grown.
    def ancestry(trak_start: int, trak_size: int, trak_header: int,
                 want: int) -> List[int]:
        chain = [moov_start, trak_start]
        for pos, btype, size, header, _ in _walk(
                data, trak_start + trak_header, trak_start + trak_size):
            if btype in ("mdia", "minf", "stbl", "stsd") and \
                    pos < want < pos + size:
                chain.append(pos)
        return chain

    injected_audio = False
    for tpos, btype, tsize, theader in _parse_boxes(
            data, moov_start + moov_header, moov_start + moov_size):
        if btype != "trak":
            continue
        handler = _track_handler(data, tpos, tsize, theader)

        if handler == "vide":
            # V1: uuid box appended to the trak itself. Full sphere only --
            # see the note in this function's docstring.
            if horizontal_fov >= 360.0:
                xml = SPHERICAL_XML_TEMPLATE.format(
                    stereo_mode=stereo_mode).encode("utf-8")
                add(tpos + tsize, _box("uuid", SPHERICAL_UUID + xml),
                    [moov_start, tpos])

            # V2: st3d + sv3d appended to the video sample entry.
            entry = _sample_entry(data, tpos, tsize, theader)
            if entry is not None:
                estart, esize = entry
                chain = ancestry(tpos, tsize, theader, estart) + [estart]
                add(estart + esize,
                    _st3d(stereo_mode) + _sv3d(horizontal_fov), chain)

        elif handler == "soun" and spatial_audio:
            entry = _sample_entry(data, tpos, tsize, theader)
            if entry is None:
                continue
            estart, esize = entry
            channels = _audio_channels(data, estart)
            order = int(round(channels ** 0.5)) - 1
            if (order + 1) ** 2 != channels or order < 1:
                raise ValueError(
                    f"--spatial-audio needs an ambiX track, but the audio has "
                    f"{channels} channel(s). ambiX is first-order upward: 4, "
                    f"9 or 16 channels (order 1, 2, 3).")
            chain = ancestry(tpos, tsize, theader, estart) + [estart]
            add(estart + esize, _sa3d(channels, order), chain)
            injected_audio = True

    if spatial_audio and not injected_audio:
        raise ValueError(
            "--spatial-audio was requested but the output has no audio track. "
            "Ambisonic audio has to be present in the source and copied "
            "through for it to be described.")

    Path(path).write_bytes(_apply(data, inserts, grow))
