"""How a pre-pass writes its working file.

Every pass that runs before the stereo conversion -- Topaz, RIFE, the shader
-- writes one video that the converter then reads and the run then deletes.
Two things about that file matter, and both were wrong:

**It was lossy.** All three preferred `hevc_nvenc` at qp 16, which measures
47.6 dB against its input at 8K -- high enough not to see, but the converter
estimates depth from this file, so any smoothing sits underneath the depth
rather than merely in the picture. Measured on twelve 8K frames:

    encoder                 s/frame   MB/frame   PSNR
    hevc_nvenc qp16            0.11        1.9   47.6
    hevc_nvenc qp10            0.11        3.4   51.8
    hevc_nvenc lossless        0.12       10.0    inf
    ffv1 level 3               0.09       10.8    inf
    libx264 qp0 ultrafast      0.08       10.6    inf

Lossless costs 0.01 s a frame. It costs five times the disk, which is the
only reason not to take it every time -- at 8K a minute of interpolated
footage is about 36 GB. So lossless is the default and near-lossless is what
happens when the file would not comfortably fit; see `wants_lossless`.

**It flattened the pixel format.** Every writer forced `yuv420p`, so a 10-bit
source became 8-bit and a 4:4:4 one became 4:2:0, before the depth model ever
saw it -- while the pipeline separately offers `--bitdepth 10` and warns when
a source is deeper than 8 bits. The source's own format is kept now, as far
as the chosen encoder can carry it.

What is *not* kept is RGB. Topaz hands back RGB, and the converter's encoder
then refuses the colour tags that come with it -- `Error setting option
colorspace to value gbr`. Anything not on the YUV list becomes yuv420p, which
is what the old blanket rule was really for.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from .ffmpeg_io import NO_CONSOLE_WINDOW

#: Bytes per pixel a lossless intermediate takes, from the table above:
#: 10 MB for a 29.5 Mpx frame. Only used to guess a file size before writing
#: it, so it is deliberately a little generous.
BYTES_PER_PIXEL = 0.36

#: Use lossless while the guess fits in this share of what the drive has
#: free. A working file is not worth filling someone's disk over.
FREE_SHARE = 0.25

#: What counts as room when the length cannot be guessed at all.
UNKNOWN_NEEDS = 50 * 1024 ** 3

#: Formats worth carrying through, in the encoder's own spelling. Anything
#: else -- RGB especially -- becomes the first of these.
_KEEP = (
    "yuv420p", "yuv422p", "yuv444p",
    "yuv420p10le", "yuv422p10le", "yuv444p10le",
    "yuv420p12le", "yuv422p12le", "yuv444p12le",
    "p010le", "p210le", "p016le",
)

#: Lossless first, then near-lossless, each tried in order. NVENC leads the
#: lossless list because it is the one that does not want the CPU -- an
#: interpolator feeding this encoder competes with it for cores, and that was
#: worth fifteen times the frame time when it went wrong before -- and it
#: happens to make the smallest file of the three: 121 MB against ffv1's 130
#: for the same twelve 8K frames.
#:
#: Every "lossless" here was checked against its own source and came back
#: infinite. That check is easy to get wrong: comparing a short encode with a
#: longer source makes ffmpeg's psnr filter pair each frame with its
#: neighbour, which reported 56.7 dB for a codec that is bit exact and 32.5
#: for ffv1, which is lossless by construction. Normalise both timelines with
#: `setpts=N/FRAME_RATE/TB` before believing any number out of it.
_LOSSLESS = (
    ("hevc_nvenc", ("-tune", "lossless", "-preset", "p5")),
    ("ffv1", ("-level", "3",)),
    ("libx264", ("-qp", "0", "-preset", "ultrafast")),
)
_NEAR = (
    ("hevc_nvenc", ("-rc", "constqp", "-qp", "10", "-preset", "p5")),
    ("libx264", ("-qp", "10", "-preset", "veryfast")),
    ("ffv1", ("-level", "3",)),
)

_formats: Dict[Tuple[str, str], frozenset] = {}


#: The full-range spellings are the same layout with a range flag beside
#: them, and the flag travels on its own. Asking for the `j` name would push
#: the choice onto an encoder that lists it, for no gain.
_FULL_RANGE = {"yuvj420p": "yuv420p", "yuvj422p": "yuv422p",
               "yuvj444p": "yuv444p"}


def keep_format(pix_fmt: Optional[str]) -> str:
    """The pixel format a working file should aim for."""
    pix_fmt = _FULL_RANGE.get(pix_fmt, pix_fmt)
    return pix_fmt if pix_fmt in _KEEP else "yuv420p"


def supported(ffmpeg: str, encoder: str) -> frozenset:
    """Pixel formats `encoder` accepts in this ffmpeg, asked once."""
    key = (ffmpeg, encoder)
    if key in _formats:
        return _formats[key]
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-h", f"encoder={encoder}"],
                             capture_output=True, text=True, timeout=60,
                             errors="replace", **NO_CONSOLE_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    m = re.search(r"Supported pixel formats:\s*(.+)", out)
    got = frozenset(m.group(1).split()) if m else frozenset()
    _formats[key] = got
    return got


def wants_lossless(width: int, height: int, frames: Optional[int],
                   where: str) -> bool:
    """Whether a lossless working file for this job is a reasonable size.

    Lossless is the default because it costs almost no time. It costs disk,
    though, and at 8K that is about 10 MB a frame -- so it gives way when the
    guess does not fit in a quarter of the free space. When the length cannot
    be guessed, it asks for `UNKNOWN_NEEDS` of room instead.
    """
    try:
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(where))).free
    except OSError:
        return True             # unknowable; the encoder will say if it fails
    if not frames or not width or not height:
        return free >= UNKNOWN_NEEDS
    return width * height * frames * BYTES_PER_PIXEL <= free * FREE_SHARE


def encoder_args(ffmpeg: str = "ffmpeg", *, pix_fmt: Optional[str] = None,
                 lossless: bool = True) -> Tuple[List[str], str]:
    """(ffmpeg output options, a line saying what was chosen and why).

    The first encoder that this ffmpeg has *and* that can carry the wanted
    pixel format. Checked rather than assumed: hevc_nvenc takes 10-bit 4:2:0
    as p010le and does not take 4:2:2 at all, and a working file silently
    converted to something else is the thing this module exists to stop.
    """
    want = keep_format(pix_fmt)
    table = _LOSSLESS if lossless else _NEAR
    kind = "lossless" if lossless else "near-lossless"

    # Keeping the format comes before the order of the list: hevc_nvenc leads
    # it for speed, but it cannot do 4:2:2 and ffv1 can, and dropping a
    # source's chroma to keep the faster encoder is the trade this module
    # exists to refuse.
    for name, opts in table:
        formats = supported(ffmpeg, name)
        fmt = None
        if want in formats:
            fmt = want
        elif want == "yuv420p10le" and "p010le" in formats:
            fmt = "p010le"      # the same ten bits, packed the other way up
        if fmt:
            return (["-c:v", name, *opts, "-pix_fmt", fmt],
                    f"{name}, {kind}, {fmt}")

    # Nobody can carry it, so say what was given up rather than doing it
    # quietly.
    for name, opts in table:
        if "yuv420p" in supported(ffmpeg, name):
            return (["-c:v", name, *opts, "-pix_fmt", "yuv420p"],
                    f"{name}, {kind}, yuv420p (wanted {want})")
    return (["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p"],
            "ffv1, lossless, yuv420p")


def choose(ffmpeg: str = "ffmpeg", *, pix_fmt: Optional[str] = None,
           width: int = 0, height: int = 0, frames: Optional[int] = None,
           where: str = ".") -> Tuple[List[str], str]:
    """Everything above in one call: what to encode a working file with."""
    return encoder_args(ffmpeg, pix_fmt=pix_fmt,
                        lossless=wants_lossless(width, height, frames, where))
