"""Which video encoders can actually encode a given frame size, here, now.

Presence in `ffmpeg -encoders` proves nothing: this build lists NVENC, QSV, AMF
and VAAPI encoders regardless of what hardware is fitted. And availability is
resolution-dependent in a way that matters for this tool -- measured on real
hardware, `hevc_amf` encodes 3840x3840 happily and rejects 7680x7680, and
`h264_nvenc` stops at 4096x4096 whatever the driver. A top-bottom stereo frame
is twice the height of its source, so it lands on the wrong side of those
limits far more often than ordinary video does.

So each candidate is asked to encode one frame at the exact target size. That
costs about a second for one that works and less for one that does not, which
is cheap next to discovering it mid-render.

Hardware encoders are reported but not recommended: they are the speed option,
not the quality one. Measured at 8K, hevc_nvenc at cq19 runs at 0.28 s/frame
against libx265 slow crf13's 1.90 -- but they are not the same picture, and
for anything being kept the CPU encoders win on quality per bit.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import List, NamedTuple, Optional, Tuple

from .ffmpeg_io import NO_CONSOLE_WINDOW

#: name, is_hardware, vendor/summary
#:
#: This is also the order the dropdown shows, so it runs best quality first:
#: software before hardware, and H.265 before H.264 within each group, since
#: HEVC gives the better picture per bit at these sizes. Grouping by codec
#: rather than by vendor means the list reads as a quality ranking, which is
#: what someone scanning it is actually choosing between -- only one vendor's
#: entries can work on any given machine anyway.
#:
#: Ordering here is presentation only. What the tool *defaults* to is
#: `_DEFAULT_ORDER` below, deliberately kept separate: this list is about what
#: looks best, that one is about what is safe to pick unattended.
CANDIDATES: Tuple[Tuple[str, bool, str], ...] = (
    ("libx265", False, "CPU · H.265 · better compression"),
    ("libx264", False, "CPU · H.264 · most compatible"),
    ("hevc_nvenc", True, "NVIDIA NVENC · H.265"),
    ("hevc_qsv", True, "Intel Quick Sync · H.265"),
    ("hevc_videotoolbox", True, "VideoToolbox · H.265 · macOS"),
    ("hevc_amf", True, "AMD AMF · H.265"),
    ("hevc_vaapi", True, "VAAPI · H.265 · Linux"),
    ("h264_nvenc", True, "NVIDIA NVENC · H.264"),
    ("h264_qsv", True, "Intel Quick Sync · H.264"),
    ("h264_videotoolbox", True, "VideoToolbox · H.264 · macOS"),
    ("h264_amf", True, "AMD AMF · H.264"),
    ("h264_vaapi", True, "VAAPI · H.264 · Linux"),
)

#: What `recommended()` reaches for, in order. Named rather than inferred from
#: position in CANDIDATES, because the two answer different questions and the
#: display order is now the reverse of this one. Leaving it implicit would
#: mean a reordering of the dropdown silently changing what the tool encodes
#: with -- libx264 stays the default for its compatibility, which is the right
#: property for a choice nobody made.
_DEFAULT_ORDER: Tuple[str, ...] = ("libx264", "libx265")


class EncoderInfo(NamedTuple):
    name: str
    available: bool
    hardware: bool
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "available": self.available,
                "hardware": self.hardware, "detail": self.detail}


def _compiled_in() -> set:
    """Encoder names this ffmpeg was built with."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return set()
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True,
                             timeout=30, **NO_CONSOLE_WINDOW).stdout
    except (subprocess.SubprocessError, OSError):
        return set()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0][:1] == "V":
            names.add(parts[1])
    return names


#: What ffmpeg says -> what it means. Its own wording is accurate but assumes
#: you know the vendor SDK; "Error creating a MFX session: -9" is not a useful
#: thing to show someone choosing an encoder from a list.
_PLAIN = (
    ("no capable devices", "no GPU here can encode this size"),
    ("mfx session", "no Intel Quick Sync device"),
    ("mfx implementation", "no Intel Quick Sync device"),
    ("encoder->init() failed", "the AMD encoder rejected this size"),
    ("cannot load", "driver or runtime library not found"),
    ("failed to initialise vaapi", "no VAAPI device"),
    ("failed to open the drm device", "no VAAPI device"),
    ("impossible to convert between the formats",
     "no VAAPI device"),
    ("no device", "no device"),
)


def _reason(stderr: str, name: str) -> str:
    """A short cause from ffmpeg's complaint.

    Takes the encoder's *first* message rather than the last line. ffmpeg
    ends with generic wrappers -- "Terminating thread with return code -22
    (Invalid argument)" is what every one of these failures looks like at the
    bottom, which distinguishes nothing. The specific cause is always the
    first line the encoder itself logged.
    """
    # Tags vary: hevc_amf logs as "[hevc_amf @ ..]" but VAAPI logs as
    # "[VAAPI @ ..]", so match the encoder name *or* its family rather than
    # an exact prefix, and strip whatever bracket tag is there either way.
    family = name.rsplit("_", 1)[-1].lower()
    first = ""
    for raw in stderr.splitlines():
        line = raw.strip()
        if not line:
            continue
        tag = line[1:line.index("]")].lower() if line.startswith("[") \
            and "]" in line else ""
        body = line.split("] ", 1)[-1] if tag else line
        if tag.startswith(name.lower()) or tag.startswith(family):
            first = body
            break
        if not first:
            first = body
    low = first.lower()
    for needle, plain in _PLAIN:
        if needle in low:
            return plain
    return first[:120] or "rejected this frame size"


def _try_encode(name: str, width: int, height: int, timeout: float) -> \
        Tuple[bool, str]:
    """Ask the encoder to do exactly what VideoEncoder would ask of it.

    The invocation is built from the same per-family helpers the real encoder
    uses, so a probe cannot report an encoder usable under flags the render
    will not use -- VAAPI in particular needs a device and an hwupload filter,
    not merely a different quality flag.
    """
    from .ffmpeg_io import encoder_family, hw_filter_args, hw_input_args

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False, "ffmpeg not on PATH"
    family = encoder_family(name)
    cmd = [ffmpeg, "-hide_banner", "-v", "error"]
    cmd += hw_input_args(family)
    cmd += ["-f", "lavfi", "-i", f"color=black:s={width}x{height}:d=0.05",
            "-frames:v", "1"]
    cmd += hw_filter_args(family)
    cmd += ["-c:v", name, "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, **NO_CONSOLE_WINDOW)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    except OSError as exc:
        return False, str(exc)[:160]
    if proc.returncode == 0:
        return True, ""
    return False, _reason(proc.stderr, name)


def probe(width: int, height: int, timeout: float = 60.0,
          include_software: bool = True,
          include_missing: bool = False) -> List[EncoderInfo]:
    """Encoders that can encode `width` x `height` on this machine.

    Pass the *output* size -- for top-bottom stereo that is the source height
    doubled, which is exactly where the hardware limits bite.
    """
    compiled = _compiled_in()
    out: List[EncoderInfo] = []
    for name, hardware, summary in CANDIDATES:
        if name not in compiled:
            # Not built in at all -- almost always a platform that cannot have
            # it (VideoToolbox off macOS, say). Listing it would be noise.
            if include_missing:
                out.append(EncoderInfo(name, False, hardware,
                                       "not built into this ffmpeg"))
            continue
        if not hardware and not include_software:
            out.append(EncoderInfo(name, True, hardware, summary))
            continue
        ok, why = _try_encode(name, width, height, timeout)
        if ok:
            detail = summary if not hardware else f"{summary} · not recommended"
        else:
            detail = f"{summary} · {why}"
        out.append(EncoderInfo(name, ok, hardware, detail))
    return out


def recommended(infos: List[EncoderInfo]) -> Optional[str]:
    """The encoder to default to: the most compatible available software one.

    Asks `_DEFAULT_ORDER` by name before falling back to list order, so that
    reordering CANDIDATES for display cannot quietly change what a run with
    no explicit `--codec` encodes with.
    """
    by_name = {i.name: i for i in infos}
    for name in _DEFAULT_ORDER:
        info = by_name.get(name)
        if info is not None and info.available:
            return name
    for info in infos:
        if info.available and not info.hardware:
            return info.name
    return next((i.name for i in infos if i.available), None)
