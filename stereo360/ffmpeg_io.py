"""FFmpeg-based video decoding/encoding using subprocess pipes (no PyAV dependency).

Frames are exchanged as raw RGB24 numpy arrays, which keeps us codec-agnostic
and lets FFmpeg's native libraries do all heavy lifting.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Iterator, List, NamedTuple, Optional, Tuple

import numpy as np


class ColorTags(NamedTuple):
    """How a stream declares its colour encoding, or None where it doesn't.

    Worth carrying because we do not pass YUV through: frames are decoded to
    RGB, processed, and re-encoded. The decode uses the source's matrix; if the
    encode is untagged, a player has to guess, and for a 7680-wide file it
    guesses BT.709. Measured against a smpte170m source that guess costs
    rmse 2.03 and up to 8 levels of error -- a systematic shift over the whole
    frame, for nothing.

    All four are passed to ffmpeg, but only `space` (the matrix) and `range`
    actually survive into the file: neither libx264 nor libx265 writes the
    transfer or primaries into the VUI in this build, whatever value is
    requested. Those two are also the pair that decides how the image decodes,
    so nothing that matters is lost -- but do not expect a round trip to
    reproduce all four.
    """

    range: Optional[str] = None          # 'tv' | 'pc'
    space: Optional[str] = None          # matrix, e.g. 'bt709', 'smpte170m'
    transfer: Optional[str] = None
    primaries: Optional[str] = None

    @property
    def declared(self) -> bool:
        return any(v is not None for v in self)

    def encoder_args(self) -> List[str]:
        """Flags that both tag the output and steer the RGB->YUV matrix."""
        args: List[str] = []
        for value, flag in ((self.range, "-color_range"),
                            (self.space, "-colorspace"),
                            (self.transfer, "-color_trc"),
                            (self.primaries, "-color_primaries")):
            if value:
                args += [flag, value]
        return args


#: Chroma layouts we can encode, and the pix_fmt for each bit depth.
CHROMA_PIX_FMT = {
    ("4:2:0", 8): "yuv420p", ("4:2:0", 10): "yuv420p10le",
    ("4:2:2", 8): "yuv422p", ("4:2:2", 10): "yuv422p10le",
    ("4:4:4", 8): "yuv444p", ("4:4:4", 10): "yuv444p10le",
}

_CHROMA_BY_PREFIX = {"nv12": "4:2:0", "nv21": "4:2:0", "nv16": "4:2:2",
                     "nv20": "4:2:2", "nv24": "4:4:4", "nv42": "4:4:4"}


def chroma_from_pix_fmt(pix_fmt: Optional[str]) -> Optional[str]:
    """'yuv422p10le' -> '4:2:2'. None when it cannot be determined.

    Named formats are checked first, then the digits in the planar names,
    which covers yuv/yuvj/yuva variants at any bit depth. RGB and packed
    formats have no subsampling, hence 4:4:4.
    """
    if not pix_fmt:
        return None
    name = pix_fmt.lower()
    for prefix, chroma in _CHROMA_BY_PREFIX.items():
        if name.startswith(prefix):
            return chroma
    if name.startswith("gray"):
        return "4:0:0"
    if name.startswith(("gbr", "rgb", "bgr", "argb", "abgr", "x2rgb", "x2bgr")):
        return "4:4:4"
    match = re.match(r"yuva?j?(\d)(\d)(\d)", name)
    if match:
        return ":".join(match.groups())
    return None


#: Semi-planar hardware surface formats, whose digits are a format code rather
#: than a component width: p010 is 10-bit, p016 is 16-bit.
_SEMIPLANAR_DEPTH = {"p010": 10, "p012": 12, "p016": 16, "p210": 10,
                     "p216": 16, "p410": 10, "p416": 16}


def bit_depth_from_pix_fmt(pix_fmt: Optional[str]) -> Optional[int]:
    """'yuv420p10le' -> 10. None when it cannot be determined.

    Only the planar YUV, gray and semi-planar hardware names are parsed, which
    is what a video file actually carries. Packed RGB names are deliberately
    left alone: their digits are the *total* bits per pixel, not per component
    (rgb24 is 8-bit, rgb48 is 16-bit), and guessing wrong here would produce a
    warning that is worse than no warning.
    """
    if not pix_fmt:
        return None
    name = pix_fmt.lower()
    for prefix, depth in _SEMIPLANAR_DEPTH.items():
        if name.startswith(prefix):
            return depth
    if not name.startswith(("yuv", "yuvj", "yuva", "gray")):
        return None
    stem = re.sub(r"(le|be)$", "", name)
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    # 'yuv420p' / 'gray': no width suffix means 8.
    return 8 if stem.endswith("p") or stem == "gray" else None


def supported_chroma(codec: str) -> Tuple[str, ...]:
    """Chroma layouts this encoder is allowed to produce here.

    NVENC advertises 4:2:2 and 4:4:4 surfaces, but whether the hardware
    accepts them at 7680x7680 is untested, and a hardware encoder is the speed
    choice rather than the mastering one. So it stays on 4:2:0 and the caller
    degrades with a message rather than failing a run.
    """
    if encoder_family(codec) != "sw":
        return ("4:2:0",)
    return ("4:2:0", "4:2:2", "4:4:4")


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: Optional[int]
    duration: Optional[float]
    has_audio: bool
    color: ColorTags = field(default_factory=ColorTags)
    pix_fmt: Optional[str] = None
    #: 'equirectangular' | 'cubemap' | ... as declared by the file's sv3d box,
    #: or None when it declares nothing. Untagged is the common case, so it
    #: cannot mean "not spherical" -- only "did not say".
    projection: Optional[str] = None
    #: Pixels padded onto each cube face edge, from the cbmp box.
    cubemap_padding: int = 0
    #: Pixels cropped from a notional full sphere either side, from the `equi`
    #: box. Zero on a full sphere, and on anything that declares nothing.
    bound_left: int = 0
    bound_right: int = 0
    #: What `st3d` says the frame contains: "2D", "side by side",
    #: "top and bottom", or None when the file declares nothing.
    stereo_layout: Optional[str] = None
    #: Channels in the first audio stream, or None when there is no audio.
    #: 4, 9 or 16 is what an ambiX track looks like -- see stereo360.ambisonics.
    audio_channels: Optional[int] = None

    @property
    def chroma(self) -> Optional[str]:
        """The source's chroma subsampling, e.g. '4:2:0'."""
        return chroma_from_pix_fmt(self.pix_fmt)

    @property
    def bit_depth(self) -> Optional[int]:
        """Bits per component in the source, or None if undetermined."""
        return bit_depth_from_pix_fmt(self.pix_fmt)

    @property
    def horizontal_fov(self) -> Optional[float]:
        """Degrees of longitude the frame covers, if the file says.

        The `equi` bounds are pixels cropped from a full sphere, so the frame
        plus the two bounds is the sphere. Returns None when nothing is
        declared -- which is the common case, and means "assume 360" rather
        than "not spherical".
        """
        if self.projection is None:
            return None
        total = self.width + self.bound_left + self.bound_right
        if total <= 0:
            return None
        return 360.0 * self.width / total

    @property
    def is_stereo(self) -> bool:
        """True only when the file positively declares two views."""
        return bool(self.stereo_layout) and self.stereo_layout != "2D"


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise RuntimeError(f"'{tool}' not found on PATH. Please install FFmpeg.")
    return path


def probe(path: str) -> VideoInfo:
    ffprobe = _require("ffprobe")
    cmd = [
        ffprobe, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)

    vstream = next(s for s in data["streams"] if s["codec_type"] == "video")

    # ffprobe surfaces the sv3d projection box as stream side data. Asked for
    # separately because -show_streams alone does not include it.
    projection = None
    padding = 0
    bound_left = bound_right = 0
    stereo_layout = None
    side = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-select_streams",
         "v:0", "-show_entries", "stream_side_data", path],
        capture_output=True, text=True)
    if side.returncode == 0:
        try:
            streams = json.loads(side.stdout).get("streams") or [{}]
            for entry in streams[0].get("side_data_list", []):
                kind = entry.get("side_data_type")
                if kind == "Spherical Mapping":
                    projection = entry.get("projection")
                    padding = int(entry.get("padding") or 0)
                    # Pixels cropped from a notional full sphere. Present only
                    # on a partial projection, which ffprobe calls "tiled
                    # equirectangular".
                    bound_left = int(entry.get("bound_left") or 0)
                    bound_right = int(entry.get("bound_right") or 0)
                elif kind == "Stereo 3D":
                    stereo_layout = entry.get("type")
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    astream = next((s for s in data["streams"]
                    if s["codec_type"] == "audio"), None)
    has_audio = astream is not None
    try:
        audio_channels = int(astream["channels"]) if astream else None
    except (KeyError, TypeError, ValueError):
        audio_channels = None

    num, den = vstream["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0

    frame_count = vstream.get("nb_frames")
    frame_count = int(frame_count) if frame_count and frame_count != "N/A" else None
    duration = data.get("format", {}).get("duration")
    duration = float(duration) if duration else None

    # ffprobe omits these keys when a stream declares nothing, and reports
    # "unknown" in some builds; both mean "not declared", and inventing a
    # value would be worse than leaving the output untagged.
    def tag(key: str) -> Optional[str]:
        value = vstream.get(key)
        return value if value and value != "unknown" else None

    return VideoInfo(
        width=int(vstream["width"]),
        height=int(vstream["height"]),
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        has_audio=has_audio,
        color=ColorTags(range=tag("color_range"), space=tag("color_space"),
                        transfer=tag("color_transfer"),
                        primaries=tag("color_primaries")),
        pix_fmt=vstream.get("pix_fmt"),
        projection=projection,
        cubemap_padding=padding,
        bound_left=bound_left,
        bound_right=bound_right,
        stereo_layout=stereo_layout,
        audio_channels=audio_channels,
    )


def stream_durations(path: str) -> Tuple[Optional[float], Optional[float]]:
    """(video, audio) durations in seconds, either None if absent."""
    ffprobe = _require("ffprobe")
    out = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_streams",
         path], capture_output=True, text=True, check=True).stdout
    video = audio = None
    for stream in json.loads(out).get("streams", []):
        try:
            value = float(stream.get("duration"))
        except (TypeError, ValueError):
            continue
        if stream.get("codec_type") == "video":
            video = value
        elif stream.get("codec_type") == "audio":
            audio = value
    return video, audio


def write_thumbnail(path: str, out_path: str, frame_index: int = 0,
                    width: int = 960) -> bool:
    """One source frame as a small JPEG. False if none could be read.

    For the interface's VR180 direction picker, which has to show what the
    source looks like *before* any depth work has been done -- the whole point
    of choosing a direction is to do it before committing to an hour of render.

    Seeks by timestamp rather than counting frames. Landing a frame or two out
    is invisible at 960 px wide, where counting to frame 9000 would mean
    decoding 9000 frames to throw them all away.
    """
    ffmpeg = _require("ffmpeg")
    info = probe(path)
    seconds = frame_index / info.fps if info.fps > 0 else 0.0
    even = max(2, int(width) // 2 * 2)       # odd widths cannot be 4:2:0 JPEG

    def grab(seek: float) -> bool:
        cmd = [ffmpeg, "-v", "error", "-y"]
        if seek > 0:
            cmd += ["-ss", f"{seek:.3f}"]
        cmd += ["-i", path, "-frames:v", "1",
                "-vf", f"scale={even}:-2:flags=area", "-q:v", "4", out_path]
        subprocess.run(cmd, capture_output=True)
        return os.path.isfile(out_path) and os.path.getsize(out_path) > 0

    if grab(seconds):
        return True
    # Asked for a frame past the end -- which the interface can do, since it
    # lets the frame number run free when the source does not declare a count.
    # The first frame is the wrong picture but the right shape, and a picker
    # with a picture in it beats an empty box.
    return seconds > 0 and grab(0.0)


def trim_audio_to_video(path: str, fps: float) -> bool:
    """Cut audio back when it outlasts the picture. True if it did.

    Audio is copied from the source rather than re-encoded, so a run that
    stops early -- Stop pressed, or --max-frames -- still writes the source's
    *entire* audio track. A 12-frame render of a 60-second clip came out as a
    60-second file: 0.4s of video, then the last frame held for a minute.

    Deliberately not ffmpeg's `-shortest`, which cuts whichever stream ends
    first. Source audio routinely stops a few hundredths of a second before
    the picture, so on a complete render -shortest trimmed the *video*: it
    dropped 3 of 92 frames on real footage. This only ever shortens audio, and
    only when it is genuinely longer, so a normal render is untouched and pays
    nothing.
    """
    video, audio = stream_durations(path)
    if video is None or audio is None:
        return False
    # Half a frame of slack so the last picture is never at risk, and a
    # threshold below which the mismatch is not worth a rewrite.
    if audio <= video + 0.25:
        return False

    ffmpeg = _require("ffmpeg")
    keep = video + (0.5 / fps if fps > 0 else 0.02)
    tmp = f"{path}.trim.mp4"
    proc = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", path, "-t", f"{keep:.6f}",
         "-map", "0", "-c", "copy", "-movflags", "-faststart", tmp],
        capture_output=True, text=True)
    if proc.returncode != 0:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise RuntimeError(f"Could not trim audio to the video length: "
                           f"{proc.stderr.strip()[-300:]}")
    os.replace(tmp, path)
    return True


def decode_frames(path: str, max_frames: Optional[int] = None,
                  skip_frames: int = 0) -> Iterator[np.ndarray]:
    """Yield frames of `path` as (H, W, 3) uint8 RGB arrays, one at a time.

    skip_frames: frame-accurate output-side seek (ffmpeg decodes and discards,
    so it is exact but not instant for large N).
    """
    ffmpeg = _require("ffmpeg")
    info = probe(path)
    w, h = info.width, info.height
    frame_size = w * h * 3

    cmd = [ffmpeg, "-v", "error", "-i", path]
    if skip_frames > 0:
        cmd += ["-ss", f"{skip_frames / info.fps:.6f}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    # Unbuffered, and read in ~1 MB pieces straight into the destination array.
    #
    # Asking for a whole 8K frame (88 MB) in one call, through Python's
    # BufferedReader, cost 445 ms per frame -- while ffmpeg alone needs 93 ms
    # and `ffmpeg | cat > /dev/null` runs at 98 ms. So the pipe was never the
    # problem: a single enormous read is. Chunking at 1 MB, which is roughly
    # how `cat` reads, gives 84 ms and puts decoding back at ffmpeg's own
    # floor. Larger chunks are worse again (8 MB measured 115 ms).
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    assert proc.stdout is not None
    stream = getattr(proc.stdout, "raw", proc.stdout)
    chunk = 1 << 20

    produced = 0
    reached_eof = False
    try:
        while True:
            if max_frames is not None and produced >= max_frames:
                break
            # A fresh array per frame: callers (the chunked path) hold several
            # at once, so the buffer cannot be reused.
            frame = np.empty((h, w, 3), dtype=np.uint8)
            view = memoryview(frame.reshape(-1))
            filled = 0
            while filled < frame_size:
                got = stream.readinto(view[filled:filled + chunk])
                if not got:
                    break
                filled += got
            if filled < frame_size:
                reached_eof = True
                break
            yield frame
            produced += 1
    finally:
        # Anything other than natural EOF -- a frame cap, a cancelled render,
        # an exception upstream -- leaves ffmpeg holding frames for a pipe
        # nobody will drain. Kill it rather than wait on a decoder that is
        # blocked writing into a full buffer.
        if not reached_eof:
            proc.kill()
        proc.stdout.close()
        proc.wait()


# Rough number of uncompressed frames x264/x265 hold at once (worker threads
# plus the rate-control lookahead). Only used to decide whether the encoder
# needs restraining on this machine, so an approximation is enough.
_ENCODER_BUFFERED_FRAMES = 50

# swscale performs the RGB->YUV conversion on every frame, and its default
# rounding is the fast one. Measured on a near-flat ramp, RGB in to RGB out
# through a real x265 encode at 4:2:0 limited range:
#
#   default              8-bit rmse 1.510
#   +accurate_rnd
#   +full_chroma_int     8-bit rmse 1.285      (16% better)
#
# Free at 8K -- 0.99x wall clock and byte-identical output -- so there is no
# reason not to. Kept as a module constant so a test can turn it off and prove
# it still matters; see tests/test_color.py.
_SWS_FLAGS = "+accurate_rnd+full_chroma_int"

# x264/x265 speed names -> NVENC's p1 (fastest) .. p7 (slowest) scale, so the
# same --preset works for either family.
_NVENC_PRESETS = {
    "ultrafast": "p1", "superfast": "p1", "veryfast": "p2", "faster": "p3",
    "fast": "p3", "medium": "p4", "slow": "p5", "slower": "p6",
    "veryslow": "p7", "placebo": "p7",
}

# AMF has three quality levels rather than a preset ladder.
_AMF_QUALITY = {
    "ultrafast": "speed", "superfast": "speed", "veryfast": "speed",
    "faster": "speed", "fast": "speed", "medium": "balanced",
    "slow": "quality", "slower": "quality", "veryslow": "quality",
    "placebo": "quality",
}


#: Render node VAAPI encodes through. Overridable because the numbering
#: shifts on machines with more than one GPU.
VAAPI_DEVICE = os.environ.get("STEREO360_VAAPI_DEVICE", "/dev/dri/renderD128")


def encoder_family(codec: str) -> str:
    """'hevc_amf' -> 'amf'; anything unsuffixed is a software encoder."""
    for suffix in ("nvenc", "qsv", "amf", "vaapi", "videotoolbox"):
        if codec.endswith("_" + suffix):
            return suffix
    return "sw"


def hw_input_args(family: str) -> List[str]:
    """Options that must precede the input, per family.

    Only VAAPI needs any: it encodes from GPU surfaces, so a device has to be
    opened before the graph is built. NVENC, QSV, AMF and VideoToolbox all
    accept ordinary software frames.
    """
    if family == "vaapi":
        return ["-vaapi_device", VAAPI_DEVICE]
    return []


def hw_filter_args(family: str) -> List[str]:
    """Filters needed to hand frames to the encoder.

    VAAPI's only supported input pixel format is `vaapi`, so frames have to be
    converted and uploaded explicitly -- without this ffmpeg fails with
    "Impossible to convert between the formats supported by the filter".
    """
    if family == "vaapi":
        return ["-vf", "format=nv12,hwupload"]
    return []


def pix_fmt_for(family: str, chroma: str, bitdepth: int) -> Optional[str]:
    """The pix_fmt to request, or None when the filter chain decides.

    Hardware encoders take surface formats rather than planar YUV: p010le for
    10-bit, nv12 for 8-bit on QSV and AMF. VAAPI is fed by hwupload, so naming
    a format here would fight the filter.
    """
    if family == "vaapi":
        return None
    if family == "sw":
        return CHROMA_PIX_FMT[(chroma, bitdepth)]
    if bitdepth == 10:
        return "p010le"
    if family in ("nvenc", "videotoolbox"):
        return "yuv420p"
    return "nv12"


def _quality_args(family: str, codec: str, crf: int, preset: str) -> List[str]:
    """One quality number and one speed name, in each family's own dialect.

    Only x264/x265 have -crf. The hardware encoders each spell constant
    quality differently, and their scales are close enough to CRF that reusing
    the same number keeps one vocabulary for the caller.
    """
    if family == "nvenc":
        return ["-rc", "vbr", "-cq", str(crf),
                "-preset", _NVENC_PRESETS.get(preset, preset)]
    if family == "qsv":
        # QSV takes the x264-style preset names directly.
        return ["-global_quality", str(crf), "-preset", preset]
    if family == "amf":
        return ["-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf),
                "-quality", _AMF_QUALITY.get(preset, "balanced")]
    if family == "vaapi":
        # VAAPI's QP scale is H.264/HEVC's own 0-51, near enough to CRF to
        # reuse the number.
        return ["-rc_mode", "CQP", "-qp", str(max(0, min(51, crf)))]
    if family == "videotoolbox":
        # VideoToolbox wants a 1-100 quality where higher is better, the
        # opposite direction to CRF. Approximate, and untested -- no macOS
        # here -- but in the right region: crf 13 -> 74, crf 18 -> 64.
        return ["-q:v", str(max(1, min(100, 100 - crf * 2)))]
    args = ["-crf", str(crf), "-preset", preset]
    return args


class VideoEncoder:
    """Encode RGB frames to H.264/H.265 MP4, optionally copying audio from a source file.

    Supports the NVENC hardware encoders as well as x264/x265. They take
    different flags for the same things -- `-cq` instead of `-crf`, `p1`..`p7`
    instead of `medium`/`slow`, `p010le` instead of `yuv420p10le` -- so the
    caller keeps using one vocabulary and the translation happens here.
    """

    def __init__(
        self,
        out_path: str,
        width: int,
        height: int,
        fps: float,
        audio_source: Optional[str] = None,
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "medium",
        bitdepth: int = 8,
        color: Optional[ColorTags] = None,
        chroma: str = "4:2:0",
        audio_filter: Optional[str] = None,
        audio_args: Optional[List[str]] = None,
    ) -> None:
        ffmpeg = _require("ffmpeg")
        self._out_path = out_path

        family = encoder_family(codec)
        cmd = [ffmpeg, "-y", "-v", "error"]
        cmd += hw_input_args(family)
        cmd += [
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
            "-i", "pipe:0",
        ]
        if audio_source:
            cmd += ["-i", audio_source]
        cmd += ["-map", "0:v"]
        if audio_source:
            # Copy audio only if the source actually has an audio stream.
            cmd += ["-map", "1:a?"]
            if audio_filter:
                # Rotating an ambisonic soundfield is the only thing that puts
                # a filter here, and it forces a re-encode: there is no way to
                # remix channels without decoding them. Costs one generation,
                # which is why the default path stays a straight copy.
                cmd += ["-filter:a", audio_filter]
                cmd += list(audio_args or ["-c:a", "aac"])
            else:
                cmd += ["-c:a", "copy"]
        if bitdepth not in (8, 10):
            raise ValueError(f"Unsupported bit depth: {bitdepth}")
        if chroma not in supported_chroma(codec):
            raise ValueError(
                f"{codec} cannot encode {chroma} here; supported: "
                f"{', '.join(supported_chroma(codec))}")
        pix_fmt = pix_fmt_for(family, chroma, bitdepth)

        if _SWS_FLAGS:
            cmd += ["-sws_flags", _SWS_FLAGS]

        # Carry the source's colour declaration onto the output. See ColorTags.
        if color is not None and color.declared:
            cmd += color.encoder_args()

        # h264_nvenc tops out at 4096x4096 in hardware, whatever the driver;
        # hevc_nvenc reaches 8192x8192 (verified at 7680x7680). Say so here
        # rather than let it surface as "No capable devices". Every other
        # hardware limit is resolution- and vendor-specific enough that
        # `stereo360.encoders.probe` measures it instead of guessing.
        if codec == "h264_nvenc" and max(width, height) > 4096:
            raise ValueError(
                f"h264_nvenc cannot encode {width}x{height}: NVENC's H.264 "
                "engine is limited to 4096x4096. Use hevc_nvenc (good to "
                "8192x8192) or a CPU encoder.")

        cmd += hw_filter_args(family)
        cmd += ["-c:v", codec]
        cmd += _quality_args(family, codec, crf, preset)
        if pix_fmt is not None:
            cmd += ["-pix_fmt", pix_fmt]
        # 10-bit H.264 needs High 10 (x265 handles it automatically).
        if family == "sw" and bitdepth == 10 and codec == "libx264":
            cmd += ["-profile:v", "high10"]

        # Bound encoder memory when it would actually be a problem. x264/x265
        # buffer roughly (threads + lookahead) uncompressed frames, so the
        # question is frame size against *this machine's* free memory, not a
        # pixel count: a fixed 16 Mpx threshold left 4K top-bottom (3840x3840
        # = 14.7 Mpx) just below the line on every machine, including ones
        # with no headroom. NVENC keeps its buffers in VRAM and needs none of
        # this.
        from .warp import available_memory

        frame_bytes = width * height * 1.5          # yuv420p
        avail = available_memory()
        heavy = (family == "sw" and avail is not None
                 and frame_bytes * _ENCODER_BUFFERED_FRAMES > 0.25 * avail)
        if heavy:
            if codec == "libx264":
                cmd += ["-threads", "4", "-x264-params",
                        "rc-lookahead=10:sync-lookahead=10"]
            elif codec == "libx265":
                cmd += ["-x265-params", "pools=2:frame-threads=1"]

        cmd += [
            # moov at end (default) lets us inject spherical metadata without
            # having to rewrite stco/co64 chunk offsets.
            "-movflags", "-faststart",
            out_path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("write() called after close()")
        assert self._proc.stdin is not None
        # Hand the array's own memory to the pipe. Callers already build the
        # exact bytes ffmpeg wants (uint8 RGB, C-contiguous), and
        # `astype(np.uint8).tobytes()` copied that twice regardless -- astype
        # copies even when the dtype already matches. At 8K top-bottom that
        # was 2 x 169 MiB of allocation per frame, ~32 ms of it memcpy, to
        # produce bytes identical to the ones we were handed. The conversion
        # still happens for anything that is not already in the right form.
        if frame.dtype != np.uint8 or not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
        try:
            self._proc.stdin.write(memoryview(frame.reshape(-1)))
        except (BrokenPipeError, OSError):
            # ffmpeg died mid-run -- a rejected pixel format, an unsupported
            # resolution for a hardware encoder, a full disk. The bare
            # BrokenPipeError says none of that, and ffmpeg's own diagnostic
            # has already gone to stderr, so name the real failure here.
            rc = self._proc.wait()
            raise RuntimeError(
                f"ffmpeg encoder exited with code {rc} while frames were "
                f"still being written; see its output above for the cause"
            ) from None

    def close(self) -> None:
        """Finish encoding: close the pipe and let ffmpeg finalize the file."""
        if self._closed:
            return
        self._closed = True
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encoder exited with code {rc}")

    def abort(self) -> None:
        """Stop encoding and discard whatever ffmpeg has not yet written.

        A *cancelled* render deliberately does not come here: it calls
        `close()`, because ffmpeg writes its moov box on a clean EOF and the
        partial file is then playable -- which is what someone who pressed
        Stop an hour into a render wants. `abort()` is for the failure path,
        where waiting on an encoder that is already broken would only hang,
        and where `close()` raising would mask the original exception.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except OSError:
            pass
        self._proc.kill()
        self._proc.wait()

    def __enter__(self) -> "VideoEncoder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()
