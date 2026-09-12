"""FSRCNNX upscaling: a learned resampler that runs as a shader.

The free option for *video*, where the others are not. Put through the same
test as everything else -- real 8K frames halved, put back, scored against
what was really there:

    method              PSNR   SSIM   temporal   ms/frame
    Lanczos            35.93  0.979        87%         16
    spline             35.70  0.978        86%          -
    ewa_lanczossharp   35.40  0.975        86%          -
    FSRCNNX x2         36.12  0.980       101%         83
    Real-ESRGAN x4v3   26.10  0.920       135%       2600
    Artemis MQ (Topaz) 27.81  0.907       122%        790

`temporal` is the sequence's own frame-to-frame change as a percentage of the
real footage's, and 100% is the target rather than the floor: Lanczos sits at
87% because it is soft, so there is less detail to change between frames, not
because it is admirably steady. FSRCNNX at 101% moves almost exactly as much
as reality -- the steadiest result measured here, ahead of Topaz's own most
settled model.

What it is not is a detail inventor. It beats Lanczos by 0.2 dB and looks
crisper on foliage, but it recovers what is recoverable rather than guessing,
which is the same property that stops it crawling. Anyone expecting a 4K
source to come out looking like 8K wants Topaz, and will pay for it in both
senses.

Three things make it cheap: it is a filter in the ffmpeg this project already
installs, it runs on Vulkan so it works on any GPU rather than only CUDA, and
at 83 ms a frame it is ten times faster than the cheapest Topaz model.

No wrap padding, deliberately. The shader's receptive field is a few pixels,
so the +/-180 seam costs 0.04 dB against a wrapped version -- inside the
noise, and not worth the filtergraph. That is measured, not assumed: an
interpolator in the same position loses 12 dB there, because motion
estimation needs context the way a resampler does not.

The shader is LGPL-3.0, by igv, and is fetched rather than vendored.

Any mpv user shader can be run here -- `--fsrcnnx-shader` takes a path, and
`chain` does not care whose file it is. One thing decides whether a given one
is usable, and it is not quality: **ffmpeg's libplacebo ignores `//!OFFSET`**.
A shader that declares a half-pixel grid shift for the host to take out does
not get it taken out, and the result sits half a pixel from where it belongs
-- which scores like a bad upscaler, looks like a fine one, and would bake a
geometry error into the depth pass. Grep a candidate before measuring it:

    shader                    //!OFFSET   aligned
    FSRCNNX, ArtCNN, CuNNy         no       yes
    ravu-lite                      no       yes
    ravu-zoom (OFFSET ALIGN)      ---       yes
    NNEDI3 (mpv-prescalers)       yes        no     -6 dB
    ravu (plain)                  yes        no     -6 dB

Both failures cost about six decibels of pure misregistration. The tell is
that a shifted score beats the unshifted one: an aligned image gets worse
whichever way you push it, so scoring each candidate at zero and at plus and
minus half a pixel says in one run whether its row means anything.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Optional

from .ffmpeg_io import NO_CONSOLE_WINDOW

#: Alongside the other fetched models. The 8-0-4-1 variant: the 16-0-4-1 one
#: is four times the size and measured 36.13 against 36.12, so it buys
#: nothing here.
DEFAULT_SHADER = os.path.join("models", "FSRCNNX_x2_8-0-4-1.glsl")

CODE = "fsrcnnx"
NAME = "FSRCNNX (shader)"
DESC = ("Runs in ffmpeg on any GPU, and the steadiest upscaler measured. "
        "A better resampler rather than a detail inventor.")

#: What the shader does. Other amounts still work -- libplacebo resamples
#: after it -- but 2x is what it was trained for, and 4K to 8K is 2x.
NATIVE_SCALE = 2

_FRAME_RE = re.compile(r"frame=\s*(\d+)")

#: Whether this ffmpeg can actually run it, which needs libplacebo *and* a
#: Vulkan device. Probed once: it costs a process launch.
_usable: Optional[bool] = None
#: Why not, when `_usable` is False. Cached with it.
_reason: str = ""


class ShaderError(RuntimeError):
    """The pass could not be done. The message is for the user."""


def shader_path(explicit: Optional[str] = None) -> str:
    return explicit or DEFAULT_SHADER


#: Hand libplacebo a Vulkan device instead of letting it make its own.
#:
#: Without these the filter segfaults before it reaches a frame -- measured on
#: a Radeon 780M with a healthy driver: `-init_hw_device vulkan` on its own
#: builds a device and lists queue families happily, and the very same
#: filtergraph then dies with an access violation the moment libplacebo has to
#: create one for itself. Supplying it explicitly is the whole fix; the
#: filtergraph does not change, and no `hwupload` is needed.
#:
#: The failure it replaces is the expensive kind: the filter is present, the
#: driver is current, Vulkan works, and the only symptom is a crash with an
#: empty log, which reads as "this machine cannot do libplacebo" when the
#: machine is fine.
VULKAN_ARGS = ["-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk"]


def _has_filter(ffmpeg: str) -> bool:
    """Whether this build carries the libplacebo filter at all."""
    try:
        done = subprocess.run([ffmpeg, "-hide_banner", "-filters"],
                              capture_output=True, text=True, timeout=60,
                              errors="replace", **NO_CONSOLE_WINDOW)
        return "libplacebo" in (done.stdout or "")
    except (OSError, subprocess.SubprocessError):
        return False


def problem(ffmpeg: str = "ffmpeg", recheck: bool = False) -> Optional[str]:
    """Why the shader cannot run here, or None if it can.

    Separated from `usable` because the two failures need different advice
    and used to be reported as one sentence. A build without the filter is
    fixed by installing a different ffmpeg. A build *with* it that crashes is
    a driver problem, and saying "this ffmpeg has no libplacebo" about a
    build whose `-filters` plainly lists it sends the reader somewhere there
    is nothing to find -- measured on a Radeon 780M, where the filter is
    present, a Vulkan device initialises, and the filter then segfaults.
    """
    global _usable, _reason
    if _usable is not None and not recheck:
        return _reason or None
    try:
        done = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             *VULKAN_ARGS,
             "-f", "lavfi", "-i", "testsrc2=size=64x64", "-frames:v", "1",
             "-vf", "format=yuv420p,libplacebo=w=128:h=128", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120, errors="replace",
            **NO_CONSOLE_WINDOW)
        if done.returncode == 0:
            _usable, _reason = True, ""
        elif not _has_filter(ffmpeg):
            _usable, _reason = False, (
                "this ffmpeg was not built with libplacebo, so the shader "
                "cannot run; a full build provides it")
        else:
            # Negative on POSIX is a signal; the large positive is a Windows
            # NTSTATUS. Either way the filter is there and it died.
            code = done.returncode
            how = (f"signal {-code}" if code < 0
                   else f"0x{code:08X}" if code > 0xC0000000 else f"exit {code}")
            _usable, _reason = False, (
                f"this ffmpeg has libplacebo but it failed to run ({how}) -- "
                f"usually a Vulkan driver problem rather than a missing build")
    except (OSError, subprocess.SubprocessError) as e:
        _usable, _reason = False, f"could not run ffmpeg ({type(e).__name__})"
    return _reason or None


def usable(ffmpeg: str = "ffmpeg", recheck: bool = False) -> bool:
    """Whether libplacebo is present and a Vulkan device answers.

    Asked by running it, not by reading the build flags: a build can carry
    the filter and still have no device to run it on, and the failure then
    lands in the middle of someone's render rather than before it.
    """
    problem(ffmpeg, recheck)
    return bool(_usable)


def available(explicit: Optional[str] = None, ffmpeg: str = "ffmpeg") -> bool:
    return os.path.exists(shader_path(explicit)) and usable(ffmpeg)


def describe(explicit: Optional[str] = None, ffmpeg: str = "ffmpeg") -> dict:
    """What the interface needs to decide whether to offer this."""
    path = shader_path(explicit)
    if not os.path.exists(path):
        return {"available": False, "shader": path,
                "reason": f"no FSRCNNX shader at {path}"}
    why = problem(ffmpeg)
    if why:
        return {"available": False, "shader": path, "reason": why}
    return {"available": True, "shader": path, "reason": ""}


#: Pass as `shader` for a resample with no model in front of it, which is
#: what the interface's first dropdown set to "None" asks for. Distinct from
#: `None`, which still means "the default shader".
NO_SHADER = "-"


#: What a source's pixel format has to become before libplacebo sees it.
#:
#: Something must be named here, because every shader in `models/` is a
#: `//!HOOK LUMA` shader: handed RGB there is no luma plane to hook, the pass
#: runs to completion, and the output is a plain resample with no sign that
#: the model never ran. So an RGB source is converted rather than left alone.
#:
#: What must *not* be named is `yuv420p` for everything, which is what stood
#: here. It is the source's own layout for all camera video, and a silent
#: downgrade for anything better: a 4:4:4 still lost three quarters of its
#: chroma (1.5 dB, measured against the 8K truth) and a 10-bit source was
#: truncated to 8 (878 distinct luma codes down to 440). The working file is
#: built to carry both -- `intermediate.choose` asks for yuv444p and
#: yuv420p10le when the source has them -- so the loss happened one filter
#: before the container designed to prevent it.
#:
#: Anything that cannot be named as a plain planar YUV layout falls back to
#: `yuv420p`, which is what every source got before: RGB, the hardware
#: surface formats, 12-bit, and an unprobed source with no format at all.
def filter_format(pix_fmt: Optional[str]) -> str:
    """The `format=` a pre-pass should use to preserve `pix_fmt`."""
    from . import ffmpeg_io

    chroma = ffmpeg_io.chroma_from_pix_fmt(pix_fmt)
    depth = ffmpeg_io.bit_depth_from_pix_fmt(pix_fmt)
    return ffmpeg_io.CHROMA_PIX_FMT.get((chroma, depth), "yuv420p")


def chain(width: int, height: int, scale: float = 2.0,
          shader: Optional[str] = None,
          resampler: Optional[str] = None,
          pix_fmt: Optional[str] = None) -> str:
    """The -vf for one pass over frames `width` x `height`.

    The shader doubles; libplacebo then resamples to whatever was actually
    asked for, so a scale other than 2 still lands on the right size. That
    second step was always happening and was never named -- a 2x shader asked
    for 4x has always handed half the job to libplacebo's default spline36 --
    and `resampler` is what lets the caller say which filter does it.

    It only bites where there is something left to do. At a scale the shader
    lands on exactly, the option is measurably a no-op: 0.099 levels out of
    255 between spline36 and ewa_lanczos4sharpest at 2x, against 0.000 for
    the same command run twice. At 3x or 4x it is doing half the work.

    With `shader=NO_SHADER` there is no model at all and the named filter
    does the whole factor, which is the safest choice on a source that is
    already sharpened -- it won every such rung measured.
    """
    out_w = int(round(width * scale))
    out_h = int(round(height * scale))
    parts = [f"w={out_w}", f"h={out_h}"]
    if resampler:
        parts.append(f"upscaler={resampler}")
    if shader != NO_SHADER:
        parts.append(f"custom_shader_path={_escape(shader_path(shader))}")
    return f"format={filter_format(pix_fmt)},libplacebo=" + ":".join(parts)


def _escape(path: str) -> str:
    """A path as an ffmpeg filter option value.

    Quoted *and* colon-escaped, because a Windows path needs both and either
    alone fails. The filtergraph parser splits options on ':' and filters on
    ',', which the quotes settle; the option parser then unescapes what is
    inside them, which is what the backslash is for. Measured, since the
    plain, quoted-only and escaped-only spellings all fail the same way --
    `Invalid argument`, with nothing to say a path was the problem.
    """
    return "'" + path.replace("\\", "/").replace(":", r"\:") + "'"


def run(src: str, dst: str, *, width: int, height: int, scale: float = 2.0,
        shader: Optional[str] = None, total: Optional[int] = None,
        pix_fmt: Optional[str] = None,
        trim_from: int = 0, frames: Optional[int] = None,
        name: str = NAME, code: str = CODE, resampler: Optional[str] = None,
        reporter=None, cancel=None, ffmpeg: str = "ffmpeg") -> None:
    """One pass over `src`, writing `dst`. Raises `ShaderError`.

    `shader=NO_SHADER` runs `resampler` alone, with no model -- there is then
    no file to look for, which is why the existence check is skipped for it
    rather than being handed a path it would not find.
    """
    if shader != NO_SHADER:
        path = shader_path(shader)
        if not os.path.exists(path):
            raise ShaderError("\n".join((
                f"FSRCNNX shader not found: {path}",
                "It is not shipped with the repository. Fetch it once:",
                "    python scripts/fetch_fsrcnnx.py")))
    if not usable(ffmpeg):
        raise ShaderError(
            "This ffmpeg cannot run libplacebo, or there is no Vulkan device "
            "for it. Upscaling with a shader needs both.")

    vf = chain(width, height, scale, shader, resampler, pix_fmt)
    if trim_from > 0:
        vf = f"trim=start_frame={trim_from},setpts=PTS-STARTPTS,{vf}"
    # Audio is copied, never re-encoded. Without this ffmpeg maps the source's
    # track and encodes it with whatever the container defaults to -- and the
    # working file is Matroska, whose default is Vorbis, so an AAC source came
    # out the far end as low-bitrate Vorbis. Nothing announced it: the pre-pass
    # is a video stage and audio was never mentioned in the command, so the
    # default applied in silence and the converter then copied the result into
    # the output as if it were the original.
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y", *VULKAN_ARGS,
           "-i", src, "-vf", vf, "-c:a", "copy"]
    if frames:
        cmd += ["-frames:v", str(int(frames))]
    from . import intermediate

    out_args, _ = intermediate.choose(
        ffmpeg, pix_fmt=pix_fmt, width=int(round(width * scale)),
        height=int(round(height * scale)), frames=total, where=dst)
    cmd += [*out_args, dst]

    if reporter is not None:
        # The model that ran, not the module it lives in. This said
        # "FSRCNNX (shader)" for every shader once there were four of them,
        # so an ArtCNN pre-pass announced itself as FSRCNNX and only the
        # filename disagreed -- the same fault e6f1674 fixed on the onnx
        # side, left behind here because that path takes a name and this
        # one did not.
        reporter.info(f"Upscaling with {name}: {scale:g}x from "
                      f"{width}x{height}", stage="upscale", model=code)
        reporter.start(total, stage="upscale")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True,
                            errors="replace", **NO_CONSOLE_WINDOW)
    seen, tail = 0, []
    assert proc.stderr is not None
    try:
        for line in proc.stderr:
            tail.append(line.rstrip())
            del tail[:-15]
            m = _FRAME_RE.search(line)
            if m and reporter is not None:
                n = int(m.group(1))
                if n > seen:
                    reporter.advance(n - seen)
                    seen = n
            if cancel is not None and cancel():
                proc.kill()
                proc.wait(timeout=10)
                raise ShaderError("cancelled")
        code = proc.wait()
    finally:
        if reporter is not None:
            reporter.finish(stage="upscale")
    if code != 0 or not os.path.exists(dst):
        raise ShaderError(f"ffmpeg exited {code}:\n" + "\n".join(tail[-6:]))
