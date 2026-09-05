"""RIFE frame interpolation through onnxruntime -- the option needing no Topaz.

Raising the frame rate is worth more in a headset than on a monitor: 30 fps
judders when your head keeps moving and there is no shutter to hide it. Topaz
does it well, but it is a paid Windows install, so this is the same job on the
runtime stereo360 already ships -- CUDA, DirectML or CPU, no torch.

Measured on this project's own footage, holding out every second frame of a 4K
clip and scoring only the frames each method had to invent:

    duplicate frames        34.51 dB      ffmpeg fps=
    cross-fade              37.44 dB      ffmpeg framerate=
    block motion            37.46 dB      ffmpeg minterpolate=
    RIFE v4.25              39.12 dB      here
    Topaz Chronos           39.61 dB

A tripod clip cannot separate those, so the same test on a synthetic 0.56
deg/frame yaw pan, where the truth is known exactly:

                          whole frame   mid-frame   at the seam
    block motion             34.81         43.25        30.71
    Topaz Chronos            43.67         45.55        43.10
    RIFE, frame as flat      47.61         53.19        40.33
    RIFE, frame as a sphere  52.83         53.19        52.34

Three things that table is saying:

* RIFE is level with Chronos on real footage and ahead of it on a pan, which
  is the motion 360 footage mostly contains -- and it is the only free option
  that gets anywhere near either.
* The +/-180 seam is worth 12 dB, and costs nothing to fix. A flat
  interpolator has to invent the columns either side of the seam with the
  other half of the picture out of reach; taking each tile's context with
  wraparound hands it back. Mid-frame is identical either way, which is how
  you know that is really what the difference is.
* Every method here is invention. None of them recovers what the camera did
  not record, so this runs on the mono source, before the stereo pass and
  never per eye -- interpolating the two eyes separately would estimate two
  independent flows, and their difference is the disparity, so an error that
  would be invisible in mono becomes a depth error that is not.

It runs *inside* the frame stream rather than over the file first: see
`Streamer`. That is not a speed decision -- measured at 8K over 30 source
frames it came to 175 s against 172 s for making a smoothed copy and
converting that, which is a wash. It is a decision about what happens while
you wait and what you keep when you stop.

Speed, on this machine's DirectML (there is no CUDA execution provider in the
build stereo360 installs on Windows, so these are a floor, not a ceiling):
0.24 s at 4K and 0.94 s at 8K per frame it has to invent, on an idle machine.
Topaz Chronos is 0.19 s at 4K. ffmpeg's minterpolate is 0.91 s at 4K and
cannot run at 8K at all -- it fails to allocate, with 16 GB free.

Inside a render it is slower than that, and by more than the GPU work
explains: the encoder is the other thing wanting the machine. libx264 at
preset medium on a 7680x7680 frame saturates every core, and what this does
between model calls is host-side array work. Measured over ten 8K frames, the
interpolation went 37 s a frame against a CPU encode and 2.5 s against
hevc_nvenc -- fifteen times, for a difference that has nothing to do with
interpolation. Halving the array work (see `_window` and `_fill`) took the
CPU-encode case to 5 s. If a render with this on is slow, the encoder is the
first place to look.

The tiling is not an optimisation. DirectML returns *wrong pixels* rather than
an error once a frame passes 2^22 pixels: correct at 2816x1408, corrupt at
3072x1536 and everything above, same model and inputs, while the CPU provider
is correct at 3840x1920. 4K is 7.4 Mpx and 8K is 29.5, so every frame is cut
into tiles that stay under the limit. Their seams do not show -- the mid-frame
crop scored above straddles one and is the best-scoring region of the frame.
"""

from __future__ import annotations

import gc
import math
import os
import subprocess
from typing import List, Optional

import numpy as np

from .ffmpeg_io import NO_CONSOLE_WINDOW

#: Where the graph is expected, by the same convention as the ONNX depth
#: models: not in the repository, fetched once by a script in scripts/.
DEFAULT_MODEL = os.path.join("models", "rife_v4.25.onnx")

#: What --interpolate takes to mean this rather than a Topaz model.
CODE = "rife"
NAME = "RIFE v4.25"
DESC = ("Runs on any GPU through onnxruntime. Level with Chronos on real "
        "footage and ahead of it on camera pans.")

#: Above 30 fps there is nothing to gain: the judder interpolation exists to
#: fix is a low-frame-rate artifact, and doubling 60 only doubles the frames
#: the stereo pass then has to convert.
MAX_SOURCE_FPS = 30.0

#: DirectML corrupts a frame larger than this. See the module docstring.
_DML_LIMIT = 1 << 22

#: Tile core and the context taken around it. 1920x960 plus 64 either side is
#: 2048x1088, comfortably under the limit, and big enough that the flow has
#: the moving thing and where it came from in the same tile.
_TILE = (1920, 960)
_OVERLAP = 64

_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider",
              "CoreMLExecutionProvider", "CPUExecutionProvider")

#: Typed, so `uint8 / _SCALE` lands in float32 rather than going through
#: float64 and being cast back.
_SCALE = np.float32(255.0)

class InterpolateError(RuntimeError):
    """Interpolation could not be done. The message is for the user."""


def model_path(explicit: Optional[str] = None) -> str:
    """Where to look for the graph: what was asked for, or the default."""
    return explicit or DEFAULT_MODEL


def available(explicit: Optional[str] = None) -> bool:
    """Whether this machine can interpolate: the graph and a runtime."""
    if not os.path.exists(model_path(explicit)):
        return False
    try:
        import onnxruntime                                    # noqa: F401
    except ImportError:
        return False
    return True


def offered_for(fps: float) -> bool:
    """Whether raising the rate is worth offering for a source this fast."""
    return 0 < float(fps) <= MAX_SOURCE_FPS


def describe(fps: float = 0.0, explicit: Optional[str] = None) -> dict:
    """What the interface needs to decide whether to offer this."""
    path = model_path(explicit)
    if not os.path.exists(path):
        return {"available": False, "model": path,
                "reason": f"no RIFE model at {path}"}
    try:
        import onnxruntime                                    # noqa: F401
    except ImportError:
        return {"available": False, "model": path,
                "reason": "onnxruntime is not installed"}
    return {"available": True, "model": path, "reason": "",
            "offered": offered_for(fps) if fps else True}


def _session(path: str, provider: Optional[str] = None):
    import onnxruntime as ort

    have = set(ort.get_available_providers())
    order = [p for p in _PROVIDERS if p in have]
    if provider:
        if provider not in have:
            raise InterpolateError(
                f"onnxruntime provider {provider!r} is not available here; "
                f"this build has {sorted(have)}")
        order = [provider] + [p for p in order if p != provider]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(path, opts, providers=order)
    except Exception as e:                                    # noqa: BLE001
        raise InterpolateError(f"could not load {path}: {e}") from e
    return sess, sess.get_providers()[0]


def _tile_for(provider: str) -> tuple:
    """Tile core for this provider.

    Only DirectML actually needs the cap, but every provider gets the same
    tiles: bounded memory at 8K is worth having everywhere, and one code path
    means the tiling is exercised on the machine it was written on.
    """
    tw, th = _TILE
    while (tw + 2 * _OVERLAP) * (th + 2 * _OVERLAP) > _DML_LIMIT:
        tw, th = tw // 2, th // 2
    return tw, th


def _window(img: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """`img[y0:y1, x0:x1]`, wrapped left to right and clamped at the poles.

    The wrap is the whole reason a tile at the seam works: on an equirect
    frame the column before the first is the last one, and a tile that knows
    that can follow motion straight across +/-180 degrees.

    Sliced rather than gathered. The obvious spelling of this -- index arrays
    through `np.ix_` -- copies every tile whether or not it needs it, and at
    8K that is 16 copies of 6 MB per frame *per input*. A tile that touches
    neither the seam nor a pole is most of them, and for those this returns a
    view of the frame it was given, costing nothing at all.
    """
    h, w = img.shape[:2]
    ya, yb = max(y0, 0), min(y1, h)
    if 0 <= x0 and x1 <= w:
        got = img[ya:yb, x0:x1]                     # a view, the common case
    elif x0 < 0:
        got = np.concatenate((img[ya:yb, w + x0:], img[ya:yb, :x1]), axis=1)
    else:
        got = np.concatenate((img[ya:yb, x0:], img[ya:yb, :x1 - w]), axis=1)
    top, bottom = ya - y0, y1 - yb
    if top or bottom:
        # Above the north pole there is nothing to wrap to, so the edge row
        # repeats -- which is what the projection does with it anyway.
        got = np.pad(got, ((top, bottom), (0, 0), (0, 0)), mode="edge")
    return got


def _fill(x: np.ndarray, a: np.ndarray, b: np.ndarray, t: float) -> None:
    """Load one tile pair into the model's input buffer, in place.

    Written as three ufunc calls with `out=` rather than the arithmetic it
    reads as, because `a.transpose(2, 0, 1).astype(np.float32) / 255.0` builds
    two whole temporaries before anything is assigned. At 8K that is the
    difference between one pass over the tile and three, and this is the part
    of the work that runs on the CPU while the encoder wants it.
    """
    np.divide(a.transpose(2, 0, 1), _SCALE, out=x[0, 0:3])
    np.divide(b.transpose(2, 0, 1), _SCALE, out=x[0, 3:6])
    if x[0, 6, 0, 0] != t:
        x[0, 6] = t          # constant across a frame's tiles; filled once


def _store(y: np.ndarray, dst: np.ndarray, top: int, left: int) -> None:
    """Write the model's output into `dst`, taking the core of the tile."""
    h, w = dst.shape[:2]
    got = y[0, :3, top:top + h, left:left + w]
    np.clip(got, 0.0, 1.0, out=got)
    np.multiply(got, _SCALE, out=got)
    np.add(got, 0.5, out=got)
    # One cast, straight into the frame being built. The transpose is a view;
    # the assignment is the only pass over the pixels.
    dst[...] = got.transpose(1, 2, 0)


def between(sess, a: np.ndarray, b: np.ndarray, t: float = 0.5,
            tile: Optional[tuple] = None) -> np.ndarray:
    """The frame `t` of the way from `a` to `b`, a tile at a time."""
    name = sess.get_inputs()[0].name
    h, w = a.shape[:2]
    tw, th = tile or _TILE
    out = np.empty((h, w, 3), np.uint8)
    # One buffer for every tile of a frame. They are all the same size unless
    # the frame does not divide by the tile, so this is normally one
    # allocation per call rather than one per tile -- and a single shape also
    # spares the runtime recompiling its kernels for a second one.
    buffers: dict = {}
    for y0 in range(0, h, th):
        for x0 in range(0, w, tw):
            y1, x1 = min(y0 + th, h), min(x0 + tw, w)
            ya, xa = y0 - _OVERLAP, x0 - _OVERLAP
            wa = _window(a, ya, y1 + _OVERLAP, xa, x1 + _OVERLAP)
            wb = _window(b, ya, y1 + _OVERLAP, xa, x1 + _OVERLAP)
            shape = wa.shape[:2]
            x = buffers.get(shape)
            if x is None:
                x = buffers[shape] = np.empty((1, 7) + shape, np.float32)
                x[0, 6] = t
            _fill(x, wa, wb, t)
            _store(sess.run(None, {name: x})[0], out[y0:y1, x0:x1],
                   y0 - ya, x0 - xa)
    return out


class Streamer:
    """Interpolation as a filter in the frame stream, not a pass over a file.

    Given the frames a source produced, it yields those plus the ones that
    belong between them, on the output grid the target rate implies. All the
    arithmetic of the job lives here -- which source frames a range of output
    frames needs, and which instant each output frame sits at -- and `run`
    wraps it in a decoder and an encoder to make the pre-pass.

    It was written to sit inside the renderer's own frame stream instead,
    which is nicer to wait for and four to seven times slower; `run` records
    the measurements. The shape survived the change because it is the right
    shape either way: a generator of frames, ignorant of where they go.
    """

    def __init__(self, src_fps: float, target_fps: Optional[float] = None,
                 model: Optional[str] = None, provider: Optional[str] = None,
                 reporter=None) -> None:
        self.src_fps = float(src_fps)
        self.target_fps = float(target_fps or self.src_fps * 2)
        if not offered_for(self.src_fps):
            raise InterpolateError(
                f"{self.src_fps:g} fps is already smooth enough; "
                f"interpolation is for sources at {MAX_SOURCE_FPS:g} fps or "
                f"below.")
        if self.target_fps <= self.src_fps:
            raise InterpolateError(
                f"asked for {self.target_fps:g} fps from a {self.src_fps:g} "
                f"fps source, which would drop frames rather than add them.")
        path = model_path(model)
        if not os.path.exists(path):
            raise InterpolateError("\n".join((
                f"RIFE model not found: {path}",
                "It is not shipped with the repository. Fetch it once:",
                "    python scripts/fetch_rife.py")))
        self._sess, self.provider = _session(path, provider)
        self._tile = _tile_for(self.provider)
        self._reporter = reporter

    @property
    def ratio(self) -> float:
        return self.target_fps / self.src_fps

    def close(self) -> None:
        """Drop the model. Idempotent, and safe to call twice."""
        self._sess = None

    def announce(self) -> None:
        if self._reporter is not None:
            self._reporter.info(
                f"Smoothing motion before the render: {NAME} on "
                f"{self.provider}, {self.src_fps:g} to {self.target_fps:g} fps",
                stage="interpolate", model=CODE)

    def total(self, source_frames: Optional[int]) -> Optional[int]:
        """Output frames for a source this long, for the progress bar."""
        if not source_frames:
            return source_frames
        return int(math.floor((source_frames - 1) * self.ratio)) + 1

    def window(self, start_output: int,
               count: Optional[int]) -> tuple:
        """(skip, take) source frames needed for this run of output frames.

        A frame range means output frames -- it is what someone counts when
        they ask for a test render -- so it has to be turned back into the
        source frames those come from. One extra either side, because an
        output frame between two real ones needs both.
        """
        step = self.src_fps / self.target_fps
        skip = int(math.floor(start_output * step))
        take = None
        if count is not None:
            last = int(math.ceil((start_output + count) * step))
            take = max(1, last - skip + 2)
        return skip, take

    def stream(self, pairs, first_output: int = 0, first_source: int = 0,
               count: Optional[int] = None):
        """Yield (frame, payload) with invented frames between the real ones.

        `pairs` is (frame, payload) as the decoder produced them. A payload
        belongs to a frame the camera really shot -- cube faces, in practice
        -- so an invented frame carries None and the depth stage rebuilds
        what it needs from the equirect, as it does for equirect input.

        Positions are computed on the *global* output grid rather than from
        the first frame handed over, so a run starting at output frame 101
        lands on exactly the instants it would have if it had started at zero.
        """
        step = self.src_fps / self.target_fps
        it = iter(pairs)
        prev = next(it, None)
        if prev is None:
            return
        nxt = next(it, None)
        index = first_source          # which source frame `prev` is
        made = 0
        j = 0
        while count is None or made < count:
            pos = (first_output + j) * step
            i = int(math.floor(pos))
            t = pos - i
            # Walk forward until `prev` is the frame this instant sits on. One
            # pass over the source, two frames held at once, which is what
            # makes 8K affordable at all.
            while i > index and nxt is not None:
                prev, nxt = nxt, next(it, None)
                index += 1
            if i > index:
                return                # ran off the end of the source
            if nxt is None and t >= 1e-4:
                return                # past the last real frame
            if t < 1e-4:
                yield prev
            elif t > 1 - 1e-4:
                yield nxt
            else:
                yield (between(self._sess, prev[0], nxt[0], t, self._tile),
                       None)
            made += 1
            j += 1


def _encoder(ffmpeg: str = "ffmpeg", *, pix_fmt=None, width=0, height=0,
             frames=None, where=".") -> List[str]:
    """What to write the smoothed copy with.

    Lossless where it fits, in the source's own pixel format. The interpolated
    file is what the converter estimates depth from, so a qp-16 version of it
    put compression underneath the depth as well as in the picture. See
    stereo360/intermediate.py.
    """
    from . import intermediate

    args, _ = intermediate.choose(ffmpeg, pix_fmt=pix_fmt, width=width,
                                  height=height, frames=frames, where=where)
    return args


def run(src: str, dst: str, *, info=None, fps: Optional[float] = None,
        model: Optional[str] = None, provider: Optional[str] = None,
        start: int = 0, count: Optional[int] = None,
        reporter=None, cancel=None, ffmpeg: str = "ffmpeg") -> int:
    """Smooth `src` into `dst` before anything else runs. Returns the frames.

    A pass over the file rather than a filter inside the render, and that is
    the whole design decision here. Interpolating as the renderer pulls frames
    is nicer to wait for -- work starts at once, stopping keeps what was
    rendered, no intermediate exists -- and it was *much* slower, because the
    interpolator then spends the whole render competing with the encoder for
    the machine. Measured over 59 output frames of 8K:

                              streamed    pre-pass
        libx264 (default)       1046 s       139 s
        hevc_nvenc               722 s       157 s

    In isolation an 8K frame takes 0.94 s to invent; inside a render sharing
    the machine with x264 it took 14. Two phases that each get the machine
    beat one phase where they fight over it, by four to seven times, and no
    amount of nicety is worth that.

    `start` and `count` are *output* frames, so the intermediate holds exactly
    the range asked for and the renderer reads it from the beginning.
    """
    if info is None:
        from . import ffmpeg_io

        info = ffmpeg_io.probe(src)
    streamer = Streamer(info.fps, fps, model=model, provider=provider,
                        reporter=reporter)
    streamer.announce()

    skip, take = streamer.window(start, count)
    total = count if count is not None else streamer.total(info.frame_count)

    # The original alongside, for its audio: the converter reads this file
    # rather than the one chosen, so a track dropped here is dropped from the
    # finished render. Only when the range starts at zero -- further in, the
    # audio would no longer line up with the frames, and a test render is not
    # the place to guess at an offset.
    keep_audio = start == 0 and info.has_audio
    enc = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{info.width}x{info.height}",
         "-r", f"{streamer.target_fps:g}", "-i", "-"]
        + (["-i", src, "-map", "0:v", "-map", "1:a?", "-c:a", "copy"]
           if keep_audio else [])
        + _encoder(ffmpeg, pix_fmt=info.pix_fmt, width=info.width,
                   height=info.height, frames=total, where=dst) + [dst],
        stdin=subprocess.PIPE, **NO_CONSOLE_WINDOW)

    if reporter is not None:
        reporter.start(total, stage="interpolate")
    made = 0
    source = _decode(src, info.width, info.height, skip, take, ffmpeg)
    try:
        pairs = ((frame, None) for frame in source)
        for frame, _ in streamer.stream(pairs, first_output=start,
                                        first_source=skip, count=count):
            enc.stdin.write(frame.tobytes())
            made += 1
            if reporter is not None:
                reporter.advance(1)
            if cancel is not None and cancel():
                raise InterpolateError("cancelled")
    finally:
        source.close()
        try:
            enc.stdin.close()
        except OSError:
            pass
        code = enc.wait()
        if reporter is not None:
            reporter.finish(stage="interpolate")
        # Let the model go before the renderer loads its own. Waiting for the
        # collector is not good enough: this session holds a working set on
        # the GPU for the whole of an 8K render otherwise, and the render that
        # follows is competing with it for memory it will never use again.
        streamer.close()
        del streamer
        gc.collect()
    if code != 0 or not os.path.exists(dst):
        raise InterpolateError(f"the encoder exited {code} writing {dst}")
    return made


def _decode(path: str, width: int, height: int, skip: int,
            take: Optional[int], ffmpeg: str = "ffmpeg"):
    """Frames of `path` as uint8 HWC, from `skip`, at most `take` of them.

    `-fps_mode passthrough` matters: a clip whose first timestamp is not zero
    -- which a seek leaves behind -- otherwise gets its first frame duplicated
    to pad the gap, and every frame after that is off by one.
    """
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", path, "-fps_mode", "passthrough"]
    # One trim carrying both bounds, not two chained. Chained, the second
    # counts from the first one's output rather than from the source, so
    # `start_frame=50,end_frame=56` reads 56 frames *after* the 50 rather than
    # stopping at 56. Harmless here -- the stream stops at `count` either way
    # -- but it decodes most of the file to render a handful of frames.
    bounds = ([f"start_frame={skip}"] if skip else []) + \
             ([f"end_frame={skip + take}"] if take else [])
    if bounds:
        cmd += ["-vf", f"trim={':'.join(bounds)},setpts=PTS-STARTPTS"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, **NO_CONSOLE_WINDOW)
    n = width * height * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                return
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        # Killed rather than waited for when the caller stops early, which a
        # cancel does. Closing the pipe alone leaves the decoder writing into
        # it and filling the log with broken-pipe errors on its way out.
        try:
            p.stdout.close()
        except OSError:
            pass
        if p.poll() is None:
            p.kill()
        p.wait()
