"""Real-ESRGAN upscaling for stills, on the runtime stereo360 already ships.

For photos only, and that is a measured decision rather than a limitation.
Run against the same experiment as the Topaz models -- real 8K frames halved,
put back, and scored against what was really there -- it is the least stable
option there is once anything moves:

    model              PSNR   SSIM   temporal   s/frame
    Lanczos           36.26  0.980        87%      0.06
    Artemis HQ        27.81  0.911       104%      1.53
    Medium Halo       27.39  0.898       102%      1.53
    Artemis MQ        27.81  0.907       122%      0.79
    Artemis LQ        27.92  0.913       126%      1.56
    Real-ESRGAN x4v3  26.10  0.920       135%      2.60

`temporal` is the sequence's own frame-to-frame change as a percentage of the
real footage's. 135% means it invents a third more movement than the scene
has, which in a headset reads as texture crawling over everything while your
head is still. That is what a per-frame model does to video: each frame
invents its detail with no knowledge of the last one, so the detail changes
even where the picture does not.

None of that applies to a photograph. There is no next frame to disagree
with, and on the two columns that still mean something it does well -- 0.920
SSIM is ahead of every Artemis variant in that table. So it is offered for
stills and refused for video, which also gives a machine with no Topaz on it
something real to upscale photos with.

The model is `realesr-general-x4v3`, the small SRVGGNetCompact variant, under
BSD-3-Clause. It is 4x and the usual job is 2x, so tiles are taken down to
the wanted size as they come back -- which also means a 16K frame never has
to exist: at 8K output the intermediate would be 15360x7680.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import numpy as np

from .ffmpeg_io import NO_CONSOLE_WINDOW

#: Alongside the other graphs, and fetched the same way.
DEFAULT_MODEL = os.path.join("models", "realesr-general-x4v3.onnx")

#: What --upscale takes to mean this rather than a Topaz model.
CODE = "esrgan"
NAME = "Real-ESRGAN (photos)"
DESC = ("For photos only: on video it invents a third more movement than "
        "the scene has, which reads as crawling.")

#: What the graph does, whatever is asked for.
NATIVE_SCALE = 4

#: Input core and the context around it. 480 in is 1920 out, which keeps the
#: largest tensor under the 2^22 elements DirectML silently corrupts past.
_TILE = 480
_OVERLAP = 32

_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider",
              "CoreMLExecutionProvider", "CPUExecutionProvider")

_SCALE_255 = np.float32(255.0)


class EsrganError(RuntimeError):
    """Upscaling could not be done. The message is for the user."""


def model_path(explicit: Optional[str] = None) -> str:
    return explicit or DEFAULT_MODEL


def available(explicit: Optional[str] = None) -> bool:
    """Whether this machine can upscale a still: the graph and a runtime."""
    if not os.path.exists(model_path(explicit)):
        return False
    try:
        import onnxruntime                                    # noqa: F401
    except ImportError:
        return False
    return True


def describe(explicit: Optional[str] = None) -> dict:
    """What the interface needs to decide whether to offer this."""
    path = model_path(explicit)
    if not os.path.exists(path):
        return {"available": False, "model": path,
                "reason": f"no Real-ESRGAN model at {path}"}
    try:
        import onnxruntime                                    # noqa: F401
    except ImportError:
        return {"available": False, "model": path,
                "reason": "onnxruntime is not installed"}
    return {"available": True, "model": path, "reason": "", "stills_only": True}


def _session(path: str, provider: Optional[str] = None):
    import onnxruntime as ort

    have = set(ort.get_available_providers())
    order = [p for p in _PROVIDERS if p in have]
    if provider:
        if provider not in have:
            raise EsrganError(
                f"onnxruntime provider {provider!r} is not available here; "
                f"this build has {sorted(have)}")
        order = [provider] + [p for p in order if p != provider]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(path, opts, providers=order)
    except Exception as e:                                    # noqa: BLE001
        raise EsrganError(f"could not load {path}: {e}") from e
    return sess, sess.get_providers()[0]


def _window(img: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """`img[y0:y1, x0:x1]`, wrapped sideways and clamped at the poles.

    A 360 photo has the same seam a frame does: the column before the first is
    the last one, and a tile that does not know it invents an edge that is not
    there. Sliced rather than gathered, so an interior tile is a view.
    """
    h, w = img.shape[:2]
    ya, yb = max(y0, 0), min(y1, h)
    if 0 <= x0 and x1 <= w:
        got = img[ya:yb, x0:x1]
    elif x0 < 0:
        got = np.concatenate((img[ya:yb, w + x0:], img[ya:yb, :x1]), axis=1)
    else:
        got = np.concatenate((img[ya:yb, x0:], img[ya:yb, :x1 - w]), axis=1)
    top, bottom = ya - y0, y1 - yb
    if top or bottom:
        got = np.pad(got, ((top, bottom), (0, 0), (0, 0)), mode="edge")
    return got


def upscale(sess, frame: np.ndarray, scale: float = 2.0,
            tile: int = _TILE, overlap: int = _OVERLAP) -> np.ndarray:
    """`frame` at `scale` times its size, a tile at a time.

    The graph is 4x, so each tile is resampled down to the wanted size as it
    comes back rather than after -- a whole 4x frame of an 8K photo is 354 MB
    that nothing ever needs.
    """
    import cv2

    if scale <= 0:
        raise EsrganError(f"scale must be positive, not {scale:g}")
    name = sess.get_inputs()[0].name
    h, w = frame.shape[:2]
    out_h, out_w = int(round(h * scale)), int(round(w * scale))
    out = np.empty((out_h, out_w, 3), np.uint8)
    buffers: dict = {}
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, xa = y0 - overlap, x0 - overlap
            win = _window(frame, ya, y1 + overlap, xa, x1 + overlap)
            th, tw = win.shape[:2]
            x = buffers.get((th, tw))
            if x is None:
                x = buffers[(th, tw)] = np.empty((1, 3, th, tw), np.float32)
            np.divide(win.transpose(2, 0, 1), _SCALE_255, out=x[0])

            big = sess.run(None, {name: x})[0][0].transpose(1, 2, 0)
            np.clip(big, 0.0, 1.0, out=big)
            # The tile's own share of the output, rounded the same way the
            # whole frame was, so the pieces tile the result exactly.
            ox0, oy0 = int(round(x0 * scale)), int(round(y0 * scale))
            ox1, oy1 = int(round(x1 * scale)), int(round(y1 * scale))
            sized = cv2.resize(
                big, (int(round(tw * scale)), int(round(th * scale))),
                interpolation=cv2.INTER_AREA)
            top = int(round((y0 - ya) * scale))
            left = int(round((x0 - xa) * scale))
            core = sized[top:top + (oy1 - oy0), left:left + (ox1 - ox0)]
            out[oy0:oy1, ox0:ox1] = (core * 255.0 + 0.5).astype(np.uint8)
    return out


def run_still(src: str, dst: str, *, scale: float = 2.0,
              model: Optional[str] = None, provider: Optional[str] = None,
              reporter=None, ffmpeg: str = "ffmpeg") -> tuple:
    """Upscale one image into `dst`. Returns the size written.

    Read and written through ffmpeg rather than an image library, so whatever
    the converter can open -- HEIC and AVIF included -- this can too.
    """
    path = model_path(model)
    if not os.path.exists(path):
        raise EsrganError("\n".join((
            f"Real-ESRGAN model not found: {path}",
            "It is not shipped with the repository. Fetch it once:",
            "    python scripts/fetch_esrgan.py")))

    from . import ffmpeg_io

    info = ffmpeg_io.probe(src)
    sess, chosen = _session(path, provider)
    if reporter is not None:
        reporter.info(
            f"Upscaling the photo: {NAME} on {chosen}, {scale:g}x from "
            f"{info.width}x{info.height}", stage="upscale", model=CODE)

    read = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
         "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, **NO_CONSOLE_WINDOW)
    want = info.width * info.height * 3
    if read.returncode != 0 or len(read.stdout) < want:
        raise EsrganError(f"could not read {src}")
    frame = np.frombuffer(read.stdout[:want], np.uint8).reshape(
        info.height, info.width, 3)

    got = upscale(sess, frame, scale)
    write = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{got.shape[1]}x{got.shape[0]}", "-i", "-", dst],
        stdin=subprocess.PIPE, **NO_CONSOLE_WINDOW)
    write.stdin.write(got.tobytes())
    write.stdin.close()
    if write.wait() != 0 or not os.path.exists(dst):
        raise EsrganError(f"could not write {dst}")
    return got.shape[1], got.shape[0]
