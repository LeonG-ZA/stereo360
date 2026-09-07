"""ArtCNN's ONNX builds, which upscale luma and nothing else.

Every other graph in `models/` takes three channels and hands back three.
ArtCNN's do not: they are `[1, 1, H, W]` in and `[1, 1, 2H, 2W]` out, because
the GLSL versions are `//!HOOK LUMA` shaders and the ONNX exports are the same
network. Feeding one an RGB frame does not fail loudly -- onnxruntime rejects
the shape, but only after the run has started and a pre-pass is under way.

So the frame is split the way the shader path splits it: luma through the
model, chroma resampled, recombined. That is not a shortcut. It is what the
shader does inside libplacebo, and measuring the ONNX build any other way
would compare two different pipelines and call the difference a model.

Measured against the same twelve frames as the rest of the table, on an
RTX 5070 Ti through DirectML:

    model            luma dB   SSIM   temporal   amplify   s/frame
    FSRCNNX 16         36.61  0.980       102%     1.10x      0.30
    ArtCNN R8F64       38.47  0.985       102%     1.09x      3.73

The second row is the reason this module exists: it is the only model here
that resolves more than the shader *and* amplifies less. Sharpness normally
costs steadiness -- Siax buys 4.24x amplification for its detail -- and this
one does not, which is what makes it safe in front of video despite being
twelve times the shader's cost.
"""

from __future__ import annotations

import numpy as np

#: BT.709 full range, which is what `format=yuv420p` hands libplacebo for HD
#: material -- so the luma this model sees is the luma the shader sees.
_M = np.array([[0.2126, 0.7152, 0.0722],
               [-0.1146, -0.3854, 0.5000],
               [0.5000, -0.4542, -0.0458]], np.float32)
_INV = np.linalg.inv(_M).astype(np.float32)

#: A whole 8K luma plane is 29.5 Mpx and DirectML returns corrupted tensors
#: above 2**22. 768 with 32 of overlap puts an output tile at 2.77 Mpx, which
#: leaves room rather than sitting on the line.
_TILE = 768
_OVERLAP = 32


def is_luma_model(sess) -> bool:
    """Whether this graph wants one channel rather than three.

    Anything that cannot say is treated as three, which is what every model
    here was before ArtCNN arrived. Real onnxruntime sessions always carry a
    shape; test doubles and future stand-ins need not, and a missing
    attribute is not a reason to take the newer path.
    """
    inputs = getattr(sess, "get_inputs", None)
    shape = getattr(inputs()[0], "shape", None) if inputs else None
    return bool(shape) and len(shape) == 4 and shape[1] == 1


def to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32) @ _M.T
    out[..., 1:] += 128.0
    return out


def to_rgb(ycc: np.ndarray) -> np.ndarray:
    out = ycc.copy()
    out[..., 1:] -= 128.0
    return np.clip(out @ _INV.T, 0, 255)


def _resized(cv2, tile, want_w: int, want_h: int):
    """`tile` at the wanted size, with the filter that suits the direction.

    INTER_AREA is a shrinking filter -- it averages source pixels into each
    output one, which is right for a 4x graph asked for 2x and wrong for a 2x
    graph asked for 4x, where it behaves about like nearest neighbour. Asking
    it to enlarge is what put Compact ldl below plain Lanczos on a 2K source.
    """
    if (tile.shape[1], tile.shape[0]) == (want_w, want_h):
        return tile                      # the graph already did exactly this
    shrinking = want_w * want_h < tile.shape[1] * tile.shape[0]
    return cv2.resize(tile, (want_w, want_h),
                      interpolation=cv2.INTER_AREA if shrinking
                      else cv2.INTER_LANCZOS4)


def upscale(sess, frame: np.ndarray, scale: float = 2.0,
            tile: int = _TILE, overlap: int = _OVERLAP) -> np.ndarray:
    """`frame` at `scale` times its size, luma through the model."""
    import cv2

    if scale <= 0:
        raise ValueError(f"scale must be positive, not {scale:g}")
    name = sess.get_inputs()[0].name
    ycc = to_ycbcr(frame)
    h, w = frame.shape[:2]
    oh, ow = int(round(h * scale)), int(round(w * scale))
    out_y = np.empty((oh, ow), np.float32)

    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, xa = max(0, y0 - overlap), max(0, x0 - overlap)
            yb, xb = min(h, y1 + overlap), min(w, x1 + overlap)
            win = ycc[ya:yb, xa:xb, 0] / 255.0
            big = sess.run(None, {name: win[None, None].astype(np.float32)}
                           )[0][0, 0] * 255.0
            # Bring the tile to the size actually asked for *before* cutting
            # its share out of it. Slicing at the graph's own scale and
            # resizing afterwards is only right when the two agree: asked for
            # 4x from a 2x graph it cut a window twice too large out of a
            # tile half the size, and the frame came back at 17.6 dB.
            big = _resized(cv2, big, int(round((xb - xa) * scale)),
                           int(round((yb - ya) * scale)))
            top = int(round((x0 - xa) * scale))
            oy0, ox0 = int(round(y0 * scale)), int(round(x0 * scale))
            oy1, ox1 = int(round(y1 * scale)), int(round(x1 * scale))
            up = int(round((y0 - ya) * scale))
            core = big[up:up + (oy1 - oy0), top:top + (ox1 - ox0)]
            out_y[oy0:oy1, ox0:ox1] = core

    chroma = cv2.resize(ycc[..., 1:], (ow, oh),
                        interpolation=cv2.INTER_LANCZOS4)
    return to_rgb(np.dstack([out_y, chroma])).astype(np.uint8)
