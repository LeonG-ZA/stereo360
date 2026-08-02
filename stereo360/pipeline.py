"""Conversion pipeline: decode -> depth -> DIBR -> top-bottom stack -> encode.

With no depth backend, the right eye is the left eye passed through the
cubemap round-trip (M1 passthrough mode). With a depth backend, per-face
depth is estimated, assembled into an equirect inverse-depth map, and the
right eye is synthesized via DIBR warping with inpainting (M2).
"""

from __future__ import annotations

import os
from typing import Callable, NamedTuple, Optional

import numpy as np

from . import ffmpeg_io, projection, spherical, warp
from .depth.base import DepthBackend
from .events import Cancelled, Reporter


def right_eye_passthrough(frame: np.ndarray, face_size: int) -> np.ndarray:
    """Round-trip a frame through the cubemap stage (M1 placeholder mode)."""
    h, w = frame.shape[:2]
    faces = projection.equirect_to_cubemap(frame, face_size)
    return projection.cubemap_to_equirect(faces, w, h)


def stereo_pair(
    frame: np.ndarray,
    disp: np.ndarray,
    strength: float,
    split: bool,
    **kw,
) -> tuple:
    """Return (left, right) for one frame given its inverse-depth map.

    Without `split` the left eye is the untouched source and the right eye
    carries the whole baseline. That is the worst arrangement for
    disocclusion: hole area grows far faster than linearly with warp distance
    (measured on 8K footage, roughly cubic -- 0.002% of the frame at strength
    0.3 against 0.071% at 1.0), so putting the entire baseline in one eye
    maximises it.

    With `split` each eye is warped half as far in opposite directions. The
    disparity between the eyes -- and therefore the depth effect -- is
    unchanged, but each eye's holes shrink by about 8x, and they land in
    *different places* in each eye, so wherever one eye has a hole the other
    has real content and binocular fusion suppresses it. The cost is that the
    left eye is no longer pristine.
    """
    if not split:
        right, _ = warp.right_eye_from_disparity(frame, disp, strength, **kw)
        return frame, right
    half = strength * 0.5
    left, _ = warp.right_eye_from_disparity(frame, disp.copy(), -half, **kw)
    right, _ = warp.right_eye_from_disparity(frame, disp, half, **kw)
    return left, right


class DepthRange:
    """A disparity normalisation range that moves slowly, not per frame.

    Depth models return *relative* depth, so the map has to be normalised
    before it can be a disparity. Recomputing the percentiles from scratch on
    every frame means the whole field is rescaled every frame: measured on 8K
    footage the 1st-99th percentile span swung from 11.7 to 14.1 across six
    consecutive frames, a 20% change. Scaling every disparity by 20% moves
    every stationary object horizontally, which is seen as the image wobbling
    left and right frame to frame.

    The chunked path already avoided this by sharing one range across a chunk
    (see `_chunk_normalize`); this is the streaming equivalent for the
    single-frame path, which never got the fix. An exponential average rather
    than a fixed range so a genuine scene change is still followed, just over
    a second or so instead of instantly.
    """

    #: ~1s at 30fps: slow enough to kill per-frame jitter, fast enough that a
    #: cut or a new subject is tracked without a visible drift.
    DEFAULT_ALPHA = 0.05

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha
        self.lo: Optional[float] = None
        self.hi: Optional[float] = None

    def update(self, depth: np.ndarray) -> tuple:
        """Fold this frame's percentiles in and return the range to use."""
        sample = depth[::16, ::16]
        lo = float(np.percentile(sample, 1))
        hi = float(np.percentile(sample, 99))
        if self.lo is None:
            self.lo, self.hi = lo, hi
        else:
            self.lo += self.alpha * (lo - self.lo)
            self.hi += self.alpha * (hi - self.hi)
        return self.lo, self.hi


class DepthStabiliser:
    """Hold static depth still between frames, without smearing motion.

    The chunked path already does this with `temporal_fill.stabilize_depth`,
    which blends each pixel toward its chunk median. The single-frame path is
    streaming and has no future frames, so this is the causal equivalent: each
    pixel is pulled toward its own previous value by an amount that falls to
    zero as the pixel starts actually moving.

        w = cap * clip((2*tau - |d - prev|) / tau, 0, 1)
        d <- w*prev + (1 - w)*d

    Two constants, both chosen by measurement rather than taste:

    `tau` matches stabilize_depth's, so the two paths agree about what counts
    as "not moving". At tau=0.02 the frame-to-frame change of depth at fixed
    points on real footage fell to 45% of unfiltered.

    `cap` is why the weight never reaches 1. With w=1 a pixel is replaced
    outright by its history and can never update, so content drifting slower
    than tau per frame -- a slow pan, a cloud -- would freeze at its first
    value forever. Measured on a 0.001/frame drift, an uncapped filter lagged
    by 33-50% of the total movement; capped at 0.9 it lags 15%, a 10-frame
    time constant.

    Costs one extra full-resolution depth map in memory (118 MB at 8K).
    """

    DEFAULT_TAU = 0.02
    CAP = 0.9

    def __init__(self, tau: float = DEFAULT_TAU, cap: float = CAP) -> None:
        self.tau = float(tau)
        self.cap = float(cap)
        self._prev: Optional[np.ndarray] = None

    def apply(self, depth: np.ndarray) -> np.ndarray:
        """Smooth `depth` in place against the previous frame."""
        if self.tau <= 0:
            return depth
        prev = self._prev
        if prev is None or prev.shape != depth.shape:
            self._prev = depth.copy()
            return depth
        delta = np.abs(depth - prev)
        w = np.clip((2.0 * self.tau - delta) / self.tau, 0.0, 1.0)
        w *= self.cap
        # depth += w * (prev - depth), in place to avoid a second full frame.
        np.subtract(prev, depth, out=delta)
        np.multiply(w, delta, out=w)
        np.add(depth, w, out=depth)
        self._prev = depth.copy()
        return depth


def right_eye_from_depth(
    frame: np.ndarray,
    face_size: int,
    backend: DepthBackend,
    strength: float,
    fg_erode: int = 2,
    inpaint_mode: str = "simple",
    depth_tiles: int = 1,
    split_baseline: bool = False,
    gradient_limit: float = 0.0,
    faces: Optional[dict] = None,
    depth_range: Optional["DepthRange"] = None,
    stabiliser: Optional["DepthStabiliser"] = None,
    face_overlap: float = projection.FACE_OVERLAP,
) -> tuple:
    """M2 path: per-face depth estimation -> equirect inverse depth -> DIBR.

    `depth_range` carries the normalisation range between frames. Without it
    each frame normalises itself and the result wobbles; see `DepthRange`.
    """
    disp = depth_map_for_frame(frame, face_size, backend, depth_tiles, faces,
                               face_overlap)
    extra = {}
    if depth_range is not None:
        lo, hi = depth_range.update(disp)
        warp.normalize_inv_depth_with(disp, lo, hi)
        # Only after normalising: the stabiliser compares against the previous
        # frame, so both have to be on the same scale to be comparable.
        if stabiliser is not None:
            disp = stabiliser.apply(disp)
        extra["normalize"] = False
    return stereo_pair(frame, disp, strength, split_baseline,
                       fg_erode=fg_erode, inpaint_mode=inpaint_mode,
                       gradient_limit=gradient_limit, **extra)


def _tile_boxes(face_size: int, split: int, overlap: float = 0.125) -> list:
    """(y0, y1, x0, x1) boxes splitting a face into split x split tiles with
    fractional overlap on interior edges."""
    stride = face_size // split
    pad = int(stride * overlap)
    boxes = []
    for ty in range(split):
        for tx in range(split):
            y0 = max(0, ty * stride - pad)
            x0 = max(0, tx * stride - pad)
            y1 = min(face_size, (ty + 1) * stride + pad)
            x1 = min(face_size, (tx + 1) * stride + pad)
            # Last tile extends to the face edge if stride doesn't divide evenly.
            if ty == split - 1:
                y1 = face_size
            if tx == split - 1:
                x1 = face_size
            boxes.append((y0, y1, x0, x1))
    return boxes


def _taper(n: int, pad: int, left: bool, right: bool) -> np.ndarray:
    """1D feather weights: 1 in the center, cosine ramp over `pad` at the
    ends flagged for tapering."""
    w = np.ones(n, dtype=np.float32)
    if pad > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(pad) / pad)
        if left and n > pad:
            w[:pad] = ramp
        if right and n > pad:
            w[-pad:] = ramp[::-1]
    return w


def _tile_weight(box: tuple, face_size: int, split: int) -> np.ndarray:
    """2D feather window for a tile; edges on the face border are NOT tapered
    (no neighbor to blend with there — they keep weight 1)."""
    y0, y1, x0, x1 = box
    stride = face_size // split
    pad = int(stride * 0.125)
    wy = _taper(y1 - y0, pad, left=y0 > 0, right=y1 < face_size)
    wx = _taper(x1 - x0, pad, left=x0 > 0, right=x1 < face_size)
    return wy[:, None] * wx[None, :]


def estimate_tiled(backend: DepthBackend, face_imgs: list, split: int) -> list:
    """Depth for a sequence of same-face images, estimated per overlapping
    tile and feather-blended. face_imgs: K (F, F, 3) uint8. Returns K (F, F)
    float32 depth maps.

    Each tile is inferred independently, so relative-depth backends give
    each tile its own scale. A coarse full-face pass provides a reference
    and every tile is scale-aligned to it (least squares) before blending.
    """
    f = face_imgs[0].shape[0]
    k = len(face_imgs)
    ref = backend.estimate_chunk(face_imgs)  # coarse scale reference
    acc = [np.zeros((f, f), np.float32) for _ in range(k)]
    wacc = np.zeros((f, f), np.float32)
    for box in _tile_boxes(f, split):
        y0, y1, x0, x1 = box
        tiles = [img[y0:y1, x0:x1] for img in face_imgs]
        depths = backend.estimate_chunk(tiles)
        w = _tile_weight(box, f, split)
        for i, d in enumerate(depths):
            d = d.astype(np.float32)
            r = ref[i][y0:y1, x0:x1]
            denom = float((d * d).sum())
            if denom > 1e-12:
                s = float((d * r).sum()) / denom
                s = min(max(s, 0.25), 4.0)
                d = d * s
            acc[i][y0:y1, x0:x1] += d * w
        wacc[y0:y1, x0:x1] += w
    np.maximum(wacc, 1e-6, out=wacc)
    return [a / wacc for a in acc]


def source_faces_for_depth(frame: np.ndarray, face_size: int,
                           faces: Optional[dict],
                           face_overlap: float) -> dict:
    """The six face images to run depth on.

    `faces` short-circuits the projection when the *source* was already a
    cubemap: estimating depth on the file's own faces avoids resampling it to
    equirect and straight back again, which is two interpolations the
    geometry never needed. Widening those faces is still a single resample,
    so the short-circuit survives the overlap.
    """
    if face_overlap <= 0.0:
        return (projection.equirect_to_cubemap(frame, face_size)
                if faces is None else faces)
    if faces is not None:
        return projection.cubemap_to_overlapping_faces(faces, face_overlap)
    return projection.equirect_to_overlapping_faces(frame, face_size,
                                                    face_overlap)


def assemble_depth(disp_faces: dict, w: int, h: int,
                   face_overlap: float) -> np.ndarray:
    """Reconcile six per-face scales and lay them back onto an equirect map.

    Faces are inferred independently, so each carries its own arbitrary
    relative-depth scale. Overlapping faces are fitted over the band they
    share and cross-faded across it; exact faces have only their shared edge
    to work with, and reassemble through a neighbour-padded atlas so bilinear
    taps at a face border read the adjacent face rather than an unrelated
    atlas block.
    """
    if face_overlap > 0.0:
        projection.align_overlapping_faces(disp_faces, face_overlap)
        return projection.overlapping_faces_to_equirect(disp_faces, w, h,
                                                        face_overlap)
    projection.align_face_scales(disp_faces)
    return projection.cubemap_to_equirect(disp_faces, w, h)[..., 0]


def depth_map_for_frame(
    frame: np.ndarray,
    face_size: int,
    backend: DepthBackend,
    depth_tiles: int = 1,
    faces: Optional[dict] = None,
    face_overlap: float = projection.FACE_OVERLAP,
) -> np.ndarray:
    """Single-frame equirect inverse-depth map via the cubemap stage."""
    h, w = frame.shape[:2]
    faces = source_faces_for_depth(frame, face_size, faces, face_overlap)
    if depth_tiles > 1:
        disp_faces = {f: estimate_tiled(backend, [faces[f]], depth_tiles)[0]
                      for f in projection.FACES}
    else:
        # One batched call, not six single-image ones: a lone 518x518 forward
        # pass leaves a discrete GPU mostly idle, and transformers warns about
        # exactly this ("using the pipelines sequentially on GPU").
        depths = backend.estimate_chunk([faces[f] for f in projection.FACES])
        disp_faces = dict(zip(projection.FACES, depths))
    del faces  # free ~66 MB at 8K before assembling the disparity map
    return assemble_depth(disp_faces, w, h, face_overlap)


def depth_maps_for_chunk(
    frames: list,
    face_size: int,
    backend: DepthBackend,
    depth_tiles: int = 1,
    face_sets: Optional[list] = None,
    face_overlap: float = projection.FACE_OVERLAP,
) -> list:
    """M3 path: estimate equirect inverse depth for a chunk of consecutive
    frames, using the backend's temporal chunk API per cubemap face.

    `face_sets` supplies the source's own faces per frame; see
    `source_faces_for_depth`.
    """
    h, w = frames[0].shape[:2]
    sets = face_sets if face_sets is not None else [None] * len(frames)
    face_imgs = [source_faces_for_depth(fr, face_size, s, face_overlap)
                 for fr, s in zip(frames, sets)]
    per_face: dict = {}
    for f in projection.FACES:
        seq = [imgs[f] for imgs in face_imgs]
        if depth_tiles > 1:
            per_face[f] = estimate_tiled(backend, seq, depth_tiles)
        else:
            per_face[f] = backend.estimate_chunk(seq)
        for imgs in face_imgs:
            imgs[f] = None  # release uint8 face storage as we go
    del face_imgs
    maps = []
    for i in range(len(frames)):
        # Per-face scales drift independently frame to frame, so every frame
        # gets its own reconciliation rather than one for the chunk.
        maps.append(assemble_depth({f: per_face[f][i] for f in projection.FACES},
                                   w, h, face_overlap))
    return maps


def _chunk_normalize(maps: list) -> None:
    """Normalize all depth maps of a chunk with ONE shared range, in place.

    Per-frame percentile normalization rescales each frame's disparity
    slightly differently, which makes the entire right eye "pump" even with
    temporally consistent depth (a moving close object shifts the 99th
    percentile every frame). Percentiles are computed over a subsample of all
    chunk maps for speed.
    """
    sample = np.concatenate([m[::16, ::16].ravel() for m in maps])
    lo, hi = np.percentile(sample, 1), np.percentile(sample, 99)
    for m in maps:
        warp.normalize_inv_depth_with(m, lo, hi)


def _blend_overlap(prev: list, cur: list) -> list:
    """Linear-ramp blend of overlapping depth maps at a chunk boundary,
    favoring the estimate from the later chunk (more temporal context)."""
    n = len(prev)
    out = []
    for i, (a, b) in enumerate(zip(prev, cur)):
        w = (i + 1) / (n + 1)
        out.append((1.0 - w) * a + w * b)
    return out


# Live bytes per frame held by a chunk: the source frame, its depth map, the
# pre-erosion copy, the warped eye and its hole mask, measured at 8K.
_CHUNK_BYTES_PER_PX = 14


def fit_chunk_size(chunk_size: int, w: int, h: int, eyes: int = 1,
                   reporter: Optional[Reporter] = None) -> int:
    """Reduce the chunk if it would not fit in the memory actually free.

    A chunk buffers whole frames, so its cost scales with resolution as well
    as length: 8 frames of 8K is around 3 GB, which is fine on a workstation
    and fatal on a laptop. Rather than pick one number for everyone, the
    default is capped to what the machine has, and says so when it does.
    """
    from .warp import available_memory

    if chunk_size <= 1:
        return chunk_size
    avail = available_memory()
    if avail is None:
        return chunk_size
    per_frame = w * h * _CHUNK_BYTES_PER_PX * max(eyes, 1)
    fits = int((avail * 0.4) // max(per_frame, 1))
    safe = max(2, min(chunk_size, fits))
    if safe < chunk_size:
        (reporter or Reporter()).info(
            f"Reducing --chunk-size {chunk_size} -> {safe}: {w}x{h} needs "
            f"~{per_frame / 2**20:.0f} MB per buffered frame and only "
            f"{avail / 2**30:.1f} GB is free.",
            chunk_size=safe, requested_chunk_size=chunk_size,
            available_bytes=avail)
    return safe


#: Output chroma unless the caller asks to follow the source. 4:2:0 is the only
#: layout headset hardware decoders handle at 8K, so it cannot be the thing a
#: user gives up by accident.
DEFAULT_CHROMA = "4:2:0"

#: How the two eyes are laid out, and how much of the sphere they cover.
#:
#: "360" is the original: a full equirect per eye, stacked top over bottom.
#: "vr180" keeps the middle 180 degrees of longitude and puts the eyes side by
#: side, which is what VR180 players expect.
#:
#: Only reachable with 360 input. The tool does not accept 180 input — VR180
#: cameras already shoot stereo, so a monoscopic half-equirect barely exists,
#: and cropping from a full sphere gives a better result than processing one
#: would: the depth stage sees six real faces and the warp has content beyond
#: the crop to shift in. See plans/vr180.md.
OUTPUT_MODES = ("360", "vr180")
DEFAULT_OUTPUT_MODE = "360"

#: Degrees of longitude a VR180 frame covers. Not a knob: the format is 180,
#: and the number appears here so the crop and the metadata cannot disagree.
VR180_FOV = 180.0


def check_yaw(output_mode: str, yaw: float) -> None:
    """A yaw only means something when part of the sphere is being discarded.

    For 360 output the whole sphere is kept, so a yaw would have to rotate it
    -- which is possible (rolling the columns of a full equirect is lossless)
    but is not what this flag does. Silently ignoring it would be worse than
    refusing: the render would take an hour and come out pointing the wrong way.
    """
    if yaw and output_mode != "vr180":
        raise ValueError(
            f"A yaw of {yaw:g} degrees was given with --output-mode "
            f"{output_mode}, which keeps the whole sphere and has no direction "
            f"to choose. Yaw applies to vr180 output only.")


def output_geometry(w: int, h: int, output_mode: str) -> tuple:
    """(width, height) of the encoded frame for a `w` x `h` equirect source.

    360 stacks two full equirects vertically. VR180 halves each eye
    horizontally and puts them side by side, so an 8K source gives 7680x3840
    either way round — the same pixel count, spent on half the sphere at twice
    the angular resolution.
    """
    _check_output_mode(output_mode)
    if output_mode == "vr180":
        half = (w // 2) // 2 * 2      # even, or the encoder rejects it
        return half * 2, h
    return w, h * 2


def _check_output_mode(output_mode: str) -> None:
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"Unknown output_mode {output_mode!r}; expected one "
                         f"of {list(OUTPUT_MODES)}")


def vr180_crop(w: int, yaw: float = 0.0) -> tuple:
    """(first column, width) of the 180-degree field centred on `yaw`.

    Yaw is degrees of longitude, positive to the right, and wraps, so any value
    is legal and -200 means the same as +160.

    Because this is a column range rather than a rotation, pointing the field
    somewhere else costs nothing and loses nothing: no resampling, no
    interpolation, no softening. That matters for the interface, where it means
    the direction can be dragged with live feedback once a frame has been
    rendered, without recomputing any depth.
    """
    half = (w // 2) // 2 * 2
    centre = (((yaw + 180.0) % 360.0) / 360.0) * w
    return int(round(centre - half / 2.0)) % w, half


def _crop_columns(a: np.ndarray, x0: int, width: int) -> np.ndarray:
    """`width` columns from `x0`, wrapping round the +/-180 degree seam."""
    end = x0 + width
    if end <= a.shape[1]:
        return a[:, x0:end]
    return np.concatenate([a[:, x0:], a[:, :end - a.shape[1]]], axis=1)


def stack_eyes(left: np.ndarray, right: np.ndarray,
               output_mode: str = DEFAULT_OUTPUT_MODE,
               yaw: float = 0.0) -> np.ndarray:
    """Combine the two eyes into the frame that gets encoded.

    The one place that decides layout, so the encoder's frame size, the written
    pixels and the spherical metadata cannot drift apart.
    """
    _check_output_mode(output_mode)
    if output_mode == "360":
        return np.concatenate([left, right], axis=0)

    x0, half = vr180_crop(left.shape[1], yaw)
    return np.concatenate([_crop_columns(left, x0, half),
                           _crop_columns(right, x0, half)],
                          axis=1)


def warn_if_source_is_deeper_than_8_bit(info, bitdepth: int,
                                        reporter: Reporter) -> bool:
    """Say so when the source carries more than 8 bits per component.

    The pipeline decodes to 8-bit RGB and hands 8-bit RGB to the encoder, so a
    10-bit source is flattened before any stage sees it. Measured on a 10-bit
    gradient, 161 distinct levels arrived and 35 came out.

    `--bitdepth 10` does not rescue it and is the more misleading case, because
    the file really is Main10 and looks like it kept something: the same test
    gave 41 levels, with a minimum step of one whole 8-bit code. So the warning
    is louder, not quieter, when 10-bit output was asked for.

    Dithering the reduction would recover the banding (measured: back to the
    10-bit figure) for 0-4% bitrate on real footage, but it is not implemented
    and is not a one-line change -- where in the pipeline it is applied decides
    whether the two eyes receive the same noise or independent noise, and
    independent noise between the eyes is the one thing VR punishes hardest.
    See "Future work: dithering a 10-bit source" in the README.

    Returns whether it warned, which is what the tests assert on.
    """
    depth = info.bit_depth
    if depth is None or depth <= 8:
        return False
    if bitdepth >= 10:
        reporter.warning(
            f"Source is {depth}-bit but the pipeline decodes to 8-bit RGB, so "
            f"the extra precision is dropped before anything sees it. "
            f"--bitdepth 10 still gives a real 10-bit file, but it holds "
            f"8-bit-quantised data -- it cannot bring back what decoding "
            f"discarded.",
            source_bit_depth=depth, output_bit_depth=bitdepth,
            precision_preserved=False)
    else:
        reporter.warning(
            f"Source is {depth}-bit; output will be 8-bit. The pipeline "
            f"decodes to 8-bit RGB, so the extra precision is lost here rather "
            f"than at the encoder -- most visible as banding in skies, fog and "
            f"water. --bitdepth 10 would give a 10-bit file but cannot recover "
            f"the source's precision.",
            source_bit_depth=depth, output_bit_depth=bitdepth,
            precision_preserved=False)
    return True


def _resolve_chroma(info, codec: str, follow_source: bool,
                    reporter: Reporter) -> str:
    """Pick the output chroma layout, saying what was decided and why.

    Following the source is a preference rather than a requirement: if the
    layout cannot be produced -- an exotic source format, or a hardware
    encoder -- the run continues at 4:2:0 with a warning. Failing a render
    because a checkbox could not be honoured would be the wrong trade.

    Worth knowing before ticking it: measured against a real 8K frame from
    this pipeline, 4:4:4 beat 4:2:0 by 4% (rmse 0.841 vs 0.873) at 10-bit and
    cost 25% more encode time, because the source was already 4:2:0 and its
    chroma detail was halved before we saw it. Going 8-bit -> 10-bit was worth
    2.2x by comparison.
    """
    if not follow_source:
        return DEFAULT_CHROMA

    source = info.chroma
    allowed = ffmpeg_io.supported_chroma(codec)
    if source is None:
        reporter.warning(
            f"Could not determine the source's chroma subsampling "
            f"(pix_fmt {info.pix_fmt!r}); encoding {DEFAULT_CHROMA}.",
            chroma=DEFAULT_CHROMA)
        return DEFAULT_CHROMA
    if source == DEFAULT_CHROMA:
        reporter.info(f"Chroma subsampling: {source} (matches the source)",
                      chroma=source, source_chroma=source)
        return source
    if source not in allowed:
        reporter.warning(
            f"Source is {source}, which {codec} cannot produce here "
            f"(supported: {', '.join(allowed)}); encoding {DEFAULT_CHROMA}.",
            chroma=DEFAULT_CHROMA, source_chroma=source)
        return DEFAULT_CHROMA
    reporter.info(f"Chroma subsampling: {source} (matching the source)",
                  chroma=source, source_chroma=source)
    return source


#: Projections we can read. Anything else declared -- mesh, or a layout we do
#: not know -- is reported rather than silently treated as equirectangular.
READABLE_PROJECTIONS = ("equirectangular", "cubemap")


class SourceFrame(NamedTuple):
    """A decoded frame as equirect, plus the source's own faces if it had any.

    Cubemap input carries its faces through instead of discarding them: the
    depth stage wants faces, and rebuilding them from a reconstructed equirect
    would resample twice for nothing.
    """

    equirect: np.ndarray
    faces: Optional[dict] = None


def resolve_projection(info, requested: str, reporter: Reporter) -> str:
    """What projection to treat the input as, and why.

    Files usually declare nothing, and an untagged file is overwhelmingly
    equirectangular -- that is the only projection YouTube accepts on upload
    and the only one Spherical Video V1 can express. So absence means
    "assume equirectangular", while a declaration we cannot read is worth
    stopping for: treating a cubemap as equirect produces output that is
    geometrically nonsense from the first frame.
    """
    if requested != "auto":
        if info.projection and info.projection != requested:
            reporter.warning(
                f"Input declares {info.projection!r} but --input-projection "
                f"says {requested!r}; using {requested!r}.",
                declared=info.projection, using=requested)
        return requested

    declared = info.projection
    if declared is None:
        return "equirectangular"
    if declared not in READABLE_PROJECTIONS:
        raise ValueError(
            f"Input declares a {declared!r} projection, which this tool "
            f"cannot read. Supported: {', '.join(READABLE_PROJECTIONS)}. "
            f"Convert it first (ffmpeg -vf v360=...=e) or override with "
            f"--input-projection if the tag is wrong.")
    if declared == "cubemap":
        reporter.info("Input is a 3x2 cubemap; using its own faces for depth.",
                      projection=declared, padding=info.cubemap_padding)
    return declared


def source_geometry(info, projection_name: str, face_size: Optional[int],
                    reporter: Reporter):
    """(face_size, equirect width, equirect height) for this input."""
    if projection_name == "cubemap":
        tile = info.width // 3 - 2 * info.cubemap_padding
        if face_size is not None and face_size != tile:
            reporter.warning(
                f"--face-size {face_size} ignored: cubemap input supplies "
                f"{tile}px faces, and resampling them would throw away the "
                f"reason to read the faces at all.", face_size=tile)
        # 4x the face size matches the cubemap's angular sampling at the
        # equator, the same rule the equirect path uses in reverse.
        return tile, tile * 4, tile * 2
    if face_size is None:
        face_size = max(1, info.width // 4)
        reporter.info(f"Auto face size: {face_size} "
                      f"(input {info.width}x{info.height})",
                      face_size=face_size, width=info.width,
                      height=info.height)
    return face_size, info.width, info.height


def read_source(input_path: str, info, projection_name: str, out_w: int,
                out_h: int, max_frames: Optional[int], start_frame: int):
    """Yield SourceFrame, converting cubemap input as it goes."""
    frames = ffmpeg_io.decode_frames(input_path, max_frames=max_frames,
                                     skip_frames=start_frame)
    if projection_name != "cubemap":
        for frame in frames:
            yield SourceFrame(frame, None)
        return
    for frame in frames:
        faces = projection.cubemap_3x2_to_faces(frame, info.cubemap_padding)
        yield SourceFrame(projection.cubemap_to_equirect(faces, out_w, out_h),
                          faces)


class ConvertResult(NamedTuple):
    """Outcome of `convert`. A cancelled run still produced a playable file."""

    output_path: str
    frames_written: int
    cancelled: bool


class _Sink:
    """Everything that happens per output frame, in one place.

    Stacking, encoding, counting, progress and the cancellation check all fire
    together and always in that order, so there is no path that writes a frame
    without reporting it or that checks for cancellation only sometimes.
    """

    def __init__(self, encoder, reporter: Reporter,
                 cancel: Optional[Callable[[], bool]],
                 output_mode: str = DEFAULT_OUTPUT_MODE,
                 yaw: float = 0.0) -> None:
        self._encoder = encoder
        self._reporter = reporter
        self._cancel = cancel
        self._output_mode = output_mode
        self._yaw = yaw
        self.written = 0

    def check(self) -> None:
        """Raise `Cancelled` if the caller has asked us to stop.

        Called before expensive work as well as before each write: a chunk's
        depth estimation is several seconds, and a Stop button that only
        responds on frame boundaries feels broken during it.
        """
        if self._cancel is not None and self._cancel():
            raise Cancelled()

    def write(self, left: np.ndarray, right: np.ndarray) -> None:
        self.check()
        self._encoder.write(
            stack_eyes(left, right, self._output_mode, self._yaw))
        self.written += 1
        self._reporter.advance(1)


def convert(
    input_path: str,
    output_path: str,
    face_size: Optional[int] = None,
    crf: int = 18,
    preset: str = "medium",
    codec: str = "libx264",
    bitdepth: int = 8,
    max_frames: Optional[int] = None,
    use_cubemap: bool = True,
    depth_backend: Optional[DepthBackend] = None,
    strength: float = 1.0,
    chunk_size: int = 1,
    chunk_overlap: int = 0,
    fg_erode: int = 2,
    start_frame: int = 0,
    inpaint_mode: str = "simple",
    temporal_fill: bool = False,
    depth_tiles: int = 1,
    split_baseline: bool = False,
    gradient_limit: float = 0.0,
    spatial_audio: bool = False,
    source_subsampling: bool = False,
    input_projection: str = "auto",
    temporal_depth: float = DepthStabiliser.DEFAULT_TAU,
    face_overlap: float = projection.FACE_OVERLAP,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    yaw: float = 0.0,
    reporter: Optional[Reporter] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> ConvertResult:
    """Convert `input_path` to a top-bottom stereo 360 MP4 at `output_path`.

    reporter: where progress and messages go. Defaults to silence, so using
              this as a library prints nothing unless asked.
    cancel:   called between frames; returning True stops the run cleanly.
              A `threading.Event`'s `.is_set` works directly. The output is
              finalized rather than discarded, so a cancelled run leaves a
              playable file containing the frames completed so far.
    """
    reporter = reporter or Reporter()
    info = ffmpeg_io.probe(input_path)

    # What the file says it is decides how it is read. Everything downstream
    # works in equirect, so cubemap input is converted once as it is decoded --
    # but its faces are carried along, because the depth stage wants faces and
    # rebuilding them from the reconstruction would resample twice for nothing.
    source_projection = resolve_projection(info, input_projection, reporter)
    face_size, w, h = source_geometry(info, source_projection, face_size,
                                      reporter)

    total = info.frame_count
    if total:
        total = max(0, total - start_frame)
    if max_frames is not None:
        total = min(total, max_frames) if total else max_frames

    audio_source = input_path if info.has_audio else None
    chroma = _resolve_chroma(info, codec, source_subsampling, reporter)
    warn_if_source_is_deeper_than_8_bit(info, bitdepth, reporter)

    _check_output_mode(output_mode)
    check_yaw(output_mode, yaw)
    out_w, out_h = output_geometry(w, h, output_mode)
    if output_mode != DEFAULT_OUTPUT_MODE:
        aim = (f"centred {yaw:+g} degrees from the source's forward direction"
               if yaw else "centred on the source's forward direction")
        reporter.info(
            f"Output: VR180, {out_w}x{out_h} side-by-side, {aim}.",
            output_mode=output_mode, width=out_w, height=out_h, yaw=yaw)
    if yaw and spatial_audio:
        # The picture turned; the soundfield did not. Loud rather than silent,
        # because a listener has no way to tell it is happening.
        reporter.warning(
            f"The view is turned {yaw:+g} degrees but the ambisonic soundfield "
            f"is NOT rotated to match, so every sound will be {abs(yaw):g} "
            f"degrees out of place -- and in a 180 field a source that should "
            f"be visible in front can end up behind you. Rotating ambiX is not "
            f"implemented yet; use --yaw 0, or drop --spatial-audio and treat "
            f"the track as plain audio.",
            yaw=yaw, ambisonic_rotation=False)

    encoder = ffmpeg_io.VideoEncoder(
        output_path, width=out_w, height=out_h, fps=info.fps,
        audio_source=audio_source, codec=codec, crf=crf, preset=preset,
        bitdepth=bitdepth, color=info.color, chroma=chroma,
    )
    frames = read_source(input_path, info, source_projection, w, h,
                         max_frames, start_frame)
    sink = _Sink(encoder, reporter, cancel, output_mode, yaw)
    cancelled = False

    reporter.start(total, width=out_w, height=out_h, fps=info.fps,
                   input=input_path, output=output_path, face_size=face_size)
    try:
        chunk_size = fit_chunk_size(chunk_size, w, h,
                                    2 if split_baseline else 1, reporter)
        if depth_backend is not None and chunk_size > 1:
            _convert_chunked(frames, sink, face_size, depth_backend,
                             strength, chunk_size, chunk_overlap,
                             fg_erode, inpaint_mode, temporal_fill, depth_tiles,
                             split_baseline, gradient_limit, face_overlap)
        else:
            depth_range = DepthRange()
            stabiliser = (DepthStabiliser(temporal_depth)
                          if temporal_depth > 0 else None)
            for source in frames:
                left = source.equirect
                if depth_backend is not None:
                    left, right = right_eye_from_depth(
                        source.equirect, face_size, depth_backend, strength,
                        fg_erode, inpaint_mode, depth_tiles, split_baseline,
                        gradient_limit, source.faces, depth_range,
                        stabiliser, face_overlap)
                elif use_cubemap:
                    right = right_eye_passthrough(source.equirect, face_size)
                else:
                    right = source.equirect
                sink.write(left, right)
    except (Cancelled, KeyboardInterrupt):
        # Ctrl-C is the terminal's version of the same request, and deserves
        # the same treatment: stop feeding frames, then let the encoder close
        # the file properly instead of leaving a truncated mdat behind.
        cancelled = True
    except BaseException:
        frames.close()
        encoder.abort()
        reporter.finish(output=output_path, frames=sink.written,
                        cancelled=False, failed=True)
        raise
    finally:
        # Kills the decoder immediately rather than waiting for the generator
        # to be collected, which on a cancelled 8K run means one fewer ffmpeg
        # holding a pipe open.
        frames.close()

    encoder.close()

    # Before the metadata: injecting rewrites the file, and a remux would not
    # carry the boxes through.
    if audio_source and ffmpeg_io.trim_audio_to_video(output_path, info.fps):
        reporter.info("Trimmed the copied audio to the rendered length.",
                      frames=sink.written)

    # Spherical + stereoscopic are always true of this output, so they are
    # never optional. Ambisonic audio is a property of the *source*, so it has
    # to be declared.
    spherical.inject_spherical_metadata(
        output_path,
        stereo_mode="left-right" if output_mode == "vr180" else "top-bottom",
        spatial_audio=spatial_audio,
        horizontal_fov=VR180_FOV if output_mode == "vr180" else 360.0)
    reporter.finish(output=output_path, frames=sink.written,
                    cancelled=cancelled)
    return ConvertResult(output_path, sink.written, cancelled)


class PreviewResult(NamedTuple):
    """Where the preview went, and at what size it was actually written."""

    output_path: str
    frame_index: int
    width: int
    height: int


# Written through cv2.imencode, so this is what OpenCV's encoders accept.
_PREVIEW_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def preview_frame(
    input_path: str,
    output_path: str,
    frame_index: int = 0,
    face_size: Optional[int] = None,
    use_cubemap: bool = True,
    depth_backend: Optional[DepthBackend] = None,
    strength: float = 1.0,
    fg_erode: int = 2,
    inpaint_mode: str = "simple",
    depth_tiles: int = 1,
    split_baseline: bool = False,
    gradient_limit: float = 0.0,
    width: int = 2048,
    input_projection: str = "auto",
    face_overlap: float = projection.FACE_OVERLAP,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    yaw: float = 0.0,
    reporter: Optional[Reporter] = None,
) -> PreviewResult:
    """Render one source frame to an image instead of converting the video.

    The point is to make settings decidable. `--strength`, `--gradient-limit`
    and `--depth-tiles` all change how the result looks and none of them can
    be judged from their numbers, so without this the only way to see a choice
    is to sit through a full render and then start again.

    Deliberately the same code path as `convert`: the frame is warped by
    `right_eye_from_depth` and stacked top-bottom exactly as it would be in
    the output. A preview produced any other way would eventually disagree
    with the render it is supposed to predict.
    """
    import cv2

    reporter = reporter or Reporter()

    suffix = os.path.splitext(output_path)[1].lower()
    if suffix not in _PREVIEW_SUFFIXES:
        raise ValueError(
            f"--preview-frame writes an image, but the output path "
            f"{output_path!r} has no image extension. Expected one of "
            f"{', '.join(_PREVIEW_SUFFIXES)}.")
    if frame_index < 0:
        raise ValueError(f"--preview-frame must not be negative, got "
                         f"{frame_index}")

    check_yaw(output_mode, yaw)
    info = ffmpeg_io.probe(input_path)
    source_projection = resolve_projection(info, input_projection, reporter)
    # A preview is for judging settings, so the same caveat applies: what it
    # shows has already been through the 8-bit decode.
    warn_if_source_is_deeper_than_8_bit(info, 8, reporter)
    face_size, w, h = source_geometry(info, source_projection, face_size,
                                      reporter)
    if info.frame_count is not None and frame_index >= info.frame_count:
        raise ValueError(
            f"--preview-frame {frame_index} is past the end of {input_path}, "
            f"which has {info.frame_count} frames (0-{info.frame_count - 1}).")

    reporter.start(1, width=w, height=h * 2, input=input_path,
                   output=output_path, face_size=face_size,
                   frame_index=frame_index, preview=True, desc="Preview")

    frames = read_source(input_path, info, source_projection, w, h, 1,
                         frame_index)
    try:
        source = next(frames)
    except StopIteration:
        raise ValueError(
            f"Could not read frame {frame_index} from {input_path}."
        ) from None
    finally:
        frames.close()

    left = source.equirect
    if depth_backend is not None:
        left, right = right_eye_from_depth(
            source.equirect, face_size, depth_backend, strength, fg_erode,
            inpaint_mode, depth_tiles, split_baseline, gradient_limit,
            source.faces, face_overlap=face_overlap)
    elif use_cubemap:
        right = right_eye_passthrough(source.equirect, face_size)
    else:
        right = source.equirect
    stacked = stack_eyes(left, right, output_mode, yaw)

    # Downscaled by default. A full 8K top-bottom preview is a 7680x7680 PNG
    # that takes seconds to encode and tens of megabytes to hold, for an image
    # nothing will display at more than a fraction of that -- which defeats
    # the purpose of a fast look.
    if width and 0 < width < stacked.shape[1]:
        height = max(1, int(round(stacked.shape[0] * width / stacked.shape[1])))
        stacked = cv2.resize(stacked, (width, height),
                             interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(suffix, cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"OpenCV could not encode a {suffix} image")
    # imencode plus a plain write, not cv2.imwrite: on Windows imwrite goes
    # through the ANSI API and silently fails on paths with non-ASCII
    # characters, which a user's own folder name may well have.
    with open(output_path, "wb") as fh:
        fh.write(buf)

    reporter.advance(1)
    reporter.finish(output=output_path, frames=1, cancelled=False,
                    preview=True, frame_index=frame_index,
                    width=stacked.shape[1], height=stacked.shape[0])
    return PreviewResult(output_path, frame_index, stacked.shape[1],
                         stacked.shape[0])


def _convert_chunked(
    frames,
    sink: "_Sink",
    face_size: int,
    backend: DepthBackend,
    strength: float,
    chunk_size: int,
    chunk_overlap: int,
    fg_erode: int = 2,
    inpaint_mode: str = "simple",
    temporal_fill: bool = False,
    depth_tiles: int = 1,
    split_baseline: bool = False,
    gradient_limit: float = 0.0,
    face_overlap: float = projection.FACE_OVERLAP,
) -> None:
    """M3 streaming: buffer chunk_size frames, estimate depth with temporal
    context per cubemap face, blend chunk overlaps, warp, encode in order.

    Overlap frames are estimated twice (tail of chunk N, head of chunk N+1)
    and ramp-blended; the later estimate has more temporal context and gets
    more weight. Chunks advance by (chunk_size - chunk_overlap) frames.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    step = chunk_size - chunk_overlap

    buf: list = []
    prev_tail: list = []  # depth maps of previous chunk's overlap tail

    def flush(sources: list) -> int:
        nonlocal prev_tail
        # Before the expensive part, not just before the writes: depth for a
        # chunk is seconds of work with no output, and a Stop pressed during
        # it should not have to wait for the whole chunk to finish.
        sink.check()
        chunk = [s.equirect for s in sources]
        face_sets = ([s.faces for s in sources]
                     if sources[0].faces is not None else None)
        maps = depth_maps_for_chunk(chunk, face_size, backend, depth_tiles,
                                    face_sets, face_overlap)
        if prev_tail:
            head = _blend_overlap(prev_tail, maps[:len(prev_tail)])
            maps = head + maps[len(prev_tail):]
        # The last `chunk_overlap` maps belong to frames that will be
        # re-estimated by the next chunk; hold them back for blending.
        keep = chunk_overlap if len(chunk) == chunk_size else 0
        emit = len(maps) - keep

        # Normalize the whole chunk with ONE depth range (see _chunk_normalize).
        _chunk_normalize(maps[:emit])
        # Median-lock temporally stable pixels to stop warp edge-flapping.
        if emit > 2:
            from .temporal_fill import stabilize_depth

            stabilize_depth(maps[:emit])

        # Warp without inpainting first so holes can be filled temporally
        # from other frames in the chunk before any spatial fill happens.
        # With split_baseline both eyes are synthesized, each at half the
        # baseline and in opposite directions (see stereo_pair); the eyes are
        # warped, temporally filled and spatially filled as two independent
        # streams, since their holes fall in different places by construction.
        eyes = ((-0.5 * strength,), (0.5 * strength,)) if split_baseline \
            else ((strength,),)
        streams = []
        for (eye_strength,) in eyes:
            dn_pres, rights, holes = [], [], []
            for i in range(emit):
                dn_pres.append(maps[i].copy())  # pre-erosion, chunk-normalized
                # The warp erodes its depth map in place, so when both eyes
                # are synthesized the second one needs an intact copy.
                dn_warp = maps[i].copy() if split_baseline else maps[i]
                right, hole = warp.right_eye_from_disparity(
                    chunk[i], dn_warp, eye_strength, fg_erode=fg_erode,
                    inpaint=False, normalize=False,
                    gradient_limit=gradient_limit)
                rights.append(right)
                holes.append(hole)

            if temporal_fill and emit > 1:
                from .temporal_fill import temporal_fill as _tfill

                _tfill(rights, holes)

            sign = 1.0 if eye_strength >= 0 else -1.0
            for i in range(emit):
                rights[i] = warp.fill_holes(rights[i], holes[i], dn_pres[i],
                                            inpaint_mode=inpaint_mode,
                                            baseline_sign=sign)
            streams.append(rights)
            del dn_pres, holes

        for i in range(emit):
            left = streams[0][i] if split_baseline else chunk[i]
            right = streams[-1][i]
            sink.write(left, right)
        prev_tail = maps[emit:] if keep else []
        return keep

    pending: list = []  # frames held back (overlap) for the next chunk
    for source in frames:
        buf.append(source)
        if len(buf) == chunk_size:
            held = flush(buf)
            pending = buf[len(buf) - held:] if held else []
            buf = pending
    if buf:  # final partial chunk and/or held-back overlap frames
        flush(buf)
