"""Depth-image-based rendering (DIBR): synthesize the right-eye view.

Geometry
--------
The left eye sits at the origin; the right eye is offset by a virtual baseline
to the side *of whichever direction is being looked at* -- omnidirectional
stereo, the convention 360 rigs record in and headsets expect. See
`_eye_offset`: offsetting along one fixed axis instead leaves the disparity
correct only near the equirect centre, zero at lon +/-90deg and inverted
behind the viewer.

Rendering is a two-pass *inverse* warp. Pass 1 lifts each left pixel to a 3D
point along its ray using relative inverse depth, translates it to the
right eye, and forward-splats only the resulting **distance** into a z-buffer
to resolve visibility (nearest wins). Pass 2 walks every right-eye pixel,
back-projects it with that distance, transforms it into the left-eye frame,
and samples the left image at **sub-pixel** coordinates.

Splatting colour forward (the naive approach) quantizes every output pixel to
an integer target, so sub-pixel depth jitter re-rasterizes thin structures
(railings, wires) differently on each frame. Sampling backwards removes that
quantization entirely: output geometry varies continuously with depth.

Pixels for which pass 1 resolved no distance are genuine disocclusion holes;
they are dilated slightly and filled by inpainting.

Warping is done in equirect space but with full 3D ray geometry, so it is
geometrically exact for every latitude and handles the +/-180deg seam by
construction (direction vectors are continuous across it).

Disparity is therefore uniform over the whole sphere: the same longitude shift
for a given depth at every longitude and latitude, and no vertical disparity
beyond second order in baseline/distance.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import namedtuple
from typing import Optional, Tuple

import cv2
import numpy as np

from .projection import (_dir_to_equirect_uv, _equirect_uv_to_dir,
                         equirect_rows_to_dir, points_to_equirect_uv)

# Row bands are processed on a thread pool. This is worth doing in Python
# because the per-band cost is almost entirely numpy ufuncs (sin/cos/arctan2/
# hypot/norm over millions of elements), and numpy releases the GIL for those,
# so threads get true parallelism. Threads rather than processes: an 8K frame
# is 88 MB, and shipping it to worker processes would cost more than the work.
#
# Every in-flight band holds its own temporaries -- direction vectors, the
# translated points, the reprojected coordinates -- so concurrency is a memory
# multiplier, not just a speed one. Roughly this many bytes of working set per
# pixel of a band, measured across the warp's two passes.
_BAND_BYTES_PER_PX = 64

# Upper bound on threads regardless of the machine. Pass 1's scatter is
# serialised behind a lock, so the gain flattens well before the core count
# does while the memory keeps climbing.
#
# Measured on 16 cores at 8K, and 8 is not leaving anything on the table:
# 4 -> 2.27 s, 8 -> 2.09 s, 12 -> 2.13 s, 16 -> 2.08 s, with byte-identical
# output at every count. The flatness is not the lock -- pass 1 is the
# *cheapest* of the three passes (0.23 s of 2.21 s). It is memory bandwidth,
# which also explains why the payload dtype barely registers: the same warp
# over uint8x3 and float32x3, a 4x difference in bytes touched, measured
# 2.11 s against 2.22 s. The cost is the geometry, not the pixels, so raising
# this or narrowing the pixel type are both dead ends.
_MAX_WORKERS = 8

_WORKERS = max(1, min(int(os.environ.get("STEREO360_WORKERS", "0")) or
                      _MAX_WORKERS, (os.cpu_count() or 4)))


_gpu_probe: list = []


def gpu_device() -> str | None:
    """Device for the GPU warp, or None to use the numpy path.

    The warp is ~70% of a frame once depth is off the CPU, and it is almost
    all transcendentals and resampling -- on an 8K frame, CUDA does
    atan2+asin+sqrt 274x faster than numpy. So it runs on the GPU when torch
    has one, and on numpy when it does not, which covers AMD on Windows (no
    ROCm build) with no change in behaviour.

    STEREO360_GPU_WARP=0 forces the numpy path, =1 insists on the GPU.
    """
    import os

    setting = os.environ.get("STEREO360_GPU_WARP", "auto").lower()
    if setting in ("0", "off", "false", "no"):
        return None
    if not _gpu_probe:
        from . import warp_torch

        _gpu_probe.append(warp_torch.device_available())
    dev = _gpu_probe[0]
    if dev is None and setting in ("1", "on", "true", "yes"):
        raise RuntimeError(
            "STEREO360_GPU_WARP=1 but torch reports no CUDA or MPS device.\n"
            "Windows has no ROCm torch, so AMD GPUs have no CUDA path. "
            "torch-directml does\nprovide a device, but it is deliberately not "
            "used: measured on a Radeon 780M,\n"
            "`scatter_reduce_` (the z-buffer) falls back to the CPU, and the "
            "results differ\nfrom the reference on 13% of pixels. See "
            "scripts/check_directml_warp.py.\n"
            "Unset the variable to use the CPU path.")
    return dev


def available_memory() -> int | None:
    """Free physical memory in bytes, or None if it cannot be determined.

    Deliberately dependency-free and best-effort: this runs on whatever
    machine a user has, and a wrong guess must degrade to a safe default
    rather than raise.
    """
    try:                                            # Linux, and most BSDs
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, AttributeError, OSError):
        pass
    try:                                            # Windows
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys)
    except Exception:
        pass
    return None


def plan_workers(row_bytes: int, chunk_rows: int) -> int:
    """Threads that fit both the core count and the memory actually free.

    The thread count cannot be a constant. It was pinned at 4 for a 16-core
    laptop with ~7 GB free and an 8K frame; that number is wrong for a 4 GB
    machine (too many) and wrong for a workstation (too few). So it is derived
    per call from the real band size and the real free memory, and capped at a
    quarter of what is free so the rest of the pipeline still has room.

    STEREO360_WORKERS overrides everything, for reproducible benchmarking.
    """
    cores = os.cpu_count() or 4
    env = os.environ.get("STEREO360_WORKERS")
    if env and env.isdigit() and int(env) > 0:
        return max(1, min(int(env), cores))
    avail = available_memory()
    if avail is None:
        budget = 4                                  # unknown machine: be modest
    else:
        band = max(int(row_bytes) * max(chunk_rows, 1), 1)
        budget = int((avail * 0.25) // band)
    return max(1, min(cores, _MAX_WORKERS, budget))


def _map_bands(fn, h: int, chunk_rows: int, row_bytes: int = 0) -> None:
    """Run fn(y0, y1) over every row band, in parallel."""
    bands = [(y0, min(y0 + chunk_rows, h)) for y0 in range(0, h, chunk_rows)]
    workers = plan_workers(row_bytes, chunk_rows) if row_bytes else _WORKERS
    if len(bands) == 1 or workers == 1:
        for y0, y1 in bands:
            fn(y0, y1)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda b: fn(*b), bands))

# Minimum inverse depth used when mapping normalized disparity to relative
# depth. Prevents infinite depth (and thus huge splat coordinates) at disp=0.
_MIN_INV_DEPTH = 0.05

# Baseline scale: strength=1.0 maps to this virtual eye separation in relative
# depth units. Tuned so default strength gives comfortable parallax for
# typical indoor/outdoor scenes; user-adjustable via --strength.
_BASELINE_SCALE = 0.03

# Pass 2 treats a point as hidden from the left eye when the nearest surface
# along that left-eye ray sits closer than this fraction of the distance to the
# point itself. Deliberately loose: real occlusions differ by the whole
# foreground/background ratio, continuous surfaces agree to interpolation
# error, and nothing lands in between.
_VIS_RATIO = 0.9

# Crack closing pulls a pixel to its 3x3 neighbourhood minimum when it stands
# this far above it. Module-level so it can be measured and disabled.
_CRACK_MARGIN = 0.05

# Distinguishes "no bands passed, work them out" from an explicit None, which
# is `detail_bands` saying the split is off.
_UNSET = object()


def _eye_offset(dist, d: np.ndarray, baseline: float):
    """Move a point between the two eyes. Returns the new (x, z).

    The other eye sits `baseline` to the side *of the direction being looked
    at*, not to the side of one fixed forward axis. That distinction is the
    whole of omnidirectional stereo, and getting it wrong is not a subtle
    error: offsetting along a fixed world +X gives a longitude shift of
    `-b*cos(lon)/dist`, so the disparity is full size at the equirect centre,
    **zero at lon +/-90 degrees**, and full size with the **opposite sign**
    behind the viewer. Opposite sign is pseudoscopic -- near reads as far --
    and since occlusion, perspective and texture all still say the near thing
    is near, the eyes are handed a depth that contradicts every other cue.
    It is worst at a high-contrast occluding edge, which is exactly where it
    is least ignorable, and it made the far half of a 360 scene painful to
    look at while the front half was fine.

    Sideways-of-the-view-direction is `(cos lon, 0, -sin lon)`, and the
    horizontal part of a unit direction is `(cos(lat)*sin lon, _,
    cos(lat)*cos lon)` -- so the offset is just the direction's own horizontal
    components, swapped and one negated, times the baseline. No trig, and the
    cos(latitude) taper falls out for free rather than being applied
    separately.

    That taper matters and is kept: equirect meridians converge, so a constant
    world offset costs 1/cos(lat) pixels of longitude -- at lat 89 degrees, 58x
    the equator's shift (measured: 28.6 px against 0.49 px at 512 px wide) --
    and the polar cap does not translate but *folds*, driving opposite
    longitudes onto one meridian, many-to-one, which is why nadir detail used
    to vanish rather than shift. Tapering converges both eyes at the poles,
    which is also correct: looking straight down there is no horizontal
    direction to separate along.

    Algebraically the result is a rotation of the horizontal plane by
    `atan(baseline/dist)` composed with a scale of `hypot(dist, baseline)`, so
    the longitude shift is uniform over the entire sphere and the latitude
    shift is second order in `baseline/dist` (measured: 0.04 px at 8K).
    """
    px = dist * d[..., 0] + baseline * d[..., 2]
    pz = dist * d[..., 2] - baseline * d[..., 0]
    return px, pz


def limit_disparity_gradient(dn: np.ndarray, max_slope: float,
                             max_slope_y: float | None = None) -> np.ndarray:
    """Clamp the depth gradient, lowering peaks only. In place.

    Disocclusion is not an accident of the renderer -- it is where the warp
    stops being injective. A horizontal map x -> x + s(x) is one-to-one iff
    1 + ds/dx > 0, and holes are exactly the regions where that fails. Since
    the shift is proportional to depth (s = baseline * dn, so ds/d(dn) is a
    constant), the condition reduces to a bound on the depth GRADIENT. Clamp
    it and the holes cannot form in the first place -- nothing is filled,
    because nothing is missing.

    The clamp is a cone erosion: `dn[x] <= dn[x-1] + s` is the same as saying
    `dn - s*x` never increases, and the mirrored condition makes `dn + s*x`
    never decrease, so two running minima do it in O(W) per row. It only ever
    lowers values, so depth ordering is preserved and nothing moves nearer.

    What it costs is depth on *sharp* structures, and only there: a step
    steeper than the limit gets ramped, so a thin near object loses some of
    its pop. Measured on an 8K frame at strength 1.0, clamping at exactly the
    critical gradient cut hole area 7.8x (0.0705% -> 0.0090%) while keeping
    100% of the overall depth range and 81% of the depth amplitude on the
    sharpest structures. That is a far better exchange than lowering
    --strength, which buys the same hole reduction by scaling *every* depth
    cue down, sharp or not.

    `max_slope_y` clamps the vertical gradient too, and is needed for depth
    edges that run near-horizontally in the image -- a handrail crossing the
    frame has its cliff in the vertical direction, where a horizontal-only
    clamp does nothing. The warp is mostly horizontal but not purely: the
    pole taper makes latitude shift as well, by a factor of sin(lat)*d_x of
    the horizontal shift. That peaks at 0.5 around +-45 degrees and falls to
    zero at both the equator and the poles, so the vertical limit can be
    twice as permissive as the horizontal one for the same safety. Left at
    None the vertical direction is not clamped at all.
    """
    _clamp_axis(dn, max_slope, axis=1)
    if max_slope_y is not None:
        _clamp_axis(dn, max_slope_y, axis=0)
    return dn


def _clamp_axis(dn: np.ndarray, max_slope: float, axis: int) -> None:
    """One cone erosion along `axis`, in place.

    The scans are sequential along `axis` but completely independent across
    it, so the work is split across threads on the *other* axis. Worth doing:
    `np.minimum.accumulate` is single-threaded and memory-bound, and at 8K the
    four scans this function performs cost about a second per frame -- a third
    of the whole warp.
    """
    n = dn.shape[axis]
    s = np.float32(max_slope)
    shape = (1, n) if axis == 1 else (n, 1)
    x = (np.arange(n, dtype=np.float32) * s).reshape(shape)
    flip = (slice(None), slice(None, None, -1)) if axis == 1 \
        else (slice(None, None, -1), slice(None))

    def run(band: np.ndarray) -> None:
        u = band - x
        np.minimum.accumulate(u, axis=axis, out=u)
        np.add(u, x, out=band)
        v = band + x
        v = np.minimum.accumulate(v[flip], axis=axis)[flip]
        np.subtract(v, x, out=band)

    other = 1 - axis
    m = dn.shape[other]
    workers = plan_workers(dn.shape[1] * 12, dn.shape[0] // max(_MAX_WORKERS, 1))
    step = max(1, (m + workers - 1) // workers)
    slabs = [slice(i, min(i + step, m)) for i in range(0, m, step)]
    if len(slabs) == 1 or workers == 1:
        run(dn)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda sl: run(dn[sl] if other == 0 else dn[:, sl]), slabs))


def _directional_fill(img: np.ndarray, hole: np.ndarray,
                      baseline_sign: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fill disocclusion holes by continuing the background into them.

    A disocclusion is not a generic hole. It opens because a near surface
    moved further than the background behind it, so the missing content is a
    continuation of that background -- and which side the background lies on
    is fixed by geometry, not by the scene.

    Under omnidirectional stereo the shift is `-atan(baseline/dist)` of
    longitude, the *same* direction at every longitude and latitude (see
    `_eye_offset`), so everything slides one way and the fill direction is one
    constant for the whole frame. The foreground slides further than the
    background, so the hole opens on the side the foreground vacated and the
    background continues in from the opposite side -- no extra state, and no
    dependence on the noisy depth map.

    This used to read the direction per column as `cos(lon) * baseline_sign`,
    which was correct for an offset along a fixed world +X and is wrong for
    ODS: it reversed the fill across half the sphere and had no direction at
    all near lon +/-90, so exactly where disocclusions are widest the
    background was continued in from the foreground side.

    Each hole run is then filled by mirroring the neighbouring background
    across the hole boundary, which continues texture instead of smearing a
    single colour. The whole operation is a deterministic function of the
    image and the hole mask, so a stable input gives a stable fill -- unlike
    Telea/LaMa, which re-solve a diffusion from scratch every frame and
    therefore crawl. Measured against Telea on a textured scene with the real
    depth noise: 290 vs 716 pixels flickering by more than 8 levels.

    Returns (filled, done) where `done` marks the pixels it handled; anything
    left (a hole with no valid pixel either side) falls through to inpainting.
    """
    h, w = hole.shape
    rows = np.flatnonzero(hole.any(axis=1))
    done = np.zeros((h, w), dtype=bool)
    if rows.size == 0:
        return img, done

    # Only rows that actually contain holes -- at 8K that is a small fraction,
    # and the index arrays below are full-width, so this bounds both cost and
    # peak memory.
    n = rows.size
    sub_hole = hole[rows] > 0
    sub_img = img[rows]
    x = np.arange(w, dtype=np.int32)[None, :]
    valid = ~sub_hole

    # Nearest valid column to the left / right of every pixel.
    li = np.where(valid, x, np.int32(-1))
    np.maximum.accumulate(li, axis=1, out=li)
    ri = np.where(valid, x, np.int32(w))
    ri = np.minimum.accumulate(ri[:, ::-1], axis=1)[:, ::-1]
    has_l, has_r = li >= 0, ri < w

    # Everything slides toward -longitude for a positive baseline, so the
    # foreground vacates its low-longitude side and the background must come in
    # from the high side.
    from_right = bool(baseline_sign > 0)
    use_right = (has_r & from_right) | ~has_l

    bound = np.where(use_right, np.clip(ri, 0, w - 1), np.clip(li, 0, w - 1))
    src = np.where(use_right, 2 * np.clip(ri, 0, w - 1), 2 * bound) - x
    np.mod(src, w, out=src)

    rr = np.broadcast_to(np.arange(n, dtype=np.int32)[:, None], (n, w))
    m = sub_hole & (has_l | has_r)
    # A mirrored source can itself land in a neighbouring hole; fall back to
    # the boundary pixel there rather than copying a hole into a hole.
    bad = sub_hole[rr, src]
    src = np.where(bad, bound, src)

    out = img.copy()
    osub = out[rows]
    osub[m] = sub_img[rr[m], src[m]]
    out[rows] = osub
    done[rows] = m
    return out, done


def _inpaint_telea_cropped(img: np.ndarray, paint_mask: np.ndarray,
                           seed_mask: np.ndarray, radius: float,
                           margin: int = 48) -> np.ndarray:
    """cv2.inpaint run only around the pixels whose fill is actually kept.

    Two different masks are in play and conflating them is expensive.
    `paint_mask` is the component-expanded mask -- it exists so Telea's fill
    boundary lands on background rather than on bright foreground. `seed_mask`
    is where genuine disocclusion holes are, i.e. the only pixels fill_holes
    composites back. Measured at 8K: seed 0.4% of the frame, paint mask 18.3%.
    Handing the whole frame plus the expanded mask to Telea cost 2.92 s against
    0.17 s for the holes alone, and ~99% of that work was then discarded by the
    composite.

    So crops are placed on the seed components but painted with the expanded
    mask, which preserves the background bias exactly. A crop is grown if the
    expanded mask covers all of it -- Telea needs some unmasked pixel to draw
    from -- up to a cap, beyond which the whole frame is used.
    """
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(
        seed_mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return img.copy()

    h, w = paint_mask.shape
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        m = margin
        for _ in range(4):   # grow until the crop contains drawable pixels
            x0, y0 = max(0, x - m), max(0, y - m)
            x1, y1 = min(w, x + bw + m), min(h, y + bh + m)
            if (paint_mask[y0:y1, x0:x1] == 0).any():
                break
            m *= 4
        boxes.append([x0, y0, x1, y1])

    # Merge overlapping boxes (repeat until stable; component counts are small).
    merged = True
    while merged:
        merged = False
        out: list = []
        for b in boxes:
            for o in out:
                if not (b[2] <= o[0] or b[0] >= o[2] or
                        b[3] <= o[1] or b[1] >= o[3]):
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    merged = True
                    break
            else:
                out.append(list(b))
        boxes = out

    # If the crops cover most of the frame anyway, the whole-frame call wins.
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes)
    if area > 0.5 * h * w:
        return cv2.inpaint(img, paint_mask, radius, cv2.INPAINT_TELEA)

    result = img.copy()
    for x0, y0, x1, y1 in boxes:
        sub = np.ascontiguousarray(img[y0:y1, x0:x1])
        sub_m = np.ascontiguousarray(paint_mask[y0:y1, x0:x1])
        result[y0:y1, x0:x1] = cv2.inpaint(sub, sub_m, radius,
                                           cv2.INPAINT_TELEA)
    return result


def _erode_foreground(dn: np.ndarray, k: int, thresh: float = 0.05) -> None:
    """Blend inverse depth at near-side edges toward the local background
    level, in place (dn must be normalized to [0, 1]).

    At a depth discontinuity the foreground object's boundary pixels warp far
    from their background neighbors, opening disocclusion gaps that inpainting
    then fills with bright foreground smear. Pulling the near-side boundary
    depth toward the local background depth makes those pixels warp *with* the
    background, so holes open behind the object and fill from background
    colors instead. Thin structures (railings, wires) must be exempt: they are
    *entirely* boundary, so eroding them replaces their whole depth with the
    background's and they lose their parallax.

    Continuity is the hard requirement here, not just correctness on a clean
    depth map. Monocular depth carries per-frame noise of roughly 1e-3 in
    these normalized units. Any operator that makes a *binary* decision from
    dn -- a threshold, a connected-component label, a mask intersection --
    turns that noise into a finite, frame-varying change in which pixels get
    rewritten, which the warp renders as thin structures changing shape every
    frame. (The previous implementation classified foreground via connected
    components of `dn >= local_max - 1e-6`; a 1e-6 tolerance against 1e-3
    noise shatters those plateaus into speckle, so its thin-structure mask was
    re-randomized every frame -- measurably 4x its own area in flipped pixels
    at noise levels far below the model's floor, and saturated, i.e. the same
    magnitude no matter how small the perturbation.)

    So every term below is built only from grayscale morphology (erode /
    dilate / open are compositions of min and max, hence 1-Lipschitz),
    division by a floored range, clipping, and products of bounded factors.
    The result is Lipschitz in dn: a depth perturbation of size d moves the
    output by O(d) instead of flipping it. No thresholds, no labels, no masks.

      near  -- where the pixel sits within its local depth range (1 on the
               foreground side of an edge, 0 on the background side);
      wide  -- how much local depth contrast there is at all (0 in flat
               regions, so interiors are untouched);
      thin  -- how much of this pixel's depth a grayscale opening removes.
               An opening deletes ridges narrower than the kernel, so
               (dn - opened) is large exactly on thin structures and ~0 on
               wide objects, including their boundary rings.
    """
    # The three morphological maps come from OpenCV, which already threads
    # them. Everything after is elementwise, so it runs in threaded row bands
    # against band-sized scratch -- both to use the other cores and to stop
    # allocating five full-frame temporaries (~590 MB at 8K) for values that
    # are consumed immediately.
    kern = np.ones((2 * k + 1,) * 2, np.uint8)
    local_min = cv2.erode(dn, kern)
    local_max = cv2.dilate(dn, kern)
    opened = cv2.morphologyEx(dn, cv2.MORPH_OPEN, kern)

    def band(y0: int, y1: int) -> None:
        d = dn[y0:y1]
        lmin = local_min[y0:y1]
        rng = local_max[y0:y1] - lmin
        near = (d - lmin) / np.maximum(rng, 1e-6)
        wide = np.clip((rng - thresh) / thresh, 0.0, 1.0)
        thin = np.clip((d - opened[y0:y1]) / thresh, 0.0, 1.0)
        np.subtract(1.0, thin, out=thin)
        wide *= near
        wide *= thin                      # wide = wide * near * (1 - thin)
        np.subtract(lmin, d, out=near)    # reuse: near = local_min - dn
        wide *= near
        d += wide

    _map_bands(band, dn.shape[0], 256, dn.shape[1] * 24)


#: Detail split radius that measured best, at the width it was measured on.
#: The band is defined in pixels, so the default scales with the frame the
#: warp actually sees -- which is the source's native width, not the delivery
#: width: each eye is rendered full size and downsampled afterwards.
DETAIL_SIGMA_AT_8K = 12.0
DETAIL_REFERENCE_WIDTH = 7680


def detail_sigma_for(width: int) -> float:
    """The default split radius for a frame this wide."""
    return DETAIL_SIGMA_AT_8K * float(width) / DETAIL_REFERENCE_WIDTH


#: Residual sigma to aim for after downsampling in `_blur`. Small enough that
#: the remaining kernel is cheap, large enough that the downsample is not
#: itself doing most of the smoothing.
_BLUR_TARGET_SIGMA = 4.0


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass, via a pyramid when the kernel would be large.

    A separable Gaussian is linear in the kernel width, and these kernels are
    not small: the detail split wants sigma 12 at 8K, a 73-tap kernel, and the
    depth smoothing wants sigma 40, a 241-tap one. Over 29.5M pixels that
    measured 1.8 s and 1.4 s.

    Both are low-passes, which is exactly the case where downsampling first is
    nearly free of consequence -- the frequencies being discarded are the ones
    the blur exists to discard. Shrinking so the remaining sigma is about
    `_BLUR_TARGET_SIGMA`, blurring there and scaling back is 19x and 54x
    faster respectively.

    Verified on the render rather than on the blur: against the direct
    kernel, a rendered eye differs on 0.015% of pixels by more than one level
    and on 0.000% by more than four, worst case 7. `detail_bands` as a whole
    goes 3.32 s to 0.30 s. `INTER_AREA` on the way down matters -- it is a box
    prefilter, so it antialiases rather than point-sampling.
    """
    fdn = int(max(1, min(8, round(sigma / _BLUR_TARGET_SIGMA))))
    h, w = img.shape[:2]
    if fdn > 1 and min(h // fdn, w // fdn) >= 16:
        small = cv2.resize(img, (w // fdn, h // fdn),
                           interpolation=cv2.INTER_AREA)
        s2 = sigma / fdn
        k2 = int(2 * round(3 * s2) + 1)
        small = cv2.GaussianBlur(small, (k2, k2), s2)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


#: What `right_eye_banded` splits a frame into, kept as one object so a caller
#: rendering both eyes can compute it once. Fields are read-only to the warp:
#: `base` and `detail` are only ever sampled, and `dn_pre` is only ever read by
#: `fill_holes` -- but `smooth` is *not*, see the copy at its use site.
_Bands = namedtuple("_Bands", "base detail smooth dn_pre")


def detail_bands(left_rgb: np.ndarray, inv_depth: np.ndarray,
                 detail_sigma: float | None = None,
                 depth_sigma: float = 40.0) -> Optional["_Bands"]:
    """The frame/depth split `right_eye_banded` warps, computed once.

    Both eyes of a pair split the frame identically -- the only thing that
    differs between them is the sign and size of the baseline -- so a caller
    rendering both can do this once and hand the result to both calls. It
    saves a blur of the RGB frame and a blur of the depth map per frame, and
    the latter is a 241-tap kernel at 8K, so it is not a rounding error.

    Returns None when the split is off, which means "use the ordinary warp".
    """
    if detail_sigma is None:
        detail_sigma = detail_sigma_for(left_rgb.shape[1])
    if detail_sigma <= 0:
        return None
    f = left_rgb.astype(np.float32)
    base = _blur(f, detail_sigma)
    return _Bands(base, f - base,
                  _blur(inv_depth.astype(np.float32), depth_sigma),
                  inv_depth.copy())


def right_eye_banded(
    left_rgb: np.ndarray,
    inv_depth: np.ndarray,
    strength: float,
    detail_sigma: float | None = None,
    depth_sigma: float = 40.0,
    bands: Optional["_Bands"] = _UNSET,
    **kw,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp coarse structure and fine detail on different depth fields.

    Thin structures disagree between the eyes because they sit on depth
    discontinuities the model misplaces by 10-25 px: each eye shears them by a
    different amount and the pair stops fusing. Placing those boundaries
    correctly was tried at length and did not work.

    So the frame is split into a blurred base and the detail that blur
    removed. The base is warped with the real depth, keeping every
    discontinuity the coarse depth percept needs -- and being blurred, a
    boundary error of a few pixels moves smooth content and barely shows. The
    detail is warped with a *smoothed* depth: a field with no steps cannot
    tear or shear a thin structure, so the detail arrives whole and arrives
    the same way in both eyes.

    Fine detail then sits at a slightly wrong depth. That is the trade and it
    is the point: detail both eyes agree on fuses more easily than detail at
    the right depth they disagree about, and coarse disparity -- which the
    base still carries correctly -- is what the depth percept mostly rests on.

    Returns the same `(rgb, hole)` as `right_eye_from_disparity`, with the
    base's hole mask, so temporal fill and everything downstream is unchanged.

    `detail_sigma` of None means "scale it to this frame", which is the
    default; an explicit 0 turns the split off and falls back to the ordinary
    warp.

    `bands` takes a split already computed by `detail_bands`, for a caller
    rendering both eyes of one frame; left alone, this works it out itself and
    behaves exactly as before. An explicit None means the split is off.
    """
    if bands is _UNSET:
        bands = detail_bands(left_rgb, inv_depth, detail_sigma, depth_sigma)
    if bands is None:
        return right_eye_from_disparity(left_rgb, inv_depth, strength, **kw)
    base, detail, smooth, dn_pre = bands

    inpaint = kw.pop("inpaint", True)
    inpaint_mode = kw.get("inpaint_mode", "simple")

    # Both warps consume the depth they are given -- normalisation, foreground
    # erosion and the gradient clamp all write in place -- so each gets its own
    # copy. That matters more than it used to: with `bands` shared across the
    # two eyes of a pair, handing `smooth` over uncopied would leave the second
    # eye warping a depth field the first had already eroded and clamped.
    b, hole = right_eye_from_disparity(base, inv_depth.copy(), strength,
                                       inpaint=False, **kw)
    d, dhole = right_eye_from_disparity(detail, smooth.copy(), strength,
                                        inpaint=False, **kw)
    # A hole in the detail layer means "no detail known here", not black:
    # adding nothing leaves the base showing through, which is right.
    d = d.astype(np.float32)
    d[dhole > 0] = 0.0
    out = np.clip(b.astype(np.float32) + d, 0, 255).astype(left_rgb.dtype)
    if inpaint and hole.any():
        out = fill_holes(out, hole, dn_pre, inpaint_mode=inpaint_mode,
                         baseline_sign=1.0 if strength >= 0 else -1.0)
    return out, hole


def right_eye_from_disparity(
    left_rgb: np.ndarray,
    inv_depth: np.ndarray,
    strength: float = 1.0,
    inpaint: bool = True,
    inpaint_radius: float = 3.0,
    hole_dilate: int = 2,
    chunk_rows: int = 256,
    fg_erode: int = 2,
    inpaint_mode: str = "simple",
    normalize: bool = True,
    directional_fill: bool = True,
    gradient_limit: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize the right-eye equirect frame from the left frame + inverse depth.

    left_rgb:  (H, W, 3) uint8 equirect left-eye frame.
    inv_depth: (H, W) float, larger = closer (relative scale, any range).
    strength:  multiplier on the virtual baseline (1.0 = default comfort).
    inpaint:   fill disocclusion holes with OpenCV Telea inpainting.

    Returns (right_rgb, hole_mask) where hole_mask is 255 at disoccluded pixels
    (before inpainting).
    """
    if inpaint_mode not in ("simple", "learned"):
        raise ValueError(f"Unknown inpaint_mode '{inpaint_mode}'")
    h, w = inv_depth.shape
    # normalize=False: caller already normalized with a chunk-consistent
    # scale (temporal pipeline); per-frame percentiles would pump the scale.
    dn = normalize_inv_depth(inv_depth) if normalize else inv_depth

    # Keep the pre-erosion depth for the near-ring mask in the inpaint stage:
    # erosion lowers boundary depth, which would otherwise hide exactly the
    # foreground pixels the mask needs to exclude.
    dn_pre = dn.copy() if (inpaint and fg_erode > 0) else dn

    gpu = gpu_device()
    if gpu is not None:
        from . import warp_torch

        right, hole = warp_torch.right_eye_from_disparity(
            left_rgb, dn, strength, _BASELINE_SCALE, _MIN_INV_DEPTH,
            _VIS_RATIO, _CRACK_MARGIN, fg_erode, gradient_limit, gpu)
        if inpaint and hole.any():
            right = fill_holes(right, hole, dn_pre, inpaint_mode=inpaint_mode,
                               inpaint_radius=inpaint_radius,
                               hole_dilate=hole_dilate,
                               directional=directional_fill,
                               baseline_sign=1.0 if strength >= 0 else -1.0)
        return right, hole

    if fg_erode > 0:
        _erode_foreground(dn, fg_erode)

    baseline = strength * _BASELINE_SCALE

    # Applied AFTER the erosion, which re-sharpens boundaries and would
    # otherwise undo the clamp. `gradient_limit` is in units of the critical
    # gradient 1 / (baseline * w / 2pi) -- the slope at which the warp stops
    # being injective -- so 1.0 means "just injective" and smaller is safer,
    # independent of resolution and strength.
    if gradient_limit > 0.0 and baseline != 0.0:
        critical = (2.0 * np.pi) / (abs(baseline) * w)
        limit_disparity_gradient(dn, gradient_limit * critical,
                                 gradient_limit * critical * 2.0)
    n = h * w

    # strength=0 => no baseline => the right eye is identical to the left.
    if baseline == 0.0:
        return left_rgb.copy(), np.zeros((h, w), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Inverse (backward) warp.
    #
    # Forward point-splatting quantizes *every* output pixel to the nearest
    # integer target, so sub-pixel depth jitter re-rasterizes thin structures
    # differently each frame (the shape-changing-railing artifact). Instead:
    #
    #   Pass 1 forward-splats only the right-eye *distance* (lambda_r) into a
    #          z-buffer to resolve visibility (nearest surface wins). A single
    #          scalar per pixel is far more robust to quantization than RGB,
    #          and the resulting hole set is identical to the old warp.
    #   Pass 2 back-projects every resolved right-eye pixel to 3D, translates
    #          it into the left-eye frame, and samples the left image with
    #          sub-pixel bilinear coordinates (cv2.remap). Stable geometry no
    #          longer depends on integer rounding of the output position.
    #
    # Disocclusion holes (no distance resolved) are left for the inpaint /
    # temporal-fill stage, exactly as before.
    # ------------------------------------------------------------------

    # float32 is sufficient: unlike the old forward warp (which needed exact
    # float64 equality to recover which source pixel won the z-test, plus a
    # full-frame int32 `winner` array) this buffer only ever takes a minimum.
    # Dropping both halves peak memory -- ~176 MB vs ~354 MB at 8K.
    # One guard row above and below (see the erosion note after pass 1), so the
    # buffer is (h + 2) rows and row r of the image lives at padded row r + 1.
    zp = np.full((h + 2) * w, np.inf, dtype=np.float32)

    # The scatter below writes to one shared buffer from arbitrary target
    # positions, so it cannot run concurrently; the trig/geometry ahead of it
    # can. Threads therefore compute their band in parallel and serialise only
    # the scatter, which profiled at ~40% of this pass.
    scatter_lock = threading.Lock()

    def pass1(y0: int, y1: int) -> None:
        dn_c = dn[y0:y1]
        lam = 1.0 / (dn_c + _MIN_INV_DEPTH)  # near -> small lam

        d = equirect_rows_to_dir(y0, y1, w, h)

        # Lift to 3D, translate to the right-eye position, reproject.
        px, pz = _eye_offset(lam, d, -baseline)
        tu, tv, norm = points_to_equirect_uv(
            px, lam * d[..., 1], pz, w, h)

        lam_r = norm.astype(np.float32).ravel()
        # 2x2 bilinear-footprint splat: near foreground is magnified in the
        # target view, so a single-pixel point splat undersamples it and
        # background splats through the gaps, cracking thin structures in a
        # sub-pixel-phase-dependent (frame-varying) pattern. Splatting to all
        # four integer neighbors guarantees a contiguous footprint regardless
        # of phase. Longitude wraps (seam continuity); latitude clamps.
        # int32, not int64: the largest index is (h + 2) * w - 1, which is
        # 29.5M at 8K against int32's 2.1B headroom, and tu/tv are bounded by
        # the equirect extent before the floor. Halves two of the largest
        # per-band temporaries (15.7 -> 7.9 MB each), which matters because
        # every in-flight thread holds a set.
        u0 = np.floor(tu).astype(np.int32).ravel()
        v0 = np.floor(tv).astype(np.int32).ravel()
        np.mod(u0, w, out=u0)
        # Clamp to [-1, h-1] then shift into the guard band. Clamping to the
        # guard row rather than to row 0 is what makes this exact: a sample at
        # v0 = -1 must reach image row 0 and no further, whereas clamping it to
        # row 0 first would let the erosion carry it into row 1 as well.
        np.clip(v0, -1, h - 1, out=v0)
        v0 += 1
        u0 += v0 * w
        with scatter_lock:
            np.minimum.at(zp, u0, lam_r)

    _map_bands(pass1, h, chunk_rows, w * _BAND_BYTES_PER_PX)

    # Complete the 2x2 footprint. Splatting each sample to all four integer
    # neighbours and taking the min is identical to splatting to the floor
    # position alone and then taking, for every output pixel, the min over the
    # 2x2 block ending at it:
    #     out[y, x] = min(Z[y, x], Z[y, x-1], Z[y-1, x], Z[y-1, x-1])
    # That is a 2x2 grayscale erosion, and erosion by a rectangle is separable,
    # so it costs two shifted minimums over the frame instead of four scattered
    # ones. np.minimum.at is unbuffered and cannot be threaded here (arbitrary
    # targets in a shared buffer), so it set the floor on this pass; doing it
    # once instead of four times cuts that floor by 4x.
    #
    # Longitude wraps, which the modulo above already handles. Latitude uses
    # the guard rows: after eroding, image row r is padded row r + 1, and a
    # sample clamped onto a guard row contributes to exactly the one edge row
    # the four-way splat would have clamped it to.
    # Both passes run band by band against one reused band-sized scratch
    # buffer. A full-frame `shifted` array would be another 118 MB at 8K on
    # top of the z-buffer itself; a 256-row band is 7.9 MB.
    zpad = zp.reshape(h + 2, w)
    band = min(chunk_rows, h + 2)
    scratch = np.empty((band, w), np.float32)

    # Horizontal: out[y, x] = min(z[y, x], z[y, x-1]), wrapping in longitude.
    # Rows are independent, so band order does not matter.
    for y0 in range(0, h + 2, band):
        y1 = min(y0 + band, h + 2)
        t = scratch[:y1 - y0]
        t[:, 0] = zpad[y0:y1, -1]
        t[:, 1:] = zpad[y0:y1, :-1]
        np.minimum(zpad[y0:y1], t, out=zpad[y0:y1])

    # Vertical: out[y] = min(z[y], z[y-1]). Rows are *not* independent, and
    # zpad[y0:y1] overlaps zpad[y0-1:y1-1], so the source rows are copied into
    # the scratch first. Bands run bottom-up so the rows being copied have not
    # been overwritten yet. Row 0 is a guard row and needs no update.
    for y0 in reversed(range(1, h + 2, band)):
        y1 = min(y0 + band, h + 2)
        t = scratch[:y1 - y0]
        np.copyto(t, zpad[y0 - 1:y1 - 1])
        np.minimum(zpad[y0:y1], t, out=zpad[y0:y1])

    del scratch
    depth_r = np.ascontiguousarray(zpad[1:h + 1]).reshape(-1)
    del zp, zpad

    # Occlusion-compatible crack closing: pixels whose distance exceeds the
    # 3x3 neighborhood min by a disparity-significant margin are residual
    # splat cracks (background distances leaking *through* magnified
    # foreground). Pull them down to the local min. Genuine holes (inf) are
    # excluded via a finite sentinel and restored afterwards.
    z = depth_r.reshape(h, w)
    finite = np.isfinite(z)
    if finite.any():
        sentinel = z[finite].max() * 2.0
        zw = np.where(finite, z, sentinel)
        zmin = cv2.erode(zw, np.ones((3, 3), np.float32))
        crack = finite & ((zw - zmin) > _CRACK_MARGIN * zmin)
        if crack.any():
            z[crack] = zmin[crack]

    miss = np.isinf(depth_r)
    hole = np.where(miss, 255, 0).astype(np.uint8).reshape(h, w)

    # Pass 2: back-project resolved pixels and sample the left image. Holes get
    # a placeholder distance (in place, no second full-frame buffer) so the map
    # stays finite; those pixels are zeroed afterward.
    depth_r[miss] = 1.0
    depth_2d = depth_r.reshape(h, w)  # view, not a copy
    right = np.zeros_like(left_rgb)

    # Set by pass 2 where the point the z-buffer chose is hidden from the left
    # eye, so no valid colour exists for it (see the visibility check below).
    occluded = np.zeros((h, w), dtype=bool)

    # Pass 2 writes disjoint output row bands, so it parallelises outright.
    def pass2(y0: int, y1: int) -> None:
        d_r = equirect_rows_to_dir(y0, y1, w, h)
        # 3D point along the right-eye ray, then right-eye frame -> world/left.
        # The offset is evaluated on the *target* ray, which is what makes this
        # omnidirectional rather than a single stereo pair: each output column
        # is its own camera on the circle, exactly as an ODS rig records it.
        z = depth_2d[y0:y1]
        px, pz = _eye_offset(z, d_r, baseline)
        su, sv, pn = points_to_equirect_uv(
            px, z * d_r[..., 1], pz, w, h)
        map_x = (su.astype(np.float32) % w)
        map_y = sv.astype(np.float32)
        right[y0:y1] = cv2.remap(
            left_rgb, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )

        # Left-eye visibility check.
        #
        # The z-buffer resolves what is visible from the *right* eye, but
        # nothing has checked that the point it settled on is visible from the
        # *left* -- and the left image is the only place colour can come from.
        # Beside a thin near structure those disagree: an output pixel legitimately
        # sees background, yet the ray from the left eye to that background
        # passes through the railing, so the sample returns railing colour and
        # a bright fragment appears detached from the structure. It moves with
        # depth noise, which reads as the railing changing shape.
        #
        # `pn` is already the left-eye distance to the point we want. Sampling
        # the depth map at the same place gives the distance to whatever the
        # left eye actually sees first. If that is materially nearer, the point
        # is occluded and there is no honest colour for it -- mark it a hole
        # and let the fill stage handle it from background neighbours.
        #
        # Nearest-neighbour on purpose: interpolating across the silhouette
        # would average foreground and background depth and blunt the test. The
        # ratio is not a delicate threshold -- at a real occlusion the two
        # distances differ by the foreground/background ratio (4.5x in the case
        # that motivated this), while on any continuous surface they agree to
        # within interpolation error, so 0.9 sits in a wide empty valley rather
        # than on top of the noise.
        dn_src = cv2.remap(dn, map_x, map_y,
                           interpolation=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_WRAP)
        lam_src = 1.0 / (dn_src + _MIN_INV_DEPTH)
        occluded[y0:y1] = lam_src < _VIS_RATIO * pn

    _map_bands(pass2, h, chunk_rows, w * _BAND_BYTES_PER_PX)

    # Occluded pixels join the disocclusion holes: same treatment, same mask.
    if occluded.any():
        miss = miss | occluded.reshape(-1)
        hole[occluded] = 255

    right.reshape(-1, 3)[miss] = 0  # disocclusion holes -> left for inpainting

    if inpaint and hole.any():
        right = fill_holes(right, hole, dn_pre, inpaint_mode=inpaint_mode,
                           inpaint_radius=inpaint_radius,
                           hole_dilate=hole_dilate,
                           directional=directional_fill,
                           baseline_sign=1.0 if baseline >= 0 else -1.0)

    return right, hole


def normalize_inv_depth_with(inv_depth: np.ndarray, lo: float,
                             hi: float) -> np.ndarray:
    """Normalize with a caller-supplied (chunk-consistent) range, in place."""
    if hi <= lo:
        hi = lo + 1e-6
    np.subtract(inv_depth, lo, out=inv_depth)
    inv_depth /= (hi - lo)
    np.clip(inv_depth, 0.0, 1.0, out=inv_depth)
    return inv_depth


def normalize_inv_depth(inv_depth: np.ndarray) -> np.ndarray:
    """Normalize inverse depth to [0, 1] robustly, in place (the caller's
    buffer is consumed) to avoid duplicating a 113 MB array at 8K."""
    lo, hi = np.percentile(inv_depth, 1), np.percentile(inv_depth, 99)
    return normalize_inv_depth_with(inv_depth, lo, hi)
def fill_holes(
    right: np.ndarray,
    hole: np.ndarray,
    dn_pre: np.ndarray,
    inpaint_mode: str = "simple",
    inpaint_radius: float = 3.0,
    hole_dilate: int = 2,
    directional: bool = True,
    baseline_sign: float = 1.0,
) -> np.ndarray:
    """Fill disocclusion holes in a warped right-eye image.

    With `directional` (the default) the holes are first continued from the
    background side by `_directional_fill`, which is deterministic and so
    temporally stable. Only what it cannot reach -- a hole with no valid pixel
    either side -- falls through to the inpainting path below. Set
    `directional=False` to get the pure Telea/LaMa behaviour for comparison.

    Background-biased: Telea/LaMa draw from the boundary of the inpaint mask,
    so bright foreground near a hole smears into it. Disocclusion holes open
    *behind* foreground objects, so the mask is extended over the whole
    foreground component(s) touching the hole; the fill boundary then lies
    entirely on background. Holes with a depth-uniform ring (interior speckle
    gaps, or disocclusions enclosed by one surface) first widen their search
    ring until it samples a depth contrast, so the near/far split stays
    meaningful.

    That expansion decides where the inpainter may *read from*. It must not
    decide what gets *overwritten*: the expanded region is real, correctly
    warped foreground, and the whole point of moving the boundary off it is to
    keep it. So the fill is composited back only over the genuine holes
    (`keep` below) and the foreground is restored untouched.

    Compositing is not a refinement here, it is load-bearing. The expansion
    grows by connected component, so a near-camera object -- a hand holding
    the rig at the nadir, a coat, a railing post -- is one blob that reaches
    every hole near it. Measured on a real frame: 0.02% of pixels were genuine
    holes and none at all in the bottom 9deg, yet the expanded mask covered
    17% of the frame and 89% of the nadir cap, one component of 87k px. Every
    one of those pixels was being deleted and replaced with Telea diffusion,
    which is why near-camera subjects came back detail-free in the right eye
    while the left eye kept them.

    right:  (H, W, 3) uint8 warped image (hole pixels may contain anything).
    hole:   (H, W) uint8, 255 at hole pixels.
    dn_pre: (H, W) float32 normalized pre-erosion inverse depth.
    """
    if not hole.any():
        return right
    hit2d = hole == 0
    hole_d = cv2.dilate(hole, np.ones((hole_dilate * 2 + 1,) * 2, np.uint8))
    # Everything outside this is restored verbatim, whatever the mask grows to.
    keep = hole_d > 0

    if directional:
        right, done = _directional_fill(right, hole_d, baseline_sign)
        if done.all() or not (keep & ~done).any():
            return right
        # Whatever the background extension could not reach goes to inpainting.
        hole = np.where(done, 0, hole).astype(np.uint8)
        hole_d = np.where(done, 0, hole_d).astype(np.uint8)
        keep = hole_d > 0
        if not hole.any():
            return right
        hit2d = hole == 0
    ring = (hole_d > 0) & hit2d
    if ring.any():
        dn_ring = dn_pre[ring]
        if dn_ring.max() - dn_ring.min() < 1e-6:
            # Depth-uniform ring: the hole sits fully inside one surface (e.g.
            # pole disocclusions enclosed by foreground), so the background
            # bias cannot engage. Widen the search ring until it samples a
            # depth contrast.
            for dil in (2 * hole_dilate + 2, 4 * hole_dilate + 4,
                        8 * hole_dilate + 8):
                hd = cv2.dilate(hole, np.ones((dil * 2 + 1,) * 2, np.uint8))
                r = (hd > 0) & hit2d
                if r.any() and dn_pre[r].max() - dn_pre[r].min() > 1e-6:
                    hole_d, ring, dn_ring = hd, r, dn_pre[r]
                    break
        t = 0.5 * (dn_ring.min() + dn_ring.max())
        fg = hit2d & (dn_pre > t)
        if fg.any() and (hit2d & ~fg).any():
            # Mask the whole foreground component(s) touching the hole, not
            # just a fixed ring: Telea fills from the mask boundary, so the
            # boundary must lie entirely on background — a partial ring leaves
            # bright foreground on the boundary (e.g. disocclusions enclosed
            # by a tall object) and smears it through the hole.
            extra_k = 2 * (hole_dilate + 4) + 1
            wide = cv2.dilate(hole, np.ones((extra_k, extra_k), np.uint8))
            n, lab = cv2.connectedComponents(fg.astype(np.uint8),
                                             connectivity=8)
            touch = np.unique(lab[(wide > 0) & fg])
            touch = touch[touch > 0]
            if touch.size:
                fg_full = np.isin(lab, touch)
                hole_d = np.where(fg_full, 255, hole_d).astype(np.uint8)
    if inpaint_mode == "learned":
        from .inpaint import inpaint_learned

        filled = inpaint_learned(right, hole_d)
    elif inpaint_mode == "simple":
        filled = _inpaint_telea_cropped(right, hole_d, keep, inpaint_radius)
    else:
        raise ValueError(f"Unknown inpaint_mode '{inpaint_mode}'")

    # Take the fill only where there was actually a hole; restore the rest.
    np.copyto(filled, right, where=~keep[..., None])
    return filled
