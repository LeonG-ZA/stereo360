"""Equirectangular <-> cubemap projection conversions.

Convention
----------
Cubemap faces are identified by their outward axis: +X, -X, +Y, -Y, +Z, -Z.
Each face uses local coords (a, b) in [-1, 1] where `a` runs right and `b` runs
down. Direction vectors are right-handed: +Z faces the viewer at equirect
center (lon=0), +X is to the right (lon=+90 deg), +Y is up (lat=+90 deg).

Equirect mapping: lon in [-pi, pi] left->right, lat in [pi/2, -pi/2] top->bottom.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

FACES: List[str] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

# face -> (origin, right, down) unit basis vectors
_FACE_BASIS: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    "+X": (np.array([1, 0, 0]), np.array([0, 0, -1]), np.array([0, -1, 0])),
    "-X": (np.array([-1, 0, 0]), np.array([0, 0, 1]), np.array([0, -1, 0])),
    "+Y": (np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])),
    "-Y": (np.array([0, -1, 0]), np.array([1, 0, 0]), np.array([0, 0, -1])),
    "+Z": (np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, -1, 0])),
    "-Z": (np.array([0, 0, -1]), np.array([-1, 0, 0]), np.array([0, -1, 0])),
}

# Map cache: (kind, dims...) -> (map_x, map_y) float32 arrays for cv2.remap
_map_cache: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}

# Cubemap->equirect remap tables, keyed (out_w, out_h, face_size, pad).
# Held separately from _map_cache because a single 8K entry is ~236 MB, so
# this one keeps at most one entry (see _face_to_equirect_maps).
_f2e_cache: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}


def clear_map_caches() -> None:
    """Drop cached remap tables (frees ~400 MB at 8K)."""
    _map_cache.clear()
    _f2e_cache.clear()
    _overlap_plan_cache.clear()
    _angular_cache.clear()


def _face_to_equirect_maps(out_w: int, out_h: int, face_size: int, pad: int,
                           chunk_rows: int = 512):
    """Cached (map_x, map_y) taking equirect pixels to atlas coordinates.

    These tables depend only on the output geometry, never on pixel data, so
    they are identical for every frame of a video. Rebuilding them per frame
    dominated the assembly stage: profiled at 8K, a 512-row band cost 0.195 s
    in `_face_local_coords` plus 0.066 s in `_equirect_uv_to_dir` against
    0.005 s for the `cv2.remap` they feed -- 98% table construction, ~2.1 s
    per frame thrown away.

    Only one entry is kept: at 7680x3840 the pair costs ~236 MB, and the
    pipeline uses a single output geometry throughout a run. Tables are filled
    band by band so the transient direction array stays band-sized rather than
    allocating a 354 MB (H, W, 3) buffer.
    """
    key = (out_w, out_h, face_size, pad)
    cached = _f2e_cache.get(key)
    if cached is not None:
        return cached

    _f2e_cache.clear()
    fs = face_size + 2 * pad
    map_x = np.empty((out_h, out_w), np.float32)
    map_y = np.empty((out_h, out_w), np.float32)
    for y0 in range(0, out_h, chunk_rows):
        y1 = min(y0 + chunk_rows, out_h)
        v, u = np.meshgrid(np.arange(y0, y1), np.arange(out_w), indexing="ij")
        d = _equirect_uv_to_dir(u.astype(np.float32), v.astype(np.float32),
                                out_w, out_h)
        face_idx, a, b = _face_local_coords(d)
        col = face_idx % 3
        row = face_idx // 3
        map_x[y0:y1] = ((a + 1.0) * 0.5 * face_size - 0.5 + pad) + col * fs
        map_y[y0:y1] = ((b + 1.0) * 0.5 * face_size - 0.5 + pad) + row * fs

    _f2e_cache[key] = (map_x, map_y)
    return map_x, map_y


def _dir_to_equirect_uv(d: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unit direction vectors (..., 3) -> equirect pixel coords (u, v)."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lon = np.arctan2(x, z)                      # [-pi, pi]
    lat = np.arcsin(np.clip(y, -1.0, 1.0))      # [-pi/2, pi/2]
    u = (lon / np.pi * 0.5 + 0.5) * w - 0.5
    v = (0.5 - lat / np.pi) * h - 0.5
    return u, v


def _equirect_uv_to_dir(u: np.ndarray, v: np.ndarray, w: int, h: int) -> np.ndarray:
    """Equirect pixel coords -> unit direction vectors (..., 3) float32."""
    lon = (((u + 0.5) / w - 0.5) * 2.0 * np.pi).astype(np.float32)
    lat = ((0.5 - (v + 0.5) / h) * np.pi).astype(np.float32)
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    return np.stack([x, y, z], axis=-1)


# Separable trig factors per (w, h): four 1-D arrays, a few dozen KB.
_trig_cache: Dict[Tuple[int, int], Tuple[np.ndarray, ...]] = {}


def _equirect_trig(w: int, h: int):
    """(sin_lon, cos_lon, sin_lat, cos_lat) for whole-pixel equirect coords."""
    key = (w, h)
    hit = _trig_cache.get(key)
    if hit is not None:
        return hit
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    lon = (((u + 0.5) / w - 0.5) * 2.0 * np.pi).astype(np.float32)
    lat = ((0.5 - (v + 0.5) / h) * np.pi).astype(np.float32)
    out = (np.sin(lon), np.cos(lon), np.sin(lat), np.cos(lat))
    _trig_cache[key] = out
    return out


def equirect_rows_to_dir(y0: int, y1: int, w: int, h: int) -> np.ndarray:
    """Direction vectors for whole rows [y0, y1) of an equirect image.

    Identical output to `_equirect_uv_to_dir` on an integer meshgrid, but the
    equirect parametrisation is separable -- longitude depends only on the
    column and latitude only on the row -- so the transcendentals collapse to
    four 1-D tables of a few dozen KB, cached per (w, h), and the per-pixel
    work becomes two broadcast multiplies.

    This matters because the warp evaluates sin/cos over the full frame twice
    per frame (once per pass) with arguments that are the same for every frame
    of the video. At 8K that was ~0.034 s per 256-row band per pass. It also
    skips the meshgrid and the `np.stack`, which allocated three extra
    full-band temporaries.
    """
    sin_lon, cos_lon, sin_lat, cos_lat = _equirect_trig(w, h)
    cl = cos_lat[y0:y1, None]
    d = np.empty((y1 - y0, w, 3), np.float32)
    np.multiply(cl, sin_lon[None, :], out=d[..., 0])
    d[..., 1] = sin_lat[y0:y1, None]
    np.multiply(cl, cos_lon[None, :], out=d[..., 2])
    return d


def points_to_equirect_uv(px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                          w: int, h: int):
    """Equirect pixel coords for arbitrary 3D points. Returns (u, v, norm).

    Takes the three components separately and unnormalized, which avoids two
    things the (..., 3) formulation pays for on every band of every frame: a
    full stacked array to hold the point, and a division of all three
    components by the norm before projecting.

    That division is pure waste. Longitude is `arctan2(px, pz)`, which is
    scale-invariant -- scaling both arguments cannot change the angle -- so it
    can be taken straight from the raw components. Only latitude needs the
    norm, and only for its own single division. Measured at 8K this is 1.6x
    faster than building the vector, normalizing it, and calling
    `_dir_to_equirect_uv`, and agrees with it to 1e-3 px, which is float32
    reordering rather than approximation.

    `norm` is returned because callers want it anyway: it is the distance to
    the point, which the warp uses as its z-buffer value.

    Deliberately NOT using cv2.phase here. It is a further 4.5x on the
    arctan2, but it is a polynomial approximation -- measured up to 0.2 px of
    longitude error at 8K, which is a real geometric distortion even though a
    deterministic one.
    """
    norm = px * px
    norm += py * py
    norm += pz * pz
    np.sqrt(norm, out=norm)
    lon = np.arctan2(px, pz)
    safe = np.where(norm == 0.0, np.float32(1.0), norm)
    lat = np.arcsin(np.clip(py / safe, -1.0, 1.0))
    lon *= np.float32(w / (2.0 * np.pi))
    lon += np.float32(w * 0.5 - 0.5)
    lat *= np.float32(-h / np.pi)
    lat += np.float32(h * 0.5 - 0.5)
    return lon, lat, norm


def equirect_rows_cos_lat(y0: int, y1: int, w: int, h: int) -> np.ndarray:
    """cos(latitude) as a (rows, 1) column, for broadcasting.

    Equals `hypot(d[..., 0], d[..., 2])` for directions from
    `equirect_rows_to_dir` -- those components are cos_lat*sin_lon and
    cos_lat*cos_lon, so the hypot just recovers cos_lat. Reading it from the
    cached table skips a full-frame hypot per warp pass, and is marginally
    more accurate since it avoids the round trip through the product.
    """
    return _equirect_trig(w, h)[3][y0:y1, None]


def _face_dirs(face: str, face_size: int) -> np.ndarray:
    """Unit direction vector for every pixel of a cubemap face. (F, F, 3)"""
    origin, right, down = _FACE_BASIS[face]
    coords = (np.arange(face_size) + 0.5) / face_size * 2.0 - 1.0  # [-1, 1]
    a, b = np.meshgrid(coords, coords)  # a: columns (right), b: rows (down)
    d = (origin[None, None, :]
         + a[..., None] * right[None, None, :]
         + b[..., None] * down[None, None, :])
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _equirect_to_face_maps(equirect_w: int, equirect_h: int, face_size: int):
    key = ("e2f", equirect_w, equirect_h, face_size)
    cached = _map_cache.get(key)
    if cached is not None:
        return cached
    maps = {}
    for face in FACES:
        d = _face_dirs(face, face_size)
        u, v = _dir_to_equirect_uv(d, equirect_w, equirect_h)
        # Wrap u so sampling across the +/-180 deg seam interpolates correctly.
        maps[face] = (u.astype(np.float32) % equirect_w,
                      v.astype(np.float32))
    _map_cache[key] = maps
    return maps


#: Google Spherical Video V2 `cbmp` layout 0 -- "a grid with 3 columns and 2
#: rows", top row right/left/up and bottom row down/front/back. Mapped onto our
#: axis names, and every tile lands at rotation 0.
#:
#: Derived by matching ffmpeg's v360 c3x2 output tile by tile against
#: `equirect_to_cubemap` of the same frame rather than reasoned from the spec
#: prose: the best match scored 1-2 mean levels against runner-ups at 30-37, so
#: the assignment is unambiguous. A wrong face order or rotation looks fine in
#: a thumbnail and is badly wrong in a headset, which is not a thing to guess.
CUBEMAP_3X2_LAYOUT = (
    ("+X", "-X", "+Y"),      # right, left, up
    ("-Y", "+Z", "-Z"),      # down, front, back
)


def cubemap_3x2_to_faces(frame: np.ndarray,
                         padding: int = 0) -> Dict[str, np.ndarray]:
    """Split a packed 3x2 cubemap frame into our per-face dict.

    `padding` is the cbmp box's value: pixels added around each face, which
    are duplicated edge samples and must be cropped away before use.
    """
    h, w = frame.shape[:2]
    if w % 3 or h % 2:
        raise ValueError(
            f"A 3x2 cubemap needs dimensions divisible by 3 and 2; got {w}x{h}")
    tile_w, tile_h = w // 3, h // 2
    if tile_w != tile_h:
        raise ValueError(
            f"3x2 cubemap faces must be square; {w}x{h} gives {tile_w}x{tile_h} "
            f"tiles. Is this really a 3x2 cubemap?")
    if padding * 2 >= tile_w:
        raise ValueError(f"Cubemap padding {padding} is too large for "
                         f"{tile_w}px tiles")

    faces: Dict[str, np.ndarray] = {}
    for row, names in enumerate(CUBEMAP_3X2_LAYOUT):
        for col, name in enumerate(names):
            y0, x0 = row * tile_h + padding, col * tile_w + padding
            y1, x1 = (row + 1) * tile_h - padding, (col + 1) * tile_w - padding
            faces[name] = np.ascontiguousarray(frame[y0:y1, x0:x1])
    return faces


def equirect_to_cubemap(equirect: np.ndarray, face_size: int) -> Dict[str, np.ndarray]:
    """Sample an equirect image (H, W, C) into six cubemap faces."""
    h, w = equirect.shape[:2]
    maps = _equirect_to_face_maps(w, h, face_size)
    out = {}
    for face in FACES:
        map_x, map_y = maps[face]
        out[face] = cv2.remap(
            equirect, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
    return out


def _face_local_coords(d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For direction vectors (..., 3), return (face_index, a, b) of the owning face."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)

    # Guard against division by zero at exact axis-aligned directions; those
    # pixels are claimed by another face's mask anyway.
    ax = np.where(ax == 0, 1.0, ax)
    ay = np.where(ay == 0, 1.0, ay)
    az = np.where(az == 0, 1.0, az)

    # Dominant axis selection, matching FACES ordering: 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z
    # (computed on the *original* magnitudes so zero components lose the tie-break)
    ax0, ay0, az0 = np.abs(x), np.abs(y), np.abs(z)
    face = np.zeros(x.shape, dtype=np.int32)
    a = np.zeros(x.shape, dtype=np.float32)
    b = np.zeros(x.shape, dtype=np.float32)

    def assign(mask, idx, a_expr, b_expr):
        face[mask] = idx
        a[mask] = a_expr[mask] if isinstance(a_expr, np.ndarray) else a_expr
        b[mask] = b_expr[mask] if isinstance(b_expr, np.ndarray) else b_expr

    m = (ax0 >= ay0) & (ax0 >= az0) & (x > 0)
    assign(m, 0, -z / ax, -y / ax)
    m = (ax0 >= ay0) & (ax0 >= az0) & (x <= 0)
    assign(m, 1, z / ax, -y / ax)
    m = (ay0 > ax0) & (ay0 >= az0) & (y > 0)
    assign(m, 2, x / ay, z / ay)
    m = (ay0 > ax0) & (ay0 >= az0) & (y <= 0)
    assign(m, 3, x / ay, -z / ay)
    m = (az0 > ax0) & (az0 > ay0) & (z > 0)
    assign(m, 4, x / az, -y / az)
    m = (az0 > ax0) & (az0 > ay0) & (z <= 0)
    assign(m, 5, -x / az, -y / az)

    return face, a, b


def cubemap_to_equirect(
    faces: Dict[str, np.ndarray], out_w: int, out_h: int,
    chunk_rows: int = 512,
    nearest: bool = False,
) -> np.ndarray:
    """Reassemble an equirect image from six cubemap faces (all same size).

    Faces are stacked into a single atlas and sampled with one remap call per
    row band. Remap tables are computed per band (not cached full-frame) to
    keep memory bounded at 8K: a cached 7680x3840 map pair costs ~236 MB.

    nearest: sample with INTER_NEAREST instead of INTER_LINEAR. When False
        (bilinear), each face block is first padded by 2 px with the
        *neighbouring face's* edge pixels (fetched via `_face_local_coords`
        on directions just outside each face edge, and mirrored along the
        edge so directions project back into the padded band). Bilinear taps
        near a face-block border then straddle the geometrically adjacent
        neighbour's values instead of an unrelated atlas block or the zero
        border, avoiding phantom depth gradients along the cube seams while
        keeping depth edges alias-free (nearest-neighbour reassembly
        stair-steps thin structures with frame-varying phase).
    """
    face_size = faces[FACES[0]].shape[0]
    sample = faces[FACES[0]]
    channels = sample.shape[2] if sample.ndim == 3 else 1

    # Build the atlas: order matches FACES list order. For bilinear sampling
    # of single-channel (depth) maps, pad each block with 2 px of neighbour
    # content so cross-block taps stay geometrically meaningful.
    pad = 0 if (nearest or channels > 1) else 2
    if pad:
        blocks = [_pad_face_cached(f, faces, pad) for f in FACES]
        fs = face_size + 2 * pad
    else:
        blocks = [faces[f] for f in FACES]
        fs = face_size

    top = np.concatenate(blocks[:3], axis=1)
    bottom = np.concatenate(blocks[3:], axis=1)
    atlas = np.concatenate([top, bottom], axis=0)

    out_shape = (out_h, out_w) + ((channels,) if channels > 1 else ())
    out = np.empty(out_shape, dtype=sample.dtype)

    # Frame-invariant tables, built once per output geometry.
    map_x, map_y = _face_to_equirect_maps(out_w, out_h, face_size, pad)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR

    for y0 in range(0, out_h, chunk_rows):
        y1 = min(y0 + chunk_rows, out_h)
        # Row-band slices of a C-contiguous array are themselves contiguous.
        out[y0:y1] = cv2.remap(
            atlas, map_x[y0:y1], map_y[y0:y1],
            interpolation=interp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    if channels == 1:
        out = out[..., None]
    return out


def _sample_face(face_img: np.ndarray, a: np.ndarray,
                 b: np.ndarray) -> np.ndarray:
    """Bilinearly sample a face at local coords (a, b) in [-1, 1]. (N,) -> (N,)"""
    f = face_img.shape[0]
    x = ((np.asarray(a) + 1.0) * 0.5 * f - 0.5).astype(np.float32).reshape(1, -1)
    y = ((np.asarray(b) + 1.0) * 0.5 * f - 0.5).astype(np.float32).reshape(1, -1)
    out = cv2.remap(np.ascontiguousarray(face_img, dtype=np.float32), x, y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return out.ravel()


_pad_plan_cache: Dict[tuple, tuple] = {}


def _pad_plan(face: str, f: int, pad: int) -> tuple:
    """Where each pad-ring pixel of `face` reads from. Cached.

    Which neighbour a ring pixel comes from, and at what local coordinates, is
    pure geometry -- it depends only on (face, face_size, pad), never on the
    pixel data. Recomputing it per frame dominated the depth assembly: at 8K
    the six `_pad_face` calls cost 0.235 s of a 0.307 s `cubemap_to_equirect`,
    almost all of it building a full `np.mgrid` over the padded face (two
    1924x1924 index arrays, 59 MB) to locate a 15,376-pixel ring.

    Returns (ring_ys, ring_xs, [(source_face, sel, map_x, map_y), ...]) so the
    per-frame work is a handful of small remaps and one scatter.
    """
    key = (face, f, pad)
    hit = _pad_plan_cache.get(key)
    if hit is not None:
        return hit

    fs = f + 2 * pad
    # The ring as four strips, without materialising a full grid.
    rows_top = np.arange(pad)
    rows_bot = np.arange(fs - pad, fs)
    rows_mid = np.arange(pad, fs - pad)
    cols_all = np.arange(fs)
    cols_side = np.concatenate([np.arange(pad), np.arange(fs - pad, fs)])
    ys = np.concatenate([
        np.repeat(rows_top, fs), np.repeat(rows_bot, fs),
        np.repeat(rows_mid, cols_side.size)])
    xs = np.concatenate([
        np.tile(cols_all, pad), np.tile(cols_all, pad),
        np.tile(cols_side, rows_mid.size)])

    origin, right, down = _FACE_BASIS[face]
    step = 2.0 / f
    a = ((xs - pad) + 0.5) * step - 1.0
    b = ((ys - pad) + 0.5) * step - 1.0
    d = (origin[None, :].astype(np.float64)
         + a[:, None] * right[None, :]
         + b[:, None] * down[None, :])
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    fj, aj, bj = _face_local_coords(d)

    plan = []
    for j, other in enumerate(FACES):
        sel = np.flatnonzero(fj == j)
        if sel.size == 0:
            continue
        mx = ((aj[sel] + 1.0) * 0.5 * f - 0.5).astype(np.float32).reshape(1, -1)
        my = ((bj[sel] + 1.0) * 0.5 * f - 0.5).astype(np.float32).reshape(1, -1)
        plan.append((other, sel, mx, my))

    out = (ys.astype(np.int32), xs.astype(np.int32), plan)
    _pad_plan_cache[key] = out
    return out


def _pad_face_cached(face: str, faces: Dict[str, np.ndarray],
                     pad: int) -> np.ndarray:
    """Pad `faces[face]` with its neighbours' content, using a cached plan."""
    img = faces[face]
    f = img.shape[0]
    fs = f + 2 * pad
    ys, xs, plan = _pad_plan(face, f, pad)

    out = np.zeros((fs, fs), dtype=np.float32)
    out[pad:pad + f, pad:pad + f] = img
    vals = np.empty(ys.size, dtype=np.float32)
    for other, sel, mx, my in plan:
        src = np.ascontiguousarray(faces[other], dtype=np.float32)
        vals[sel] = cv2.remap(src, mx, my, interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE).ravel()
    out[ys, xs] = vals
    return out


def _pad_face(face_img: np.ndarray, faces: Dict[str, np.ndarray],
              pad: int) -> np.ndarray:
    """Pad a face block with `pad` px of its geometric neighbours' content.

    For each pixel of the pad ring, build the world direction just outside
    the face (local coord mirrored outwards by up to `pad` pixels), find the
    owning face via `_face_local_coords`, and sample that face bilinearly.
    Result: (F + 2*pad, F + 2*pad) block whose border region carries the
    adjacent faces' edge values, so bilinear taps near the block edge never
    read an unrelated atlas block or the zero border.
    """
    f = face_img.shape[0]
    fs = f + 2 * pad
    out = np.zeros((fs, fs), dtype=np.float32)
    out[pad:pad + f, pad:pad + f] = face_img

    face = FACES[[i for i, name in enumerate(FACES)
                  if faces[name] is face_img or
                  np.array_equal(faces[name], face_img)][0]]
    origin, right, down = _FACE_BASIS[face]

    step = 2.0 / f  # local-coord units per pixel
    ys, xs = np.mgrid[0:fs, 0:fs]
    ring = (ys < pad) | (ys >= fs - pad) | (xs < pad) | (xs >= fs - pad)
    if not ring.any():
        return out

    # Pixel centers in local coords: interior pixel i -> ((i - pad) + 0.5)
    # mapped into [-1, 1]; pad pixels extend beyond +-1.
    a = ((xs[ring] - pad) + 0.5) * step - 1.0
    b = ((ys[ring] - pad) + 0.5) * step - 1.0
    d = (origin[None, :].astype(np.float64)
         + a[:, None] * right[None, :]
         + b[:, None] * down[None, :])
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    fj, aj, bj = _face_local_coords(d)
    vals = np.empty(a.shape, dtype=np.float32)
    for j, other in enumerate(FACES):
        m = fj == j
        if m.any():
            vals[m] = _sample_face(faces[other], aj[m], bj[m])
    out[ring] = vals
    return out


# (axis, sign) of each of a face's four edges in local (a, b) coords.
_EDGE_SPECS = (("a", 1.0), ("a", -1.0), ("b", 1.0), ("b", -1.0))


def align_face_scales(
    disp_faces: Dict[str, np.ndarray],
    delta: float = 0.02,
    samples: int = 64,
    max_scale: float = 4.0,
) -> Dict[str, np.ndarray]:
    """Bring six per-face relative inverse-depth maps onto a COMMON scale.

    Monocular relative-depth backends (Depth Anything, Video Depth Anything)
    output depth whose absolute scale is arbitrary *per inference*. Estimating
    each cubemap face separately therefore gives each face its own scale: a
    structure crossing a cube edge gets two different depths, and because each
    face's scale drifts independently from frame to frame, that mismatch
    changes every frame — which the warp turns into moving geometric
    distortion.

    These backends are *affine*-invariant: predictions relate to true
    disparity by d_true ≈ s·d_pred + t with both s and t arbitrary per
    inference. A scale-only solve cannot reconcile two faces with mismatched
    offsets (the seam residual varies with depth along the edge and drifts
    per frame), so the full affine pair (s_k, t_k) is solved per face: for
    every shared edge, matching world directions just inside both faces must
    satisfy s_i·d_i + t_i = s_j·d_j + t_j. All edge sample pairs go into one
    least-squares system with gauge constraints mean(s)=1, mean(t)=0,
    refined with three Huber/IRLS iterations. Scales are clipped to
    [1/max_scale, max_scale] and shifts to |t| ≤ 0.5·median(d); a degenerate
    solve falls back to the scale-only result.

    Modifies `disp_faces` in place and returns it. A degenerate solve (uniform
    or non-positive depth) leaves the maps untouched.
    """
    eps = 1e-6
    pairs: List[Tuple[int, int, np.ndarray, np.ndarray]] = []

    t = (np.arange(samples) + 0.5) / samples * 2.0 - 1.0

    for face in FACES:
        i = FACES.index(face)
        origin, right, down = _FACE_BASIS[face]
        for axis, sign in _EDGE_SPECS:
            if axis == "a":
                aa, bb = np.full(samples, sign), t
                a_out, b_out = np.full(samples, sign * (1.0 + delta)), t
            else:
                aa, bb = t, np.full(samples, sign)
                a_out, b_out = t, np.full(samples, sign * (1.0 + delta))

            # Nudge just past the edge so the direction lands on the neighbour.
            d_out = (origin[None, :].astype(np.float64)
                     + a_out[:, None] * right[None, :]
                     + b_out[:, None] * down[None, :])
            d_out /= np.linalg.norm(d_out, axis=-1, keepdims=True)

            fj, aj, bj = _face_local_coords(d_out)
            other = fj != i
            if other.sum() < 3:
                continue
            vals, counts = np.unique(fj[other], return_counts=True)
            j = int(vals[counts.argmax()])
            sel = other & (fj == j)
            if sel.sum() < 3:
                continue

            di = _sample_face(disp_faces[face], aa[sel], bb[sel])
            dj = _sample_face(disp_faces[FACES[j]], aj[sel], bj[sel])
            good = (di > eps) & (dj > eps)
            if good.sum() < 3:
                continue

            pairs.append((i, j, di[good].astype(np.float64),
                          dj[good].astype(np.float64)))

    return _fit_and_apply(pairs, disp_faces, max_scale)


def _fit_and_apply(
    pairs: List[Tuple[int, int, np.ndarray, np.ndarray]],
    disp_faces: Dict[str, np.ndarray],
    max_scale: float,
) -> Dict[str, np.ndarray]:
    """Fit one affine (s_k, t_k) per face from sample pairs, and apply it.

    Shared by both ways of collecting the pairs: `align_face_scales` samples a
    strip at the shared edge, `align_overlapping_faces` samples the whole band
    two widened faces have in common. The fit itself does not care which —
    every pair is just "these two faces saw this same direction".
    """
    if not pairs:
        return disp_faces

    # Scale-only candidate: log-space solve, robust medians per edge, gauge
    # sum of log-scales = 0 (preserves overall magnitude).
    scale_only = None
    rows = []
    rhs = []
    for i, j, di, dj in pairs:
        row = np.zeros(len(FACES), dtype=np.float64)
        row[i], row[j] = 1.0, -1.0
        rows.append(row)
        rhs.append(float(np.log(np.median(dj)) - np.log(np.median(di))))
    rows.append(np.ones(len(FACES), dtype=np.float64))
    rhs.append(0.0)
    try:
        x, *_ = np.linalg.lstsq(np.stack(rows), np.asarray(rhs), rcond=None)
    except np.linalg.LinAlgError:
        x = None
    if x is not None and np.all(np.isfinite(x)):
        scales = np.exp(x - x.mean())
        lo, hi = 1.0 / max_scale, max_scale
        scale_only = [(float(np.clip(scales[k], lo, hi)), 0.0)
                      for k in range(len(FACES))]

    # Affine solve: d_i*s_i + t_i = d_j*s_j + t_j over all samples.
    # A joint least-squares over (s, t) is ill-conditioned on real data (a
    # face whose shared depths lack variance trades huge offsets against its
    # scale, and occluding contours violate the continuity assumption), so it
    # is solved by robust alternation instead: median-based shift solves given
    # the scales, median-based log-scale solves given the shifts — each small
    # solve as robust as the scale-only one above.
    affine = _solve_face_affine(pairs, disp_faces, max_scale, scale_only)

    # Pick whichever candidate leaves the smaller residual (clipping of a
    # degenerate affine solution can make it worse than scale-only).
    candidates = [c for c in (affine, scale_only) if c is not None]
    if not candidates:
        return disp_faces
    best = min(candidates, key=lambda c: _seam_residual(pairs, c))
    for k, face in enumerate(FACES):
        s, tt = best[k]
        if abs(s - 1.0) > 1e-9 or abs(tt) > 1e-9:
            disp_faces[face] = np.clip(disp_faces[face] * s + tt, 0.0, None)
    return disp_faces


def _seam_residual(
    pairs: List[Tuple[int, int, np.ndarray, np.ndarray]],
    params: List[Tuple[float, float]],
) -> float:
    """RMS of s_i*d_i + t_i - (s_j*d_j + t_j) over all edge sample pairs."""
    num, den = 0.0, 0
    for i, j, di, dj in pairs:
        si, ti = params[i]
        sj, tj = params[j]
        r = si * di + ti - (sj * dj + tj)
        num += float((r * r).sum())
        den += len(r)
    return (num / max(den, 1)) ** 0.5


def _solve_face_affine(
    pairs: List[Tuple[int, int, np.ndarray, np.ndarray]],
    disp_faces: Dict[str, np.ndarray],
    max_scale: float,
    scale_only: List[Tuple[float, float]] | None = None,
    outer_iters: int = 4,
) -> List[Tuple[float, float]] | None:
    """Solve per-face affine corrections (s_k, t_k) from edge sample pairs.

    Robust alternation, starting from the scale-only solution (or identity):
      * shift step: t_i - t_j = median(s_j*d_j - s_i*d_i) per edge — a small
        well-conditioned lstsq exactly like the scale-only log solve;
      * scale step: log-space ratio solve on the shift-corrected depths
        s_k*d_k + t_k (positive samples only).
    Both steps use per-edge medians, so occluding-contour outliers at seams
    cannot dominate, and neither step can trade scale against offset along an
    unidentifiable direction. Returns [(s_k, t_k)] or None if degenerate.
    """
    nf = len(FACES)
    eps = 1e-6
    lo, hi = 1.0 / max_scale, max_scale
    # Only a magnitude, used to cap |t|. Taking it over every pixel of all six
    # faces cost 0.118 s of a 0.218 s solve at 8K; a strided subsample gives
    # the same number to far more precision than a cap needs.
    med = float(np.median(np.concatenate(
        [np.asarray(disp_faces[f], dtype=np.float64).ravel()[::64]
         for f in FACES])))
    if not np.isfinite(med) or med <= 0:
        return None
    t_cap = 0.5 * med

    s = np.array([sc for sc, _ in scale_only], dtype=np.float64) \
        if scale_only is not None else np.ones(nf)
    t = np.zeros(nf, dtype=np.float64)
    diff = np.eye(nf)  # row template

    for _ in range(outer_iters):
        # ---- shift step ----
        rows, rhs = [], []
        for i, j, di, dj in pairs:
            r_ij = float(np.median(s[j] * dj) - np.median(s[i] * di))
            row = np.zeros(nf)
            row[i], row[j] = 1.0, -1.0   # t_i - t_j = median(s_j d_j - s_i d_i)
            rows.append(row)
            rhs.append(r_ij)
        rows.append(np.ones(nf))         # gauge: mean(t) = 0
        rhs.append(0.0)
        try:
            t, *_ = np.linalg.lstsq(np.stack(rows), np.asarray(rhs), rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(t)):
            return None
        t = np.clip(t, -t_cap, t_cap)

        # ---- scale step ----
        rows, rhs = [], []
        for i, j, di, dj in pairs:
            vi = s[i] * di + t[i]
            vj = s[j] * dj + t[j]
            good = (vi > eps) & (vj > eps)
            if good.sum() < 3:
                continue
            mi, mj = float(np.median(vi[good])), float(np.median(vj[good]))
            if mi <= eps or mj <= eps:
                continue
            row = np.zeros(nf)
            row[i], row[j] = 1.0, -1.0   # g_i - g_j = log(mj / mi)
            rows.append(row)
            rhs.append(np.log(mj) - np.log(mi))
        if not rows:
            break
        rows.append(np.ones(nf))         # gauge: mean log-gain = 0
        rhs.append(0.0)
        try:
            g, *_ = np.linalg.lstsq(np.stack(rows), np.asarray(rhs), rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(g)):
            return None
        s = np.clip(s * np.exp(g - g.mean()), lo, hi)

    # Final consistency: re-solve shifts for the final scales.
    rows, rhs = [], []
    for i, j, di, dj in pairs:
        row = np.zeros(nf)
        row[i], row[j] = 1.0, -1.0
        rows.append(row)
        rhs.append(float(np.median(s[j] * dj) - np.median(s[i] * di)))
    rows.append(np.ones(nf))
    rhs.append(0.0)
    try:
        t, *_ = np.linalg.lstsq(np.stack(rows), np.asarray(rhs), rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(t)):
        return None
    t = np.clip(t, -t_cap, t_cap)
    return [(float(s[k]), float(t[k])) for k in range(nf)]


# ---------------------------------------------------------------------------
# Overlapping faces (depth only)
# ---------------------------------------------------------------------------
#
# Six exact 90-degree faces tile the sphere with no slack, so neighbours share
# nothing but a line. That is fine for resampling an image, where the pixels
# either side of the line are the same pixels. It is not fine for depth:
# relative-depth models emit an arbitrary affine scale per inference, and with
# only a one-pixel strip to fit it from, whatever disagreement survives the fit
# lands exactly on the seam. Measured on 8K footage the depth step across a
# seam was 52x the step between ordinary neighbouring pixels (240x at the
# front/down edge), which reads in a headset as the ground creasing -- a patch
# of it detaching and floating forward.
#
# Widening each face past 90 degrees gives neighbours a real region in common.
# The fit gets a 2-D sample instead of a border strip, and the assembly
# cross-fades across the shared band, so any residual is spread over degrees
# rather than concentrated into a line. It costs angular resolution -- the
# model resizes its input to a fixed size regardless of the field of view it
# covers -- which is why the overlap is no wider than it needs to be.

#: Extra reach past the nominal face edge, in tangent units: a face spans
#: [-(1+FACE_OVERLAP), 1+FACE_OVERLAP] instead of [-1, 1]. 0.15 is 98 degrees
#: per face.
#:
#: Two things bound this from opposite sides. Too little and the seam step
#: survives. Too much and the tangent projection stretches the corners past
#: what a monocular depth model was trained on -- 90 degrees is close to a
#: normal lens, and the corner of a face sits at 54.7 degrees off-axis with a
#: 5.2x area stretch, rising to 58.4 degrees and 7.0x here and 60.5 degrees
#: and 8.4x at 103 degrees.
#:
#: Measured both, on two 8K clips and three model sizes. Ground-plane fidelity
#: (how well predicted depth over real flat ground stays linear in the view
#: direction, which is exact geometry for any plane) is unchanged at 98
#: degrees for every model -- 1.00-1.07x of the 90-degree error for Large,
#: better than 90 degrees for Small and Base. At 103 degrees Large loses
#: 13-20%. Meanwhile the seam step is already dead at 98 degrees: 3.5x the
#: ordinary neighbour step against 313x at 90 degrees, and the 95th percentile
#: (0.0037) is if anything better than at 103 degrees (0.0038).
#:
#: So the wider face buys nothing on the seam and costs real accuracy on the
#: largest model. 98 degrees also gives up less angular resolution -- 9%
#: against 15% -- since the model resizes its input whatever field of view it
#: covers.
FACE_OVERLAP = 0.15


def face_fov_degrees(overlap: float = FACE_OVERLAP) -> float:
    """Field of view one face covers, in degrees."""
    return float(2.0 * np.degrees(np.arctan(1.0 + overlap)))


#: How much of the ray-versus-axis correction to apply to each face's depth.
#: Off by default: measured well on two scenes, never yet judged in a headset.
#:
#: Depth Anything V3 predicts its own camera, and that prediction saturates.
#: Fed the same six views at a sweep of fields of view it answers 58.9 degrees
#: for a real 61.9, then 64.3 for 73.7, then 65.7 for 90, 65.6 for our 98 and
#: 67.9 for 106.9 -- it tracks the truth to about 62 degrees and then stops.
#: So at the 98-degree faces this pipeline uses it under-reads by a third, and
#: reconstructs depth for a much longer lens than it was given.
#:
#: The consequence is geometric and one-sided. A ray genuinely 45 degrees off
#: the face axis -- which is exactly where the cube seam falls -- is treated as
#: though it were about 30 degrees off, so the distance it must travel to reach
#: a surface is under-estimated, and the surface is placed too near. On flat
#: ground the estimate is faithful at the face centre and lifts steadily toward
#: the edge: measured against exact plane geometry it reaches 0.18 camera
#: heights of false elevation by the seam, which is the ground visibly bulging
#: up toward the viewer at about one camera height out.
#:
#: The correction divides by 1 + strength*(sec(theta) - 1), where sec(theta) is
#: the ray's own foreshortening: 1.00 at the face centre, 1.41 at the nominal
#: cube edge, 1.91 at the widened corner. strength=1 is the full ray-versus-
#: axis conversion and overshoots, landing the ground 0.12 camera heights
#: *below* true. Between them:
#:
#:   strength   worst floor error out to 1.2 camera heights
#:      0.0     0.196   (today)
#:      0.4     0.079
#:      0.55    0.032
#:      0.7     0.036
#:      1.0     0.127
#:
#: On the reference photo's scorecard, 0.55 takes wall wobble from 19.6% to
#: 8.5% and floor rms from 27.7% to 24.5%, and 0.7 is better again on the wall.
#: Depth span falls with it -- 1.30 to 1.07 -- but that is the point rather
#: than a cost: by latitude the loss is 3-5% on the bands sitting at face
#: centres and 13-15% on the bands at 45 degrees, so what is being removed is
#: the false nearness at the peripheries. The other three scores are ratios and
#: scale-invariant, so `--strength 1.2` restores the parallax for nothing.
#:
#: 0.55 is where the outdoor floor geometry lands and 0.7 where the indoor wall
#: does; the optimum is not sharp and anything in that band is a large win on
#: both. It is an empirical constant, not a derived one -- the model's own
#: predicted intrinsics imply about 0.36 -- so it is tied to V3 at this face
#: width and should not be assumed to transfer to another backend.
ANGULAR_CORRECTION = 0.0

_angular_cache: Dict[tuple, np.ndarray] = {}


def angular_correction_table(face_size: int, overlap: float,
                             strength: float) -> np.ndarray:
    """Per-pixel divisor 1 + strength*(sec(theta) - 1) for a widened face.

    Depends only on the face geometry, never on pixel data, so it is the same
    for every face and every frame of a run -- the same reasoning that keeps
    the remap tables cached. One entry per geometry, and a face-sized float32
    array is small beside the tables it sits next to (15 MB at an 8K face
    against ~540 MB for the blend plan).
    """
    key = (int(face_size), round(float(overlap), 6), round(float(strength), 6))
    hit = _angular_cache.get(key)
    if hit is not None:
        return hit
    lim = 1.0 + overlap
    t = ((np.arange(face_size, dtype=np.float32) + 0.5) / face_size
         * 2.0 - 1.0) * lim
    a, b = np.meshgrid(t, t)
    sec = np.sqrt(1.0 + a * a + b * b, dtype=np.float32)
    out = (1.0 + strength * (sec - 1.0)).astype(np.float32)
    _angular_cache.clear()
    _angular_cache[key] = out
    return out


def apply_angular_correction(depth_faces: Dict[str, np.ndarray],
                             overlap: float = FACE_OVERLAP,
                             strength: float = ANGULAR_CORRECTION,
                             ) -> Dict[str, np.ndarray]:
    """Pull each face's periphery back onto its true rays. In place.

    Runs *before* `align_overlapping_faces`, not after: the correction changes
    what each face says in the band it shares with its neighbours, so fitting
    the faces together first would fit them on values about to move. Doing it
    in this order also improves the seam agreement it is not aiming at -- the
    spread between faces over the lower hemisphere fell from 6.9% to 4.5% of
    local depth, because the faces were disagreeing precisely where each was
    least reliable.
    """
    if strength <= 0.0:
        return depth_faces
    face_size = depth_faces[FACES[0]].shape[0]
    div = angular_correction_table(face_size, overlap, strength)
    for face in FACES:
        depth_faces[face] = (np.asarray(depth_faces[face], dtype=np.float32)
                             / div)
    return depth_faces


def _axis_component(vec: np.ndarray) -> Tuple[int, float]:
    """(index, sign) of an axis-aligned basis vector.

    Every entry of `_FACE_BASIS` is a signed unit axis, so projecting a
    direction onto one is a strided read and a sign flip rather than a dot
    product over a full (rows, w, 3) band.
    """
    idx = int(np.argmax(np.abs(vec)))
    return idx, float(np.sign(vec[idx]))


def _overlap_face_dirs(face: str, face_size: int,
                       overlap: float) -> np.ndarray:
    """Unit directions for every pixel of a widened face. (F, F, 3)"""
    origin, right, down = _FACE_BASIS[face]
    lim = 1.0 + overlap
    coords = ((np.arange(face_size) + 0.5) / face_size * 2.0 - 1.0) * lim
    a, b = np.meshgrid(coords, coords)
    d = (origin[None, None, :]
         + a[..., None] * right[None, None, :]
         + b[..., None] * down[None, None, :])
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _overlap_weight(x: np.ndarray, overlap: float) -> np.ndarray:
    """Cross-fade weight: 1 inside the nominal face, 0 at the outer reach.

    A raised cosine over the shared band, so the blend is smooth and no
    derivative discontinuity survives at either end of the fade. At the
    nominal seam both neighbours weigh 0.5, and the face that owns a
    direction never weighs less than 0.25, so the weight sum cannot vanish.
    """
    lo, hi = 1.0 - overlap, 1.0 + overlap
    t = np.clip((np.abs(x) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return 0.5 + 0.5 * np.cos(np.pi * t)


def equirect_to_overlapping_faces(
    equirect: np.ndarray, face_size: int,
    overlap: float = FACE_OVERLAP,
) -> Dict[str, np.ndarray]:
    """Sample an equirect image into six overlapping cubemap faces."""
    h, w = equirect.shape[:2]
    key = ("e2fov", w, h, face_size, round(overlap, 6))
    maps = _map_cache.get(key)
    if maps is None:
        maps = {}
        for face in FACES:
            d = _overlap_face_dirs(face, face_size, overlap)
            u, v = _dir_to_equirect_uv(d, w, h)
            maps[face] = (u.astype(np.float32) % w, v.astype(np.float32))
        _map_cache[key] = maps
    return {face: cv2.remap(equirect, maps[face][0], maps[face][1],
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_WRAP)
            for face in FACES}


_gather_plan_cache: Dict[tuple, list] = {}


def _overlap_gather_plan(face: str, face_size: int, overlap: float) -> list:
    """Where each pixel of a widened face reads from in an exact cubemap.

    Pure geometry, so cached. Returns [(source_face, flat_indices, map_x,
    map_y)] covering every pixel exactly once -- the core of the face comes
    from its own source face and the widened border from the neighbours.
    """
    key = (face, face_size, round(overlap, 6))
    hit = _gather_plan_cache.get(key)
    if hit is not None:
        return hit

    d = _overlap_face_dirs(face, face_size, overlap)
    fj, aj, bj = _face_local_coords(d.reshape(-1, 3))
    plan = []
    for j, other in enumerate(FACES):
        sel = np.flatnonzero(fj == j)
        if sel.size == 0:
            continue
        mx = ((aj[sel] + 1.0) * 0.5 * face_size - 0.5)
        my = ((bj[sel] + 1.0) * 0.5 * face_size - 0.5)
        plan.append((other, sel, mx.astype(np.float32).reshape(1, -1),
                     my.astype(np.float32).reshape(1, -1)))
    _gather_plan_cache[key] = plan
    return plan


def cubemap_to_overlapping_faces(
    faces: Dict[str, np.ndarray],
    overlap: float = FACE_OVERLAP,
) -> Dict[str, np.ndarray]:
    """Widen exact 90-degree faces in place of a trip through equirect.

    A cubemap source already carries its own faces; going face -> equirect ->
    widened face would resample twice for a geometry that needs one step.
    """
    face_size = faces[FACES[0]].shape[0]
    sample = faces[FACES[0]]
    channels = sample.shape[2] if sample.ndim == 3 else 1
    out = {}
    for face in FACES:
        buf = np.empty((face_size * face_size, channels), dtype=sample.dtype)
        for other, sel, mx, my in _overlap_gather_plan(face, face_size,
                                                       overlap):
            got = cv2.remap(faces[other], mx, my,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
            buf[sel] = got.reshape(-1, channels)
        widened = buf.reshape(face_size, face_size, channels)
        out[face] = widened if channels > 1 else widened[..., 0]
    return out


def _face_ab(face: str, dirs: np.ndarray, lim: float):
    """(a, b, keep) for one face over a block of direction vectors.

    The basis vectors are signed unit axes, so projecting onto them is a
    strided read and a sign flip rather than a dot product over the block.
    """
    (oi, osg), (ri, rsg), (di, dsg) = (_axis_component(v)
                                       for v in _FACE_BASIS[face])
    t = dirs[..., oi] * np.float32(osg)
    # Behind the face plane: no amount of scaling brings it into view.
    front = t > 1e-6
    safe = np.where(front, t, np.float32(1.0))
    a = (dirs[..., ri] * np.float32(rsg)) / safe
    b = (dirs[..., di] * np.float32(dsg)) / safe
    return a, b, front & (np.abs(a) <= lim) & (np.abs(b) <= lim)


def _true_runs(mask: np.ndarray) -> list:
    """Contiguous [start, stop) runs of True in a 1-D mask."""
    edges = np.flatnonzero(np.diff(
        np.concatenate(([False], mask, [False])).astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


_overlap_plan_cache: Dict[tuple, list] = {}


def _overlap_blend_plan(out_w: int, out_h: int, face_size: int,
                        overlap: float, chunk_rows: int = 512) -> list:
    """Cached per-face blend tables: [(face, y0, y1, x0, x1, map_x, map_y,
    weight)], where `weight` is already divided by the total weight at that
    pixel, so assembling a frame is a remap and a multiply-add per entry.

    Frame-invariant, and rebuilding it per frame is what made the first
    version of this 2.5 s a frame at 8K against 0.07 s for the exact-face
    assembly -- 36x, and essentially all of it table construction rather than
    the six remaps the tables feed.

    Each entry covers only the rows and columns its face actually reaches, so
    the six of them together span about 1.5 frames rather than 6. The back
    face straddles the +/-180 degree wrap and gets two column runs; that is
    why an entry is a box rather than a face.

    One geometry is kept, as in `_face_to_equirect_maps` and for the same
    reason: at 8K the tables are ~540 MB and a run uses a single geometry
    throughout.
    """
    key = (out_w, out_h, face_size, round(overlap, 6))
    hit = _overlap_plan_cache.get(key)
    if hit is not None:
        return hit
    _overlap_plan_cache.clear()

    lim = 1.0 + overlap
    bands = [(y, min(y + chunk_rows, out_h))
             for y in range(0, out_h, chunk_rows)]

    # Pass 1: how far each face reaches, so pass 2 can allocate exactly.
    extent = {}
    for face in FACES:
        rows = np.zeros(out_h, bool)
        cols = np.zeros(out_w, bool)
        for y0, y1 in bands:
            _, _, keep = _face_ab(
                face, equirect_rows_to_dir(y0, y1, out_w, out_h), lim)
            rows[y0:y1] = keep.any(axis=1)
            cols |= keep.any(axis=0)
        extent[face] = (rows, cols)

    # Pass 2: fill the tables, and the weight total they will be normalised by.
    total = np.zeros((out_h, out_w), np.float32)
    plan = []
    for face in FACES:
        rows, cols = extent[face]
        if not rows.any():
            continue
        ys = np.flatnonzero(rows)
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        for x0, x1 in _true_runs(cols):
            shape = (y1 - y0, x1 - x0)
            mx = np.empty(shape, np.float32)
            my = np.empty(shape, np.float32)
            wt = np.empty(shape, np.float32)
            for by0, by1 in bands:
                by0, by1 = max(by0, y0), min(by1, y1)
                if by0 >= by1:
                    continue
                d = equirect_rows_to_dir(by0, by1, out_w, out_h)[:, x0:x1]
                a, b, keep = _face_ab(face, d, lim)
                s = slice(by0 - y0, by1 - y0)
                np.multiply(_overlap_weight(a, overlap),
                            _overlap_weight(b, overlap), out=wt[s])
                wt[s] *= keep
                mx[s] = (a / lim + 1.0) * 0.5 * face_size - 0.5
                my[s] = (b / lim + 1.0) * 0.5 * face_size - 0.5
            total[y0:y1, x0:x1] += wt
            plan.append([face, y0, y1, x0, x1, mx, my, wt])

    # Fold the normalisation into the weights: every direction is covered, so
    # the total is never zero (the owning face alone contributes at least
    # 0.25), and pre-dividing removes a full-frame division per frame.
    np.maximum(total, 1e-6, out=total)
    for entry in plan:
        _, y0, y1, x0, x1, _, _, wt = entry
        wt /= total[y0:y1, x0:x1]

    plan = [tuple(e) for e in plan]
    _overlap_plan_cache[key] = plan
    return plan


def overlapping_faces_to_equirect(
    depth_faces: Dict[str, np.ndarray], out_w: int, out_h: int,
    overlap: float = FACE_OVERLAP,
) -> np.ndarray:
    """Cross-fade six overlapping single-channel faces into an equirect map.

    Every direction is covered by the face that owns it and, near a nominal
    edge, by one or two neighbours that reach past it; each contributes in
    proportion to how far inside its own field of view the direction sits.
    Because the weight reaches zero before the face does, the outermost ring
    of pixels -- where a depth model is least reliable, and where `remap`
    would otherwise have to invent border values -- contributes nothing.
    """
    face_size = depth_faces[FACES[0]].shape[0]
    out = np.zeros((out_h, out_w), np.float32)
    for face, y0, y1, x0, x1, mx, my, wt in _overlap_blend_plan(
            out_w, out_h, face_size, overlap):
        got = cv2.remap(depth_faces[face], mx, my,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        got *= wt
        out[y0:y1, x0:x1] += got
    return out


def align_overlapping_faces(
    depth_faces: Dict[str, np.ndarray],
    overlap: float = FACE_OVERLAP,
    samples: int = 96,
    max_scale: float = 4.0,
) -> Dict[str, np.ndarray]:
    """Bring six overlapping relative-depth faces onto a common scale.

    Same fit as `align_face_scales` and the same reason for it, but the sample
    pairs come from the whole band two faces share rather than a strip at
    their shared edge. That matters more than it sounds: a border strip is
    where a monocular model is least reliable, it is thin enough for a single
    occluding contour to dominate, and the values it holds sit at the extreme
    end of each face's own depth distribution -- the front face's bottom row
    is its nearest content while the down face's top row is its farthest.

    `samples` is the grid resolution across a face, not a pixel stride: what
    the fit needs is a certain number of sample pairs, and that must not
    depend on how many pixels the face happens to have. Roughly a third of a
    grid of 96 lands in the shared band, so each face pair contributes several
    hundred samples at any face size, for a fit that costs milliseconds.
    """
    face_size = depth_faces[FACES[0]].shape[0]
    lim = 1.0 + overlap
    eps = 1e-6
    stride = max(1, face_size // max(samples, 1))
    coords = ((np.arange(0, face_size, stride) + 0.5) / face_size * 2.0 - 1.0)
    coords = coords * lim
    a0, b0 = np.meshgrid(coords, coords)

    pairs: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    for face in FACES:
        i = FACES.index(face)
        origin, right, down = _FACE_BASIS[face]
        d = (origin[None, None, :].astype(np.float64)
             + a0[..., None] * right[None, None, :]
             + b0[..., None] * down[None, None, :])
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        fj, aj, bj = _face_local_coords(d)
        di_all = depth_faces[face][::stride, ::stride].astype(np.float64)
        for j, other in enumerate(FACES):
            if j == i:
                continue
            sel = fj == j
            # A handful of samples cannot pin down a scale; that is a
            # degeneracy guard, not a resolution one -- `samples` above is
            # what keeps the count healthy at any face size.
            if sel.sum() < 16:
                continue
            xs = ((aj[sel] / lim + 1.0) * 0.5 * face_size - 0.5)
            ys = ((bj[sel] / lim + 1.0) * 0.5 * face_size - 0.5)
            dj = cv2.remap(np.ascontiguousarray(depth_faces[other],
                                                dtype=np.float32),
                           xs.astype(np.float32).reshape(1, -1),
                           ys.astype(np.float32).reshape(1, -1),
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE).ravel()
            di = di_all[sel]
            good = (di > eps) & (dj > eps)
            if good.sum() < 16:
                continue
            pairs.append((i, j, di[good], dj[good].astype(np.float64)))

    return _fit_and_apply(pairs, depth_faces, max_scale)
