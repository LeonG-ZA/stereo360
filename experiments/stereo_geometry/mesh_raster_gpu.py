"""The mesh warp as a scanline rasteriser: exact coverage, not scattered points.

`mesh_warp_gpu` moved the prototype's algorithm to the GPU and got 2.9x. The
profile said why that was the ceiling: 90% of the time was `round+mask`,
`zbuffer+scatter` and `blend` -- the brute-force stand-in for a rasteriser --
against 0.7% for the geometry itself. Faster hardware running the wrong
algorithm.

**What the wrong algorithm was.** Each 1x1 source quad was sampled at 16 fixed
points, each point rounded to the nearest output pixel and scattered. That is
wasteful where the warp compresses (16 samples landing on one pixel), starved
where it stretches (4 samples spread across 2.5 px), and wrong everywhere: the
texture coordinate comes from where a *sample* happened to fall rather than
from where the *output pixel* is, so it carries the rounding error into the
resampled image.

**What replaces it.** Take the segment between horizontally adjacent vertices
and ask which output columns it actually covers -- the integers in
`[u0, u1)` -- then solve for the source position at each of those columns
exactly. Three things fall out.

*It tiles.* Segment i ends where segment i+1 begins, so on a connected surface
every output column lies in exactly one segment and is written exactly once.
The z-buffer stops being a contest and becomes a formality except where two
surfaces genuinely overlap. That is what kills the hairlines at longitudes
+/-45 and +/-135 degrees: they came from every sample tying on constant depth
and float32 jitter picking a different winner column by column. With exact
coverage there is no tie to break.

*It is bounded.* A segment wider than `max_stretch` is cut, so coverage never
exceeds `ceil(max_stretch) + 1` columns. The variable-length expansion a
general rasteriser needs -- and which DirectML has no good primitive for --
collapses to a fixed four masked passes.

*It is exact horizontally.* The output column is an integer by construction,
never rounded, and the source coordinate is interpolated to it.

**Why per-row is legitimate.** `_eye_offset` moves a point in the horizontal
plane only: `py` is untouched and the distance grows to `hypot(dist,
baseline)`, so latitude shifts by `-tan(lat) * b^2/(2 dist^2)` -- second order,
and measured below half a pixel over the test frames. The production splat
relies on the same fact, giving its z-buffer a single guard row above and
below. Rows are still allowed to move: the destination row is interpolated and
rounded like any other, with margin rows so a shifted edge is not lost at a
band boundary. It is the *coverage* that is computed per row, not the geometry
that is assumed flat. `_TV_DRIFT` records the largest drift actually seen.
"""
from __future__ import annotations

import numpy as np

try:
    from . import mesh_warp
except ImportError:  # run as a loose script
    import mesh_warp
from stereo360 import projection, warp

#: Rows per band.
BAND = 256

#: Rows of overlap above and below a band, so an edge whose latitude rounds
#: into the band is still generated. One row covers the measured drift; the
#: rest is slack, and costs only geometry, which is 0.7% of the work.
MARGIN = 3

#: Largest |tv - y| seen, in pixels, filled in as a side effect of rendering.
#: If this ever approaches 1.0 the single-guard-row assumption -- which the
#: production splat shares -- needs revisiting rather than trusting.
_TV_DRIFT = {"max": 0.0}

#: Set to a dict to collect z-buffer statistics; None to skip the syncs.
STATS: dict | None = None


def _device():
    """The DirectML device, or None if it is not usable."""
    try:
        import torch  # noqa: F401
        import torch_directml
    except ImportError:
        return None
    try:
        return torch_directml.device()
    except Exception:
        return None


def render_full(dn: np.ndarray, rgb: np.ndarray, baseline: float,
                cut_ratio: float = mesh_warp.CUT_RATIO, subdiv: int = 4,
                band: int = BAND, margin: int = MARGIN,
                min_cut: int = mesh_warp.MIN_CUT,
                max_stretch: float = mesh_warp.MAX_STRETCH_PX,
                dn_cut: np.ndarray | None = None,
                device=None):
    """Same contract as `mesh_warp.render_full`. Returns (image, cut mask).

    `subdiv` is accepted and ignored: there is nothing to subdivide when
    coverage is computed exactly. It stays in the signature so this is a
    drop-in for the two renderers it is measured against.
    """
    import cv2
    import torch

    dev = device if device is not None else _device()
    if dev is None:
        return mesh_warp.render_full(dn, rgb, baseline, cut_ratio, subdiv,
                                     band, margin, min_cut, max_stretch,
                                     dn_cut)

    h, w = dn.shape
    dnx = np.concatenate([dn, dn[:, :1]], axis=1)
    cutx = dnx if dn_cut is None else np.concatenate(
        [dn_cut, dn_cut[:, :1]], axis=1)
    rgbx = np.concatenate([rgb, rgb[:, :1]], axis=1)

    map_x = np.full((h, w), -1.0, np.float32)
    map_y = np.full((h, w), -1.0, np.float32)
    bl = -baseline
    # Coverage of a kept segment is at most its width plus one, and a segment
    # wider than `max_stretch` is cut before it is drawn.
    kmax = int(np.ceil(max_stretch)) + 1
    drift = 0.0

    def project(depth_row, dt):
        lam = 1.0 / (depth_row + warp._MIN_INV_DEPTH)
        px = lam * dt[..., 0] + bl * dt[..., 2]
        pz = lam * dt[..., 2] - bl * dt[..., 0]
        py = lam * dt[..., 1]
        dist = torch.sqrt(px * px + py * py + pz * pz)
        safe = torch.where(dist == 0.0, torch.ones_like(dist), dist)
        u = torch.atan2(px, pz) * (w / (2.0 * np.pi)) + (w * 0.5 - 0.5)
        v = (torch.asin((py / safe).clamp(-1.0, 1.0)) * (-h / np.pi)
             + (h * 0.5 - 0.5))
        return u, v, dist

    for y0 in range(0, h, band):
        y1 = min(y0 + band, h)
        a, b = max(0, y0 - margin), min(h, y1 + margin)

        d = projection.equirect_rows_to_dir(a, b, w, h)
        d = np.concatenate([d, d[:, :1]], axis=1)
        dt = torch.as_tensor(d, dtype=torch.float32, device=dev)
        sub = torch.as_tensor(dnx[a:b], dtype=torch.float32, device=dev)
        tu, tv, dist = project(sub, dt)

        # Cuts are decided from the un-eroded depth when the caller supplies
        # it; see the note in `mesh_warp.render_full`.
        if dn_cut is None:
            tu_c = tu
        else:
            sc = torch.as_tensor(cutx[a:b], dtype=torch.float32, device=dev)
            tu_c, _, _ = project(sc, dt)

        u0, u1 = tu[:, :-1], tu[:, 1:]
        c0, c1 = tu_c[:, :-1], tu_c[:, 1:]
        v0, v1 = tv[:, :-1], tv[:, 1:]
        d0, d1 = dist[:, :-1], dist[:, 1:]

        # Longitude wraps, so the segment crossing the seam would otherwise
        # read as almost a full turn wide.
        def unwrap(x):
            return torch.where(x > w / 2, x - w,
                               torch.where(x < -w / 2, x + w, x))

        delta = unwrap(u1 - u0)
        dcut = unwrap(c1 - c0)
        # A non-positive width means the surface has folded back through
        # itself and what would be drawn is its mirrored back face; too
        # positive and the warp has pulled it into a smear. Both are cut.
        cut = (dcut <= 0) | (dcut > max_stretch)

        cut_np = cut.cpu().numpy().astype(np.uint8)
        if min_cut > 1 and cut_np.any():
            nlab, lbl, st, _ = cv2.connectedComponentsWithStats(cut_np, 8)
            if nlab > 1:
                small = np.zeros(nlab, bool)
                small[1:] = st[1:, cv2.CC_STAT_AREA] < min_cut
                cut_np[small[lbl]] = 0
        keep = torch.as_tensor(cut_np == 0, device=dev)
        if not bool(keep.any().cpu()):
            continue

        rows = b - a
        xs = torch.arange(w, dtype=torch.float32,
                          device=dev).unsqueeze(0).expand(rows, w)
        ys = torch.arange(a, b, dtype=torch.float32,
                          device=dev).unsqueeze(1).expand(rows, w)

        # `delta` is only used as a divisor where the segment was kept, and a
        # kept segment has delta > 0.
        safe_delta = torch.where(keep, delta, torch.ones_like(delta))
        start = torch.ceil(u0)
        end = u0 + delta

        flats, sds, sxs, sys = [], [], [], []
        for k in range(kmax):
            uk = start + k
            # Half-open [u0, u1): the column at u1 belongs to the next
            # segment. Closed on both ends would write shared endpoints twice
            # and break the exact tiling.
            inside = keep & (uk < end)
            if not bool(inside.any().cpu()):
                break
            t = ((uk - u0) / safe_delta).clamp(0.0, 1.0)
            sv = v0 + t * (v1 - v0)
            iv = torch.round(sv).long()
            ok = inside & (iv >= y0) & (iv < y1)
            if not bool(ok.any().cpu()):
                continue
            drift = max(drift, float((sv - ys)[inside].abs().max().cpu()))
            okf = ok.reshape(-1)
            tk = t.reshape(-1)[okf]
            ivk = iv.reshape(-1)[okf]
            # Wrap in float and convert after, not the other way round.
            # `torch.remainder` on int64 returned w itself for some columns
            # under DirectML, which put `(iv - y0) * w + iu` exactly one past
            # the end of the band buffer. `uk` is integer-valued, so a float
            # modulo is exact here and the int64 op is not needed.
            iuk = torch.remainder(uk, float(w)).reshape(-1)[okf].long()
            # Band-local *before* the multiply, never after. Building the
            # global index first and subtracting `y0 * w` at the end is the
            # obvious way to write this and it silently destroys the result:
            # the offset exceeds 2**24 from row 2184 on, and DirectML puts an
            # int64-minus-large-scalar through float32, where the spacing at
            # 20M is 2. The low bit of the target column is lost, every write
            # lands on an even column, and half of every band below that row
            # comes out empty -- 21.6% of the frame, all of it on odd columns.
            # Kept local, nothing here exceeds 256 * w, which float32 holds
            # exactly whatever the backend does internally.
            flats.append((ivk - y0) * w + iuk)
            sds.append(d0.reshape(-1)[okf]
                       + tk * (d1 - d0).reshape(-1)[okf])
            sxs.append(xs.reshape(-1)[okf] + tk)
            sys.append(ys.reshape(-1)[okf])

        if not flats:
            continue
        flat = torch.cat(flats)
        sd = torch.cat(sds)
        sx = torch.cat(sxs)
        sy = torch.cat(sys)
        del flats, sds, sxs, sys

        # Nearest surface wins. On a connected surface each output column has
        # exactly one candidate and this is a formality; it does real work
        # only where a cut has let two surfaces overlap.
        nrows = (y1 - y0) * w
        # One sync per band, and worth it: an index one past the end is the
        # failure this renderer is most likely to have, and it presents as a
        # crash in some frames and silent corruption in others.
        oob = int(((flat < 0) | (flat >= nrows)).sum().cpu())
        if oob:
            _TV_DRIFT["oob"] = _TV_DRIFT.get("oob", 0) + oob
            flat = flat.clamp(0, nrows - 1)
        zbuf = torch.full((nrows,), float("inf"), dtype=torch.float32,
                          device=dev)
        zbuf.scatter_reduce_(0, flat, sd, "amin", include_self=True)
        # `<=`, not `==`. The reduction is a selection, so in principle the
        # winner's distance survives bit-identical and equality would hold --
        # but that is a claim about DirectML's implementation, not about the
        # algorithm, and if it is false every candidate loses and the pixel is
        # left empty. `<=` costs nothing and cannot fail that way.
        win = sd <= zbuf[flat]
        if STATS is not None:
            STATS["candidates"] = STATS.get("candidates", 0) + sd.numel()
            STATS["winners"] = STATS.get("winners", 0) + int(win.sum().cpu())
            STATS["strict_eq"] = STATS.get("strict_eq", 0) + int(
                (sd == zbuf[flat]).sum().cpu())

        mx = torch.full((nrows,), -1.0, dtype=torch.float32, device=dev)
        my = torch.full((nrows,), -1.0, dtype=torch.float32, device=dev)
        mx.index_put_((flat[win],), sx[win])
        my.index_put_((flat[win],), sy[win])
        map_x[y0:y1] = mx.cpu().numpy().reshape(y1 - y0, w)
        map_y[y0:y1] = my.cpu().numpy().reshape(y1 - y0, w)
        del zbuf, mx, my, flat, sd, sx, sy, win

    _TV_DRIFT["max"] = drift
    filled = map_x >= 0
    out = cv2.remap(rgbx, np.where(filled, map_x, 0).astype(np.float32),
                    np.where(filled, map_y, 0).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out[~filled] = 0
    return out, (~filled)
