"""The mesh warp on the GPU, with a z-buffer instead of a sort.

Profiling the numpy version put the actual geometry -- projecting vertices and
deciding which quads to cut -- at 1.6% of its runtime. The other 98% was the
brute-force stand-in for a rasteriser: 46% sorting samples so the nearest could
be written last, 30% interpolating them, and the rest rounding and scattering.

Two changes, and only the second is about hardware.

**A z-buffer, not a sort.** Painter's algorithm sorts every sample by distance
so the nearest is written last. But "keep the nearest per output pixel" is what
a scatter-with-minimum does directly, in one pass and with no ordering at all.
Measured on this machine at four million samples: `scatter_reduce_` with
"amin" takes 63 ms where `argsort` takes 394. It is also the honest algorithm
rather than an emulation of one.

**DirectML.** torch has no CUDA here but `torch_directml` reaches the Radeon,
and the projection is almost all transcendentals, which is what a GPU is for:
20 rounds of atan2 over a million elements run in 21 ms against numpy's 139.

Two things deliberately stay on the CPU. `grid_sample` is not implemented on
DirectML and silently falls back, and a hand-rolled bilinear gather costs
479 ms per four million fetches, so the final texture sample is left to
`cv2.remap`, which is heavily optimised and runs once per output pixel rather
than once per sample. And the hole fill is unchanged.

Ties are still broken arbitrarily, so the four hairlines at longitudes +/-45
and +/-135 degrees survive this change -- it is a speed fix, not a correctness
one. See the module docstring in `mesh_warp.py`.
"""
from __future__ import annotations

import numpy as np

try:
    from . import mesh_warp
except ImportError:  # run as a loose script
    import mesh_warp
from stereo360 import projection, warp

#: Rows per band. Smaller than the numpy version's: every sample array lives
#: in GPU memory at once, and this machine's Radeon shares system RAM.
BAND = 128

#: Slack in the "is this sample the winner" test, as a fraction of the winning
#: distance. Zero, and deliberately so: `scatter_reduce_` with "amin" returns
#: one of the values it reduced, unaltered -- it is a selection, not an
#: arithmetic reduction -- so `sd == zbuf[flat]` is exact and needs no
#: tolerance. A defensive 1e-5 was tried first and only let extra samples
#: through to fight over the same pixel: against the numpy renderer it
#: quadrupled the disagreement (8.57% of pixels differing to 3.66%, and 0.066%
#: differing by more than 8 levels to 0.020%) while costing 11% more time.
Z_TOL = 0.0

#: Set true to accumulate per-phase timings into `PROF`. Off by default; the
#: syncs it needs to get an honest number are themselves a cost.
PROFILE = False
PROF: dict[str, float] = {}


def _tick(dev, name, t0):
    """Charge elapsed time to `name`, forcing the queue to drain first.

    DirectML dispatches asynchronously, so timing without a sync measures how
    fast work is *queued*, which is not a number anyone wants.
    """
    import time, torch
    torch.zeros(1, device=dev).cpu()
    now = time.perf_counter()
    PROF[name] = PROF.get(name, 0.0) + now - t0
    return now


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
                band: int = BAND, margin: int = 6,
                min_cut: int = mesh_warp.MIN_CUT,
                max_stretch: float = mesh_warp.MAX_STRETCH_PX,
                dn_cut: np.ndarray | None = None,
                device=None):
    """Same contract as `mesh_warp.render_full`. Returns (image, cut mask)."""
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

    t = (np.arange(subdiv, dtype=np.float32) + 0.5) / subdiv
    ty, tx = np.meshgrid(t, t, indexing="ij")
    wgt = [torch.as_tensor(q.ravel(), dtype=torch.float32, device=dev)
           for q in ((1 - ty) * (1 - tx), (1 - ty) * tx,
                     ty * (1 - tx), ty * tx)]

    def corners(a):
        return a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:]

    for y0 in range(0, h - 1, band):
        y1 = min(y0 + band, h - 1)
        a, b = max(0, y0 - margin), min(h, y1 + 1 + margin)

        import time
        t0 = time.perf_counter()
        d = projection.equirect_rows_to_dir(a, b, w, h)
        d = np.concatenate([d, d[:, :1]], axis=1)
        dt = torch.as_tensor(d, dtype=torch.float32, device=dev)
        sub = torch.as_tensor(dnx[a:b], dtype=torch.float32, device=dev)
        if PROFILE:
            t0 = _tick(dev, "upload", t0)

        lam = 1.0 / (sub + warp._MIN_INV_DEPTH)
        # _eye_offset, inline -- and note the sign. The numpy path calls it as
        # `_eye_offset(lam, d, -baseline)`, so the offset added to x is
        # *minus* the baseline. Inlining it with the sign the other way still
        # renders a clean, plausible image: it is simply the other eye. That
        # cost a debugging pass, because nothing about the output looks broken.
        bl = -baseline
        px = lam * dt[..., 0] + bl * dt[..., 2]
        pz = lam * dt[..., 2] - bl * dt[..., 0]
        py = lam * dt[..., 1]
        dist = torch.sqrt(px * px + py * py + pz * pz)
        # points_to_equirect_uv, inline so the whole projection stays on
        # the device. The offsets are w/2 - 1/2 and h/2 - 1/2, not w/2 and
        # h/2: pixel centres, not corners. Getting that half pixel wrong
        # displaces every sample by half a pixel in both axes, which changes
        # every rounding decision -- 89% of output pixels differed, a quarter
        # of them by more than 8 levels.
        safe = torch.where(dist == 0.0, torch.ones_like(dist), dist)
        tu = torch.atan2(px, pz) * (w / (2.0 * np.pi)) + (w * 0.5 - 0.5)
        tv = (torch.asin((py / safe).clamp(-1.0, 1.0)) * (-h / np.pi)
              + (h * 0.5 - 0.5))

        if dn_cut is None:
            tu_c = tu
        else:
            sc = torch.as_tensor(cutx[a:b], dtype=torch.float32, device=dev)
            lc = 1.0 / (sc + warp._MIN_INV_DEPTH)
            pxc = lc * dt[..., 0] + bl * dt[..., 2]
            pzc = lc * dt[..., 2] - bl * dt[..., 0]
            tu_c = (torch.atan2(pxc, pzc) * (w / (2.0 * np.pi))
                    + (w * 0.5 - 0.5))

        if PROFILE:
            t0 = _tick(dev, "project", t0)

        cu = corners(tu_c)
        umax = torch.maximum(torch.maximum(cu[0], cu[1]),
                             torch.maximum(cu[2], cu[3]))
        umin = torch.minimum(torch.minimum(cu[0], cu[1]),
                             torch.minimum(cu[2], cu[3]))
        du = umax - umin
        du = torch.minimum(du, w - du)
        delta = cu[1] - cu[0]
        delta = torch.where(delta > w / 2, delta - w,
                            torch.where(delta < -w / 2, delta + w, delta))
        cut = ((du > max_stretch) | (delta < 0))

        # The small-group rejection is a connected-components pass, which has
        # no DirectML equivalent worth writing; it is cheap on a band-sized
        # mask, so it goes back to the host for that one step.
        if PROFILE:
            t0 = _tick(dev, "cut-decide", t0)
        cut_np = cut.cpu().numpy().astype(np.uint8)
        if min_cut > 1 and cut_np.any():
            nlab, lbl, st, _ = cv2.connectedComponentsWithStats(cut_np, 8)
            if nlab > 1:
                small = np.zeros(nlab, bool)
                small[1:] = st[1:, cv2.CC_STAT_AREA] < min_cut
                cut_np[small[lbl]] = 0
        keep = torch.as_tensor(cut_np == 0, device=dev).reshape(-1)
        if not bool(keep.any().cpu()):
            continue

        if PROFILE:
            t0 = _tick(dev, "cut-components (host round trip)", t0)

        gx, gy = np.meshgrid(np.arange(w + 1, dtype=np.float32),
                             np.arange(a, b, dtype=np.float32))
        gxt = torch.as_tensor(gx, dtype=torch.float32, device=dev)
        gyt = torch.as_tensor(gy, dtype=torch.float32, device=dev)

        def blend(arr):
            c = corners(arr)
            flat = [q.reshape(-1)[keep].unsqueeze(1) for q in c]
            return sum(wt * f for wt, f in zip(wgt, flat)).reshape(-1)

        if PROFILE:
            t0 = _tick(dev, "grid upload", t0)

        su, sv = blend(tu), blend(tv)
        sx, sy, sd = blend(gxt), blend(gyt), blend(dist)

        if PROFILE:
            t0 = _tick(dev, "blend", t0)

        iu = torch.remainder(torch.round(su).long(), w)
        iv = torch.round(sv).long()
        ok = (iv >= y0) & (iv < y1) & torch.isfinite(sd)
        iu, iv, sx, sy, sd = iu[ok], iv[ok], sx[ok], sy[ok], sd[ok]
        del su, sv, ok
        if iu.numel() == 0:
            continue

        if PROFILE:
            t0 = _tick(dev, "round+mask", t0)

        flat = (iv - y0) * w + iu
        rows = (y1 - y0) * w
        # One pass: keep the nearest distance per output pixel. No ordering.
        zbuf = torch.full((rows,), float("inf"), dtype=torch.float32,
                          device=dev)
        zbuf.scatter_reduce_(0, flat, sd, "amin", include_self=True)
        win = sd <= zbuf[flat] * (1.0 + Z_TOL)

        mx = torch.full((rows,), -1.0, dtype=torch.float32, device=dev)
        my = torch.full((rows,), -1.0, dtype=torch.float32, device=dev)
        mx.index_put_((flat[win],), sx[win])
        my.index_put_((flat[win],), sy[win])
        if PROFILE:
            t0 = _tick(dev, "zbuffer+scatter", t0)
        map_x[y0:y1] = mx.cpu().numpy().reshape(y1 - y0, w)
        map_y[y0:y1] = my.cpu().numpy().reshape(y1 - y0, w)
        if PROFILE:
            t0 = _tick(dev, "download maps", t0)
        del zbuf, mx, my, flat, iu, iv, sx, sy, sd, win

    import time
    t0 = time.perf_counter()
    filled = map_x >= 0
    out = cv2.remap(rgbx, np.where(filled, map_x, 0).astype(np.float32),
                    np.where(filled, map_y, 0).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out[~filled] = 0
    if PROFILE:
        _tick(dev, "remap", t0)
    return out, (~filled)
