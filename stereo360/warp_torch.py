"""GPU implementation of the DIBR warp, via torch.

Mirrors `warp.right_eye_from_disparity` exactly -- same geometry, same
gradient clamp, same 2x2 splat footprint, same visibility test -- but runs the
whole thing on the device so nothing round-trips mid-pass.

Why this is worth a second implementation. Measured on an 8K frame:

    numpy: arctan2 + arcsin + sqrt      0.405 s
    cuda : atan2 + asin + sqrt          0.0015 s      (274x)

The warp is ~70% of a frame once depth is on the GPU, and it is almost
entirely transcendentals and resampling -- exactly what a GPU is for. The two
operations that could have forced a round trip both exist on-device:
`scatter_reduce_(reduce="amin")` for the z-buffer and `grid_sample` for the
sub-pixel resample, so the only transfers are the depth map and image in, and
the result out.

DirectML is deliberately NOT used, though torch-directml does expose AMD and
Intel GPUs as a torch device. Measured on a Radeon 780M: `scatter_reduce_` --
the z-buffer, the hottest operation in pass 1 -- is unsupported and silently
falls back to the CPU, which copies the whole frame off the device and back
mid-pass; and the output disagreed with the numpy reference on 13% of pixels
(mean 1.54 levels) against 0.0008% on CUDA. `scripts/check_directml_warp.py`
re-runs both checks if a future driver or plugin changes that.

torch rather than OpenCL deliberately. OpenCV's OpenCL path would cover AMD
too, but its only atan2 is `cv2.phase`, a polynomial approximation measuring
up to 0.2 px of longitude error at 8K -- a real geometric distortion. torch's
atan2 is exact, and torch reaches CUDA and Apple MPS. Where torch has no GPU
(notably AMD on Windows, which has no ROCm build) the numpy path in `warp` is
used unchanged, so nothing regresses there.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .projection import _equirect_trig

_trig_cache: dict = {}


def device_available() -> Optional[str]:
    """'cuda', 'mps', or None if torch has no GPU here."""
    try:
        import torch
    except ImportError:
        return None
    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        return None
    return None


def _trig(w: int, h: int, device, torch):
    """Separable trig tables on device, matching the numpy path bit for bit.

    Built from `projection._equirect_trig` rather than recomputed in torch, so
    the two implementations start from identical values and any difference
    downstream is float arithmetic order, not a different sin/cos.
    """
    key = (w, h, str(device))
    hit = _trig_cache.get(key)
    if hit is not None:
        return hit
    sin_lon, cos_lon, sin_lat, cos_lat = _equirect_trig(w, h)
    out = tuple(torch.from_numpy(np.ascontiguousarray(a)).to(device)
                for a in (sin_lon, cos_lon, sin_lat, cos_lat))
    _trig_cache[key] = out
    return out


def _cone_clamp(dn, max_slope: float, dim: int, torch) -> None:
    """Cone erosion along `dim`, in place -- the gradient limit, on device.

    `dn[i] <= dn[i-1] + s` is the same as `dn - s*i` never increasing, which
    is a running minimum; `torch.cummin` provides it in both directions.
    """
    n = dn.shape[dim]
    shape = [1, 1]
    shape[dim] = n
    ramp = (torch.arange(n, device=dn.device, dtype=dn.dtype)
            * float(max_slope)).reshape(shape)
    dn.sub_(ramp)
    dn.copy_(torch.cummin(dn, dim=dim).values)
    dn.add_(ramp)
    dn.add_(ramp)
    dn.copy_(torch.cummin(dn.flip(dim), dim=dim).values.flip(dim))
    dn.sub_(ramp)


def _erode_foreground(dn, k: int, thresh: float, torch) -> None:
    """The continuous near-edge softening, on device.

    Grayscale morphology becomes max_pool2d (and its negation for erosion),
    which keeps the operator 1-Lipschitz exactly as the numpy version is --
    the property that stopped thin structures changing shape between frames.
    """
    import torch.nn.functional as F

    ks = 2 * k + 1
    x = dn[None, None]
    local_max = F.max_pool2d(x, ks, stride=1, padding=k)
    local_min = -F.max_pool2d(-x, ks, stride=1, padding=k)
    opened = F.max_pool2d(local_min, ks, stride=1, padding=k)

    local_max, local_min, opened = (t[0, 0] for t in (local_max, local_min,
                                                     opened))
    rng = local_max - local_min
    near = (dn - local_min) / rng.clamp_min(1e-6)
    wide = ((rng - thresh) / thresh).clamp_(0.0, 1.0)
    thin = ((dn - opened) / thresh).clamp_(0.0, 1.0)
    weight = wide * near * (1.0 - thin)
    dn.add_(weight * (local_min - dn))


def right_eye_from_disparity(
    left_rgb: np.ndarray,
    inv_depth: np.ndarray,
    strength: float,
    baseline_scale: float,
    min_inv_depth: float,
    vis_ratio: float,
    crack_margin: float,
    fg_erode: int,
    gradient_limit: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """(right_rgb, hole_mask) computed on `device`. Numpy in, numpy out."""
    import torch
    import torch.nn.functional as F

    h, w = inv_depth.shape
    dev = torch.device(device)
    sin_lon, cos_lon, sin_lat, cos_lat = _trig(w, h, dev, torch)
    cl = cos_lat[:, None]

    dn = torch.from_numpy(np.ascontiguousarray(inv_depth)).to(dev)
    if fg_erode > 0:
        _erode_foreground(dn, fg_erode, 0.05, torch)

    baseline = strength * baseline_scale
    if baseline == 0.0:
        return left_rgb.copy(), np.zeros((h, w), np.uint8)

    if gradient_limit > 0.0:
        critical = (2.0 * np.pi) / (abs(baseline) * w)
        _cone_clamp(dn, gradient_limit * critical, 1, torch)
        _cone_clamp(dn, gradient_limit * critical * 2.0, 0, torch)

    def project(depth, sign):
        """Point along each ray, offset to the other eye, reprojected.

        The offset is sideways of *the direction being looked at*, which is the
        direction's own horizontal components swapped and one negated -- see
        `warp._eye_offset` for why that is not the same as a fixed world +X and
        what goes wrong when it is.
        """
        dx = cl * sin_lon[None, :]
        dz = cl * cos_lon[None, :]
        px = depth * dx + sign * baseline * dz
        py = depth * sin_lat[:, None]
        pz = depth * dz - sign * baseline * dx
        norm = torch.sqrt(px * px + py * py + pz * pz)
        u = torch.atan2(px, pz) * (w / (2.0 * np.pi)) + (w * 0.5 - 0.5)
        v = torch.asin((py / norm.clamp_min(1e-12)).clamp(-1.0, 1.0)) \
            * (-h / np.pi) + (h * 0.5 - 0.5)
        return u, v, norm

    # ---- pass 1: splat distance into a z-buffer -------------------------
    tu, tv, norm = project(1.0 / (dn + min_inv_depth), -1.0)
    u0 = torch.floor(tu).to(torch.int64).remainder_(w)
    v0 = torch.floor(tv).to(torch.int64).clamp_(-1, h - 1).add_(1)
    idx = (v0 * w + u0).reshape(-1)

    zp = torch.full(((h + 2) * w,), float("inf"), device=dev,
                    dtype=norm.dtype)
    zp.scatter_reduce_(0, idx, norm.reshape(-1), reduce="amin")
    del idx, u0, v0, tu, tv

    # Complete the 2x2 footprint: a separable min over the block ending at
    # each pixel. Longitude wraps, latitude uses the guard rows.
    # Shifts by slicing rather than torch.roll: DirectML has no `aten::roll`
    # and silently falls back to the CPU for it, which would copy the whole
    # frame off the device and back in the middle of the pass.
    zpad = zp.view(h + 2, w)
    shifted = torch.empty_like(zpad)
    shifted[:, 0] = zpad[:, -1]          # longitude wraps
    shifted[:, 1:] = zpad[:, :-1]
    zpad = torch.minimum(zpad, shifted)
    shifted[0] = zpad[0]                 # latitude uses the guard row
    shifted[1:] = zpad[:-1]
    zpad = torch.minimum(zpad, shifted)
    z = zpad[1:h + 1].contiguous()
    del zp, zpad, shifted

    # Crack closing: pull residual background leaking through magnified
    # foreground down to its 3x3 neighbourhood minimum.
    finite = torch.isfinite(z)
    if bool(finite.any()):
        sentinel = z[finite].max() * 2.0
        zw = torch.where(finite, z, sentinel)
        zmin = -F.max_pool2d(-zw[None, None], 3, stride=1, padding=1)[0, 0]
        crack = finite & ((zw - zmin) > crack_margin * zmin)
        z = torch.where(crack, zmin, z)

    miss = torch.isinf(z)
    z = torch.where(miss, torch.ones_like(z), z)

    # ---- pass 2: back-project and sample the left image -----------------
    su, sv, pn = project(z, 1.0)

    # grid_sample has no wrap mode, so the image carries one wrapped column on
    # each side and the coordinates shift to match.
    img = torch.from_numpy(np.ascontiguousarray(left_rgb)).to(dev)
    img = img.permute(2, 0, 1)[None].to(torch.float32)
    img = torch.cat((img[..., -1:], img, img[..., :1]), dim=3)
    wp = w + 2

    gx = ((su.remainder(w) + 1.0) * 2.0 + 1.0) / wp - 1.0
    gy = (sv.clamp(0, h - 1) * 2.0 + 1.0) / h - 1.0
    grid = torch.stack((gx, gy), dim=-1)[None]
    right = F.grid_sample(img, grid, mode="bilinear", padding_mode="border",
                          align_corners=False)[0]
    # Match the input's dtype instead of assuming uint8. `warp.right_eye_banded`
    # warps a *signed* detail layer through here -- the frame minus its own blur,
    # centred on zero -- and rounding that to uint8 clamps away every negative
    # value, i.e. the whole darkening half of the detail. What comes back is a
    # one-sided residual that only ever adds, so the recombined frame reads
    # bright and overexposed wherever it has texture. The numpy path gets this
    # right for free via `np.zeros_like(left_rgb)`; here it has to be asked for.
    right = right.permute(1, 2, 0)
    if left_rgb.dtype == np.uint8:
        right = right.round_().clamp_(0, 255).to(torch.uint8)

    # Left-eye visibility: the z-buffer resolved what the RIGHT eye sees, but
    # the colour can only come from the left, and beside a thin near structure
    # those disagree. Sample the depth where we sampled colour; if the left eye
    # meets something materially nearer, the point is hidden and there is no
    # honest colour for it.
    dnp = torch.cat((dn[:, -1:], dn, dn[:, :1]), dim=1)[None, None]
    dn_src = F.grid_sample(dnp, grid, mode="nearest", padding_mode="border",
                           align_corners=False)[0, 0]
    occluded = (1.0 / (dn_src + min_inv_depth)) < vis_ratio * pn

    hole = (miss | occluded)
    right[hole] = 0
    return (right.cpu().numpy(),
            (hole.to(torch.uint8) * 255).cpu().numpy())
