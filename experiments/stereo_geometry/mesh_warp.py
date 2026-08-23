"""Render the eye warp as a connected surface instead of a cloud of points.

`warp.right_eye_from_disparity` treats every source pixel as an independent
point: lift it to 3D by its inverse depth, translate to the other eye,
reproject, scatter it into a 2x2 footprint, keep the nearest in a z-buffer.
Nothing in that says two neighbouring pixels belong to the same surface, so a
small object's pixels land unevenly and shear along the warp direction -- which
is opposite in the two eyes, which is why the lamp's finial leans right in one
eye and left in the other.

A mesh keeps them joined. Vertices come from the depth samples, quads join
neighbours, and the surface between two vertices is *interpolated* rather than
left to chance. Two consequences matter here:

  * Magnification cannot open gaps. Between neighbours the surface is
    continuous, so the 2x2 footprint that exists to stop background splatting
    through the cracks is not needed.

  * The foreground/background decision becomes explicit. At a depth cliff the
    quads bridging near to far stretch into rubber sheets, and each one is
    either kept (a smear) or cut (a clean hole on the silhouette, to inpaint).
    That is the same call `gradient_limit` makes today -- but today it makes it
    by *rewriting the depth values* to keep the warp injective, which on this
    lamp spread the depth step over about 12 px and ate a fifth of a 63 px ball
    from either side. Cutting geometry leaves the depth untouched.

Rasterised by dense sampling rather than a scanline fill: each surviving quad
is sampled on an SxS grid by bilinearly interpolating its corners' warped
positions, texture coordinates and distances, and the samples are composited
far-to-near. That is a painter's algorithm, and for a prototype it is far
easier to get right than a barycentric rasteriser while giving the property
being tested -- samples generated *from the surface* rather than only at
original pixel centres.

Prototype. Crop only, CPU, no seam or pole handling: the caller rolls the
equirect so the region of interest is nowhere near either.
"""
from __future__ import annotations

import numpy as np

from stereo360 import projection, warp

#: Cut a quad when its nearest and furthest corner differ by more than this
#: ratio. 1.25 is a 25% depth jump -- far above surface noise, far below a real
#: silhouette, and on this scene the lamp/grass step is a factor of 1.6.
CUT_RATIO = 1.25

#: Samples per quad edge. Enough that neighbouring samples land within a pixel
#: of each other in the target even where the surface is magnified.
SUBDIV = 5


def render(dn: np.ndarray, rgb: np.ndarray, baseline: float, x0: int, y0: int,
           w: int, h: int, cut_ratio: float = CUT_RATIO, subdiv: int = SUBDIV):
    """Warp one eye over a crop. Returns (image, filled_mask).

    `dn` and `rgb` are the crop; `x0, y0` place it in the full `w` x `h`
    equirect, which is what the direction of every vertex depends on.
    """
    ch, cw = dn.shape
    d = projection.equirect_rows_to_dir(y0, y0 + ch, w, h)[:, x0:x0 + cw]
    lam = 1.0 / (dn + warp._MIN_INV_DEPTH)

    px, pz = warp._eye_offset(lam, d, -baseline)
    tu, tv, dist = projection.points_to_equirect_uv(px, lam * d[..., 1], pz,
                                                    w, h)
    # Texture coordinates are simply where each vertex came from.
    gx, gy = np.meshgrid(np.arange(cw, dtype=np.float32),
                         np.arange(ch, dtype=np.float32))

    # Quad corners: 00 = top-left, 11 = bottom-right.
    def corners(a):
        return a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:]

    l00, l01, l10, l11 = corners(lam)
    lo = np.minimum(np.minimum(l00, l01), np.minimum(l10, l11))
    hi = np.maximum(np.maximum(l00, l01), np.maximum(l10, l11))
    keep = (hi / np.maximum(lo, 1e-6)) <= cut_ratio
    ky, kx = np.nonzero(keep)

    # Bilinear weights for an SxS grid inside each surviving quad.
    t = (np.arange(subdiv, dtype=np.float32) + 0.5) / subdiv
    ty, tx = np.meshgrid(t, t, indexing="ij")
    ty, tx = ty.ravel(), tx.ravel()
    wgt = ((1 - ty) * (1 - tx), (1 - ty) * tx, ty * (1 - tx), ty * tx)

    def blend(a):
        c = corners(a)
        return sum(w_ * c_[ky, kx][:, None] for w_, c_ in zip(wgt, c))

    su = blend(tu).ravel()
    sv = blend(tv).ravel()
    sx = blend(gx).ravel()
    sy = blend(gy).ravel()
    sd = blend(dist).ravel()

    # Painter's algorithm: composite far to near so the nearest sample wins.
    iu = np.rint(su).astype(np.int64) - x0
    iv = np.rint(sv).astype(np.int64) - y0
    ok = (iu >= 0) & (iu < cw) & (iv >= 0) & (iv < ch) & np.isfinite(sd)
    iu, iv, sx, sy, sd = iu[ok], iv[ok], sx[ok], sy[ok], sd[ok]
    order = np.argsort(-sd, kind="stable")

    map_x = np.full((ch, cw), -1.0, np.float32)
    map_y = np.full((ch, cw), -1.0, np.float32)
    flat = iv * cw + iu
    map_x.ravel()[flat[order]] = sx[order]
    map_y.ravel()[flat[order]] = sy[order]

    import cv2
    filled = map_x >= 0
    out = cv2.remap(rgb, np.where(filled, map_x, 0).astype(np.float32),
                    np.where(filled, map_y, 0).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out[~filled] = 0
    return out, filled


#: Refuse to cut a group of quads smaller than this. A silhouette is a
#: connected curve tens or hundreds of quads long; depth noise that happens
#: to cross the ratio is a single quad on its own in the middle of a
#: surface. Without this the full frame cut 1131 separate regions of which
#: 818 were 4 px or smaller and 450 were single pixels -- each one a dot the
#: hole filler then painted in from the background, which is what reads as
#: objects being full of holes. Coverage itself was never the problem: with
#: cutting disabled entirely the frame missed 0.002% of its pixels.
MIN_CUT = 12

#: Cut a quad when the warp stretches it wider than this many output pixels.
#: This replaces the depth-ratio test, which was scale-free and so fired just
#: as hard on a tree forty metres away -- where a leaf and the gap behind it
#: differ by a fraction of a pixel and the stretch is invisible -- as on the
#: silhouette of a van two metres away. Cutting on the ratio left the canopy
#: riddled with holes; cutting on how far the quad is actually pulled apart
#: only removes geometry that would have been a visible rubber sheet.
MAX_STRETCH_PX = 2.5


def render_full(dn: np.ndarray, rgb: np.ndarray, baseline: float,
                cut_ratio: float = CUT_RATIO, subdiv: int = 4,
                band: int = 256, margin: int = 6,
                min_cut: int = MIN_CUT,
                max_stretch: float = MAX_STRETCH_PX,
                dn_cut: np.ndarray | None = None,
                want_depth: bool = False):
    """Whole-sphere version. Returns (image, cut_mask).

    Two things the crop version ducked.

    Longitude wraps: the quad joining the last column to the first is a real
    quad, so both arrays get one extra column copied from the front, texture
    coordinates are allowed to run to `w`, and output columns are taken mod w.

    Bands: a vertex per pixel at 7680x3840 is 29.5M quads, and at 16 samples
    each that is half a billion samples -- tens of gigabytes if built at once.
    Banding works here for a reason specific to this projection: `_eye_offset`
    moves a point almost purely in longitude, the latitude shift being second
    order in baseline/distance (0.4 px at the very nearest thing in this
    scene). So a sample stays in the row it started in, and a band can be
    composited independently, with a few rows of margin for safety.
    """
    import cv2

    h, w = dn.shape
    dnx = np.concatenate([dn, dn[:, :1]], axis=1)
    # Cuts are decided from `dn_cut` when given, and the surface is warped with
    # `dn`. They differ when the caller has eroded the foreground: erosion
    # works by ramping the near side of an edge down to the background, and
    # this renderer cuts ramps, so the two fight. Measured on a 17 px railing
    # post, eroding by 5 px took its depth transition from 6 px to 12 px --
    # most of the post -- and the cut then removed nearly all of it. Deciding
    # from the un-eroded depth keeps the cuts on the true silhouettes while
    # the erosion still does its job of pulling the halo back.
    cutx = dnx if dn_cut is None else np.concatenate(
        [dn_cut, dn_cut[:, :1]], axis=1)
    rgbx = np.concatenate([rgb, rgb[:, :1]], axis=1)

    map_x = np.full((h, w), -1.0, np.float32)
    map_y = np.full((h, w), -1.0, np.float32)
    zbuf = np.full((h, w), np.inf, np.float32)

    t = (np.arange(subdiv, dtype=np.float32) + 0.5) / subdiv
    ty, tx = np.meshgrid(t, t, indexing="ij")
    ty, tx = ty.ravel(), tx.ravel()
    wgt = ((1 - ty) * (1 - tx), (1 - ty) * tx, ty * (1 - tx), ty * tx)

    for y0 in range(0, h - 1, band):
        y1 = min(y0 + band, h - 1)
        a, b = max(0, y0 - margin), min(h, y1 + 1 + margin)
        sub = dnx[a:b]
        subc = cutx[a:b]
        d = projection.equirect_rows_to_dir(a, b, w, h)
        d = np.concatenate([d, d[:, :1]], axis=1)
        lam = 1.0 / (sub + warp._MIN_INV_DEPTH)
        px, pz = warp._eye_offset(lam, d, -baseline)
        tu, tv, dist = projection.points_to_equirect_uv(
            px, lam * d[..., 1], pz, w, h)

        gx, gy = np.meshgrid(np.arange(w + 1, dtype=np.float32),
                             np.arange(a, b, dtype=np.float32))

        def corners(arr):
            return arr[:-1, :-1], arr[:-1, 1:], arr[1:, :-1], arr[1:, 1:]

        # How far the warp pulls each quad apart, in output pixels. The
        # quad is one pixel wide to start with, so anything much over 1 is
        # stretch. Longitude wraps, so a quad straddling the seam would
        # otherwise read as almost a full turn wide.
        if dn_cut is None:
            tu_c = tu
        else:
            lam_c = 1.0 / (subc + warp._MIN_INV_DEPTH)
            px_c, pz_c = warp._eye_offset(lam_c, d, -baseline)
            tu_c, _, _ = projection.points_to_equirect_uv(
                px_c, lam_c * d[..., 1], pz_c, w, h)
        cu = corners(tu_c)
        umax = np.maximum(np.maximum(cu[0], cu[1]), np.maximum(cu[2], cu[3]))
        umin = np.minimum(np.minimum(cu[0], cu[1]), np.minimum(cu[2], cu[3]))
        du = umax - umin
        du = np.minimum(du, w - du)

        # A quad also has to be discarded when it turns inside out: the warp
        # has folded the surface back through itself and what would be drawn
        # is its back face, mirrored. The stretch test cannot see this -- a
        # fold makes a quad *narrow*, not wide -- so without an explicit
        # orientation check a folded object is drawn a second time, reversed,
        # beside itself. Measured on one indoor scene at 40 mm, 12043 quads
        # fold, nearly all of them on a single object.
        #
        # The splat path never meets this because `gradient_limit` keeps the
        # warp injective by clamping depth, at the cost of sharpness. A mesh
        # keeps the depth and culls the fold instead, which costs nothing.
        left, right = corners(tu_c)[0], corners(tu_c)[1]
        delta = right - left
        delta = np.where(delta > w / 2, delta - w,
                         np.where(delta < -w / 2, delta + w, delta))
        cut = ((du > max_stretch) | (delta < 0)).astype(np.uint8)
        if min_cut > 1 and cut.any():
            # Keep only cuts that are part of a large enough connected
            # group; the rest are noise and are stitched back up.
            nlab, lbl, st, _ = cv2.connectedComponentsWithStats(cut, 8)
            if nlab > 1:
                small = np.zeros(nlab, bool)
                small[1:] = st[1:, cv2.CC_STAT_AREA] < min_cut
                cut[small[lbl]] = 0
        ky, kx = np.nonzero(cut == 0)
        if ky.size == 0:
            continue

        def blend(arr):
            cc = corners(arr)
            return sum(wt * q[ky, kx][:, None] for wt, q in zip(wgt, cc))

        su = blend(tu).ravel(); sv = blend(tv).ravel()
        sx = blend(gx).ravel(); sy = blend(gy).ravel()
        sd = blend(dist).ravel()
        del cu, umax, umin, du

        iu = np.mod(np.rint(su).astype(np.int64), w)
        iv = np.rint(sv).astype(np.int64)
        ok = (iv >= y0) & (iv < y1) & np.isfinite(sd)
        iu, iv, sx, sy, sd = iu[ok], iv[ok], sx[ok], sy[ok], sd[ok]

        # Far to near, so the nearest sample is written last and wins.
        #
        # Ties are left to generation order, and that is a known defect rather
        # than an oversight: on constant depth every sample ties, the winner
        # flips wherever float32 jitter in the shift pushes a rounding the
        # other way, and the texture coordinate steps a quarter pixel at that
        # column -- a hairline down the frame at longitudes +/-45 and +/-135.
        # Breaking ties by proximity to the pixel centre was tried and made it
        # worse (the worst column went from 2.3x the median ridge to 10.3x)
        # while costing 60% more time, so it is not in. The real answer is a
        # rasteriser that computes coverage per output pixel instead of
        # scattering rounded samples, which is out of scope for a prototype.
        order = np.argsort(-sd, kind="stable")
        flat = iv * w + iu
        map_x.ravel()[flat[order]] = sx[order]
        map_y.ravel()[flat[order]] = sy[order] - 0.0
        zbuf.ravel()[flat[order]] = sd[order]
        del iu, iv, sx, sy, sd, order, flat, su, sv, ok

    filled = map_x >= 0
    out = cv2.remap(rgbx, np.where(filled, map_x, 0).astype(np.float32),
                    np.where(filled, map_y, 0).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out[~filled] = 0
    if not want_depth:
        return out, (~filled)
    # The z-buffer holds each winning sample's distance from the *new* eye, so
    # inverting it gives that eye its own depth map -- what a second warp
    # needs in order to start from this view rather than from the original.
    return out, (~filled), zbuf
