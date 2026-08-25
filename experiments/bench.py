"""A fixed benchmark for stereo renders, so every change is scored the same way.

Written after a session in which five separate ad-hoc measurements were wrong,
all in the same way: they compared two things that were not comparable. A
warped eye is shifted tens of pixels, so **any window fixed in output
coordinates holds different physical content in the two images**. Measuring a
slope, a width or an edge over the same column range in both eyes compares
different places and manufactures a difference -- it invented a floor-line
"inversion" that did not exist, and inflated a real cost from 1.34-to-1.56
into 1.34-to-3.44.

Its mirror image is just as bad: *fitting* the shift by minimising the
quantity being reported. Searching for the offset that minimises a left/right
thickness difference and then quoting that difference as "+0.0" is circular.

So the rule this module exists to enforce: **correspondence is established
first, independently, by image cross-correlation, and only then is anything
measured.** Every measure returns the correlation it matched at, so a bad
match is visible rather than silent.

Two invariants are included because they were the only measures that held up
all session, and they held up because they have a knowable right answer rather
than a plausible one:

* rendering at zero baseline must reproduce the source exactly;
* on constant depth the sampling map must step by exactly 1 per column.

Prefer adding measures of that shape. A number with no known-correct value is
a number that can be wrong without anyone noticing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

REPO = os.environ.get("STEREO360_REPO",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W, H = 7680, 3840


# ---------------------------------------------------------------- targets

@dataclass(frozen=True)
class Band:
    """A bright specular ridge whose thickness must agree between the eyes.

    The chair rail thins in the warped eye because its topmost rows carry the
    background's depth and lag behind, shearing the bar. Thickness is measured
    at half prominence around the brightest point, which needs no colour
    threshold -- an earlier saturation-gated version skipped this very feature,
    because a specular highlight is bright and *de*saturated.
    """
    name: str
    scene: str
    cols: tuple
    yband: tuple
    match_box: tuple            # x0, x1, y0, y1 to correlate on


@dataclass(frozen=True)
class Line:
    """A straight edge in the world whose bend is a rendering artefact.

    Straight lines on a floor are curves in equirect, so the reference is the
    source's own measured turn, not zero.
    """
    name: str
    scene: str
    xwin: tuple
    yrange: tuple
    match_box: tuple


@dataclass(frozen=True)
class Shape:
    """A small feature whose bending is scored after dividing out any shift."""
    name: str
    scene: str
    box: tuple                  # x0, x1, y0, y1


TARGETS = (
    Band("chair rail", "indoor",
         cols=(5480, 5510, 5540, 5570, 5600, 5630), yband=(2440, 2600),
         match_box=(5450, 5700, 2450, 2620)),
    Line("floor grout", "indoor",
         xwin=(4770, 4900), yrange=(2600, 2780),
         match_box=(4770, 4900, 2600, 2780)),
    Shape("lamp finial", "road", box=(45, 115, 2700, 2770)),
    Shape("sign post", "road", box=(4230, 4290, 2080, 2220)),
    Shape("handrail", "road", box=(4180, 4240, 2500, 2650)),
)

SCENES = {"indoor": "indoor.jpg", "road": "7680p.jpg"}


def source(scene: str) -> np.ndarray:
    img = cv2.imread(os.path.join(REPO, SCENES[scene]))
    if img is None:
        raise FileNotFoundError(SCENES[scene])
    if img.shape[1] != W:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img


# ------------------------------------------------------- correspondence

def match_shift(ref: np.ndarray, mov: np.ndarray, box, lo=-140, hi=40):
    """Horizontal shift aligning `mov` to `ref` over `box`, by correlation.

    Never derive this from the statistic under test. Returns (shift, corr),
    and callers should treat a low corr as "this target did not match" rather
    than measuring anyway.
    """
    x0, x1, y0, y1 = box
    a = cv2.cvtColor(ref[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    a -= a.mean()
    na = np.linalg.norm(a)
    best = (0, -1.0)
    for s in range(lo, hi):
        cols = np.arange(x0 + s, x1 + s) % W
        b = cv2.cvtColor(mov[y0:y1, cols], cv2.COLOR_BGR2GRAY).astype(np.float32)
        b -= b.mean()
        d = na * np.linalg.norm(b)
        if d < 1e-6:
            continue
        c = float((a * b).sum() / d)
        if c > best[1]:
            best = (s, c)
    return best


# -------------------------------------------------------------- measures

def band_thickness(img, x, yband) -> float:
    """Width at half prominence of the brightest ridge down one column."""
    y0, y1 = yband
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[y0:y1, x % W, 2].astype(np.float32)
    v = cv2.GaussianBlur(v.reshape(-1, 1), (1, 5), 0).ravel()
    p = int(np.argmax(v))
    peak, base = v[p], float(np.percentile(v, 25))
    if peak - base < 12:
        return 0.0
    half = (peak + base) / 2.0
    i = p
    while i > 0 and v[i] > half:
        i -= 1
    j = p
    while j < len(v) - 1 and v[j] > half:
        j += 1
    return float(j - i)


def line_turn(img, xwin, yrange, shift=0) -> float:
    """How much a traced line changes direction between its halves.

    The line here is steep, so it is traced as the darkest *column* per row.
    """
    x0, x1 = xwin[0] + shift, xwin[1] + shift
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                         (5, 5), 0).astype(np.float32)
    ys, xs = [], []
    for y in range(*yrange):
        row = g[y, np.arange(x0, x1) % W]
        c = int(np.argmin(row))
        if row[c] < row.mean() - 2.0:
            ys.append(y); xs.append(x0 + c)
    if len(ys) < 60:
        return float("nan")
    ys, xs = np.array(ys, float), np.array(xs, float)
    m = ys.size // 2
    return float(abs(np.polyfit(ys[m:], xs[m:], 1)[0]
                     - np.polyfit(ys[:m], xs[:m], 1)[0]))


def shape_residual(src, img, box) -> float:
    """Lowest mean-abs difference against the source over integer shifts.

    Divides out where the feature sits, leaving how far its shape had to bend.
    """
    x0, x1, y0, y1 = box
    cols = np.arange(x0, x1)
    ref = src[y0:y1][:, cols % W].astype(np.float32)
    im = img.astype(np.float32)
    best = 1e9
    for dy in range(-8, 9):
        for dx in range(-60, 61):
            p = im[y0 + dy:y1 + dy][:, (cols + dx) % W]
            if p.shape != ref.shape:
                continue
            v = p.max(axis=2) > 0
            if v.mean() < 0.75:
                continue
            best = min(best, float(np.abs(p - ref).mean(axis=2)[v].mean()))
    return best


# ------------------------------------------------------------- invariants

def invariant_identity(render, dn, rgb) -> dict:
    """At zero baseline the warp is a no-op: any difference is noise floor."""
    img, cut = render(dn, rgb, 0.0)
    d = np.abs(img.astype(np.int16) - rgb.astype(np.int16)).max(axis=2)
    ok = ~cut
    return {"pixels differing": float(100 * (d[ok] > 0).mean()),
            "mean error": float(d[ok].mean()),
            "worst": int(d[ok].max()),
            "unfilled": float(100 * cut.mean())}


def invariant_flat(render, dn, baseline) -> dict:
    """On constant depth the sampling map must step by exactly 1 per column.

    Recovered by warping a float32 ramp whose value *is* the column index;
    bilinear interpolation of a linear ramp returns the sampling coordinate
    itself. It has to be float -- a byte-packed ramp wraps every 256 columns
    and interpolating across that wrap returns nonsense.
    """
    h, w = dn.shape
    flat = np.full_like(dn, float(np.median(dn)))
    ramp = np.zeros((h, w, 3), np.float32)
    ramp[..., 0] = np.arange(w, dtype=np.float32)[None, :]
    img, cut = render(flat, ramp, baseline)
    got, good = img[..., 0].astype(np.float64), ~cut
    sl = slice(w // 8, w - w // 8)
    step = np.diff(got[:, sl], axis=1)
    valid = good[:, sl][:, 1:] & good[:, sl][:, :-1]
    if not valid.any():
        return {"step error mean": float("nan"), "step error max": float("nan")}
    err = np.abs(step[valid] - 1.0)
    return {"step error mean": float(err.mean()),
            "step error max": float(err.max())}


# ----------------------------------------------------------------- runner

def split_tb(path):
    tb = cv2.imread(path)
    if tb is None:
        raise FileNotFoundError(path)
    h = tb.shape[0] // 2
    return tb[:h], tb[h:]


def score(left, right, scene, src=None) -> list:
    """Every target for one scene, as (name, measure, value, reference, corr)."""
    src = source(scene) if src is None else src
    rows = []
    for t in TARGETS:
        if t.scene != scene:
            continue
        if isinstance(t, Band):
            s, c = match_shift(left, right, t.match_box)
            l = float(np.mean([band_thickness(left, x, t.yband) for x in t.cols]))
            r = float(np.mean([band_thickness(right, x + s, t.yband)
                               for x in t.cols]))
            ref = float(np.mean([band_thickness(src, x, t.yband) for x in t.cols]))
            rows.append((t.name, "band R-L (px)", r - l, 0.0, c))
            rows.append((t.name, "band L vs source", l - ref, 0.0, c))
        elif isinstance(t, Line):
            s, c = match_shift(src, right, t.match_box)
            got = line_turn(right, t.xwin, t.yrange, s)
            ref = line_turn(src, t.xwin, t.yrange, 0)
            rows.append((t.name, "turn vs source", got - ref, 0.0, c))
        else:
            rows.append((t.name, "shape residual",
                         shape_residual(src, right, t.box), 0.0, 1.0))
    return rows


def report(named_rows: dict, note: str = "") -> str:
    """One table, same columns every time, so runs are comparable."""
    keys, out = [], []
    for rows in named_rows.values():
        for n, m, *_ in rows:
            if (n, m) not in keys:
                keys.append((n, m))
    wname = max(len(f"{n} {m}") for n, m in keys) + 2
    head = " " * wname + "".join(f"{k:>16s}" for k in named_rows)
    out.append(head)
    out.append("-" * len(head))
    for n, m in keys:
        line = f"{n + ' ' + m:<{wname}s}"
        for rows in named_rows.values():
            v = next((r[2] for r in rows if r[0] == n and r[1] == m), None)
            line += f"{v:>16.2f}" if v is not None else f"{'-':>16s}"
        out.append(line)
    lo = min((r[4] for rows in named_rows.values() for r in rows), default=1.0)
    out.append("")
    out.append(f"lowest correspondence correlation: {lo:.3f}"
               + ("  <-- suspect below 0.6" if lo < 0.6 else ""))
    if note:
        out.append(note)
    return "\n".join(out)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("renders", nargs="+",
                    help="top-bottom stereo files to score")
    ap.add_argument("--scene", required=True, choices=sorted(SCENES))
    ap.add_argument("--label", nargs="*", default=None,
                    help="column names, defaulting to the file stems")
    args = ap.parse_args()
    src = source(args.scene)
    labels = args.label or [os.path.splitext(os.path.basename(p))[0][:15]
                            for p in args.renders]
    named = {}
    for lbl, p in zip(labels, args.renders):
        l, r = split_tb(p)
        named[lbl] = score(l, r, args.scene, src)
    print(report(named))


if __name__ == "__main__":
    main()
