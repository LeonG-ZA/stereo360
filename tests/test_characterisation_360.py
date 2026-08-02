"""The 360 output must not move.

A safety net, not a correctness test. Nothing here says the output is *right* —
the rest of the suite does that. This says it is the *same*, which is a
different and sometimes more useful thing: unit tests check the properties
somebody thought to name, and this catches the ones nobody did.

Written before the VR180 work, because that work adds an output mode and the
one thing it must not do is disturb the path that already works.

Three deliberate choices:

* **A synthetic scene and a synthetic depth model.** Real footage is not in the
  repository and a real model would make this a test of HuggingFace. The fake
  model normalises each face to its own visible range, which is what makes the
  face alignment do real work rather than nothing.
* **The CPU warp, forced.** The GPU path is not bit-identical to it and would
  make this machine-dependent. Nothing is lost: `test_gpu_warp.py` already pins
  the GPU path to the CPU one, so the two together cover both.
* **A stored fingerprint rather than a hash.** A hash says "something moved";
  a fingerprint says how much and where, which is what you need at 2am.

Regenerate deliberately, never casually:

    STEREO360_REGEN_BASELINE=1 python -m pytest tests/test_characterisation_360.py

If that is needed, the size of the reported movement is the review. Measured
sensitivity, by deliberately perturbing the pipeline:

| change | worst cell moves |
|---|---|
| face overlap +0.3% | 0.40 levels |
| face overlap +3.3% | 1.16 |
| stereo baseline +0.2% | 0.32 |
| stereo baseline +2.0% | 1.06 |
| face overlap removed entirely | 11.5 |

So a failure near the 0.30 threshold is plausibly library drift and worth
looking at twice; a failure in the units is something real.
"""

import os
import pathlib

import numpy as np
import pytest

from stereo360 import pipeline, warp
from stereo360.depth.base import DepthBackend

BASELINE = pathlib.Path(__file__).parent / "data" / "characterisation_360.npz"

W, H = 512, 256
FACE = 128
# The CPU pipeline is bit-identical run to run -- measured, 0 difference over
# three renders -- so the tolerance is not absorbing run-to-run noise. It exists
# only for library and CPU variation: a different OpenCV SIMD path can move a
# uint8 by one in the last place, which survives the block mean as a fraction
# of a level. Hence a small but non-zero budget.
#
# The grid is as fine as is useful. Measured against a deliberate +0.7% change
# to the face overlap, the worst cell moves 0.18 levels at (12,24) and 0.49 at
# (48,96) -- so the finer grid is nearly three times more sensitive for the
# same idea, and costs only a slightly larger baseline file.
GRID = (48, 96)
TOL = 0.30          # in 0-255 levels, on the downsampled fingerprint


def scene(w=W, h=H):
    """A deterministic equirect with structure at several scales.

    Detail at more than one scale matters: a change that only moved fine detail
    would hide inside a coarse fingerprint, and one that only moved large
    shapes would hide inside a fine one.
    """
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    lon = x / w * 2 * np.pi
    lat = (0.5 - y / h) * np.pi
    r = 128 + 90 * np.sin(3 * lon) * np.cos(lat)
    g = 128 + 90 * np.cos(2 * lat) * np.cos(lon * 0.5)
    b = 128 + 70 * np.sin(9 * lon + 4 * lat) + 40 * np.sin(31 * lon)
    # A hard-edged near object, so disocclusion and the hole fill are exercised.
    blob = ((np.abs(x - w * 0.62) < w * 0.06)
            & (np.abs(y - h * 0.58) < h * 0.10))
    out = np.stack([r, g, b], -1)
    out[blob] = (240, 60, 40)
    return np.clip(out, 0, 255).astype(np.uint8)


class PerFaceNormalising(DepthBackend):
    """Rescales each face to its own visible range, as real models do.

    That per-face arbitrariness is the whole reason `align_overlapping_faces`
    exists, so a fake model without it would leave the alignment untested.
    """

    def estimate(self, frame_rgb):
        return self.estimate_chunk([frame_rgb])[0]

    def estimate_chunk(self, frames_rgb):
        out = []
        for img in frames_rgb:
            d = img.astype(np.float32).mean(axis=2) / 255.0
            lo, hi = float(d.min()), float(d.max())
            n = (d - lo) / max(hi - lo, 1e-6)
            out.append((n ** 1.4).astype(np.float32) + 0.05)
        return out


def fingerprint(a):
    """Block-mean downsample to GRID, as float32."""
    a = a.astype(np.float32)
    gh, gw = GRID
    h, w = a.shape[:2]
    a = a[:h // gh * gh, :w // gw * gw]
    a = a.reshape(gh, a.shape[0] // gh, gw, a.shape[1] // gw, *a.shape[2:])
    return a.mean(axis=(1, 3))


def render():
    """The 360 path, end to end, deterministically."""
    prev = os.environ.get("STEREO360_GPU_WARP")
    os.environ["STEREO360_GPU_WARP"] = "0"
    warp._gpu_probe.clear()
    try:
        frame = scene()
        backend = PerFaceNormalising()
        depth = pipeline.depth_map_for_frame(frame, FACE, backend)
        left, right = pipeline.right_eye_from_depth(
            frame, FACE, backend, strength=1.0,
            depth_range=pipeline.DepthRange(), stabiliser=None)
        # Exactly how _Sink.write stacks it: left eye on top.
        stacked = np.concatenate([left, right], axis=0)
        return depth, stacked
    finally:
        warp._gpu_probe.clear()
        if prev is None:
            os.environ.pop("STEREO360_GPU_WARP", None)
        else:
            os.environ["STEREO360_GPU_WARP"] = prev


@pytest.fixture(scope="module")
def rendered():
    return render()


def test_the_360_output_has_not_moved(rendered):
    depth, stacked = rendered
    got = {"depth": fingerprint(depth), "stacked": fingerprint(stacked)}

    if os.environ.get("STEREO360_REGEN_BASELINE"):
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(BASELINE, **got)
        pytest.skip(f"baseline regenerated at {BASELINE}")

    assert BASELINE.exists(), (
        f"no baseline at {BASELINE}. Create one with "
        f"STEREO360_REGEN_BASELINE=1 pytest {__file__}")
    want = np.load(BASELINE)

    for key in ("depth", "stacked"):
        a, b = got[key], want[key]
        assert a.shape == b.shape, f"{key}: shape {a.shape} vs {b.shape}"
        # Depth is relative, so compare it on its own scale rather than in
        # absolute units; the stacked eyes are already 0-255.
        if key == "depth":
            span = max(float(np.ptp(b)), 1e-6)
            a, b = a / span * 255.0, b / span * 255.0
        diff = np.abs(a - b)
        if diff.max() > TOL:
            # The stacked fingerprint carries a colour axis; the depth one does
            # not. Unravel against the real shape and report only the position.
            pos = np.unravel_index(diff.argmax(), diff.shape)
            raise AssertionError(
                f"the 360 {key} moved: worst {diff.max():.3f} levels at grid "
                f"cell {pos[:2]} of {diff.shape[:2]}, mean {diff.mean():.4f}."
                f"\nIf this change was intended, review it and regenerate with "
                f"STEREO360_REGEN_BASELINE=1.")


def test_the_stacking_convention_is_left_eye_on_top(rendered):
    """Cheap to get backwards, and invisible until someone puts on a headset.

    The VR180 work adds a second stacking convention, which is exactly the
    circumstance in which this gets swapped by accident.
    """
    _, stacked = rendered
    h = stacked.shape[0] // 2
    left, right = stacked[:h], stacked[h:]
    frame = scene()
    assert stacked.shape == (H * 2, W, 3)
    # The left eye is the source frame untouched; only the right is synthesised.
    np.testing.assert_array_equal(left, frame)
    assert not np.array_equal(right, frame), "right eye was not warped"


def test_the_two_eyes_differ_by_a_plausible_disparity(rendered):
    """A guard against the warp silently becoming a no-op or going wild."""
    _, stacked = rendered
    h = stacked.shape[0] // 2
    left = stacked[:h].astype(np.int16)
    right = stacked[h:].astype(np.int16)
    band = slice(int(h * 0.35), int(h * 0.65))       # away from the poles
    diff = np.abs(left[band] - right[band]).mean()
    assert 0.5 < diff < 60, f"mean inter-eye difference {diff:.2f} is implausible"


def test_the_baseline_is_committed():
    """A characterisation test with no baseline silently passes forever."""
    assert BASELINE.exists(), (
        "the characterisation baseline is missing; it must be committed or "
        "this test protects nothing")
    assert BASELINE.stat().st_size < 200_000, "baseline unexpectedly large"
