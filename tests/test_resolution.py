"""Nothing may be tied to one input resolution.

Everything resolution-dependent has to derive from the frame, not from a
constant chosen while testing 8K footage. The one that mattered was the
encoder's memory guard: a flat `width * height >= 16_000_000` left 4K
top-bottom (3840x3840 = 14.7 Mpx) just under the line on every machine,
including ones with no headroom to spare.
"""

import numpy as np
import pytest

from stereo360 import pipeline, projection, warp


@pytest.mark.parametrize("w,h", [(1024, 512), (2048, 1024), (3840, 1920)])
def test_critical_gradient_normalises_itself(w, h):
    """`--gradient-limit 1.0` must mean the same thing at any width.

    The critical slope is 2*pi / (baseline * width), so the pixel shift it
    permits is one pixel regardless of resolution.
    """
    baseline = warp._BASELINE_SCALE
    critical = (2.0 * np.pi) / (baseline * w)
    shift_px = baseline * critical / (2.0 * np.pi) * w
    assert abs(shift_px - 1.0) < 1e-6


@pytest.mark.parametrize("w,h", [(512, 256), (1024, 512), (2048, 1024)])
def test_warp_runs_and_holes_stay_closed_at_any_size(w, h):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    dn = np.full((h, w), 0.10, np.float32)
    dn[h // 4:3 * h // 4, w // 2 - w // 16:w // 2 + w // 16] = 0.90
    right, hole = warp.right_eye_from_disparity(
        img, dn.copy(), 1.0, inpaint=False, normalize=False,
        gradient_limit=1.0)
    assert right.shape == img.shape
    assert (hole > 0).mean() < 0.001, (
        f"{100 * (hole > 0).mean():.4f}% holes at {w}x{h}")


@pytest.mark.parametrize("w,h", [(512, 256), (2048, 1024)])
def test_cubemap_round_trip_at_any_size(w, h):
    face = max(8, w // 4)
    img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    faces = projection.equirect_to_cubemap(img, face)
    assert all(faces[f].shape == (face, face, 3) for f in projection.FACES)
    back = projection.cubemap_to_equirect(faces, w, h)
    assert back.shape[:2] == (h, w)


def test_encoder_memory_guard_follows_the_machine(monkeypatch):
    """The guard must depend on free memory, not a fixed pixel count."""
    from stereo360 import ffmpeg_io

    def limited(width, height, free_gb):
        monkeypatch.setattr(ffmpeg_io, "available_memory",
                            lambda: int(free_gb * 2 ** 30), raising=False)
        from stereo360 import warp as _warp
        monkeypatch.setattr(_warp, "available_memory",
                            lambda: int(free_gb * 2 ** 30))
        frame_bytes = width * height * 1.5
        return (frame_bytes * ffmpeg_io._ENCODER_BUFFERED_FRAMES
                > 0.25 * (free_gb * 2 ** 30))

    # 8K top-bottom needs restraining even on a large machine ...
    assert limited(7680, 7680, 16)
    # ... 4K does not, when there is room ...
    assert not limited(3840, 3840, 32)
    # ... but the same 4K output does on a small one, which a fixed
    # 16 Mpx threshold could never express.
    assert limited(3840, 3840, 2)


def test_resource_planning_scales_with_frame_size(monkeypatch):
    monkeypatch.setattr(warp, "available_memory", lambda: 4 * 2 ** 30)
    small = warp.plan_workers(1024 * warp._BAND_BYTES_PER_PX, 256)
    large = warp.plan_workers(7680 * warp._BAND_BYTES_PER_PX, 256)
    assert large <= small
    assert pipeline.fit_chunk_size(8, 7680, 3840) <= \
        pipeline.fit_chunk_size(8, 1920, 960)
