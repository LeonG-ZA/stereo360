"""Resource planning must adapt to the machine, not to the author's laptop."""

import os

import numpy as np

from stereo360 import warp


def test_available_memory_is_plausible_or_none():
    av = warp.available_memory()
    assert av is None or av > 64 * 2**20, f"implausible free memory: {av}"


def test_worker_count_falls_when_memory_is_scarce(monkeypatch):
    """The thread count is a memory multiplier, so it must track free RAM.

    Each in-flight band holds its own temporaries. A constant chosen on a
    16-core laptop with 7 GB free is simultaneously too many for a 4 GB
    machine and too few for a workstation, so the count is derived per call
    from the real band size and the real free memory.
    """
    row_bytes = 7680 * warp._BAND_BYTES_PER_PX          # an 8K frame

    monkeypatch.setattr(warp, "available_memory", lambda: 32 * 2**30)
    roomy = warp.plan_workers(row_bytes, 256)
    monkeypatch.setattr(warp, "available_memory", lambda: 512 * 2**20)
    cramped = warp.plan_workers(row_bytes, 256)

    assert cramped < roomy, f"{cramped} not fewer than {roomy} on 512 MB"
    assert cramped >= 1, "must always leave at least one worker"


def test_worker_count_falls_as_frames_get_bigger(monkeypatch):
    monkeypatch.setattr(warp, "available_memory", lambda: 2 * 2**30)
    small = warp.plan_workers(1920 * warp._BAND_BYTES_PER_PX, 256)
    huge = warp.plan_workers(15360 * warp._BAND_BYTES_PER_PX, 256)
    assert huge <= small


def test_unknown_memory_falls_back_to_a_modest_default(monkeypatch):
    monkeypatch.setattr(warp, "available_memory", lambda: None)
    n = warp.plan_workers(7680 * warp._BAND_BYTES_PER_PX, 256)
    assert 1 <= n <= 4, f"unknown machine should stay modest, got {n}"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("STEREO360_WORKERS", "3")
    monkeypatch.setattr(warp, "available_memory", lambda: 512 * 2**20)
    assert warp.plan_workers(7680 * warp._BAND_BYTES_PER_PX, 256) == 3


def test_thread_count_never_changes_the_result(monkeypatch):
    """Parallelism is an optimisation; it must not be observable in output."""
    rng = np.random.default_rng(0)
    h, w = 300, 600
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    dn = np.full((h, w), 0.05, np.float32)
    dn[:, w // 2 - 60:w // 2 + 60] = 0.95

    ref = None
    for n in (1, 2, 3, 8):
        monkeypatch.setenv("STEREO360_WORKERS", str(n))
        got, hole = warp.right_eye_from_disparity(
            img, dn.copy(), 3.0, inpaint=False, normalize=False)
        if ref is None:
            ref = (got, hole)
            continue
        assert np.array_equal(got, ref[0]), f"image differs at {n} workers"
        assert np.array_equal(hole, ref[1]), f"holes differ at {n} workers"


def test_banner_wording_matches_the_two_cases():
    """The CPU case must be conspicuous. Falling back to the CPU silently is
    the worst outcome: an order of magnitude slower with nothing saying so."""
    from stereo360.depth import autodetect

    gpu = autodetect.Runtime("onnx", "CUDAExecutionProvider",
                             "ONNX Runtime on CUDAExecutionProvider", True)
    cpu = autodetect.Runtime("depth-anything", "cpu",
                             "Depth Anything V2 on CPU (torch)", False)

    assert autodetect.banner(gpu) == (
        "GPU accelerated: ONNX Runtime on CUDAExecutionProvider")
    msg = autodetect.banner(cpu)
    assert msg.startswith("**WARNING** GPU not being utilized.")
    assert "CPU accelerated: Depth Anything V2 on CPU (torch)" in msg
    assert msg.endswith("Please read the README file.")


def test_autodetect_returns_a_usable_runtime():
    from stereo360.depth import autodetect

    rt = autodetect.detect("models/depth_anything_v2_small.onnx")
    assert rt.backend in ("depth-anything", "onnx")
    assert rt.label and isinstance(rt.gpu, bool)


def test_chunk_size_shrinks_to_fit_memory(monkeypatch):
    """A chunk buffers whole frames, so its cost scales with resolution.

    Eight frames of 8K is roughly 3 GB -- fine on a workstation, fatal on a
    laptop. One constant cannot serve both, so the ceiling comes from the
    machine.
    """
    from stereo360 import pipeline, warp

    monkeypatch.setattr(warp, "available_memory", lambda: 64 * 2**30)
    assert pipeline.fit_chunk_size(8, 7680, 3840) == 8

    monkeypatch.setattr(warp, "available_memory", lambda: 3 * 2**30)
    tight = pipeline.fit_chunk_size(8, 7680, 3840)
    assert 2 <= tight < 8, tight

    # splitting synthesizes two eyes, so it buffers twice as much
    monkeypatch.setattr(warp, "available_memory", lambda: 8 * 2**30)
    assert pipeline.fit_chunk_size(8, 7680, 3840, eyes=2) <= \
        pipeline.fit_chunk_size(8, 7680, 3840, eyes=1)

    # never reduced below a usable chunk, and never touched when already 1
    monkeypatch.setattr(warp, "available_memory", lambda: 64 * 2**20)
    assert pipeline.fit_chunk_size(8, 7680, 3840) == 2
    assert pipeline.fit_chunk_size(1, 7680, 3840) == 1
