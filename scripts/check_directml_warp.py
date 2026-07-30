"""Can the GPU warp run on DirectML? Probes each operation it needs.

The GPU warp (`stereo360/warp_torch.py`) is plain torch, so in principle it can
run on any device torch supports -- including AMD and Intel GPUs through
`torch-directml`. That would be worth a lot: on CUDA the warp went from 1.20 s
to 0.11 s per 8K frame, and the warp is the largest CPU cost left on machines
without a torch GPU.

Whether it *actually* runs depends on DirectML's operator coverage, which is
narrower than CUDA's and not introspectable. So this tries each operation the
warp depends on, one at a time, and reports what works.

    pip install torch-directml
    python scripts/check_directml_warp.py

WARNING: torch-directml pins torch==2.4.1 and torchvision==0.19.1, so
installing it will DOWNGRADE torch. That is harmless on a machine where torch
is CPU-only anyway (AMD on Windows has no ROCm build), but it will break a
working CUDA install. Do not run this on a CUDA machine without a virtualenv.

If every operation passes, the last check runs the real warp on the DirectML
device and compares it against the numpy reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch is not installed.")
        return 1

    try:
        import torch_directml
    except ImportError:
        print("torch-directml is not installed.\n"
              "    pip install torch-directml\n"
              "Note it pins torch==2.4.1, so it will downgrade torch. Safe on "
              "a machine\nwhere torch is CPU-only; it will break a CUDA "
              "install.")
        return 1

    try:
        dev = torch_directml.device()
    except Exception as exc:                                # pragma: no cover
        print(f"torch-directml found no device: {exc}")
        return 1

    print(f"device: {torch_directml.device_name(0)}")
    print(f"torch:  {torch.__version__}")
    print()

    import torch.nn.functional as F

    h, w = 64, 128
    a = torch.rand(h, w, device=dev)
    b = torch.rand(h, w, device=dev)

    # Each entry is (label, thunk). The labels name where the warp uses them so
    # a failure says what breaks rather than just which operator is missing.
    checks = [
        ("arithmetic + sqrt          (3D lift, distance)",
         lambda: torch.sqrt(a * a + b * b)),
        ("atan2                      (longitude)",
         lambda: torch.atan2(a, b)),
        ("asin + clamp               (latitude)",
         lambda: torch.asin(a.clamp(-1, 1))),
        ("floor / int64 / remainder  (splat target)",
         lambda: torch.floor(a * w).to(torch.int64).remainder(w)),
        # dtype is stated explicitly on both sides. An earlier version of this
        # probe let `torch.full` pick its own, and DirectML rejected the
        # mismatch with "Expected self.dtype to be equal to src.dtype" -- which
        # looked like a missing operator but was the probe's own bug. The real
        # warp has always passed dtype=norm.dtype here.
        ("scatter_reduce_ amin       (the z-buffer)",
         lambda: torch.full((h * w,), float("inf"), device=dev,
                            dtype=torch.float32).scatter_reduce_(
             0, torch.floor(a * (h * w - 1)).to(torch.int64).reshape(-1),
             b.reshape(-1).to(torch.float32), reduce="amin")),
        # Fallbacks, only interesting if the line above fails.
        ("  fallback: index_reduce_ amin",
         lambda: torch.full((h * w,), float("inf"), device=dev,
                            dtype=torch.float32).index_reduce_(
             0, torch.floor(a * (h * w - 1)).to(torch.int64).reshape(-1),
             b.reshape(-1).to(torch.float32), "amin")),
        ("  fallback: scatter_reduce_ on CPU, rest on device",
         lambda: torch.full((h * w,), float("inf"),
                            dtype=torch.float32).scatter_reduce_(
             0, torch.floor(a * (h * w - 1)).to(torch.int64).reshape(-1).cpu(),
             b.reshape(-1).to(torch.float32).cpu(), reduce="amin").to(dev)),
        ("slice shift + minimum      (2x2 splat footprint)",
         lambda: torch.minimum(a, torch.cat((a[:, -1:], a[:, :-1]), dim=1))),
        ("cummin + flip              (gradient clamp)",
         lambda: torch.cummin(a.flip(1), dim=1).values.flip(1)),
        ("max_pool2d                 (fg erode, crack close)",
         lambda: F.max_pool2d(a[None, None], 5, stride=1, padding=2)),
        ("grid_sample bilinear       (pass 2 resample)",
         lambda: F.grid_sample(a[None, None],
                               torch.rand(1, h, w, 2, device=dev) * 2 - 1,
                               mode="bilinear", align_corners=False)),
        ("grid_sample nearest        (visibility check)",
         lambda: F.grid_sample(a[None, None],
                               torch.rand(1, h, w, 2, device=dev) * 2 - 1,
                               mode="nearest", align_corners=False)),
        ("isfinite / isinf / where   (hole detection)",
         lambda: torch.where(torch.isfinite(a), a, torch.zeros_like(a))),
        ("boolean mask assignment    (zeroing holes)",
         lambda: a.clone().masked_fill_(a > 0.5, 0.0)),
        ("device -> cpu numpy        (returning the result)",
         lambda: a.cpu().numpy()),
    ]

    # "It ran" is not the same as "it ran on the GPU". DirectML silently
    # executes unsupported operators on the CPU, which for a full-frame tensor
    # means copying it off the device and back in the middle of a pass -- far
    # worse than not using the GPU at all. Those warnings are the only signal,
    # so they are captured rather than printed loose.
    import warnings as _warnings

    failures = 0
    fallbacks = 0
    for label, thunk in checks:
        optional = label.startswith("  fallback")
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            try:
                out = thunk()
                if hasattr(out, "cpu"):
                    out.cpu()                  # force it to actually execute
                failed = None
            except Exception as exc:
                failed = exc
        fell_back = any("fall back to run on the CPU" in str(c.message)
                        for c in caught)
        if failed is not None:
            if not optional:
                failures += 1
            print(f"  {'skip' if optional else 'FAIL'}  {label}")
            print(f"          {type(failed).__name__}: "
                  f"{str(failed).splitlines()[0][:90]}")
        elif fell_back:
            if not optional:
                fallbacks += 1
            print(f"  CPU   {label}")
            print("          runs, but on the CPU: a full-frame round trip "
                  "per call")
        else:
            print(f"  OK    {label}")

    print()
    if failures:
        print(f"{failures} operation(s) unsupported: the GPU warp cannot run on "
              "DirectML as written.\nThe numpy path is used automatically, so "
              "nothing is broken -- there is just\nno GPU warp on this machine.")
        return 2

    print("All operations supported. Running the real warp for comparison...")
    from stereo360 import warp, warp_torch

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (256, 512, 3), dtype=np.uint8)
    dn = np.full((256, 512), 0.10, np.float32)
    dn[64:192, 216:296] = 0.90

    cpu, cpu_hole = warp.right_eye_from_disparity(
        img, dn.copy(), 1.0, inpaint=False, normalize=False, gradient_limit=1.0)
    try:
        gpu, gpu_hole = warp_torch.right_eye_from_disparity(
            img, dn.copy(), 1.0, warp._BASELINE_SCALE, warp._MIN_INV_DEPTH,
            warp._VIS_RATIO, warp._CRACK_MARGIN, 2, 1.0, str(dev))
    except Exception as exc:
        print(f"  the full warp still failed: {type(exc).__name__}: {exc}")
        return 2

    diff = np.abs(cpu.astype(int) - gpu.astype(int)).max(axis=-1)
    holes_match = np.array_equal(cpu_hole > 0, gpu_hole > 0)
    print(f"  hole masks identical : {holes_match}")
    print(f"  mean |difference|    : {diff.mean():.4f} levels")
    print(f"  pixels > 2 levels    : {100 * (diff > 2).mean():.4f}%")
    print()
    if holes_match and diff.mean() < 0.05:
        print("Matches the reference. Enable it with STEREO360_GPU_WARP=1 "
              "and report the\nspeed from scripts/bench_pipeline.py.")
        return 0
    print("Runs, but disagrees with the reference -- do not use it.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
