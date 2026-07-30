"""Per-stage timings for the real pipeline, on whatever machine runs it.

Reports the environment it found and where the time actually goes, so the two
can be compared across machines (CUDA vs DirectML vs CPU) without guessing.

    python scripts/bench_pipeline.py input.mp4
    python scripts/bench_pipeline.py input.mp4 --frames 8 --face-size 960

The first frame is discarded: it builds the projection tables and loads the
model, which are per-run costs rather than per-frame ones. Encoding is excluded
because ffmpeg runs it in a separate process, overlapped with everything here --
measuring it in isolation overstates it badly.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def _describe_environment(onnx_model: str) -> None:
    import cv2

    print("environment")
    print("-" * 62)
    print(f"  python              {sys.version.split()[0]}")
    print(f"  numpy               {np.__version__}")
    print(f"  opencv              {cv2.__version__}   OpenCL={cv2.ocl.haveOpenCL()}")
    print(f"  cpu threads         {os.cpu_count()}")

    from stereo360.warp import available_memory, gpu_device

    avail = available_memory()
    print(f"  free memory         "
          f"{'unknown' if avail is None else f'{avail / 2**30:.1f} GB'}")

    try:
        import torch

        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "-"
        print(f"  torch               {torch.__version__}   cuda={cuda} {name}")
    except ImportError:
        print("  torch               not installed")

    try:
        import onnxruntime as ort

        print(f"  onnxruntime         {ort.__version__}")
        print(f"    providers         {ort.get_available_providers()}")
    except ImportError:
        print("  onnxruntime         not installed")

    print(f"  GPU warp            {gpu_device() or 'no (numpy path)'}")

    from stereo360.depth import autodetect

    rt = autodetect.detect(onnx_model)
    print(f"  depth backend       {autodetect.banner(rt)}")
    print()
    return rt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--frames", type=int, default=4,
                   help="frames to time (plus one warm-up, default 4)")
    p.add_argument("--face-size", type=int, default=None,
                   help="cubemap face size (default: input width / 4)")
    p.add_argument("--depth-tiles", type=int, default=1)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--gradient-limit", type=float, default=1.0)
    p.add_argument("--onnx-model", default="models/depth_anything_v2_small.onnx")
    args = p.parse_args()

    from stereo360 import ffmpeg_io, pipeline, projection, warp

    rt = _describe_environment(args.onnx_model)

    if rt.backend == "onnx":
        from stereo360.depth.onnx_backend import OnnxDepthBackend

        backend = OnnxDepthBackend(args.onnx_model, provider=rt.device)
    else:
        from stereo360.depth.depth_anything import DepthAnythingBackend

        backend = DepthAnythingBackend(device=rt.device)

    info = ffmpeg_io.probe(args.input)
    face = args.face_size or max(1, info.width // 4)
    print(f"input {info.width}x{info.height}, face size {face}, "
          f"{args.frames} timed frames (1 warm-up)")
    print("-" * 62)

    totals: dict = {}

    def add(name: str, seconds: float) -> None:
        totals[name] = totals.get(name, 0.0) + seconds

    frames_iter = ffmpeg_io.decode_frames(args.input,
                                          max_frames=args.frames + 1)
    for index in range(args.frames + 1):
        warm = index == 0
        t = time.perf_counter()
        try:
            frame = next(frames_iter)
        except StopIteration:
            print(f"  (input has fewer than {args.frames + 1} frames)")
            break
        decode = time.perf_counter() - t
        h, w = frame.shape[:2]

        t = time.perf_counter()
        faces = projection.equirect_to_cubemap(frame, face)
        to_faces = time.perf_counter() - t

        t = time.perf_counter()
        depths = backend.estimate_chunk([faces[f] for f in projection.FACES]) \
            if args.depth_tiles == 1 else \
            [pipeline.estimate_tiled(backend, [faces[f]], args.depth_tiles)[0]
             for f in projection.FACES]
        disp_faces = dict(zip(projection.FACES, depths))
        depth = time.perf_counter() - t

        t = time.perf_counter()
        projection.align_face_scales(disp_faces)
        align = time.perf_counter() - t

        t = time.perf_counter()
        disp = projection.cubemap_to_equirect(disp_faces, w, h)[..., 0]
        assemble = time.perf_counter() - t

        dn = warp.normalize_inv_depth(disp)
        t = time.perf_counter()
        right, hole = warp.right_eye_from_disparity(
            frame, dn.copy(), args.strength, inpaint=False, normalize=False,
            gradient_limit=args.gradient_limit)
        warp_t = time.perf_counter() - t

        t = time.perf_counter()
        warp.fill_holes(right, hole, dn)
        fill = time.perf_counter() - t

        if warm:
            continue
        add("decode", decode)
        add("equirect -> 6 faces", to_faces)
        add("depth inference", depth)
        add("align_face_scales", align)
        add("cubemap -> equirect", assemble)
        add("warp", warp_t)
        add("fill_holes", fill)

    n = max(args.frames, 1)
    grand = sum(totals.values())
    print(f"{'stage':<26} {'s/frame':>10} {'share':>8}")
    print("-" * 62)
    for name, value in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{name:<26} {value / n:>10.3f} {100 * value / grand:>7.1f}%")
    print("-" * 62)
    print(f"{'TOTAL (excludes encode)':<26} {grand / n:>10.3f}")
    print()
    print("Encoding is deliberately excluded: ffmpeg encodes in its own "
          "process,\noverlapped with the work above, so it rarely adds to wall "
          "clock.")
    backend.close()


if __name__ == "__main__":
    main()
