"""Export Depth Anything V2 (HF) to ONNX for the onnxruntime backend (M4).

Usage:
    python scripts/export_onnx.py [--model depth-anything/Depth-Anything-V2-Small-hf]
                                  [--out models/depth_anything_v2_small.onnx]
                                  [--opset 17]

The exported graph takes a normalized float32 image (1, 3, H, W), H/W dynamic
(multiples of 14), and outputs (1, H, W) relative inverse depth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _utf8_console() -> None:
    """Let torch's exporter print its status characters on Windows.

    torch.onnx logs progress with check and cross marks. Under the default
    Windows console encoding (cp1252) that raises UnicodeEncodeError from
    inside the logger, which replaces the real export error with an encoding
    traceback -- the failure this script actually needed to report ends up
    invisible.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main() -> None:
    _utf8_console()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--out", default="models/depth_anything_v2_small.onnx")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--size", type=int, default=518, metavar="N",
                   help="Input resolution, a multiple of 14 (default 518, the "
                        "model's native size). Smaller is much faster and much "
                        "coarser -- the graph bakes this in, so export a "
                        "separate file per size you want.")
    p.add_argument("--static-batch", action="store_true",
                   help="Pin the batch axis to 1. Required for DirectML, "
                        "which rejects the graph's Reshape once batch is "
                        "dynamic -- even at batch 1. Costs the batching "
                        "speedup on CUDA/CPU, so only use it for DML.")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForDepthEstimation

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = AutoModelForDepthEstimation.from_pretrained(args.model).eval()

    class Wrapper(torch.nn.Module):
        def forward(self, pixel_values):
            return model(pixel_values=pixel_values).predicted_depth

    wrapper = Wrapper().eval()
    if args.size % 14:
        raise SystemExit(f"--size must be a multiple of 14, got {args.size}")
    dummy = torch.zeros(2, 3, args.size, args.size)
    # Batch is dynamic so several cubemap faces (or --depth-tiles crops) can go
    # through in one call. At --depth-tiles 2 a frame needs 30 inferences; run
    # one at a time they are far too small to keep a discrete GPU busy, and the
    # per-launch overhead dominates. Exported with a batch of 2 so the tracer
    # cannot fold the batch dimension away as a constant.
    shapes = {}
    if not args.static_batch:
        # min=2, not min=1. torch.export traces a guard that the batch is not
        # 1 (something in the model specialises on it), so declaring min=1
        # fails the whole export with "Constraints violated (batch)". That is
        # only a tracing-time constraint: the exported graph keeps a symbolic
        # batch axis and onnxruntime still accepts 1, which the check below
        # verifies.
        shapes[0] = torch.export.Dim("batch", min=2)
    else:
        dummy = dummy[:1]
    torch.onnx.export(
        wrapper, dummy, str(out), opset_version=args.opset,
        input_names=["pixel_values"], output_names=["depth"],
        dynamic_shapes={"pixel_values": shapes} if shapes else None,
    )
    print(f"Exported {args.model} -> {out} ({out.stat().st_size / 2**20:.0f} MB)")

    # An export that cannot run the batch sizes the pipeline uses is worse than
    # a failed one, because nothing says so until a conversion is under way.
    # Six faces per frame is the normal call; 1 is what the probe tries first.
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("  (install onnxruntime to verify the graph runs)")
        return
    sess = ort.InferenceSession(str(out),
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    wanted = (1,) if args.static_batch else (1, 2, 6)
    for batch in wanted:
        probe = np.zeros((batch, 3, args.size, args.size), np.float32)
        try:
            got = sess.run(None, {name: probe})[0]
            print(f"  batch {batch}: OK {got.shape}")
        except Exception as exc:
            print(f"  batch {batch}: FAILED {type(exc).__name__}: "
                  f"{str(exc)[:160]}")


if __name__ == "__main__":
    main()
