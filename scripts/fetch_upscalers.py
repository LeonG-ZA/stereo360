"""Fetch the upscaling models into models/.

    python scripts/fetch_upscalers.py            # everything missing
    python scripts/fetch_upscalers.py span siax  # just these

Shaders are a download. The ONNX models are a download *and* an export, and
the export needs `spandrel` beside torch -- for the same reason
`fetch_esrgan.py` needs torch: nothing needs it afterwards.

Why spandrel rather than writing the architectures out, as the other fetch
scripts do. RRDBNet and SRVGGNetCompact were written out and verified against
it, tensor for tensor and score for score. SPAN was not: it was reconstructed
three times by hand and each attempt produced a picture and the wrong one,
the last collapsing to flat grey. Two traps in it, and the second is the
instructive one -- the blocks ship their fused convolution *unfused*, so a
loader that takes the single 3x3 at face value gets a plausible model made of
initialisation noise. spandrel fuses on load and knows the rest.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stereo360 import upscalers                                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# torch's ONNX exporter prints progress with emoji in it, and on Windows a
# console or a pipe defaults to cp1252, which cannot encode them. The print
# then raises UnicodeEncodeError *inside* the exporter and takes the export
# down with it -- so a working torch, a working spandrel and a model that
# loads fine still produced nothing, and the failure read as a missing
# package. Measured on an installed 1.0.7: five checkpoints downloaded, no
# graphs written, and the only clue three frames deep in someone else's
# traceback.
#
# `errors="replace"` as well as utf-8, because the point is that no character
# an upstream library decides to print may stop a model being built.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # not a real stream
        pass


def fetch(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"    fetching {url.rsplit('/', 1)[-1]}")
    urllib.request.urlretrieve(url, dest)


def export_onnx(weights: str, out: str) -> None:
    """Load whatever architecture the checkpoint is, and write a graph."""
    import torch
    from spandrel import ModelLoader

    d = ModelLoader().load_from_file(weights)
    net = d.model.eval()
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"    {d.architecture.name} x{d.scale}, {params:.2f}M parameters")
    # Dynamic height and width: the runner tiles, and the tiles at a frame's
    # right and bottom edges are not the same size as the rest.
    #
    # `dynamo=False` asks for the older tracing exporter, and it is not
    # conservatism for its own sake. torch made the dynamo exporter the
    # default, and SPAN through it *segfaults* -- exit 139, no traceback, on
    # torch 2.13 -- while the same model through the tracer exports in a
    # second. The tracer also takes `dynamic_axes`, which the new one
    # deprecates and warns about. Older torch has no such argument, so its
    # absence is not an error.
    kw = dict(input_names=["input"], output_names=["output"],
              opset_version=17,
              dynamic_axes={"input": {2: "h", 3: "w"},
                            "output": {2: "H", 3: "W"}})
    try:
        torch.onnx.export(net, torch.randn(1, 3, 64, 64), out,
                          dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(net, torch.randn(1, 3, 64, 64), out, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("which", nargs="*",
                    help=f"any of: {', '.join(v.code for v in upscalers.VARIANTS)}")
    ap.add_argument("--force", action="store_true",
                    help="fetch again even if the file is already there")
    args = ap.parse_args()

    want = upscalers.VARIANTS
    if args.which:
        unknown = [w for w in args.which if w not in upscalers.BY_CODE]
        if unknown:
            print(f"Unknown: {', '.join(unknown)}. Expected any of "
                  f"{', '.join(upscalers.BY_CODE)}.", file=sys.stderr)
            return 2
        want = [upscalers.BY_CODE[w] for w in args.which]

    todo = [v for v in want if args.force or not upscalers.present(v, ROOT)]
    if not todo:
        print("Everything asked for is already in models/.")
        return 0
    print(f"Fetching {len(todo)}, about {sum(v.mb for v in todo):.0f} MB.\n")

    failed = []
    for v in todo:
        print(f"  {v.name}")
        dest = os.path.join(ROOT, v.path)
        try:
            if v.kind == "shader":
                fetch(v.url, dest)
            else:
                # The weights keep their own name and extension beside the
                # graph: it is what says where a model came from, and the
                # export can be repeated without fetching again.
                raw = os.path.join(ROOT, "models", v.url.rsplit("/", 1)[-1])
                if args.force or not os.path.exists(raw):
                    fetch(v.url, raw)
                export_onnx(raw, dest)
            print(f"    -> {v.path} "
                  f"({os.path.getsize(dest) / 1e6:.1f} MB)")
        except Exception as e:                                   # noqa: BLE001
            # One model that cannot be built must not stop the others: the
            # ONNX export needs torch and spandrel, and a machine without
            # them should still end up with the shaders.
            failed.append((v.name, f"{type(e).__name__}: {e}"))
            print(f"    failed: {type(e).__name__}: {e}", file=sys.stderr)

    if failed:
        print(f"\n{len(failed)} of {len(todo)} could not be fetched:",
              file=sys.stderr)
        for name, why in failed:
            print(f"  {name}: {why}", file=sys.stderr)
        if any(v.kind == "onnx" for v in todo):
            print("\nThe ONNX models need torch and spandrel to export:\n"
                  "    pip install spandrel", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
