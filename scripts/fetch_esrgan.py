"""Fetch the Real-ESRGAN photo upscaler into models/.

    python scripts/fetch_esrgan.py

Downloads the official weights (4.9 MB, BSD-3-Clause) and exports them to
ONNX, which is what stereo360 runs. Needs torch for the export only, exactly
like scripts/export_onnx.py -- nothing needs it afterwards.

Only used by `--upscale esrgan`, and only for photos. See stereo360/esrgan.py
for the measurements behind that.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request

URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
       "realesr-general-x4v3.pth")
OUT = os.path.join("models", "realesr-general-x4v3.onnx")


def _build(state):
    """SRVGGNetCompact, sized from the checkpoint rather than assumed.

    Written out rather than taking on `basicsr` as a dependency: it is a
    stack of 3x3 convolutions with PReLU between them, a pixel shuffle at the
    end, and the input added back nearest-neighbour upscaled.
    """
    import torch.nn as nn

    last = max(int(k.split(".")[1]) for k in state if k.startswith("body."))
    num_conv = (last - 2) // 2
    out_ch = state[f"body.{last}.weight"].shape[0]
    scale = int(round((out_ch / 3) ** 0.5))
    feat = state["body.0.weight"].shape[0]

    import torch.nn.functional as F

    class SRVGGNetCompact(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = scale
            body = [nn.Conv2d(3, feat, 3, 1, 1), nn.PReLU(feat)]
            for _ in range(num_conv):
                body += [nn.Conv2d(feat, feat, 3, 1, 1), nn.PReLU(feat)]
            body.append(nn.Conv2d(feat, 3 * scale * scale, 3, 1, 1))
            self.body = nn.Sequential(*body)

        def forward(self, x):
            out = F.pixel_shuffle(self.body(x), self.scale)
            return out + F.interpolate(x, scale_factor=self.scale,
                                       mode="nearest")

    return SRVGGNetCompact(), scale, num_conv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT, help=f"where to write (default: {OUT})")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"{args.out} is already there.")
        return 0
    try:
        import torch
    except ImportError:
        print("This needs torch for the export step only:\n"
              "    pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        weights = os.path.join(tmp, "realesr.pth")
        print(f"Fetching {args.url}")
        urllib.request.urlretrieve(args.url, weights)

        state = torch.load(weights, map_location="cpu")
        state = state.get("params", state)
        net, scale, num_conv = _build(state)
        # Strict, because a shape that does not fit means the architecture
        # was guessed wrong and the export would be quietly useless.
        net.load_state_dict(state)
        net.eval()
        print(f"{num_conv} convolutions, {scale}x")

        torch.onnx.export(
            net, torch.rand(1, 3, 64, 64), args.out,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {2: "h", 3: "w"},
                          "output": {2: "H", 3: "W"}},
            opset_version=17)

    # Worth having only if it agrees with the model it came from.
    import numpy as np
    import onnxruntime as ort

    probe = torch.rand(1, 3, 96, 128)
    with torch.no_grad():
        want = net(probe).numpy()
    got = ort.InferenceSession(
        args.out, providers=["CPUExecutionProvider"]).run(
            None, {"input": probe.numpy()})[0]
    err = float(np.abs(got - want).max())
    if err > 1e-4:
        print(f"The export does not match the weights ({err:.2e}).",
              file=sys.stderr)
        return 1

    size = sum(os.path.getsize(args.out + s) for s in ("", ".data")
               if os.path.exists(args.out + s))
    print(f"Wrote {args.out} ({size / 1e6:.1f} MB), "
          f"matching torch to {err:.1e}")
    print("Use it with: --upscale esrgan   (photos only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
