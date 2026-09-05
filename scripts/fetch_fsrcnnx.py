"""Fetch the FSRCNNX upscaling shader into models/.

    python scripts/fetch_fsrcnnx.py

About 70 KB of GLSL, by igv, under LGPL-3.0 — fetched rather than kept in the
repository, like the other models here. Needs nothing to run it but the
ffmpeg stereo360 already uses: it is a libplacebo custom shader, so it goes
on any GPU with a Vulkan driver.

Used by `--upscale fsrcnnx`. See stereo360/fsrcnnx.py for the measurements.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

URL = ("https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/"
       "FSRCNNX_x2_8-0-4-1.glsl")
OUT = os.path.join("models", "FSRCNNX_x2_8-0-4-1.glsl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT, help=f"where to write (default: {OUT})")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"{args.out} is already there.")
        return 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Fetching {args.url}")
    urllib.request.urlretrieve(args.url, args.out)

    # A shader that is not a shader would fail deep inside a filtergraph, so
    # check the one thing that says what it is.
    with open(args.out, encoding="utf-8", errors="replace") as fh:
        head = fh.read(4096)
    if "!HOOK" not in head:
        os.remove(args.out)
        print("That download is not an mpv-format shader.", file=sys.stderr)
        return 1

    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stereo360 import fsrcnnx

    if not fsrcnnx.usable():
        print("\nNote: this ffmpeg has no libplacebo, or there is no Vulkan\n"
              "device for it, so --upscale fsrcnnx will not run here yet.",
              file=sys.stderr)
        return 0
    print("Use it with: --upscale fsrcnnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
