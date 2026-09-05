"""Fetch the RIFE interpolation graph into models/.

    python scripts/fetch_rife.py

About 20 MB. The graph is not in the repository for the same reason the depth
graphs are not: nothing in it is authored here.

It comes from the vs-mlrt model release, which packages the RIFE v4.x models
as ONNX. RIFE itself is MIT (hzwer/Practical-RIFE), so shipping renders made
with it carries no conditions.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

URL = ("https://github.com/AmusementClub/vs-mlrt/releases/download/"
       "external-models/rife_v4.25.7z")

#: Inside the archive. The v2 form takes the two frames and the timestep as
#: one 7-channel tensor and pads internally, which is why it is the one used.
MEMBER = "rife_v2/rife_v4.25.onnx"
OUT = os.path.join("models", "rife_v4.25.onnx")


def _extract(archive: str, into: str) -> None:
    """7-Zip, bsdtar or py7zr -- whichever this machine turns out to have."""
    tries = []
    seven = shutil.which("7z") or shutil.which("7za")
    if seven:
        tries.append([seven, "x", "-y", f"-o{into}", archive])
    # Windows ships bsdtar as tar.exe and it reads 7z; GNU tar does not, so
    # this is attempted rather than relied on.
    if shutil.which("tar"):
        tries.append(["tar", "-xf", archive, "-C", into])
    for cmd in tries:
        try:
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                if os.path.exists(os.path.join(into, MEMBER)):
                    return
        except OSError:
            pass
    try:
        import py7zr

        with py7zr.SevenZipFile(archive, "r") as z:
            z.extractall(into)
        if os.path.exists(os.path.join(into, MEMBER)):
            return
    except ImportError:
        pass
    raise SystemExit(
        "Could not unpack the archive. Install 7-Zip, or `pip install py7zr`, "
        f"or unpack {archive} by hand and copy {MEMBER} to {OUT}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT, help=f"where to write (default: {OUT})")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"{args.out} is already there.")
        return 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "rife.7z")
        print(f"Fetching {args.url}")
        urllib.request.urlretrieve(args.url, archive)
        print(f"{os.path.getsize(archive) / 1e6:.1f} MB, unpacking")
        _extract(archive, tmp)
        shutil.move(os.path.join(tmp, MEMBER), args.out)

    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print("Use it with: --interpolate rife")
    return 0


if __name__ == "__main__":
    sys.exit(main())
