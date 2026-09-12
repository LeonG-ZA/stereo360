"""NVIDIA Video Super Resolution: the upscaler that is not ours to ship.

RTX Video left the browser in 2026 as `nvidia-vfx`, a Python binding for the
VFX SDK. It is a per-frame effect, so it fits the same interface as the onnx
models rather than needing the sequence machinery an interpolator does.

Measured 4K to 8K against the same references as everything else:

    source                      what it does
    outdoor.jpg, clean 360      37.6-38.3 dB, 0.9 dB behind ArtCNN C4F32
    Giethoorn, clean and sharp  35.2 dB, 4.4 dB behind, oversharpens 19%
    input.mp4, the X5           reproduces the source's own texture almost
                                exactly -- scrub 24.43 against 24.57,
                                brick 12.61 against 12.68

It is trained on compressed streaming video, so it assumes degradation. Given
a degraded source that assumption is right and it is the best thing measured;
given clean 8K it adds contrast that should not be there. It belongs with the
generators, and it is worth reaching for on camera footage rather than on a
pristine master.

**It is stable, and this was measured rather than assumed.** Over 120 frames
of a near-static X5 shot the lawn window changed 1.22x as much as the Lanczos
floor -- against the *untouched 8K's* own 1.20x. The truth flickers too, from
sensor noise and wind, so a method sitting at the floor is too smooth rather
than admirably steady; Lanczos at 1.00x is 17% under reality. The name is not
the evidence: the API takes one frame and holds no temporal state.

Speed is the surprise: 18 ms a frame at 4K to 8K for ULTRA, three times
faster than ArtCNN C4F32, on a pipeline whose encode costs about 960 ms. A
streaming loop measured 62 ms because each frame crossed the PCI bus twice;
an integration that keeps frames on the GPU would not pay that.

Three things keep it optional rather than default:

- the licence is NVIDIA's, not ours, so nothing here is vendored and the user
  fetches it themselves after reading the agreement. See `AGREEMENT_URL`.
- it runs on NVIDIA tensor cores only, so most of the list stays available to
  everyone and this one entry does not.
- the wheel is 490 MB, about a hundred times the largest model here.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from .ffmpeg_io import NO_CONSOLE_WINDOW

CODE = "nvvsr"
NAME = "NVIDIA VSR"
DESC = ("NVIDIA's RTX Video upscaler. Trained on compressed video, so it "
        "suits camera footage rather than a clean master -- it reproduces a "
        "degraded source's own texture more closely than anything else here, "
        "and oversharpens a pristine one. Needs an RTX 20 series or newer "
        "card, and a licence you accept once.")

#: What the wheel is called on NVIDIA's index, which is not PyPI -- PyPI
#: carries a 2.7 KB stub and the 490 MB wheels live here.
PACKAGE = "nvidia-vfx"
INDEX = "https://pypi.nvidia.com"
AGREEMENT_URL = ("https://www.nvidia.com/en-us/agreements/enterprise-software"
                 "/nvidia-software-license-agreement/")

#: Tensor cores, which the SDK requires, arrived with Turing -- compute
#: capability 7.5. Two Turing parts shipped without them, and they are the
#: reason this cannot be a bare `>= 7.5`: the GTX 1650 and 1660 report 7.5 and
#: would be offered a model they cannot run. NVIDIA words the requirement as
#: "Turing, Ampere, Ada, Blackwell, or Hopper", and every such part with
#: tensor cores is either an RTX card or a datacentre one.
MIN_CAPABILITY = (7, 5)
_NO_TENSOR_CORES = re.compile(r"\bGTX\s*16\d\d\b", re.I)

#: Driver floors, per NVIDIA. Linux is stated as a set of branch minimums
#: rather than one number, so a 580 driver older than 580.82 is not covered by
#: "newer than 570.190" and is checked against its own branch.
_WINDOWS_MIN = (570, 65)
_LINUX_BRANCH_MIN = {570: 190, 580: 82, 590: 44}

#: Beside the models it unlocks, so deleting `models/` also forgets the
#: consent -- which is the right way round.
CONSENT = os.path.join("models", ".nvidia-vfx-accepted")


@dataclass(frozen=True)
class Gpu:
    """What `nvidia-smi` says, which is all we can ask before installing."""

    name: str
    capability: tuple
    driver: tuple

    @property
    def has_tensor_cores(self) -> bool:
        return (self.capability >= MIN_CAPABILITY
                and not _NO_TENSOR_CORES.search(self.name))

    @property
    def driver_ok(self) -> bool:
        if platform.system() == "Windows":
            return self.driver >= _WINDOWS_MIN
        floor = _LINUX_BRANCH_MIN.get(self.driver[0])
        if floor is not None:
            return self.driver[1] >= floor
        # A branch newer than any listed is assumed fine; older is not.
        return self.driver[0] > max(_LINUX_BRANCH_MIN)


def _smi() -> Optional[str]:
    """`nvidia-smi` output, or None. Never raises.

    Asked of the driver rather than of torch, because this decides whether to
    *offer* the download -- at which point nothing is installed yet.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, **NO_CONSOLE_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _version(text: str) -> tuple:
    found = tuple(int(n) for n in re.findall(r"\d+", text)[:2])
    return found or (0,)


def gpu() -> Optional[Gpu]:
    """The first NVIDIA GPU the driver reports, or None."""
    out = _smi()
    if not out:
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        return Gpu(name=parts[0], capability=_version(parts[1]),
                   driver=_version(parts[2]))
    return None


def supported() -> bool:
    """Whether this machine could run it, said before anything is downloaded."""
    g = gpu()
    return bool(g and g.has_tensor_cores and g.driver_ok)


def why_not() -> Optional[str]:
    """One line for the UI when the entry is greyed out, or None if it is not.

    A reason rather than a disabled control with nothing beside it: the likely
    reader is someone on an AMD laptop wondering what they are missing.
    """
    g = gpu()
    if g is None:
        return "Needs an NVIDIA GPU; none was detected."
    if not g.has_tensor_cores:
        return ("Needs an RTX 20 series or newer card for its tensor cores. "
                f"This is a {g.name}.")
    if not g.driver_ok:
        want = (".".join(str(n) for n in _WINDOWS_MIN)
                if platform.system() == "Windows" else "570.190")
        got = ".".join(str(n) for n in g.driver)
        return f"Needs driver {want} or newer; this machine has {got}."
    return None


def installed() -> bool:
    """Whether the wheel is present. Importing is the only honest test.

    `invalidate_caches` first, and it is load-bearing rather than tidy. This
    is asked again moments after pip has added a directory to site-packages
    *in the running process*, and Python caches each path entry's listing --
    so the import looks in a directory listing taken before the package
    existed and answers no. The interface then said the download had finished
    and quietly went back to offering it, because "not installed" is exactly
    what it had been told.
    """
    import importlib

    importlib.invalidate_caches()
    try:
        import nvvfx  # noqa: F401
    except Exception:
        return False
    return True


# --- the licence -----------------------------------------------------------
#
# Acceptance is recorded against a hash of the exact text that was shown, not
# a bare flag. NVIDIA reserves the right to update the agreement, and the
# agreement itself expects whoever presents it to "update the terms it
# presents to Customer End Users"; keying on the text means an update
# re-prompts rather than passing silently under a tick from last year.


def digest(text: str) -> str:
    """A hash of the agreement, insensitive to how it was wrapped."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def accepted(text: Optional[str] = None, where: Optional[str] = None) -> bool:
    """Whether this exact agreement has been accepted before.

    With no `text`, answers whether anything was accepted -- which is what to
    ask when offline, so a machine that cannot reach NVIDIA still honours a
    consent it already holds.

    `where` defaults inside the body rather than in the signature, so that
    `CONSENT` is read when the call happens. As a parameter default it is
    bound once at import, and anything that later moved the file -- a test, or
    a per-user config directory -- would be ignored without saying so.
    """
    where = where or CONSENT
    try:
        with open(where, encoding="utf-8") as fh:
            stored = fh.read().split()
    except OSError:
        return False
    if not stored:
        return False
    return True if text is None else digest(text) in stored


def record(text: str, where: Optional[str] = None) -> None:
    """Note the acceptance, beside the models it unlocks.

    `where` is resolved here rather than in the signature, for the reason
    given in `accepted`.
    """
    from datetime import datetime, timezone

    where = where or CONSENT
    parent = os.path.dirname(where)
    if parent:
        os.makedirs(parent, exist_ok=True)
    when = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(where, "a", encoding="utf-8") as fh:
        fh.write(f"{digest(text)} {when}\n")


#: The agreement is shown from the PDFs inside the wheel rather than from
#: NVIDIA's web page, and the choice is not laziness about scraping.
#:
#: The page and the PDFs are not the same document. Fetched 2026-09-09, the
#: page's text carries "GOVERNING LAW" and "Free SDKs" but neither "1.1 Grant"
#: nor "17.22", both of which the PDF has. Presenting the page would ask
#: someone to accept text that is not the agreement shipped with their copy.
#:
#: Reading the bundled files means consent can only be asked *after* the
#: download, which turns out to match the agreement anyway -- it binds on
#: use, not on acquisition ("By registering to use or using the Software
#: Offerings, Customer is affirming that it has read the Agreement"). So the
#: download carries a short notice and a link, and the full text is presented
#: for acceptance before the first render.
_PDFS = ("NVIDIA-Software-License-Agreement-*.pdf",
         "product-specific-terms-for-nvidia-ai-products-*.pdf")


def agreement_files() -> list:
    """The licence PDFs shipped in the installed wheel, newest name first."""
    import glob

    try:
        import nvvfx
    except Exception:
        return []
    site = os.path.dirname(os.path.dirname(os.path.abspath(nvvfx.__file__)))
    found = []
    for pattern in _PDFS:
        found += sorted(glob.glob(os.path.join(
            site, "nvidia_vfx-*.dist-info", "licenses", "**", pattern),
            recursive=True))
    return found


#: Extracting the PDFs costs about half a second, which is tolerable once on
#: a deliberate click and not tolerable every time a dialog reopens. Keyed on
#: the files and their timestamps so a reinstall is not served a stale copy.
_CACHE: dict = {}


def agreement() -> Optional[str]:
    """The text to present, or None if it cannot be read.

    None is a refusal, not a warning: a dialog that cannot show the terms
    must not offer to accept them.
    """
    import logging

    files = agreement_files()
    key = tuple((f, os.path.getmtime(f)) for f in files if os.path.exists(f))
    if key and key in _CACHE:
        return _CACHE[key]
    # NVIDIA's PDFs have malformed cross-reference entries, and pypdf says so
    # once per object: six lines of "Ignoring wrong pointing object" every
    # time this is read. It recovers and the text comes out whole, so the
    # complaint is noise -- but this is reached from `present`, which every
    # probe and every render calls, so the noise lands in the render log
    # where someone reasonably wonders what is wrong with their footage.
    if not files:
        return None
    try:
        from pypdf import PdfReader
    except ImportError:
        return None

    quiet = logging.getLogger("pypdf")
    was = quiet.level
    quiet.setLevel(logging.ERROR)
    try:
        parts = []
        for path in files:
            try:
                pages = PdfReader(path).pages
            except Exception:                                 # noqa: BLE001
                return None
            text = "\n".join(page.extract_text() or "" for page in pages)
            if not text.strip():
                return None
            parts.append(f"### {os.path.basename(path)}\n\n{text}")
    finally:
        # Restored on every path out, including the two that give up early.
        # Lowering a logger for the rest of the process because a parse
        # failed halfway would hide someone else's errors.
        quiet.setLevel(was)
    whole = "\n\n".join(parts)
    if key:
        _CACHE[key] = whole
    return whole


def consented() -> bool:
    """Consent to the agreement that is actually installed here.

    Not `accepted()` with no text, which answers "has anything ever been
    accepted" -- and that is a different and far weaker question. A marker
    left by anything at all then unlocks the model, which is precisely what
    the dialog exists to prevent. It happened: a test wrote consent to the
    string "THE TERMS" into the real `models/` directory, and NVIDIA VSR was
    offered as though its licence had been read.

    A missing agreement is a refusal rather than a pass. If the terms cannot
    be read they cannot have been agreed to, and the dialog says the same by
    leaving Accept disabled.
    """
    text = agreement()
    return bool(text) and accepted(text)


def describe() -> dict:
    """What the probe reports, so the UI never has to guess."""
    g = gpu()
    return {"code": CODE, "name": NAME, "desc": DESC,
            "supported": supported(), "installed": installed(),
            "accepted": consented(), "why_not": why_not(),
            "gpu": g.name if g else None,
            "package": PACKAGE, "index": INDEX,
            "agreement_url": AGREEMENT_URL}


# --- running it ------------------------------------------------------------


class NvvsrError(RuntimeError):
    """Anything that stops a pass, phrased for someone reading a log."""


def _levels() -> list:
    """The quality names this wheel offers, or [] if it cannot be loaded."""
    try:
        from nvvfx import VideoSuperRes
    except Exception:
        return []
    return [n for n in dir(VideoSuperRes.QualityLevel) if not n.startswith("_")]


def run(src: str, dst: str, *, width: int, height: int, scale: float = 2.0,
        level: str = "ULTRA", total: Optional[int] = None,
        pix_fmt: Optional[str] = None, trim_from: int = 0,
        frames: Optional[int] = None, name: str = NAME, code: str = CODE,
        reporter=None, cancel=None, ffmpeg: str = "ffmpeg") -> None:
    """One pass over `src`, writing `dst`. Raises `NvvsrError`.

    Shaped like `fsrcnnx.run` so the caller does not care which it got.

    Frames cross between ffmpeg and the GPU one at a time, which is what
    makes this cost about 62 ms a frame where the model itself takes 18.
    The transfer is the price of reading from and writing to a container
    without holding an 8K sequence in memory; keeping frames resident would
    need the whole pipeline to be GPU-side, which it is not.
    """
    import subprocess

    import numpy as np

    if not consented():
        raise NvvsrError(
            "NVIDIA VSR needs its licence accepted before it can run. Open "
            "the interface and read it there; it is asked once.")
    try:
        import torch
        from nvvfx import VideoSuperRes
    except Exception as e:                                    # noqa: BLE001
        raise NvvsrError(
            f"NVIDIA VSR is installed but will not load ({e}). It needs "
            f"torch with CUDA as well as the wheel.")
    if not torch.cuda.is_available():
        raise NvvsrError(
            "NVIDIA VSR needs a CUDA device and torch cannot see one.")
    have = _levels()
    if level not in have:
        raise NvvsrError(f"No VSR quality level called {level!r}. "
                         f"This wheel has: {', '.join(sorted(have))}.")

    out_w, out_h = int(round(width * scale)), int(round(height * scale))
    effect = VideoSuperRes(quality=getattr(VideoSuperRes.QualityLevel, level))
    effect.output_width, effect.output_height = out_w, out_h
    effect.load()

    read_cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
                "-i", src]
    if trim_from > 0:
        read_cmd += ["-vf", f"trim=start_frame={trim_from},setpts=PTS-STARTPTS"]
    if frames:
        read_cmd += ["-frames:v", str(int(frames))]
    read_cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    from . import intermediate

    out_args, _ = intermediate.choose(ffmpeg, pix_fmt=pix_fmt, width=out_w,
                                      height=out_h, frames=total, where=dst)
    # Audio is copied from the source rather than lost, the same way the
    # shader pass does it -- a pre-pass that silently drops the soundtrack is
    # a fault nobody notices until the far end of the render.
    write_cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
                 "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{out_w}x{out_h}", "-i", "-", "-i", src,
                 "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy",
                 *out_args, dst]

    if reporter is not None:
        reporter.info(f"Upscaling with {name}: {scale:g}x from "
                      f"{width}x{height}", stage="upscale", model=code)
        reporter.start(total, stage="upscale")
    reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, **NO_CONSOLE_WINDOW)
    writer = subprocess.Popen(write_cmd, stdin=subprocess.PIPE,
                              stderr=subprocess.PIPE, **NO_CONSOLE_WINDOW)
    need = width * height * 3
    seen = 0
    try:
        assert reader.stdout is not None and writer.stdin is not None
        while True:
            raw = reader.stdout.read(need)
            if len(raw) < need:
                break
            if cancel is not None and cancel():
                raise NvvsrError("cancelled")
            frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
            # `.copy()` because the buffer is read-only and torch refuses it,
            # and `.contiguous()` because permute returns a view and the SDK
            # rejects a non-contiguous tensor outright.
            t = (torch.from_numpy(frame.copy()).cuda().permute(2, 0, 1)
                 .float().div_(255.0).contiguous())
            big = torch.from_dlpack(effect.run(t).image).clone()
            arr = (big.clamp(0, 1).mul(255).round().byte()
                   .permute(1, 2, 0).cpu().numpy())
            writer.stdin.write(np.ascontiguousarray(arr).tobytes())
            seen += 1
            if reporter is not None:
                reporter.advance(1)
            del t, big, arr
    except BrokenPipeError:
        raise NvvsrError("the writer stopped early; the pass is incomplete")
    finally:
        for p in (reader, writer):
            try:
                if p.stdin:
                    p.stdin.close()
            except OSError:
                pass
        if reader.stdout:
            reader.stdout.close()
        rc_w = writer.wait()
        reader.wait()
        if reporter is not None:
            reporter.finish(stage="upscale")
        try:
            del effect
            torch.cuda.empty_cache()
        except Exception:                                     # noqa: BLE001
            pass
    if seen == 0:
        raise NvvsrError(f"no frames were read from {src}")
    if rc_w != 0 or not os.path.exists(dst):
        why = bytes(writer.stderr.read() if writer.stderr else b"").decode(
            "utf-8", "replace").strip()
        raise NvvsrError(f"ffmpeg exited {rc_w} writing the pass:\n{why[-400:]}")
