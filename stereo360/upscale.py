"""Optional upscaling through Topaz Video AI, when the machine happens to have it.

Detected, never depended on. Nothing here is imported at module load beyond the
standard library, no subprocess runs unless something asks a question, and every
entry point answers "not available" rather than raising when Topaz is absent.
A machine without it must behave exactly as it did before this file existed.

Why it is worth offering at all. A 4K equirect carries 10.7 pixels per degree
against a Quest 3's roughly 25, so 4K sources look soft in a headset in a way
no depth work can fix -- the detail is not there to render. Measured on this
repository's own 8K footage, downscaled to 4K and put back:

    model            PSNR   SSIM   temporal   s/frame
    Lanczos         36.43  0.981        85%       --
    Artemis LQ      27.92  0.913       126%     1.56
    Artemis MQ      27.81  0.907       122%     0.79
    Artemis HQ      27.81  0.911       104%     1.53
    Medium Halo     27.39  0.898       102%     1.53
    Gaia HQ         28.95  0.943       126%     1.09

Two things in that table decide how this is presented. Plain Lanczos wins PSNR
and SSIM by 8 dB, which does not mean it looks better: those scores reward not
guessing, and every model here invents texture rather than recovering it. And
the "temporal" column -- frame-to-frame change against the real 8K's own -- is
the one that matters in a headset, where invented detail that fails to hold
still reads as crawling. Artemis HQ and Medium Halo sit at 104% and 102%; the
rest add a quarter more movement than reality has.

Tested against **Topaz Video AI 7** only -- the last perpetual, offline
version before Topaz moved to the "Topaz Video" subscription. Everything here
was written against that: the install paths, the `tvai_up` and `tvai_fi`
filter names and their options, the model descriptions in ProgramData with
their `modelType`/`preflight`/`gui.name` fields, and the wording of the
authentication lines this reads out of the log. Internally that build reports
1.9.43, which `login.exe` also carries as its file version.

Any of those could have moved in the subscription version. If it has, the
failure should be the good kind -- `find()` returns None and the feature
simply is not offered -- but nothing here has been run against it, and the
model-description parsing is the part most likely to need a second shape.

Both passes run on **equirectangular** frames, before the pipeline's cubemap
stage, which exists only inside depth estimation. That is right for upscaling:
Topaz tiles at 672x576 with 48 px overlap, so it never sees a whole frame and
the projection's global distortion is largely invisible to it. It is more of a
compromise for interpolation, which estimates motion: an object crossing the
+/-180 degree seam has no continuation on the other side, because the tiling
does not wrap. Padding the frame with a wrapped strip before the pass and
cropping after would close that, and is not done yet.

Four facts about driving it that cost an afternoon to find, recorded so they
do not have to be found again:

  * the filter needs a *video* stream, and a run-up within it. A single PNG,
    a PNG looped twelve times, and an explicit `format=yuv420p` in the chain
    all emit nothing, so it is the demuxer rather than the pixel format -- but
    a one-frame *video* fails too. See `wrap_frames`.
  * the intermediate has to be yuv420p or the converter's own encoder refuses
    the colour tags that come back; see `_ENCODERS`.
  * Topaz's bundled ffmpeg has no libx264. Its encoder list is its own.
  * the authentication verdict is printed about a second in, while the whole
    run takes ~28 s of engine build. `auth_state` reads until the verdict and
    then stops, rather than waiting for work it does not want.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

#: Above this the upscaler is not offered. 8K is already past what a Quest 3
#: can decode as a 360 stereo *video*, and a 2x on an 8K source would land at
#: 16K, which nothing in this pipeline or on a headset wants.
FULL_WIDTH = 7680

#: Where the installer puts it. ProgramData holds the models and the auth
#: token; Program Files holds the binary.
_APP_DIRS = (
    r"C:\Program Files\Topaz Labs LLC\Topaz Video AI",
    r"C:\Program Files (x86)\Topaz Labs LLC\Topaz Video AI",
)
_MODEL_DIRS = (
    r"C:\ProgramData\Topaz Labs LLC\Topaz Video AI\models",
)

#: `modelType` in a model's json. 1 is an upscaler, 2 is frame interpolation,
#: and only the first sort belongs on the `tvai_up` filter.
_UPSCALE_TYPE = 1

_AUTH_OK = ("successfully authenticated",)
_AUTH_LOGIN = ("reauthentication is required", "watermark will be enabled",
               "failed to refresh auth", "authentication failed")


@dataclass(frozen=True)
class Install:
    """Where Topaz is, once it has been found."""

    ffmpeg: str
    models: str

    def env(self) -> Dict[str, str]:
        """The environment its ffmpeg needs to locate the models.

        Both variables, because the binary reads them separately and logs the
        two concatenated when one is missing.
        """
        e = dict(os.environ)
        e["TVAI_MODEL_DIR"] = self.models
        e["TVAI_MODEL_DATA_DIR"] = self.models
        return e


@dataclass(frozen=True)
class Model:
    """One upscaling model, named as the Topaz interface names it.

    `name` comes from the description's `gui.name` rather than its
    `displayName`, because the latter is ambiguous: Artemis and Gaia both
    offer a "High Quality" and only the gui name says which is which.
    """

    code: str          # what --upscale-model takes, e.g. "amq-13"
    short: str         # "amq"
    name: str          # "Artemis - High Quality"
    desc: str = ""     # one line, for a tooltip
    min_scale: float = 1.0
    max_scale: float = 4.0
    order: int = 999   # Topaz's own ordering, so the list reads as it does there
    preflight: int = 0   # frames it wants before it will emit anything
    postflight: int = 0


def find(app_dirs=_APP_DIRS, model_dirs=_MODEL_DIRS) -> Optional[Install]:
    """The Topaz install, or None. Never raises, never runs anything."""
    ffmpeg = next((os.path.join(d, "ffmpeg.exe") for d in app_dirs
                   if os.path.isfile(os.path.join(d, "ffmpeg.exe"))), None)
    models = next((d for d in model_dirs if os.path.isdir(d)), None)
    if ffmpeg is None or models is None:
        return None
    return Install(ffmpeg=ffmpeg, models=models)


def _version(code: str) -> int:
    m = re.search(r"-(\d+)$", code)
    return int(m.group(1)) if m else -1


def models(install: Install) -> List[Model]:
    """The upscalers this install can run, newest version of each.

    Read from the model descriptions rather than hard-coded, because which
    ones are present depends on what the owner has used -- weights arrive on
    first use. A description with no weights beside it still works: the
    binary fetches them, which was confirmed by running Medium Halo without
    ever opening the Topaz application.
    """
    return _read_models(install, _UPSCALE_TYPE)


def _read_models(install: Install, model_type: int) -> List[Model]:
    """Every enabled model of one `modelType`, newest version of each."""
    best: Dict[str, Model] = {}
    try:
        names = os.listdir(install.models)
    except OSError:
        return []
    for fn in names:
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(install.models, fn), encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or d.get("modelType") != model_type:
            continue
        if not d.get("enabled", 1):
            continue
        gui = d.get("gui") if isinstance(d.get("gui"), dict) else {}
        short = d.get("shortName") or ""
        name = gui.get("name") or d.get("displayName") or short
        code = fn[:-len(".json")]
        if not short or not name:
            continue
        keep = best.get(short)
        if keep is None or _version(code) > _version(keep.code):
            best[short] = Model(
                code=code, short=short, name=str(name),
                desc=str(gui.get("desc") or ""),
                min_scale=float(gui.get("minScale") or 1.0),
                max_scale=float(gui.get("maxScale") or 4.0),
                order=int(gui.get("displayPri") or 999),
                preflight=int(d.get("preflight") or 0),
                postflight=int(d.get("postflight") or 0))
    return sorted(best.values(), key=lambda m: (m.order, m.name.lower()))


def offered_for(width: int) -> bool:
    """Whether upscaling is worth offering for a source this wide.

    Below 8K only. Offering it on a source that is already 8K invites a 16K
    intermediate for no gain a headset can show.
    """
    return 0 < int(width) < FULL_WIDTH


def auth_state(install: Install, timeout: float = 40.0) -> str:
    """'ok', 'login', or 'unknown'.

    Topaz signs in through its own application and writes an AES-encrypted
    `auth.tpz`, so the token cannot be read or dated from disk -- an expired
    one looks exactly like a good one. The only honest check is to start a job
    and see what it says, which it says about a second in.

    So this starts the smallest job there is, reads until the verdict appears,
    and kills it rather than waiting out the ~28 s engine build behind it.
    """
    if not os.path.isfile(os.path.join(install.models, "auth.tpz")):
        return "login"
    cmd = [install.ffmpeg, "-hide_banner", "-loglevel", "debug",
           "-f", "lavfi", "-i", "testsrc=size=64x64:rate=1:duration=1",
           "-vf", "tvai_up=model=amq-13:scale=1", "-f", "null", "-"]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            env=install.env(), cwd=os.path.dirname(install.ffmpeg),
            text=True, errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError:
        return "unknown"
    verdict = "unknown"
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            low = line.lower()
            if any(s in low for s in _AUTH_LOGIN):
                verdict = "login"
                break
            if any(s in low for s in _AUTH_OK):
                verdict = "ok"
                break
    finally:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:                                   # noqa: BLE001
            pass
    return verdict


def describe(width: int = 0) -> dict:
    """What the interface needs to decide whether to show any of this.

    One call, because the interface asks once at startup and the answer is
    three questions that are only meaningful together: is it here, is it
    usable, and is this particular source worth offering it for.
    """
    install = find()
    if install is None:
        return {"available": False, "reason": "Topaz Video AI was not found",
                "auth": "unknown", "models": [], "offered": False}
    auth = auth_state(install)
    return {
        "available": True,
        "reason": "",
        "auth": auth,
        "needs_login": auth == "login",
        "models": [{"code": m.code, "short": m.short, "name": m.name,
                    "desc": m.desc, "min_scale": m.min_scale,
                    "max_scale": m.max_scale}
                   for m in models(install)],
        "offered": offered_for(width) if width else True,
        "ffmpeg": install.ffmpeg,
    }


#: Artemis Medium Quality. Chosen after looking at seven renders in a headset,
#: not from the scorecard -- it reads as the most natural on grass and foliage,
#: which is what an outdoor 360 frame is mostly made of. Note that it is one of
#: the less temporally settled models (122% of the real footage's own
#: frame-to-frame movement, against 104% for Artemis High Quality); if a render
#: shimmers in motion, `ahq` is the first thing to try.
DEFAULT_UPSCALE = "amq"

#: Chronos. Apollo is the other family; both are `tvai_fi` models.
DEFAULT_INTERPOLATE = "chr"

#: Fewest frames a wrapped still is ever given, whatever the model claims.
#: Measured, feeding Artemis Medium Quality copies of one frame:
#:
#:     1 -> 0 out    2 -> 0 out    3 -> 0 out    4 -> 4 out    6 -> 6 out
#:
#: so a one-frame video does not work either -- the filter needs a run-up, not
#: merely a video container. Gaia High Quality manages on 3. The requirement
#: is per model and the models declare it, so `wrap_frames` asks them and this
#: is only the floor.
MIN_WRAP_FRAMES = 4

#: Cap, so a model with an odd declaration cannot turn one still into a long
#: render. Every model seen needs six or fewer.
MAX_WRAP_FRAMES = 16


def wrap_frames(model: Optional[Model]) -> int:
    """How many copies of a still to hand the filter.

    `preflight` is what a model says it wants before emitting, `postflight`
    what it holds back at the end; one more than the sum clears both. That
    gives 6 for Artemis and 5 for Gaia, against the 12 this used to send --
    each of those frames is a full-size upscale, so the guess was costing
    about half the time a still took.
    """
    if model is None:
        return MIN_WRAP_FRAMES
    want = model.preflight + model.postflight + 1
    return max(MIN_WRAP_FRAMES, min(want, MAX_WRAP_FRAMES))

#: Preferred intermediate encoders, in order. The pre-pass writes a whole 8K
#: intermediate that the converter then reads, so this wants to be fast and
#: near-transparent rather than perfect: NVENC at a high quality is both, and
#: ffv1 is the fallback where there is no NVIDIA card. Topaz's ffmpeg has no
#: libx264 at all, which is why the obvious choice is absent.
#: yuv420p is not incidental. Left to itself the chain hands back RGB, and the
#: converter passes a source's colour tags through to its own encoder, where
#: libx264 refuses `colorspace gbr` and the whole run dies at the last step --
#: after the upscale has been paid for. It also matches what the final output
#: is encoded as, so nothing is gained by carrying more here.
_ENCODERS = (
    ("hevc_nvenc", ("-rc", "constqp", "-qp", "16", "-preset", "p5",
                    "-pix_fmt", "yuv420p")),
    ("ffv1", ("-level", "3", "-pix_fmt", "yuv420p")),
)


def resolve(install: Install, wanted: str, kind: str = "up") -> Optional[Model]:
    """A model from a short name ('amq') or an exact code ('amq-13')."""
    want = (wanted or "").strip().lower()
    if not want:
        return None
    pool = models(install) if kind == "up" else interpolators(install)
    for m in pool:
        if m.code.lower() == want:
            return m
    for m in pool:
        if m.short.lower() == want:
            return m
    return None


def interpolators(install: Install) -> List[Model]:
    """Frame interpolation models -- `tvai_fi` rather than `tvai_up`.

    Worth having for VR beyond what it is worth for a monitor: 30 fps judders
    noticeably in a headset, where your head keeps moving even when the
    footage does not, and there is no shutter to hide it.
    """
    return _read_models(install, model_type=2)


def _encoder(install: Install) -> List[str]:
    try:
        out = subprocess.run([install.ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=60,
                             errors="replace").stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    for name, opts in _ENCODERS:
        if name in out:
            return ["-c:v", name, *opts]
    return ["-c:v", "ffv1", "-pix_fmt", "yuv420p"]


def chain(up: Optional[Model] = None, scale: float = 2.0,
          fi: Optional[Model] = None, fps: Optional[float] = None,
          device: int = 0) -> str:
    """The -vf for one Topaz pass.

    Upscale before interpolation, deliberately. Both orders work; this one is
    much cheaper, because the upscaler only sees the frames that were really
    shot rather than the doubled set -- and the upscaler is the expensive half
    (0.79 s a frame at 4K to 8K, against interpolation's fraction of that).
    """
    parts = []
    if up is not None:
        parts.append(f"tvai_up=model={up.code}:scale={scale:g}"
                     f":device={device}:vram=1:instances=1")
    if fi is not None:
        rate = f":fps={fps:g}" if fps else ""
        parts.append(f"tvai_fi=model={fi.code}:device={device}"
                     f":vram=1:instances=1{rate}")
    return ",".join(parts)


_FRAME_RE = re.compile(r"frame=\s*(\d+)")


class UpscaleError(RuntimeError):
    """Topaz was asked to do something and could not."""


def run(install: Install, src: str, dst: str, *,
        up: Optional[Model] = None, scale: float = 2.0,
        fi: Optional[Model] = None, fps: Optional[float] = None,
        total: Optional[int] = None, reporter=None, cancel=None,
        extra_in: Optional[List[str]] = None,
        out_args: Optional[List[str]] = None) -> None:
    """One Topaz pass over `src`, writing `dst`. Raises `UpscaleError`.

    Runs before anything else in the pipeline touches the footage, and that
    ordering is not negotiable: these models *invent* detail, so upscaling the
    two eyes separately would invent different detail for each and hand the
    viewer binocular rivalry rather than sharpness. Mono in, mono out, and the
    stereo pass afterwards sees one consistent picture.
    """
    vf = chain(up, scale, fi, fps)
    if not vf:
        raise UpscaleError("nothing to do: no upscale or interpolation model")
    cmd = [install.ffmpeg, "-hide_banner", "-nostdin", "-y",
           *(extra_in or []), "-i", src, "-vf", vf,
           *(out_args if out_args is not None else _encoder(install)), dst]
    if reporter is not None:
        reporter.start(total, stage="upscale")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            env=install.env(), cwd=os.path.dirname(install.ffmpeg),
            text=True, errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as e:
        raise UpscaleError(f"could not start Topaz: {e}") from e

    seen, tail = 0, []
    assert proc.stderr is not None
    for line in proc.stderr:
        tail.append(line.rstrip())
        del tail[:-15]
        low = line.lower()
        if any(s in low for s in _AUTH_LOGIN):
            proc.kill()
            proc.wait(timeout=10)
            raise UpscaleError(
                "Topaz Video AI needs signing in again -- open it and sign in")
        m = _FRAME_RE.search(line)
        if m and reporter is not None:
            n = int(m.group(1))
            if n > seen:
                reporter.advance(n - seen)
                seen = n
        if cancel is not None and cancel():
            proc.kill()
            proc.wait(timeout=10)
            raise UpscaleError("cancelled")
    code = proc.wait()
    # A "%" means an image sequence, where the name is a pattern rather than a
    # file and there is nothing to stat.
    missing = "%" not in dst and not os.path.exists(dst)
    if code != 0 or missing:
        raise UpscaleError(f"Topaz exited {code}:\n" + "\n".join(tail[-6:]))
    if reporter is not None:
        reporter.finish(stage="upscale")


def run_still(install: Install, src: str, dst: str, *,
              up: Optional[Model] = None, scale: float = 2.0,
              ffmpeg: str = "ffmpeg", reporter=None, cancel=None) -> None:
    """The same, for a single image, via a short video it can accept.

    The wrap exists because the filter will not emit anything for an image
    input; see `WRAP_FRAMES`. The system ffmpeg builds it rather than Topaz's,
    which has no libx264 -- and this pipeline already requires ffmpeg on PATH.
    """
    import tempfile

    tmp = tempfile.mkdtemp(prefix="stereo360-upscale-")
    wrapped = os.path.join(tmp, "in.mp4")
    out_dir = os.path.join(tmp, "out")
    os.makedirs(out_dir, exist_ok=True)
    try:
        n = wrap_frames(up)
        wrap = [ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y",
                "-loop", "1", "-i", src, "-frames:v", str(n),
                "-r", "30", "-pix_fmt", "yuv420p",
                "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", wrapped]
        if subprocess.run(wrap, capture_output=True, text=True).returncode:
            raise UpscaleError("could not prepare the still for Topaz")
        run(install, wrapped, os.path.join(out_dir, "%03d.png"),
            up=up, scale=scale, total=n, reporter=reporter,
            cancel=cancel, out_args=["-pix_fmt", "rgb24"])
        frames = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
        if not frames:
            raise UpscaleError("Topaz produced no frames for this still")
        # The last one: the models carry state across frames, so the final
        # copy of a repeated still is the settled answer.
        os.replace(os.path.join(out_dir, frames[-1]), dst)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
