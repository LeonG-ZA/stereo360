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
  * Topaz hands back RGB, and the converter's own encoder then refuses the
    colour tags that come with it, so the working file has to be YUV. What
    format it is otherwise, and whether it is lossless, is
    `stereo360.intermediate`'s business.
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

from .ffmpeg_io import NO_CONSOLE_WINDOW

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
    #: Whether the file declared a displayName. The tie-break between two
    #: models presenting under one name; see `_one_per_name`.
    named: bool = False
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
                named=bool(d.get("displayName")),
                preflight=int(d.get("preflight") or 0),
                postflight=int(d.get("postflight") or 0))
    return sorted(_one_per_name(best.values()),
                  key=lambda m: (m.order, m.name.lower()))


def _one_per_name(models) -> List[Model]:
    """One entry per name shown, because two of anything reads as a mistake.

    Topaz ships `chf-3` and `ifi-1` both named "Chronos Fast". They differ in
    that one declares a `displayName` and the other does not, which is the
    only signal available for which of them Topaz itself means -- so that is
    the tie-break, and the newer version breaks the remaining tie.
    """
    keep: Dict[str, Model] = {}
    for m in models:
        seen = keep.get(m.name)
        if seen is None or (m.named, _version(m.code)) > (seen.named,
                                                          _version(seen.code)):
            keep[m.name] = m
    return list(keep.values())


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
            text=True, errors="replace", **NO_CONSOLE_WINDOW)
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


def _as_dict(m: Model) -> dict:
    """A model as the interface wants it: names to show, limits to obey."""
    return {"code": m.code, "short": m.short, "name": m.name, "desc": m.desc,
            "min_scale": m.min_scale, "max_scale": m.max_scale}


def describe(width: int = 0) -> dict:
    """What the interface needs to decide whether to show any of this.

    One call, because the interface asks once per source and the answer is
    three questions that are only meaningful together: is it here, is it
    usable, and is this particular source worth offering it for.
    """
    install = find()
    if install is None:
        return {"available": False, "reason": "Topaz Video AI was not found",
                "auth": "unknown", "needs_login": False, "models": [],
                "interpolators": [], "offered": False}
    auth = auth_state(install)
    return {
        "available": True,
        "reason": "",
        "auth": auth,
        "needs_login": auth == "login",
        "models": [_as_dict(m) for m in models(install)],
        "interpolators": [_as_dict(m) for m in interpolators(install)],
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


def _encoder(install: Install, pix_fmt=None, width=0, height=0,
             frames=None, where=".") -> List[str]:
    """What to write the working file with.

    Lossless where it fits, and in the source's own pixel format. Both were
    wrong before -- qp 16 and a blanket yuv420p -- and the converter reads
    this file to estimate depth from, so neither was free. See
    stereo360/intermediate.py.
    """
    from . import intermediate

    args, _ = intermediate.choose(install.ffmpeg, pix_fmt=pix_fmt, width=width,
                                  height=height, frames=frames, where=where)
    return args


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


def trimmed(vf: str, trim_from: int = 0, trim_to=None) -> str:
    """`vf` with the source cut to a frame range before the models see it.

    A frame range is the whole point of a test render, and without this the
    pass ran over the entire file before the renderer got to apply it -- so
    asking for sixty frames of an 8K video meant waiting for all of it.

    Both bounds count *source* frames, and the cut goes at the head of the
    chain rather than `-frames:v` at the tail. Cutting the output short kills
    Topaz mid-stream: it exits 0xC0000374, a corrupted heap, having written
    nothing. Given an input that ends where it should, it reaches the end of
    the stream the way it expects to and stops cleanly. Frame-exact either
    way, and it never enhances a frame nobody asked for.
    """
    bounds = ([f"start_frame={trim_from}"] if trim_from > 0 else []) +              ([f"end_frame={trim_to}"] if trim_to else [])
    if not bounds:
        return vf
    return f"trim={':'.join(bounds)},setpts=PTS-STARTPTS,{vf}"


_FRAME_RE = re.compile(r"frame=\s*(\d+)")

#: Codecs Topaz's bundled ffmpeg decodes in software without help. Anything
#: else is checked before a job starts rather than crashed into.
_SOFTWARE_CODECS = ("h264", "hevc", "mpeg4", "mpeg2video", "vp9", "prores",
                    "dnxhd", "ffv1", "rawvideo", "mjpeg")

#: Hardware decoders to try for a codec the software build cannot read, in the
#: order they are worth trying. Whichever the machine actually has wins.
_HW_DECODERS = ("cuvid", "qsv", "amf")

#: A decode that failed rather than a model that failed. Topaz answers an
#: undecodable input by crashing, so the tail of its log is the only evidence.
_DECODE_FAILED = ("error submitting packet to decoder",
                  "function not implemented",
                  "decode error rate")


def input_args(install: Install, src: str,
               codec: Optional[str] = None) -> List[str]:
    """ffmpeg input options that let Topaz read `src`, or raise saying why.

    Its bundled ffmpeg lists an `av1` decoder that is a hardware-accelerated
    shell: asked to decode AV1 in software it answers "Function not
    implemented" and the process then dies with an access violation, which
    reaches the user as a wall of ffmpeg noise about a filter option. So a
    codec that is not known-good is settled here, before any work starts, by
    asking that ffmpeg to decode a single frame.

    Empty list means the plain path works, which is the common case.
    """
    if not codec or codec.lower() in _SOFTWARE_CODECS:
        return []

    candidates: List[List[str]] = [[]]
    listed = _decoders(install)
    for suffix in _HW_DECODERS:
        name = f"{codec.lower()}_{suffix}"
        if name in listed:
            candidates.append(["-c:v", name])
    for args in candidates:
        if _can_decode(install, src, args):
            return args
    raise UpscaleError(
        f"Topaz Video AI's ffmpeg cannot decode this {codec.upper()} source. "
        f"Its bundled build has no software {codec.upper()} decoder and no "
        f"hardware one that works here.\n"
        f"Either convert the source to H.265 first, or use --interpolate "
        f"rife, which reads the file through stereo360's own ffmpeg.")


def _decoders(install: Install) -> str:
    try:
        return subprocess.run(
            [install.ffmpeg, "-hide_banner", "-decoders"], capture_output=True,
            text=True, timeout=60, errors="replace", env=install.env(),
            **NO_CONSOLE_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _can_decode(install: Install, src: str, args: List[str]) -> bool:
    """Whether Topaz's ffmpeg gets a frame out of `src` with these options."""
    src = os.path.abspath(src)
    try:
        done = subprocess.run(
            [install.ffmpeg, "-hide_banner", "-nostdin", "-y", *args,
             "-i", src, "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120, errors="replace",
            env=install.env(), cwd=os.path.dirname(install.ffmpeg),
            **NO_CONSOLE_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


class UpscaleError(RuntimeError):
    """Topaz was asked to do something and could not."""


def run(install: Install, src: str, dst: str, *,
        up: Optional[Model] = None, scale: float = 2.0,
        fi: Optional[Model] = None, fps: Optional[float] = None,
        total: Optional[int] = None, reporter=None, cancel=None,
        codec: Optional[str] = None, stage: str = "upscale",
        pix_fmt: Optional[str] = None, width: int = 0, height: int = 0,
        trim_from: int = 0, trim_to: Optional[int] = None,
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
    vf = trimmed(vf, trim_from, trim_to)
    if out_args is None:
        out_args = _encoder(install, pix_fmt=pix_fmt, width=width,
                            height=height, frames=total, where=dst)
    if extra_in is None:
        extra_in = input_args(install, src, codec)
    # Absolute, because the command runs from Topaz's own directory: a path
    # relative to the user's would point at nothing there.
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    cmd = [install.ffmpeg, "-hide_banner", "-nostdin", "-y",
           *extra_in, "-i", src, "-vf", vf,
           *out_args, dst]
    if reporter is not None:
        reporter.start(total, stage=stage)
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            env=install.env(), cwd=os.path.dirname(install.ffmpeg),
            text=True, errors="replace", **NO_CONSOLE_WINDOW)
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
        joined = "\n".join(tail[-6:])
        if any(s in joined.lower() for s in _DECODE_FAILED):
            raise UpscaleError(
                "Topaz Video AI's ffmpeg could not decode this source, so it "
                "never reached the model. Convert the source to H.265 first, "
                "or use --interpolate rife, which reads the file through "
                "stereo360's own ffmpeg.\n" + joined)
        raise UpscaleError(f"Topaz exited {code}:\n" + joined)
    if reporter is not None:
        reporter.finish(stage=stage)


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
        if subprocess.run(wrap, capture_output=True, text=True,
                          **NO_CONSOLE_WINDOW).returncode:
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
