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

Three facts about driving it that cost an afternoon to find, recorded so they
do not have to be found again:

  * the filter needs a *video* stream. A single PNG, and a PNG looped twelve
    times, both consume their frames and emit nothing. The same still wrapped
    as a 12-frame yuv420p video upscales fine, which is what `wrap_still` is
    for.
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
        if not isinstance(d, dict) or d.get("modelType") != _UPSCALE_TYPE:
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
                order=int(gui.get("displayPri") or 999))
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
