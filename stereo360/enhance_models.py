"""Fetch the three models that enhance a source before the stereo pass.

These are not in the repository -- nothing in them is authored here, and the
three licences differ -- so they are fetched from their upstream releases. That
is a deliberate choice over bundling: downloading on someone's behalf is a
different question from redistributing weights inside an installer, and it
costs the user the same 26 MB either way.

Fetching them is not the interesting part; *who* fetches them is. The scripts
in `scripts/` already worked, and a user who knows to run them was never the
problem. The problem was a first-run experience where the Enhance panel simply
did not appear, with the reason sitting unread in a JSON payload. So this
exists to give the installer and the interface one entry point to call, rather
than teaching PowerShell, bash and QML three copies of the same knowledge.

Paths come from the modules that consume them -- `fsrcnnx.DEFAULT_SHADER` and
friends -- so there is exactly one definition of where each model lives, and a
move cannot leave the fetcher writing to a place the loader no longer reads.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence

from .ffmpeg_io import NO_CONSOLE_WINDOW
from . import interpolate, upscalers


def repo_root() -> str:
    """The directory holding `stereo360/`, `scripts/` and `models/`.

    Resolved from this file rather than from the working directory: the
    installer runs from a temporary folder and the interface from wherever it
    was launched, and both must write `models/` to the same place the loaders
    read it from.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Spec(NamedTuple):
    """One fetchable model: what it is, where it goes, what fetches it."""

    key: str
    label: str
    path: str               # relative to repo_root()
    script: str             # relative to repo_root()
    bytes_: int             # download size, for a message before the wait
    needs_torch: bool       # true when the fetch is an export, not a copy
    #: "upscale" or "interpolate" -- which question this model answers, and so
    #: which judgement decides whether it is worth offering for a given
    #: source. They are not the same judgement and must not be merged: an 8K
    #: source is past the point where upscaling helps, and that says nothing
    #: at all about whether its 30 fps is worth doubling.
    kind: str

    @property
    def full_path(self) -> str:
        return os.path.join(repo_root(), self.path)

    def present(self) -> bool:
        return os.path.exists(self.full_path)


#: Sizes are the real Content-Length of each release asset, so the interface
#: can say what a download costs before starting it rather than after.
SPECS: Sequence[Spec] = tuple(
    Spec(v.code, v.name, v.path,
         os.path.join("scripts", "fetch_upscalers.py"),
         # Needing torch is about the *export*, not the runtime: a graph
         # published as .onnx is fetched and used as it is, and saying it
         # needs torch would offer a download the machine could not do.
         int(v.mb * 1e6),
         v.kind == "onnx" and not v.url.endswith(".onnx"), "upscale")
    for v in upscalers.VARIANTS
) + (
    Spec("rife", "RIFE frame interpolation", interpolate.DEFAULT_MODEL,
         os.path.join("scripts", "fetch_rife.py"), 21_026_388, False,
         "interpolate"),
)

BY_KEY: Dict[str, Spec] = {s.key: s for s in SPECS}


def missing(keys: Optional[Sequence[str]] = None) -> List[Spec]:
    """Those of `keys` (default: all) not already on disk."""
    want = SPECS if keys is None else [BY_KEY[k] for k in keys if k in BY_KEY]
    return [s for s in want if not s.present()]


def total_bytes(specs: Sequence[Spec]) -> int:
    return sum(s.bytes_ for s in specs)


def missing_export_dependency() -> Optional[str]:
    """Why the ONNX exports cannot run here, or None if they can.

    Imports for real, in a throwaway process. `find_spec` was tried first and
    is not enough: it answers whether a name resolves, not whether the module
    loads, so a package that is installed and broken -- the ordinary result of
    a pinned dependency -- passed the check and then failed once per model
    with the tail of somebody else's stderr. The same trap as
    `backends.torch_backend_problem`, which is why that one imports the names
    it actually needs.

    A subprocess rather than an import here because the caller may be the
    interface, and pulling torch into a running window to find out whether a
    download is possible is a poor trade. It is one short process, once.
    """
    probe = "import torch, spandrel"
    try:
        done = subprocess.run([sys.executable, "-c", probe],
                              capture_output=True, text=True, timeout=180,
                              **NO_CONSOLE_WINDOW)
    except (OSError, subprocess.SubprocessError) as e:
        return f"could not check for torch and spandrel ({type(e).__name__})"
    if done.returncode == 0:
        return None
    # The last line of a traceback is the exception, which is the sentence
    # worth repeating. Everything above it is where, not what.
    lines = [l for l in (done.stderr or "").strip().splitlines() if l.strip()]
    return lines[-1] if lines else "torch and spandrel are not both usable"


def _can_export() -> bool:
    return missing_export_dependency() is None


def fetch(keys: Optional[Sequence[str]] = None,
          on_line=None) -> Dict[str, Dict[str, object]]:
    """Fetch whatever is missing. Returns {key: {ok, detail}}.

    Runs each shipped script as a subprocess rather than importing it: they are
    scripts, not a package, and one of them needs torch only for its own export
    step -- importing that into a running interface to fetch a 71 KB shader
    would be a poor trade.

    Never raises. A machine that is offline, or without the export
    dependencies, must still finish with whatever it could get; the Enhance
    panel is built to offer what it finds.

    The one missing dependency is reported once, before anything is tried,
    rather than as a failure per model. Five graphs are exported from
    checkpoints and all five need the same loader, so without it the run
    printed the same cryptic tail line five times and left the reader to
    notice they were one sentence.
    """
    root = repo_root()
    out: Dict[str, Dict[str, object]] = {}
    todo = [s_ for s_ in (SPECS if keys is None else
                          [BY_KEY[k] for k in keys if k in BY_KEY])
            if not s_.present()]
    if any(s_.needs_torch for s_ in todo) and not _can_export():
        why = missing_export_dependency()
        if on_line:
            on_line(f"Cannot export the model graphs: {why}")
            # "pip install it" is the wrong advice for a package that is
            # already installed and merely will not load, which is the
            # commoner of the two faults: it is what a pinned dependency
            # leaves behind. Say which fault this is.
            absent = "No module named" in why
            if absent:
                on_line(f"    {sys.executable} -m pip install spandrel")
            else:
                on_line("  Both are installed but one does not load. Usually a "
                        "pinned dependency; reinstall them together:")
                on_line(f"    {sys.executable} -m pip install -U --force-"
                        f"reinstall torch spandrel")
            on_line("  The shaders need nothing and are fetched anyway.")
        for s_ in todo:
            if s_.needs_torch:
                out[s_.key] = {"ok": False, "detail": why}
        keys = [s_.key for s_ in todo if not s_.needs_torch]
    for spec in (SPECS if keys is None else
                 [BY_KEY[k] for k in keys if k in BY_KEY]):
        if spec.present():
            out[spec.key] = {"ok": True, "detail": "already present"}
            continue
        script = os.path.join(root, spec.script)
        # One script fetches every upscaler and takes the code to fetch;
        # RIFE has its own and takes none. Passing the code matters: without
        # it the first missing model would drag all six down with it.
        argv = ([sys.executable, script, spec.key] if spec.kind == "upscale"
                else [sys.executable, script])
        if not os.path.exists(script):
            out[spec.key] = {"ok": False,
                             "detail": f"{spec.script} missing from this build"}
            continue
        if on_line:
            on_line(f"Fetching {spec.label} ({spec.bytes_ / 1e6:.1f} MB)...")
        try:
            # cwd at the root so a script's default `models/...` output lands
            # beside the package instead of wherever the caller happened to be.
            # The interface calls this, and on Windows a bare spawn from a
            # windowless parent opens a console of its own -- which for a
            # download that takes a minute is a black rectangle over
            # someone's work.
            # utf-8 for the child, and belt-and-braces with the script's
            # own reconfigure: torch's exporter prints emoji, the Windows
            # default for a pipe is cp1252, and the resulting
            # UnicodeEncodeError happens inside the exporter rather than
            # here -- so it looked like a missing package rather than a
            # console encoding.
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            proc = subprocess.run(argv, cwd=root, capture_output=True,
                                  text=True, timeout=900, env=env,
                                  errors="replace", **NO_CONSOLE_WINDOW)
        except (OSError, subprocess.SubprocessError) as e:
            out[spec.key] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
            continue
        if proc.returncode == 0 and spec.present():
            out[spec.key] = {"ok": True, "detail": "fetched"}
        else:
            # The most informative line, not the last one. The fetch script
            # ends its failure with an instruction, so taking the tail
            # reported "    pip install spandrel" as though that were the
            # error -- advice detached from what went wrong.
            tail = [l.strip() for l in
                    (proc.stderr or proc.stdout or "").strip().splitlines()
                    if l.strip()]
            # A warning is not the error, even when it mentions one: torch's
            # deprecation notice for `dynamic_axes` contains the text
            # "UserError", and reporting that as the failure sent the reader
            # after the wrong thing entirely.
            said = next((l for l in reversed(tail)
                         if ("Error" in l or "error" in l)
                         and "Warning" not in l), None)
            out[spec.key] = {
                "ok": False,
                "detail": said or (tail[-1] if tail else
                                   f"exit {proc.returncode}")}
        if on_line:
            r = out[spec.key]
            on_line(f"  {spec.label}: "
                    f"{'ok' if r['ok'] else 'failed -- ' + str(r['detail'])}")
    return out
