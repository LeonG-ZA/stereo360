"""No black console windows when the interface runs a job.

ffmpeg and ffprobe are console programs. A process that has no console of its
own gets Windows to make a *new* one for each child it starts -- and the
desktop interface has no console, because its shortcut launches it with
pythonw. So every probe, thumbnail and render flashed up a black window.

It never showed in development because a UI started from a terminal has a
console to inherit. It only appeared once the app was installed and launched
from the Start Menu, which is the whole reason to install it and try it.

CREATE_NO_WINDOW fixes it, and the fix has to be at *every* call site -- one
missed spawn is one flashing window. So this walks the source rather than
trusting a list.
"""

import ast
import platform
import subprocess
import sys
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parent.parent / "stereo360"
SPAWNERS = {"run", "Popen", "call", "check_call", "check_output"}


def spawn_calls():
    """Every subprocess spawn in the core, as (file, line, has_flag)."""
    for path in sorted(CORE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr in SPAWNERS
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "subprocess"):
                continue
            flagged = any(
                kw.arg is None and isinstance(kw.value, ast.Name)
                and kw.value.id == "NO_CONSOLE_WINDOW"
                or kw.arg is None and isinstance(kw.value, ast.Attribute)
                and kw.value.attr == "NO_CONSOLE_WINDOW"
                for kw in node.keywords)
            yield path.name, node.lineno, flagged


def test_there_are_spawns_to_check():
    """Guards the guard: if the walk stopped finding anything, every other
    assertion here would pass by describing an empty set."""
    calls = list(spawn_calls())
    assert len(calls) >= 8, f"only found {len(calls)} subprocess calls"


def test_every_ffmpeg_spawn_suppresses_its_console_window():
    missing = [f"{name}:{line}" for name, line, ok in spawn_calls() if not ok]
    assert not missing, (
        "these spawn a console window when the parent has none:\n  "
        + "\n  ".join(missing))


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
def test_the_flag_is_only_set_on_windows():
    """creationflags is a Windows-only argument -- passing it on Linux or
    macOS raises. An empty dict everywhere else keeps one code path."""
    from stereo360 import ffmpeg_io

    assert ffmpeg_io.NO_CONSOLE_WINDOW == {"creationflags": 0x08000000}


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
def test_a_child_of_a_windowless_parent_really_gets_no_console(tmp_path):
    """The measurement behind all of the above, rather than trust in a flag.

    Run from pythonw, which has no console, a plain child reports having been
    given one and a child started with the flag reports none.
    """
    from stereo360 import ffmpeg_io

    child = tmp_path / "child.py"
    child.write_text("import ctypes, sys\n"
                     "sys.exit(1 if ctypes.windll.kernel32.GetConsoleWindow()"
                     " else 0)\n")
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import subprocess, sys\n"
        f"sys.path.insert(0, r'{CORE.parent}')\n"
        "from stereo360 import ffmpeg_io\n"
        "exe = sys.executable.replace('pythonw', 'python')\n"
        f"a = subprocess.run([exe, r'{child}']).returncode\n"
        f"b = subprocess.run([exe, r'{child}'],\n"
        "                   **ffmpeg_io.NO_CONSOLE_WINDOW).returncode\n"
        "print(a, b)\n")

    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        pytest.skip("no pythonw.exe beside this interpreter")
    out = subprocess.run([str(pythonw), str(runner)], capture_output=True,
                         text=True, timeout=120,
                         **ffmpeg_io.NO_CONSOLE_WINDOW)
    assert out.stdout.split() == ["1", "0"], \
        f"expected a console without the flag and none with it: {out.stdout!r}"
