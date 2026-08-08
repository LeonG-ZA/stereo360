"""Things that should never be committed, checked against what is committed.

One test so far, and it exists because a hardcoded
`C:/Users/<name>/Documents/...` sat in `experiments/score.py` for weeks. It
broke nothing -- the only person running that script was the person whose path
it was -- which is exactly why nobody noticed, and why a review will not catch
the next one either.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: An absolute path into somebody's account, on any of the three platforms.
#: Deliberately not "any absolute path": `C:/x/out.mp4` in a test is a fixture,
#: and `%LOCALAPPDATA%\Programs\stereo360` in the installer is the answer.
_HOME_PATH = re.compile(
    r"""(?ix)
    (?: [a-z]:[\\/]+users[\\/]+           # C:\Users\someone
      | /(?:home|Users)/                  # /home/someone, /Users/someone
    )
    (?!                                   # ...unless the next word is a
        (?: x | y | mine | test | tmp |   # placeholder nobody could own
            user | username | you |
            someone | example )
        \b
    )
    [^\s"'<>|*?]+
    """)

#: Files that legitimately name a person or a machine.
_ALLOWED = {"LICENSE"}


def _tracked_text_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    for name in out.stdout.split("\0"):
        if not name or name in _ALLOWED:
            continue
        path = ROOT / name
        try:
            yield name, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue          # a binary asset has nothing to say here


def test_no_paths_from_anyones_machine_are_committed():
    """A checkout has to work from wherever it is cloned, and a repository is
    not a good place to publish where you keep your files."""
    hits = []
    for name, text in _tracked_text_files():
        for i, line in enumerate(text.splitlines(), 1):
            found = _HOME_PATH.search(line)
            if found:
                hits.append(f"{name}:{i}: {found.group(0)}")
    assert not hits, (
        "absolute paths into a user account:\n  " + "\n  ".join(hits)
        + "\nResolve it from __file__, or take it from the environment.")


@pytest.mark.parametrize("line,caught", [
    (r'INDOOR = "C:/Users/jsmith/Documents/CubeTest/indoor.jpg"', True),
    (r'path = "/home/jsmith/work/out.mp4"', True),
    (r'p = "/Users/jsmith/Pictures/pano.jpg"', True),
    # The fixtures and placeholders already in the suite must stay legal, or
    # the check gets switched off the first time it cries wolf.
    (r'options.build_argv({"input": r"C:\My Videos\a.mp4"})', False),
    (r'out = options.resolve_output("C:/x/pano_360_TB.jpg")', False),
    (r'"C:/mine/my_edit.mkv"', False),
    (r"$ArpKey = 'HKCU:\Software\Microsoft\Windows'", False),
    # Documentation is allowed to illustrate a path, as long as the account
    # in it is obviously nobody's.
    (r'e.g. C:\Users\you\Downloads\stereo360.bat', False),
    (r'/home/username/videos', False),
])
def test_the_check_catches_what_it_should_and_not_what_it_should_not(line,
                                                                     caught):
    """The pattern is the whole test, so it gets its own examples. A rule this
    broad is only useful while it stays quiet on the legitimate cases."""
    assert bool(_HOME_PATH.search(line)) is caught, line


def test_the_scoring_harness_finds_its_photo_from_a_checkout():
    """The specific case that prompted all this. Resolved from __file__, so
    it does not care about the working directory or about whose clone it is."""
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import score
    finally:
        sys.path.pop(0)

    assert Path(score.INDOOR).resolve() == (ROOT / "indoor.jpg").resolve()
