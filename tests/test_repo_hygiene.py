"""Things that should never be committed, checked against what is committed.

Two rules so far, and both exist because of the same commit: a hardcoded
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
    (?!                                   # ...unless what follows is
        (?: x | y | mine | test | tmp |   # a placeholder nobody could own,
            user | username | you |
            someone | example )
        \b
        | [{$%]                           # or a variable, so not a real path
    )
    [^\s"'<>|*?]+
    """)

#: Files that legitimately name a person or a machine.
_ALLOWED = {"LICENSE"}

#: A stand-in account name for the examples below. Assembled at runtime rather
#: than written out, because this file is scanned like every other one and a
#: literal example would be a finding against itself. The `[{$%]` clause above
#: is what keeps the assembled form legal, and it earns its place regardless:
#: a path with a variable in it is nobody's actual path.
_WHOEVER = "jsmith"


def _tracked_names():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return [name for name in out.stdout.split("\0") if name]


def _tracked_text_files():
    for name in _tracked_names():
        if name in _ALLOWED:
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
    (f'INDOOR = "C:/Users/{_WHOEVER}/Documents/CubeTest/indoor.jpg"', True),
    (f'path = "/home/{_WHOEVER}/work/out.mp4"', True),
    (f'p = "/Users/{_WHOEVER}/Pictures/pano.jpg"', True),
    # The templated form the examples above are written in, which must stay
    # legal or this file reports itself.
    (r'src = f"C:/Users/{_WHOEVER}/x.jpg"', False),
    (r'dest = "$HOME/videos"', False),
    (r'p = "C:/Users/%USERNAME%/Downloads"', False),
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


def _score_module():
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        import score
        return score
    finally:
        sys.path.pop(0)


def test_the_scoring_harness_looks_beside_the_checkout_not_inside_it():
    """The specific case that prompted all this. Resolved from __file__, so it
    does not care about the working directory or about whose clone it is --
    and the photo itself is not committed, being a 360 shot of a house."""
    score = _score_module()

    assert Path(score.INDOOR).resolve() == (ROOT / "indoor.jpg").resolve()
    assert "indoor.jpg" not in _tracked_names(), \
        "the reference photo must not be committed"


def test_a_missing_reference_photo_explains_itself(monkeypatch):
    """It is absent from a fresh clone by design, so that is the *normal* first
    experience of this harness rather than an edge case. cv2.imread returns
    None for a missing file, which surfaces as a TypeError several frames
    down and says nothing about what to do."""
    import numpy as np

    score = _score_module()
    monkeypatch.setattr(score, "INDOOR", str(ROOT / "no_such_photo.jpg"))

    with pytest.raises(FileNotFoundError) as excinfo:
        score.score(np.zeros((64, 128), np.float32))
    assert "STEREO360_INDOOR" in str(excinfo.value)


def test_no_large_media_is_committed():
    """A repository is a bad place for a 40 MB render: it is in every clone
    forever, and removing it later means rewriting history."""
    big = []
    for name in _tracked_names():
        path = ROOT / name
        if path.is_file() and path.stat().st_size > 2 * 1024 * 1024:
            big.append(f"{name} ({path.stat().st_size // 1024 // 1024} MB)")
    assert not big, "tracked files over 2 MB:\n  " + "\n  ".join(big)
