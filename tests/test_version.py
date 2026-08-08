"""The version the app reports, and the one release that lied about it.

v1.0.0 shipped with `__version__` reading 0.1.0. That broke nothing, because
nothing read it -- no `--version`, nothing in the interface, and the installer
recorded the GitHub tag instead. It only mattered once something needed to ask
what was installed and whether it was current.

The rule from v1.0.1: `__version__` and the git tag match, and `released_as`
carries the one historical exception.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import stereo360

ROOT = Path(__file__).resolve().parent.parent


def test_the_version_is_a_plain_release_number():
    """No suffixes. It is compared against a tag and shown to a user, and
    both of those want the same short string."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", stereo360.__version__), \
        stereo360.__version__


def test_the_rule_starts_at_the_version_it_says_it_does():
    """1.0.1 or later. A 0.x here would mean the drift came back, and 1.0.0
    would collide with the release that already used that name."""
    parts = tuple(int(n) for n in stereo360.__version__.split("."))
    assert parts >= (1, 0, 1), \
        f"{stereo360.__version__} predates the version rule"


def test_the_first_release_is_recognised_by_the_name_it_shipped_under():
    """Installs of v1.0.0 have 0.1.0 in the package. Anything reading a
    version off an existing install has to call that v1.0.0, because that is
    what the user downloaded and what Settings > Apps shows them."""
    assert stereo360.released_as("0.1.0") == "1.0.0"


def test_every_other_version_is_left_alone():
    """The mapping is one historical fact, not a translation layer. If it
    starts growing, the rule it exists to bridge has stopped working."""
    assert stereo360.released_as("1.0.1") == "1.0.1"
    assert stereo360.released_as("2.3.4") == "2.3.4"
    assert list(stereo360._RELEASED_AS) == ["0.1.0"], \
        "a second entry means __version__ drifted from the tag again"


def test_released_as_defaults_to_this_build():
    assert stereo360.released_as() == stereo360.released_as(
        stereo360.__version__)


def test_the_cli_reports_it():
    """The absence of this flag is why the constant went three releases
    without anyone noticing it was wrong."""
    proc = subprocess.run([sys.executable, "-m", "stereo360", "--version"],
                          capture_output=True, text=True, timeout=300,
                          cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"stereo360 {stereo360.__version__}"


def test_version_needs_no_input_file():
    """--version and --help are the two things that must work with nothing
    else on the command line."""
    proc = subprocess.run([sys.executable, "-m", "stereo360", "--version"],
                          capture_output=True, text=True, timeout=300,
                          cwd=str(ROOT))
    assert "required" not in proc.stderr.lower()


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="not a git checkout")
def test_no_tag_already_claims_this_version():
    """The rule is one tag per version. Bumping to something already tagged
    would give two different builds the same name, which is exactly the
    confusion 0.1.0-as-v1.0.0 caused."""
    proc = subprocess.run(["git", "tag", "--list"], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    tags = {t.strip().lstrip("v") for t in proc.stdout.splitlines() if t.strip()}
    assert stereo360.__version__ not in tags, (
        f"v{stereo360.__version__} is already tagged; bump __version__ before "
        f"releasing again")
