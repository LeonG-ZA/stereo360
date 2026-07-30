"""Command-line surface: the parts that break without any test noticing.

`--help` is the obvious one. argparse runs every help string through
`%`-expansion, so a literal `%` in one of them raises `TypeError: must be real
number, not dict` — at `--help` time only, never during a render. A help string
gained "40% less" and "+9% render time" and `--help` stayed broken until
someone tried to read it.
"""

import subprocess
import sys

import pytest

from stereo360 import cli


def run(*args):
    return subprocess.run([sys.executable, "-m", "stereo360", *args],
                          capture_output=True, text=True)


def test_help_renders():
    """Guards against a literal % in any help string."""
    r = run("--help")
    assert r.returncode == 0, r.stderr[-500:]
    assert "usage: stereo360" in r.stdout
    assert "--depth-model" in r.stdout


def test_every_help_string_survives_percent_expansion():
    """The same failure, caught without spawning a process -- and pointing at
    the offending option rather than a stack trace inside argparse."""
    parser = cli.build_parser()
    formatter = parser._get_formatter()
    for action in parser._actions:
        if not action.help:
            continue
        try:
            formatter._expand_help(action)
        except (TypeError, ValueError, KeyError) as exc:
            name = "/".join(action.option_strings) or action.dest
            pytest.fail(f"{name} help fails % expansion ({exc}); a literal "
                        f"percent sign has to be written %%")


@pytest.mark.parametrize("args", [
    ("--probe-backends",),
    ("--probe-encoders", "640x480"),
])
def test_machine_probes_need_no_input_file(args):
    """They describe the machine, not a file. Requiring a dummy positional
    just to run them made the documented command a lie."""
    r = run(*args)
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.strip().startswith("{")


def test_an_input_is_still_required_for_a_render():
    r = run("-o", "out.mp4")
    assert r.returncode != 0
    assert "input" in r.stderr.lower()


def test_an_output_is_still_required_for_a_render():
    r = run("in.mp4")
    assert r.returncode != 0
    assert "output" in r.stderr.lower()
