"""The panel must not be able to build a command line the CLI will refuse.

`cli._VIDEO_ONLY_FLAGS` is a refusal list, not an ignore list: handed
`--max-frames` for a still the converter exits 2 rather than shrugging. That
is the right behaviour for a typed command, and it makes the UI's job to
never emit one -- a switch left on from the last video would otherwise fail
every photo conversion after it.

`build_argv` had a guard saying exactly that, and it named three flags. The
refusal list later grew a fourth, `--no-supersample`, and the guard did not
grow with it, so upscaling a photo with supersample switched off died with:

    stereo360: error: the input is an image, so these do not apply:
    --no-supersample (a still is one frame, and rendering it small only
    makes it worse)

So this walks the list rather than naming its members. A fifth entry is then
covered on the day it is added.
"""

import pytest

from stereo360 import cli
from stereo360_ui import options

#: Every UI setting that maps to something the CLI refuses for a still, each
#: at a value that is *not* the default -- a guard that only ever sees
#: defaults proves nothing.
LOUDEST = {
    "input": "holiday.jpg",
    "output": "holiday_360_TB.jpg",
    "sourceWidth": 3840,
    "outputWidth": 7680,
    "upscale": True,
    "upscaleModel": "artcnn32",
    "upscaleScale": 2.0,
    "supersample": False,
    "maxFrames": 120,
    "startFrame": 30,
    "spatialAudio": True,
}


def test_no_refused_flag_reaches_the_command_line():
    argv = options.build_argv(dict(LOUDEST))
    bad = [flag for _, flag, _ in cli._VIDEO_ONLY_FLAGS if flag in argv]
    assert not bad, f"the panel emitted {bad} for a photo"


def test_the_cli_itself_accepts_what_the_panel_builds():
    """The stronger form: parse the argv and run the real refusal. Nothing
    here can drift, because it is the converter's own check."""
    argv = options.build_argv(dict(LOUDEST))
    args = cli.build_parser().parse_args(argv[2:])   # past "-m", "stereo360"
    cli._refuse_video_only_flags(args)               # raises SystemExit if not


def test_the_upscale_settings_still_survive():
    """The guard must suppress the refused flags and nothing else -- a photo
    that cannot upscale is not a fix."""
    argv = options.build_argv(dict(LOUDEST))
    assert argv[argv.index("--upscale") + 1] == "artcnn32"
    assert argv[argv.index("--output-width") + 1] == "7680"


class TestTheVideoCaseIsUnmoved:
    VIDEO = dict(LOUDEST, input="holiday.mp4", output="holiday_360_TB.mp4")

    def test_a_video_still_gets_no_supersample(self):
        assert "--no-supersample" in options.build_argv(dict(self.VIDEO))

    def test_and_still_gets_its_frame_range(self):
        argv = options.build_argv(dict(self.VIDEO))
        assert "--max-frames" in argv and "--start-frame" in argv

    def test_supersample_left_on_says_nothing(self):
        """Emitted only when off, like every other switch whose default is
        the fuller-quality one."""
        argv = options.build_argv(dict(self.VIDEO, supersample=True))
        assert "--no-supersample" not in argv


def test_the_panel_does_not_offer_the_switch_for_a_photo():
    """Suppressing the flag is the guarantee; hiding the row is what stops
    someone setting it and wondering why it had no effect."""
    from pathlib import Path

    qml = (Path(__file__).resolve().parent.parent / "stereo360_ui" / "qml"
           / "Main.qml").read_text(encoding="utf-8")
    start = qml.index('objectName: "supersampleRow"')
    block = qml[start:qml.index("hint:", start)]
    assert "photoMode" in block, "the supersample row must be hidden for a still"


@pytest.mark.parametrize("suffix", [".jpg", ".jpeg", ".png", ".webp", ".tif"])
def test_every_still_the_panel_opens_is_covered(suffix):
    argv = options.build_argv(dict(LOUDEST, input="holiday" + suffix,
                                   output="out" + suffix))
    assert "--no-supersample" not in argv
