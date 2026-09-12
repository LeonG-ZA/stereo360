"""A photo is not subject to the headset's video decode limit.

`default_output_width` has always known this -- it returns 0, meaning the
source's own size, for a still -- because 35.6 MP is what a Quest 3's *video*
decoder will take and a 59 MP stereo JPEG displays fine. A photo never reaches
a codec.

The upscaling path did not know it. It picked its default through
`preferredWidth`, which reads the `fits` flag directly, so switching upscaling
on for a photo re-imposed the ceiling that had already been lifted and capped
a 4K still at 5760 rather than letting it reach 7680.

These run the QML function itself rather than reading the file and hoping.
It is a pure function of its arguments, so a JS engine can execute it, and
only `photoMode` has to be passed in rather than read off the root object.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication          # noqa: E402
from PySide6.QtQml import QJSEngine                  # noqa: E402

from stereo360_ui import options                     # noqa: E402

MAIN = (Path(__file__).resolve().parent.parent
        / "stereo360_ui" / "qml" / "Main.qml")

#: What a 4K source offers once upscaling is on: 7680 is past the decode
#: limit, the two below it are not.
FOUR_K = ("[{width:7680,fits:false},{width:5760,fits:true},"
          "{width:4096,fits:true}]")


@pytest.fixture(scope="module")
def js():
    """`preferredWidth` out of Main.qml, callable, with photoMode a parameter.

    QJSEngine needs an application object; without one the interpreter exits
    without reaching a single assertion.
    """
    QCoreApplication.instance() or QCoreApplication(sys.argv or ["test"])
    qml = MAIN.read_text(encoding="utf-8")
    opener = "    function preferredWidth(l) {"
    assert opener in qml, "preferredWidth moved or changed its signature"
    start = qml.index(opener)
    body = qml[start:qml.index("\n    }", start) + len("\n    }")].strip()
    body = body.replace("preferredWidth(l)", "preferredWidth(photoMode, l)", 1)
    engine = QJSEngine()
    assert not engine.evaluate(body).isError()

    def call(photo_mode, listing=FOUR_K):
        got = engine.evaluate(
            "preferredWidth(%s, %s)" % ("true" if photo_mode else "false",
                                        listing))
        assert not got.isError(), got.toString()
        return got.toInt()
    return call


class TestThePhotoCase:
    def test_a_photo_takes_the_largest_size_on_offer(self, js):
        """The bug. 5760 was being chosen for a still that has no decoder to
        satisfy, quietly throwing away the resolution the upscale bought."""
        assert js(True) == 7680

    def test_even_though_nothing_in_the_list_fits(self, js):
        """`fits` is false for 7680 and it is still the right answer, which
        is the whole point: the flag is about a codec a photo never meets."""
        assert js(True, "[{width:7680,fits:false}]") == 7680

    def test_an_empty_list_is_still_nothing(self, js):
        """An 8K source has nowhere above it to go, photo or not."""
        assert js(True, "[]") == 0


class TestTheVideoCaseIsUnmoved:
    def test_a_video_still_stops_at_the_decode_limit(self, js):
        assert js(False) == 5760

    def test_and_falls_back_to_the_smallest_when_none_fit(self, js):
        assert js(False, "[{width:7680,fits:false},"
                         "{width:5760,fits:false}]") == 5760

    def test_an_empty_list_is_still_nothing(self, js):
        assert js(False, "[]") == 0


class TestTheTwoHalvesAgree:
    """The non-upscaling path answers the same question in Python, and the
    two must not disagree about what a photo is entitled to."""

    def test_python_gives_a_photo_its_own_size(self):
        assert options.default_output_width(7680, 3840, "360",
                                            is_photo=True) == 0

    def test_and_caps_a_video(self):
        assert (options.default_output_width(7680, 3840, "360",
                                             is_photo=False)
                == options.HEADSET_SAFE_WIDTH)

    def test_the_dropdown_does_not_warn_a_photo_about_codecs(self):
        """The other face of the same bug: picking 7680 for a still while the
        row under it reads "upload only" would be telling the reader their
        photo will not play."""
        qml = MAIN.read_text(encoding="utf-8")
        start = qml.index("readonly property var resolutionModel")
        block = qml[start:qml.index("\n    }", start)]
        assert "photoMode" in block, (
            "the decode warning must be suppressed for a still")
