"""Converting many files in one run.

The point of doing it in-process rather than in a shell loop is the model:
measured, 5.11 s of every run is interpreter start, imports, the backend probe
and loading the weights, against 0.91 s a frame. Twenty files in a loop spend
over a minute and a half repeating that.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from stereo360 import cli, ffmpeg_io, vr_naming

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------- the input list


def test_a_list_is_one_path_per_line(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.mp4").write_bytes(b"")
    lst = tmp_path / "list.txt"
    lst.write_text("a.mp4\nb.mp4\n")
    assert cli.batch_inputs(str(lst), ffmpeg_io) == [
        str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")]


def test_blank_lines_and_comments_are_skipped(tmp_path):
    lst = tmp_path / "list.txt"
    lst.write_text("\n# a note\n\na.mp4\n   \n# another\nb.mp4\n")
    got = [os.path.basename(p) for p in cli.batch_inputs(str(lst), ffmpeg_io)]
    assert got == ["a.mp4", "b.mp4"]


def test_a_relative_path_is_read_relative_to_the_list(tmp_path):
    """A list beside its footage should work from any working directory --
    which is the situation anyone writing one is in."""
    shots = tmp_path / "shots"
    shots.mkdir()
    lst = shots / "list.txt"
    lst.write_text("a.mp4\n")
    assert cli.batch_inputs(str(lst), ffmpeg_io) == [str(shots / "a.mp4")]


def test_an_absolute_path_in_a_list_is_left_alone(tmp_path):
    lst = tmp_path / "list.txt"
    lst.write_text(f"{tmp_path / 'elsewhere' / 'a.mp4'}\n")
    assert cli.batch_inputs(str(lst), ffmpeg_io) == [
        str(tmp_path / "elsewhere" / "a.mp4")]


def test_a_directory_takes_the_media_in_it_and_nothing_else(tmp_path):
    for name in ("b.mp4", "a.mov", "still.jpg", "notes.txt", "list.txt"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.mp4").write_bytes(b"")

    got = [os.path.basename(p) for p in cli.batch_inputs(str(tmp_path),
                                                         ffmpeg_io)]
    assert got == ["a.mov", "b.mp4", "still.jpg"], "sorted, media only"
    assert "deep.mp4" not in got, (
        "not recursive: recursing would make the output layout a question "
        "with no answer that is right for everyone")


# ----------------------------------------------------------------- the naming


def test_the_default_suffix_is_the_one_the_tool_already_suggests(tmp_path):
    """A batch should name its output the way a single conversion does."""
    out = cli.batch_output("/in/garden.mp4", "/out", "360", None,
                           vr_naming, ffmpeg_io)
    assert out == os.path.join("/out", "garden_360_TB.mp4")
    assert vr_naming.SUFFIXES["360"] == "_360_TB"


def test_vr180_gets_its_own_token():
    out = cli.batch_output("/in/garden.mp4", "/out", "vr180", None,
                           vr_naming, ffmpeg_io)
    assert out.endswith("garden_180x180_3dh.mp4")


def test_a_name_that_already_says_it_is_not_argued_with():
    """`suggest` leaves a deliberate name alone, and a batch inherits that."""
    out = cli.batch_output("/in/garden_360_TB.mp4", "/out", "360", None,
                           vr_naming, ffmpeg_io)
    assert os.path.basename(out) == "garden_360_TB.mp4"


def test_an_explicit_suffix_overrides_it():
    out = cli.batch_output("/in/garden.mp4", "/out", "360", "_MYTAG",
                           vr_naming, ffmpeg_io)
    assert os.path.basename(out) == "garden_MYTAG.mp4"


def test_video_always_comes_out_as_mp4():
    """The encoder writes one container; an input .mkv would otherwise name an
    output .mkv that it did not write."""
    out = cli.batch_output("/in/clip.mkv", "/out", "360", None,
                           vr_naming, ffmpeg_io)
    assert out.endswith(".mp4")


def test_a_still_keeps_its_extension_unless_it_cannot_be_written():
    keep = cli.batch_output("/in/a.png", "/out", "360", None,
                            vr_naming, ffmpeg_io)
    assert keep.endswith(".png")
    # Readable but not writable -- OpenCV encodes none of the ISOBMFF stills.
    heic = cli.batch_output("/in/a.heic", "/out", "360", None,
                            vr_naming, ffmpeg_io)
    assert heic.endswith(".jpg")


# ------------------------------------------------------------ what it refuses


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "stereo360", *args],
                          capture_output=True, text=True, timeout=300,
                          cwd=str(ROOT))


@pytest.mark.parametrize("extra,expected", [
    (["in.mp4"], "brings its own inputs"),
    (["-o", "out.mp4"], "use --output-dir"),
    (["--progress-json"], "single render"),
])
def test_batch_refuses_what_it_cannot_mean(extra, expected):
    """Each of these is a flag whose single-file meaning does not survive
    being pointed at many, so it is refused rather than reinterpreted."""
    proc = _cli("--batch", "list.txt", *extra)
    assert proc.returncode != 0
    assert expected in proc.stderr, proc.stderr


def test_an_empty_list_says_so_rather_than_succeeding(tmp_path):
    lst = tmp_path / "list.txt"
    lst.write_text("# nothing here\n")
    proc = _cli("--batch", str(lst))
    assert proc.returncode == 2
    assert "No files to convert" in proc.stderr, proc.stderr


def test_a_missing_list_is_an_error_not_a_traceback(tmp_path):
    proc = _cli("--batch", str(tmp_path / "nope.txt"))
    assert proc.returncode == 2
    assert "Cannot read the batch list" in proc.stderr, proc.stderr
    assert "Traceback" not in proc.stderr
