"""Reporter events, cancellation, and the machine-readable CLI mode.

These cover the seam a desktop UI sits on: it has to be able to learn what is
happening without parsing prose, and to stop a long render without killing the
process and stranding ffmpeg.
"""

import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from stereo360 import backends, ffmpeg_io, pipeline, spherical
from stereo360.events import Cancelled, ConsoleReporter, JsonReporter, Reporter
from test_end_to_end import make_test_video


def _events(stream: io.StringIO):
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


# ---------------------------------------------------------------- reporters


def test_base_reporter_is_silent_and_total():
    """The default reporter must accept every call and print nothing, so
    using the pipeline as a library stays quiet unless asked."""
    r = Reporter()
    r.info("x", a=1)
    r.warning("y")
    r.start(10, w=1)
    r.advance(3)
    r.finish(ok=True)
    r.error("boom", kind="X")


def test_json_reporter_emits_one_object_per_line():
    buf = io.StringIO()
    r = JsonReporter(buf, interval=0.0)
    r.info("hello", backend="onnx")
    r.start(2, width=64)
    r.advance()
    r.advance()
    r.finish(output="out.mp4", cancelled=False)

    ev = _events(buf)
    assert [e["type"] for e in ev] == [
        "info", "start", "progress", "progress", "done"]
    assert ev[0]["message"] == "hello" and ev[0]["backend"] == "onnx"
    assert ev[1]["total"] == 2 and ev[1]["width"] == 64
    assert ev[2]["frame"] == 1 and ev[3]["frame"] == 2
    assert ev[-1]["frames"] == 2 and ev[-1]["cancelled"] is False


def test_json_progress_is_throttled_but_keeps_the_first_and_last_frames():
    """A fast run must not flood the parent, yet a UI still needs to see that
    work started and that the bar reached 100%."""
    buf = io.StringIO()
    r = JsonReporter(buf, interval=3600.0)  # suppress everything time-based
    r.start(50)
    for _ in range(50):
        r.advance()

    progress = [e for e in _events(buf) if e["type"] == "progress"]
    assert [e["frame"] for e in progress] == [1, 50], \
        "throttling should collapse the middle but keep both ends"


def test_json_reporter_reports_eta_only_when_total_is_known():
    buf = io.StringIO()
    r = JsonReporter(buf, interval=0.0)
    r.start(None)
    r.advance()
    ev = [e for e in _events(buf) if e["type"] == "progress"][0]
    assert ev["total"] is None and ev["eta"] is None


def test_console_reporter_writes_plain_lines():
    buf = io.StringIO()
    ConsoleReporter(buf).info("Auto face size: 1920")
    assert buf.getvalue().strip() == "Auto face size: 1920"


# ------------------------------------------------------------------- build


def test_build_passthrough_skips_every_import():
    built = backends.build(passthrough=True)
    assert built.backend is None
    assert built.name == "passthrough" and built.chunk_size == 1


def test_build_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown depth backend"):
        backends.build(depth_backend="nope")


def test_build_only_chunks_for_the_temporal_model():
    """Chunking exists to give the temporal model cross-frame context; for
    every other backend it would just buffer frames for nothing."""
    class Recorder(Reporter):
        def __init__(self):
            self.msgs = []

        def info(self, message, **f):
            self.msgs.append(message)

        warning = info

    rec = Recorder()
    built = backends.build(depth_backend="auto", chunk_size=8, reporter=rec)
    assert built.chunk_size == 1
    assert built.name in ("depth-anything", "onnx")
    # Whatever it picked, it must have said so -- silent CPU fallback is the
    # failure mode this banner exists to prevent.
    assert rec.msgs and ("GPU" in rec.msgs[0] or "CPU" in rec.msgs[0])


# ------------------------------------------------------------ cancellation


def test_convert_cancel_leaves_a_playable_partial_file(tmp_path: Path):
    """Stopping half way is not an error: the frames already encoded are kept
    and the file is finalized, because throwing away an hour of work would be
    a strange response to pressing Stop."""
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=12, with_audio=False)

    result = pipeline.convert(src, dst, face_size=32, use_cubemap=False)
    assert result.frames_written == 12 and not result.cancelled

    # The predicate is consulted once per frame, so this stops after four.
    seen = {"n": 0}

    def cancel():
        seen["n"] += 1
        return seen["n"] > 4

    dst2 = str(tmp_path / "out2.mp4")
    result = pipeline.convert(src, dst2, face_size=32, use_cubemap=False,
                              cancel=cancel)

    assert result.cancelled is True
    assert result.frames_written == 4
    # Finalized, not truncated: probeable, and still carries the metadata.
    assert ffmpeg_io.probe(dst2).width == 128
    assert spherical.has_spherical_metadata(dst2)


def test_cancel_accepts_a_threading_event(tmp_path: Path):
    """`cancel=event.is_set` is the shape a GUI will actually use."""
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=6, with_audio=False)

    ev = threading.Event()
    ev.set()
    result = pipeline.convert(src, dst, face_size=32, use_cubemap=False,
                              cancel=ev.is_set)
    assert result.cancelled and result.frames_written == 0


def test_sink_check_raises_cancelled():
    sink = pipeline._Sink(None, Reporter(), lambda: True)
    with pytest.raises(Cancelled):
        sink.check()


# ------------------------------------------------------------- CLI --json


def test_progress_json_stream_is_parseable(tmp_path: Path):
    """Every line the CLI emits in this mode must be JSON -- a parent process
    has no way to recover from a stray print."""
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=6, with_audio=False)

    proc = subprocess.run(
        [sys.executable, "-m", "stereo360", src, "-o", dst, "--passthrough",
         "--face-size", "32", "--progress-json"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stderr

    ev = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "start"
    assert "progress" in kinds and "done" in kinds
    done = next(e for e in ev if e["type"] == "done")
    assert done["frames"] == 6 and done["cancelled"] is False


def test_progress_json_survives_a_closed_stdin(tmp_path: Path):
    """stdin at EOF must not read as a cancel: a parent that spawns us
    without a stdin pipe would otherwise stop the run instantly."""
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=5, with_audio=False)

    proc = subprocess.run(
        [sys.executable, "-m", "stereo360", src, "-o", dst, "--passthrough",
         "--face-size", "32", "--progress-json"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    ev = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    done = next(e for e in ev if e["type"] == "done")
    assert done["cancelled"] is False and done["frames"] == 5


def test_progress_json_reports_errors_as_events(tmp_path: Path):
    """A traceback on stderr is useless to a GUI; the failure has to arrive on
    the same channel as everything else."""
    dst = str(tmp_path / "out.mp4")
    proc = subprocess.run(
        [sys.executable, "-m", "stereo360", str(tmp_path / "missing.mp4"),
         "-o", dst, "--passthrough", "--progress-json"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 1
    ev = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    err = next(e for e in ev if e["type"] == "error")
    assert err["message"] and err["kind"]


def test_stdin_cancel_stops_the_run(tmp_path: Path):
    """The GUI-facing stop path, end to end: write 'cancel', get exit 130 and
    a finalized partial file."""
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    errlog = tmp_path / "err.txt"
    # Long enough that the cancel reliably lands mid-run, and passthrough so
    # no model is downloaded; the cubemap round-trip supplies the work.
    make_test_video(src, w=512, h=256, frames=900, fps=30, with_audio=False)

    with open(errlog, "w") as err:
        proc = subprocess.Popen(
            [sys.executable, "-m", "stereo360", src, "-o", dst,
             "--passthrough", "--face-size", "256", "--progress-json"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
            text=True,
        )
        # Drained on a thread throughout, so a regression that stops producing
        # output fails on the wait() timeout instead of hanging the suite.
        lines = []
        drain = threading.Thread(target=lambda: lines.extend(proc.stdout),
                                 daemon=True)
        drain.start()

        # Ask to stop only once real progress has been reported, so the cancel
        # lands mid-run rather than racing startup.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if any(json.loads(x)["type"] == "progress" for x in list(lines)):
                break
            time.sleep(0.02)
        else:
            proc.kill()
            pytest.fail(f"no progress event within 60s: {errlog.read_text()}")

        proc.stdin.write("cancel\n")
        proc.stdin.flush()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail(f"did not stop on cancel: {errlog.read_text()}")
        drain.join(10)

    assert proc.returncode == 130, errlog.read_text()
    ev = [json.loads(x) for x in lines if x.strip()]
    done = next(e for e in ev if e["type"] == "done")
    assert done["cancelled"] is True
    assert 0 < done["frames"] < 900
    # Finalized despite the stop: still a valid, probeable top-bottom file.
    assert ffmpeg_io.probe(dst).height == 512   # top-bottom of 256


def test_progress_json_works_with_an_open_stdin_pipe(tmp_path: Path):
    """The configuration a GUI actually uses: stdin held open, waiting to
    send a command.

    This is a regression guard, not a formality. A thread blocked reading
    stdin deadlocks `import numpy` on Windows, so starting the cancel watcher
    before those imports froze the process before it emitted a single line --
    silently, and only when spawned this way.
    """
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=6, with_audio=False)

    proc = subprocess.Popen(
        [sys.executable, "-m", "stereo360", src, "-o", dst, "--passthrough",
         "--face-size", "32", "--progress-json"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True,
    )
    lines = []
    drain = threading.Thread(target=lambda: lines.extend(proc.stdout),
                             daemon=True)
    drain.start()
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("froze with stdin held open (produced %d lines)"
                    % len(lines))
    drain.join(10)

    assert proc.returncode == 0
    ev = [json.loads(x) for x in lines if x.strip()]
    done = next(e for e in ev if e["type"] == "done")
    assert done["frames"] == 6 and done["cancelled"] is False


# ---------------------------------------------------- audio length matching


def _durations(path):
    return ffmpeg_io.stream_durations(path)


def test_a_truncated_render_does_not_keep_the_full_audio(tmp_path: Path):
    """Reported: pressing Stop left a file the length of the whole source,
    holding the last frame for the remainder.

    Audio is copied rather than re-encoded, so a run that stops early still
    wrote the source's entire track and the container took its duration.
    """
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=60, fps=10, with_audio=True)

    result = pipeline.convert(src, dst, face_size=32, use_cubemap=False,
                              max_frames=6)
    assert result.frames_written == 6

    video, audio = _durations(dst)
    assert video == pytest.approx(0.6, abs=0.15), video
    # Audio may overshoot slightly: with a stream copy it can only be cut at a
    # packet boundary. What matters is that it is not the source's 6 seconds.
    assert audio is not None and audio < video + 0.3, \
        f"audio {audio}s against video {video}s"


def test_a_complete_render_keeps_every_frame(tmp_path: Path):
    """The trap in the obvious fix. ffmpeg's -shortest ends at whichever
    stream finishes first, and source audio routinely stops a little before
    the picture -- it silently dropped 3 of 92 frames on real footage. Audio
    is only ever shortened, never video."""
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=40, fps=10, with_audio=True)
    expected = len(list(ffmpeg_io.decode_frames(src)))

    result = pipeline.convert(src, dst, face_size=32, use_cubemap=False)
    assert result.frames_written == expected

    got = list(ffmpeg_io.decode_frames(dst))
    assert len(got) == expected, "a complete render lost frames"


def test_trim_is_skipped_when_audio_already_fits(tmp_path: Path):
    """A normal render must pay nothing for this."""
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=20, fps=10, with_audio=True)
    pipeline.convert(src, dst, face_size=32, use_cubemap=False)

    before = Path(dst).stat().st_mtime_ns
    assert ffmpeg_io.trim_audio_to_video(dst, 10.0) is False
    assert Path(dst).stat().st_mtime_ns == before, "file was rewritten"


def test_silent_source_is_unaffected(tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=10, with_audio=False)
    pipeline.convert(src, dst, face_size=32, use_cubemap=False, max_frames=4)

    video, audio = _durations(dst)
    assert audio is None
    assert video is not None
