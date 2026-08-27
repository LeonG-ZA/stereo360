"""Run stereo360 as a child process and turn its NDJSON into Qt signals.

QProcess rather than subprocess: its readyReadStandardOutput signal delivers
output on the Qt event loop, so the whole progress path needs no threads and
no polling, and the UI cannot deadlock waiting on a pipe.

The core stays a separate process even though this UI is also Python. torch
and ONNX Runtime can hard-crash or exhaust memory, and a render runs for
minutes to hours -- neither belongs in the process drawing the window.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


def core_root() -> str:
    """Directory to run the child in, so relative paths like models/ resolve."""
    import stereo360

    return os.path.dirname(os.path.dirname(os.path.abspath(stereo360.__file__)))


class Runner(QObject):
    """One conversion or preview at a time."""

    started = Signal()
    #: frame, total, elapsed seconds, frames/sec, eta seconds (-1 = unknown)
    progressed = Signal(int, int, float, float, float)
    #: level ('info' | 'warning' | 'error'), text
    logged = Signal(str, str)
    #: What the run is doing before any frame exists, as a sentence to show.
    #:
    #: Everything up to the first frame used to be one silent gap labelled
    #: "Starting...", and the gap grew teeth when the default stills model
    #: became a 1.9 GB download: a first run looks identical to a hang for as
    #: long as that takes. These events already say exactly what is happening
    #: -- they just only reached the log, which is collapsed by default.
    staged = Signal(str)

    #: A frame of the render has landed at this path. Carries the frame
    #: number too, so a consumer can make the URL unique -- QML caches an
    #: Image by source, and a file that changes under an unchanged URL does
    #: not repaint.
    previewed = Signal(str, int)
    #: ok, cancelled, output path
    finished = Signal(bool, bool, str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.SeparateChannels)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)
        self._buf = b""
        self._output = ""
        self._cancelling = False
        self._failed_to_start = False
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill)

    # ---------------------------------------------------------------- state

    def is_running(self) -> bool:
        return self._proc.state() != QProcess.NotRunning

    # ----------------------------------------------------------- lifecycle

    def start(self, argv: List[str]) -> None:
        if self.is_running():
            raise RuntimeError("A render is already running.")
        self._buf = b""
        self._output = ""
        self._cancelling = False
        self._failed_to_start = False
        self._proc.setWorkingDirectory(core_root())
        self._proc.setProgram(sys.executable)
        self._proc.setArguments(argv)
        self._proc.start()
        self.started.emit()

    def cancel(self) -> None:
        """Ask for a clean stop, and insist if it is ignored.

        A line on stdin rather than a kill, so ffmpeg finalizes the file and
        the frames already encoded stay playable. The timer is the backstop
        for a child wedged somewhere that never checks.
        """
        if not self.is_running() or self._cancelling:
            return
        self._cancelling = True
        self.logged.emit("info", "Stopping after the current frame…")
        self._proc.write(b"cancel\n")
        self._kill_timer.start(30_000)

    def _force_kill(self) -> None:
        if self.is_running():
            self.logged.emit("warning",
                             "Did not stop on request; terminating.")
            self._proc.kill()

    # ------------------------------------------------------------- reading

    def _read_stdout(self) -> None:
        self._buf += bytes(self._proc.readAllStandardOutput())
        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                # Anything not JSON on this stream is unexpected, so surface
                # it rather than dropping it -- it is usually the real cause.
                self.logged.emit("info", line.decode("utf-8", "replace"))
                continue
            self._dispatch(event)

    def _read_stderr(self) -> None:
        text = bytes(self._proc.readAllStandardError()).decode("utf-8",
                                                              "replace")
        for line in text.splitlines():
            if line.strip():
                self.logged.emit("warning", line.rstrip())

    def _dispatch(self, event: Dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "progress":
            eta = event.get("eta")
            self.progressed.emit(
                int(event.get("frame") or 0),
                int(event.get("total") or 0),
                float(event.get("elapsed") or 0.0),
                float(event.get("fps") or 0.0),
                float(eta) if eta is not None else -1.0)
        elif kind in ("info", "warning", "error"):
            message = str(event.get("message", ""))
            self.logged.emit(kind, message)
            # `backend` is only ever attached by stereo360.backends, so it
            # marks the model-preparation phase precisely -- choosing a
            # runtime, downloading weights, loading them. Keyed off the field
            # rather than off the wording, which is the whole reason the
            # reporter carries structured fields next to the prose.
            if event.get("backend"):
                self.staged.emit(message)
        elif kind == "preview":
            path = str(event.get("path") or "")
            if path:
                self.previewed.emit(path, int(event.get("frame") or 0))
        elif kind == "start":
            self._output = str(event.get("output") or "")
            total = event.get("total")
            if total:
                self.logged.emit("info", f"{total} frames to render.")
        elif kind == "done":
            self._output = str(event.get("output") or self._output)

    # ------------------------------------------------------------ teardown

    def _on_error(self, err: QProcess.ProcessError) -> None:
        if err == QProcess.FailedToStart:
            self._failed_to_start = True
            self.logged.emit(
                "error",
                f"Could not start {sys.executable}. Is stereo360 installed in "
                f"this Python environment?")

    def _on_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        self._kill_timer.stop()
        self._read_stdout()
        # 130 is the cancel exit code; anything else non-zero is a real
        # failure and the error event on stdout already said why.
        cancelled = code == 130 or self._cancelling
        ok = code == 0 and status == QProcess.NormalExit
        if not ok and not cancelled and not self._failed_to_start:
            self.logged.emit("error", f"stereo360 exited with code {code}.")
        self.finished.emit(ok, cancelled, self._output)
