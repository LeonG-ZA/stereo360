"""The object QML talks to.

QML holds the settings and hands them over as one plain object, rather than
this class mirroring twenty-odd typed properties. Adding an option then means
a control in the QML and a line in `options.build_argv`, with nothing in
between to keep in sync.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, Optional

import sys

from PySide6.QtCore import (Property, QObject, QProcess, QUrl, Signal, Slot)

from . import options
from .runner import Runner, core_root


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


class Controller(QObject):
    busyChanged = Signal()
    progressChanged = Signal()
    statusChanged = Signal()
    previewChanged = Signal()
    commandChanged = Signal()
    sourceInfoChanged = Signal()
    backendsChanged = Signal()
    encodersChanged = Signal()
    #: level, text -- appended to the log view
    logged = Signal(str, str)
    #: fired when a run ends so QML can flash a result banner
    completed = Signal(bool, bool, str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._runner = Runner(self)
        self._runner.progressed.connect(self._on_progress)
        self._runner.logged.connect(self._on_log)
        self._runner.finished.connect(self._on_finished)

        self._busy = False
        self._preview_mode = False
        self._progress = 0.0
        self._status = "Ready"
        self._detail = ""
        self._preview = ""
        self._command = ""
        self._preview_path = os.path.join(
            tempfile.gettempdir(), "stereo360_preview.png")

        self._source_info: Dict[str, Any] = {}
        self._backends: list = []
        self._backend_probe = QProcess(self)
        self._backend_probe.finished.connect(self._on_backends_probed)

        self._encoders: list = []
        self._encoder_size = None
        self._encoder_probe = QProcess(self)
        self._encoder_probe.finished.connect(self._on_encoders_probed)
        # Its own process, so a probe can never disturb a running render.
        self._probe = QProcess(self)
        self._probe.finished.connect(self._on_probe_finished)

    # ---------------------------------------------------------- properties

    @Property(bool, notify=busyChanged)
    def busy(self) -> bool:
        return self._busy

    @Property(bool, notify=busyChanged)
    def previewMode(self) -> bool:
        return self._preview_mode

    @Property(float, notify=progressChanged)
    def progress(self) -> float:
        return self._progress

    @Property(str, notify=statusChanged)
    def status(self) -> str:
        return self._status

    @Property(str, notify=statusChanged)
    def detail(self) -> str:
        return self._detail

    @Property(str, notify=previewChanged)
    def previewSource(self) -> str:
        return self._preview

    @Property(str, notify=commandChanged)
    def command(self) -> str:
        return self._command

    @Property("QVariantMap", notify=sourceInfoChanged)
    def sourceInfo(self) -> Dict[str, Any]:
        """What --probe-json reported about the chosen input, or {}."""
        return self._source_info

    @Property("QVariantList", notify=encodersChanged)
    def encoders(self) -> list:
        """[{name, available, hardware, detail}] for the current output size."""
        return self._encoders

    @Property("QVariantList", notify=backendsChanged)
    def backends(self) -> list:
        """[{name, available, detail}] -- which depth backends can run here."""
        return self._backends

    # ------------------------------------------------------------- probing

    @Slot()
    def probeBackends(self) -> None:
        """Ask which backends are usable, so the UI need not offer duds.

        Out of process for the same reason as the input probe: the answer
        depends on what is importable, and finding out must not drag torch
        into the process drawing the window.
        """
        if self._backend_probe.state() != QProcess.NotRunning:
            return
        self._backend_probe.setWorkingDirectory(core_root())
        self._backend_probe.setProgram(sys.executable)
        self._backend_probe.setArguments(
            ["-m", "stereo360", "--probe-backends", "-"])
        self._backend_probe.start()

    @Slot(int, int)
    def probeEncoders(self, width: int, height: int) -> None:
        """Which encoders manage this output size on this machine.

        Resolution-dependent and worth re-asking per input: hevc_amf takes
        3840x3840 and refuses 7680x7680, so a list built once for a 4K project
        would offer an encoder that cannot touch an 8K one.
        """
        size = (int(width), int(height))
        if size == self._encoder_size or min(size) <= 0:
            return          # already known; probing costs a few seconds
        if self._encoder_probe.state() != QProcess.NotRunning:
            return
        self._encoder_size = size
        self._encoder_probe.setWorkingDirectory(core_root())
        self._encoder_probe.setProgram(sys.executable)
        self._encoder_probe.setArguments(
            ["-m", "stereo360", "-", "--probe-encoders",
             f"{size[0]}x{size[1]}"])
        self._encoder_probe.start()

    def _on_encoders_probed(self, code: int, _status) -> None:
        raw = bytes(self._encoder_probe.readAllStandardOutput()).decode(
            "utf-8", "replace").strip()
        try:
            self._encoders = json.loads(raw)["encoders"]
        except (ValueError, KeyError, TypeError):
            self._encoder_size = None       # let a later attempt retry
            return
        self.encodersChanged.emit()
        usable = [e["name"] for e in self._encoders
                  if e["available"] and e["hardware"]]
        if usable:
            self.logged.emit("info", "Hardware encoders available at "
                                     f"{self._encoder_size[0]}x"
                                     f"{self._encoder_size[1]}: "
                                     + ", ".join(usable))

    def _on_backends_probed(self, code: int, _status) -> None:
        raw = bytes(self._backend_probe.readAllStandardOutput()).decode(
            "utf-8", "replace").strip()
        try:
            self._backends = json.loads(raw)["backends"]
        except (ValueError, KeyError, TypeError):
            return
        self.backendsChanged.emit()
        for entry in self._backends:
            if not entry.get("available"):
                self.logged.emit("info", f"{entry['name']} unavailable: "
                                         f"{entry['detail']}")

    @Slot(str)
    def probeInput(self, path: str) -> None:
        """Ask stereo360 to describe the input, asynchronously.

        Out of process and off the UI thread: ffprobe on a file over a network
        share can take seconds, and blocking the window while someone picks a
        file is exactly the kind of stall that makes an app feel broken. Also
        keeps the interface from importing the core -- and numpy with it.
        """
        self._source_info = {}
        self.sourceInfoChanged.emit()
        if not path or self._probe.state() != QProcess.NotRunning:
            return
        self._probe.setWorkingDirectory(core_root())
        self._probe.setProgram(sys.executable)
        self._probe.setArguments(["-m", "stereo360", path, "--probe-json"])
        self._probe.start()

    def _on_probe_finished(self, code: int, _status) -> None:
        if code != 0:
            return
        raw = bytes(self._probe.readAllStandardOutput()).decode(
            "utf-8", "replace").strip()
        try:
            self._source_info = json.loads(raw)
        except ValueError:
            return
        self.sourceInfoChanged.emit()

    # ------------------------------------------------------------- helpers

    def _set_busy(self, value: bool, preview: bool = False) -> None:
        self._busy = value
        self._preview_mode = preview
        self.busyChanged.emit()

    def _set_status(self, status: str, detail: str = "") -> None:
        self._status = status
        self._detail = detail
        self.statusChanged.emit()

    # --------------------------------------------------------- QML entry

    @Slot(str, result=str)
    def toLocalPath(self, url: str) -> str:
        """file:// URL from a FileDialog -> a plain path for the command."""
        return QUrl(url).toLocalFile() if url.startswith("file:") else url

    @Slot(str, result=str)
    def suggestOutput(self, input_url: str) -> str:
        """A sensible default output beside the input, so the second file
        picker is usually unnecessary."""
        path = self.toLocalPath(input_url)
        if not path:
            return ""
        stem, _ = os.path.splitext(path)
        return f"{stem}_stereo.mp4"

    @Slot(str, result=str)
    def presetNote(self, quality: str) -> str:
        return options.preset_note(quality)

    @Slot("QVariantMap", result=str)
    def previewCommand(self, opts: Dict[str, Any]) -> str:
        """The command as text, for the 'equivalent command' line."""
        try:
            return options.display_command(options.build_argv(opts))
        except ValueError:
            return ""

    @Slot("QVariantMap")
    def convert(self, opts: Dict[str, Any]) -> None:
        self._start(opts, preview_frame=None)

    @Slot("QVariantMap", int)
    def preview(self, opts: Dict[str, Any], frame: int) -> None:
        self._start(opts, preview_frame=max(0, int(frame)))

    @Slot()
    def cancel(self) -> None:
        self._runner.cancel()

    def _start(self, opts: Dict[str, Any],
               preview_frame: Optional[int]) -> None:
        if self._busy:
            return
        is_preview = preview_frame is not None
        try:
            argv = options.build_argv(
                opts, preview_frame=preview_frame,
                preview_output=self._preview_path if is_preview else None)
        except ValueError as exc:
            self._on_log("error", str(exc))
            self.completed.emit(False, False, "")
            return

        self._command = options.display_command(argv)
        self.commandChanged.emit()
        self._progress = 0.0
        self.progressChanged.emit()
        self._set_busy(True, is_preview)
        self._set_status("Rendering preview…" if is_preview else "Starting…")
        self._on_log("info", "$ " + self._command)
        self._runner.start(argv)

    # ------------------------------------------------------------- signals

    def _on_progress(self, frame: int, total: int, elapsed: float,
                     fps: float, eta: float) -> None:
        if total > 0:
            self._progress = min(1.0, frame / total)
            head = f"Frame {frame} of {total}"
        else:
            self._progress = 0.0
            head = f"Frame {frame}"
        self.progressChanged.emit()

        bits = []
        if eta >= 0:
            bits.append(f"{_fmt_duration(eta)} left")
        if fps > 0:
            bits.append(f"{fps:.2f} fps" if fps >= 1
                        else f"{1 / fps:.1f}s per frame")
        bits.append(f"{_fmt_duration(elapsed)} elapsed")
        self._set_status(head, " · ".join(bits))

    def _on_log(self, level: str, text: str) -> None:
        self.logged.emit(level, text)

    def _on_finished(self, ok: bool, cancelled: bool, output: str) -> None:
        self._set_busy(False)
        if cancelled:
            self._set_status("Stopped", "The finished frames were saved.")
        elif ok:
            self._progress = 1.0
            self.progressChanged.emit()
            if self._preview_mode or output == self._preview_path:
                self._set_status("Preview ready", "")
                # Qt caches images by URL, and the preview path never
                # changes, so without a unique query the second preview shows
                # the first one's picture.
                self._preview = (QUrl.fromLocalFile(self._preview_path).toString()
                                 + f"?t={time.time():.3f}")
                self.previewChanged.emit()
            else:
                self._set_status("Finished", output)
        else:
            self._set_status("Failed", "See the log below.")
        self.completed.emit(ok, cancelled, output)
