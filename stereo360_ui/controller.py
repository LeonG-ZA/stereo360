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
    thumbnailChanged = Signal()
    #: level, text -- appended to the log view
    logged = Signal(str, str)
    #: fired when a run ends so QML can flash a result banner
    completed = Signal(bool, bool, str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._runner = Runner(self)
        self._runner.progressed.connect(self._on_progress)
        self._runner.logged.connect(self._on_log)
        self._runner.staged.connect(self._on_stage)
        self._runner.previewed.connect(self._on_live_preview)
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

        self._thumbnail = ""
        self._thumbnail_key = None
        self._pending_thumbnail = None
        self._thumbnail_path = os.path.join(
            tempfile.gettempdir(), "stereo360_thumbnail.jpg")
        self._thumbnail_proc = QProcess(self)
        self._thumbnail_proc.finished.connect(self._on_thumbnail_finished)

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

    @Property(str, notify=thumbnailChanged)
    def thumbnailSource(self) -> str:
        """A picture of the source frame, for the direction picker, or ''."""
        return self._thumbnail

    @Property(int, constant=True)
    def photoDepthTiles(self) -> int:
        """Tiles per cube face the CLI will use for a photo."""
        return options.PHOTO_DEPTH_TILES

    @Property("QVariantList", constant=True)
    def openFilters(self) -> list:
        """Name filters for the open dialog, covering video and photo both."""
        return options.open_filters()

    @Slot(bool, result="QVariantList")
    def saveFilters(self, photo: bool) -> list:
        """Name filters for the save dialog, for a photo job or a video one."""
        return options.save_filters(bool(photo))

    @Slot(str, str, str, str, result="QVariantMap")
    def resolveOutput(self, current: str, previous_suggestion: str,
                      input_url: str, mode: str) -> dict:
        """{output, suggested} for the Output box after the job changed.

        Returns both, because the caller has to remember what was suggested
        in order to know next time whether the box still holds its own idea
        or someone's.
        """
        suggestion = self.suggestOutput(input_url, mode)
        return {"output": options.resolve_output(
                    current, previous_suggestion, suggestion,
                    options.is_image(self.toLocalPath(input_url))),
                "suggested": suggestion}

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

    @Slot(int, int, str, int, result="QVariantList")
    def outputSize(self, width: int, height: int, mode: str,
                   output_width: int = 0) -> list:
        """[w, h] of the encoded frame for this source, mode and chosen size."""
        return list(options.output_size(int(width), int(height), str(mode),
                                        int(output_width) or None))

    @Slot(int, int, str, result="QVariantList")
    def resolutionChoices(self, width: int, height: int, mode: str) -> list:
        """Output sizes worth offering for this source, largest first."""
        return options.resolution_choices(int(width), int(height), str(mode))

    @Slot(int, int, str, bool, result=int)
    def defaultOutputWidth(self, width: int, height: int, mode: str,
                           is_photo: bool) -> int:
        """Width to start this source on; 0 means the source's own."""
        return options.default_output_width(int(width), int(height), str(mode),
                                            bool(is_photo))

    @Slot(int, int, result=bool)
    def exceedsLevelLimit(self, width: int, height: int) -> bool:
        """Whether a frame this size is past the top level of both codecs."""
        return int(width) * int(height) > options.MAX_LEVEL_LUMA

    @Slot(int, int, str, int)
    def probeEncoders(self, width: int, height: int, mode: str,
                      output_width: int = 0) -> None:
        """Which encoders manage this output on this machine.

        Takes the *source* size and works out the output itself, because the
        answer depends on the output mode as much as on the input: the same 8K
        source encodes as 7680x7680 in 360 mode and 7680x3840 in VR180, and
        there are encoders that take one and refuse the other.

        Resolution-dependent and worth re-asking per input: hevc_amf takes
        3840x3840 and refuses 7680x7680, so a list built once for a 4K project
        would offer an encoder that cannot touch an 8K one.
        """
        size = options.output_size(int(width), int(height), str(mode),
                                   int(output_width) or None)
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

    def _on_live_preview(self, path: str, frame: int) -> None:
        """Point the preview panel at the frame just rendered.

        The frame number goes in the URL as a query string. QML caches an
        Image by its source, so a file rewritten under an unchanged URL shows
        the first frame for the whole render; the path stays one file on disk
        and only the URL changes.
        """
        self._preview = f"{QUrl.fromLocalFile(path).toString()}?f={frame}"
        self.previewChanged.emit()

    @Slot(str)
    def explainBackend(self, name: str) -> None:
        """Say why a backend cannot be chosen, at the moment it is chosen.

        This used to be a line per unavailable backend written into the log at
        startup, before anyone had asked about depth at all. It read as a list
        of things that were wrong with the machine, and it was the first thing
        in the window on a perfectly healthy install -- most people never
        wanted the ONNX export or a clone of the Video Depth Anything repo.
        The dropdown already carries the reason under each entry it has
        disabled, so the startup lines said it a second time, sooner, and with
        less context.

        Saying it on the click keeps the one case where the log earns its
        place: the entry is disabled, so choosing it does nothing, and
        silently doing nothing is the one outcome that needs explaining.
        """
        for entry in self._backends:
            if entry.get("name") == name and not entry.get("available"):
                self.logged.emit(
                    "warning",
                    f"{name} is not available here: {entry['detail']}")
                return

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

    @Slot(str, int)
    def requestThumbnail(self, path: str, frame: int) -> None:
        """Fetch a picture of the source frame for the direction picker.

        Cheap enough to do on every frame change -- one decoded frame, no
        model, measured at 0.7 s on an 8K file -- and it has to be, because
        choosing where the VR180 field points is something you do *before*
        spending an hour on a render.
        """
        key = (path, int(frame))
        if not path or key == self._thumbnail_key:
            return
        if self._thumbnail and self._thumbnail_key \
                and self._thumbnail_key[0] != path:
            # A picture of the previous file is worse than no picture: it
            # would look like a valid answer to a question about this one.
            self._thumbnail = ""
            self.thumbnailChanged.emit()
        if self._thumbnail_proc.state() != QProcess.NotRunning:
            # Dragging the frame number outruns a 0.7 s decode easily. Keep
            # only the latest -- the intermediate frames were never wanted.
            self._pending_thumbnail = key
            return
        self._pending_thumbnail = None
        self._thumbnail_key = key
        self._thumbnail_proc.setWorkingDirectory(core_root())
        self._thumbnail_proc.setProgram(sys.executable)
        self._thumbnail_proc.setArguments(
            ["-m", "stereo360", path, "--thumbnail", self._thumbnail_path,
             "--preview-frame", str(max(0, int(frame)))])
        self._thumbnail_proc.start()

    def _on_thumbnail_finished(self, code: int, _status) -> None:
        if code == 0:
            # Same cache defeat as the preview: the path never changes, so
            # without a unique query Qt paints the previous frame's picture.
            self._thumbnail = (
                QUrl.fromLocalFile(self._thumbnail_path).toString()
                + f"?t={time.time():.3f}")
            self.thumbnailChanged.emit()
        else:
            self._thumbnail_key = None      # let a later attempt retry
        pending, self._pending_thumbnail = self._pending_thumbnail, None
        if pending:
            self.requestThumbnail(pending[0], pending[1])

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

    # Both arities registered. Declaring only `(str)` -- as this did -- does
    # not fail when QML passes two: Qt drops the extra silently and `mode`
    # falls back to its default, so every VR180 photo was named `_360_TB`.
    # The token is what the Quest gallery reads, so the file did not merely
    # look wrong, it described the wrong layout.
    @Slot(str, str, result=str)
    @Slot(str, result=str)
    def suggestOutput(self, input_url: str, mode: str = "360") -> str:
        """A sensible default output beside the input, so the second file
        picker is usually unnecessary.

        For a photo the name carries the layout tokens, since that is how
        players that read filenames rather than metadata learn what the file
        is -- and it saves the person having to know the convention. Always
        `.jpg`, whatever went in: it is the one format headsets read reliably.
        """
        path = self.toLocalPath(input_url)
        if not path:
            return ""
        stem, _ = os.path.splitext(path)
        if options.is_image(path):
            from stereo360 import vr_naming

            return vr_naming.suggest(f"{stem}.jpg", str(mode))
        return f"{stem}_stereo.mp4"

    @Slot(str, result=bool)
    def isImage(self, path: str) -> bool:
        """Whether this input makes it a photo job rather than a video one."""
        return options.is_image(self.toLocalPath(path))

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

    @Slot(str)
    def reportPreviewFailure(self, source: str) -> None:
        """Say so when the panel cannot show a picture that does exist.

        The file is written and correct; only the on-screen copy failed. That
        distinction matters enough to spell out, because "no picture" reads as
        "no output" and the natural next move is to run the whole thing again.
        """
        path = source.split("?", 1)[0]
        self.logged.emit(
            "warning",
            f"Could not display {path} in the panel. The file itself is "
            f"written and fine -- open it in a viewer or a headset.")

    def _on_log(self, level: str, text: str) -> None:
        self.logged.emit(level, text)

    def _on_stage(self, text: str) -> None:
        """Show what is happening before the first frame exists.

        Only until then: the first progress event replaces this, so nothing
        has to decide when preparation ended. The status line keeps whichever
        of "Starting..." or "Rendering preview..." was already set, because
        that is still true -- what was missing was the detail underneath.
        """
        if self._busy:
            self._set_status(self._status, text)

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
            elif options.is_image(output):
                # A photo *is* the deliverable, so the panel shows it rather
                # than the source it came from. Same cache-buster: the second
                # conversion to the same path would otherwise show the first.
                self._set_status("Finished", output)
                self._preview = (QUrl.fromLocalFile(output).toString()
                                 + f"?t={time.time():.3f}")
                self.previewChanged.emit()
            else:
                self._set_status("Finished", output)
        else:
            self._set_status("Failed", "See the log below.")
        self.completed.emit(ok, cancelled, output)
