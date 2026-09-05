"""Progress reporting and cooperative cancellation.

The pipeline used to print directly and drive tqdm itself, which works for a
terminal and nothing else. Anything that is not a terminal -- a GUI, a test, a
benchmark -- had only two options: scrape prose off stdout, or parse a
progress bar off stderr. Both are fragile, and both break the moment a message
is reworded.

`Reporter` is the seam. The pipeline says what it is doing; the caller decides
how that is presented. `ConsoleReporter` reproduces the previous terminal
behaviour exactly, so nothing changes for CLI users. `JsonReporter` writes one
JSON object per line for a parent process to read, which is what makes a
desktop UI possible without the UI knowing anything about our log format.

Cancellation belongs here for the same reason. A render can run for hours, so
"stop" cannot mean killing the process: that strands two ffmpeg children and
throws away work that is already encoded. Instead the caller supplies a
predicate, the pipeline checks it between frames, and the encoder is shut down
cleanly so the partial file is still playable.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Optional, TextIO


class Cancelled(Exception):
    """A caller-supplied cancel predicate returned True.

    Internal to the pipeline: `convert` catches it and reports the stop
    through its return value, so cancelling is an outcome rather than an
    error.
    """


class Reporter:
    """Base reporter: accepts everything, presents nothing.

    Not a logger. Each method corresponds to something the pipeline actually
    does, and carries structured `fields` alongside the human-readable text,
    so a consumer never has to parse the prose to recover the numbers.
    """

    def info(self, message: str, **fields: Any) -> None:
        """Something worth saying, e.g. which depth runtime was chosen."""

    def warning(self, message: str, **fields: Any) -> None:
        """Something the user should notice but that is not fatal."""

    def start(self, total: Optional[int], **fields: Any) -> None:
        """The frame-by-frame phase is beginning. `total` may be unknown."""

    def advance(self, n: int = 1) -> None:
        """`n` more frames have been written to the encoder."""

    def preview(self, path: str, frame: int, **fields: Any) -> None:
        """A frame of the render has been written to `path` to be looked at.

        Its own event rather than an `info` line: it happens repeatedly during
        a render and carries a path rather than a sentence, so a consumer wants
        to act on it silently and a console wants to say nothing at all.
        """

    def finish(self, **fields: Any) -> None:
        """The frame-by-frame phase has ended, cancelled or not."""

    def error(self, message: str, **fields: Any) -> None:
        """The run failed."""


class ConsoleReporter(Reporter):
    """Terminal presentation: plain lines plus a tqdm bar, as before."""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._bar = None

    def _write(self, text: str) -> None:
        # While a bar is live, tqdm.write is the only way to emit a line
        # without the bar redrawing over it.
        if self._bar is not None:
            from tqdm import tqdm

            tqdm.write(text, file=self._stream)
        else:
            print(text, file=self._stream)

    def info(self, message: str, **fields: Any) -> None:
        self._write(message)

    def warning(self, message: str, **fields: Any) -> None:
        self._write(message)

    def error(self, message: str, **fields: Any) -> None:
        print(message, file=sys.stderr)

    def start(self, total: Optional[int], **fields: Any) -> None:
        from tqdm import tqdm

        self._bar = tqdm(total=total, unit="frame",
                         desc=fields.get("desc", "Converting"))

    def advance(self, n: int = 1) -> None:
        if self._bar is not None:
            self._bar.update(n)

    def finish(self, **fields: Any) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class JsonReporter(Reporter):
    """One JSON object per line (NDJSON), for a parent process to read.

    Line-delimited rather than a single JSON document because the consumer
    reads it as it is produced, and a document cannot be parsed until it ends.
    Every line is flushed for the same reason: a UI that only learns about
    progress when the pipe buffer fills is not a progress display.

    Event `type`s: info, warning, start, progress, preview, done, error.
    """

    def __init__(self, stream: Optional[TextIO] = None,
                 interval: float = 0.1) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._interval = interval
        self._total: Optional[int] = None
        self._done = 0
        self._t0 = time.monotonic()
        self._last = 0.0
        self._emitted = False
        self._stage: Optional[str] = None

    def _emit(self, type_: str, **fields: Any) -> None:
        fields["type"] = type_
        json.dump(fields, self._stream, default=str)
        self._stream.write("\n")
        self._stream.flush()

    def info(self, message: str, **fields: Any) -> None:
        self._emit("info", message=message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._emit("warning", message=message, **fields)

    def preview(self, path: str, frame: int, **fields: Any) -> None:
        # Flushed like every other line: a preview a consumer learns about
        # once the pipe buffer fills is not a live preview.
        self._emit("preview", path=path, frame=frame, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._emit("error", message=message, **fields)

    def start(self, total: Optional[int], **fields: Any) -> None:
        self._total = total
        self._done = 0
        self._t0 = time.monotonic()
        self._last = 0.0
        self._emitted = False
        # Carried onto every progress event of this phase. Without it the
        # frames of a pre-pass are indistinguishable from the frames of the
        # render, and on an 8K source the pre-pass is the longer of the two --
        # so the interface says "rendering" for twenty minutes while nothing
        # of the sort is happening.
        self._stage = fields.get("stage")
        self._emit("start", total=total, **fields)

    def advance(self, n: int = 1) -> None:
        self._done += n
        now = time.monotonic()
        # Throttled: at 8K a frame takes about a second and every one is worth
        # reporting, but a passthrough run emits thousands per second and the
        # parent gains nothing from being flooded.
        #
        # The first and last frames always report regardless. The first so a
        # UI learns work has actually started without waiting out an interval;
        # the last so the bar reaches its total instead of stopping just shy
        # of it.
        milestone = not self._emitted or self._done == self._total
        if not milestone and now - self._last < self._interval:
            return
        self._emitted = True
        self._last = now
        elapsed = now - self._t0
        rate = self._done / elapsed if elapsed > 0 else 0.0
        eta = ((self._total - self._done) / rate
               if self._total and rate > 0 else None)
        self._emit("progress", frame=self._done, total=self._total,
                   elapsed=round(elapsed, 3), fps=round(rate, 3),
                   eta=round(eta, 1) if eta is not None else None,
                   **({"stage": self._stage} if self._stage else {}))

    def finish(self, **fields: Any) -> None:
        # setdefault, not a keyword: the caller's own frame count wins if it
        # has one, and passing `frames=` must not collide with ours.
        fields.setdefault("frames", self._done)
        fields.setdefault("elapsed", round(time.monotonic() - self._t0, 3))
        if self._stage:
            fields.setdefault("stage", self._stage)
        self._stage = None
        self._emit("done", **fields)
