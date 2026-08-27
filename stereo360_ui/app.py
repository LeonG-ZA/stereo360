"""Application bootstrap.

`--selftest` loads the interface, renders it, writes a screenshot and exits.
It is how the layout gets checked without a human having to look at it, and it
turns "the QML has a typo" from a runtime surprise into a failed command.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

from .controller import Controller

_QML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qml")


def _make_engine(app: QGuiApplication):
    # Material handles focus rings, hover and disabled states correctly, which
    # is a lot of fiddly detail to reimplement; the palette on top of it is
    # ours (see Theme.qml), so it does not read as an Android app.
    QQuickStyle.setStyle("Material")

    engine = QQmlApplicationEngine()
    engine.addImportPath(_QML_DIR)

    controller = Controller(app)
    engine.rootContext().setContextProperty("app", controller)

    engine.load(QUrl.fromLocalFile(os.path.join(_QML_DIR, "Main.qml")))
    if not engine.rootObjects():
        raise RuntimeError("QML failed to load; see the errors above.")
    return engine, controller


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    selftest = "--selftest" in argv
    shot = demo = size = None
    dump = False
    overrides: List[str] = []
    if selftest:
        argv.remove("--selftest")
        if "--dump-rows" in argv:
            argv.remove("--dump-rows")
            dump = True
        taken = {}
        for flag in ("--shot", "--demo", "--size"):
            if flag in argv:
                taken[flag] = argv.pop(argv.index(flag) + 1)
                argv.remove(flag)
        # Repeatable name=value, so a screenshot can exercise a state the
        # default view never shows -- a conditional row, an error banner.
        while "--set" in argv:
            i = argv.index("--set")
            overrides.append(argv.pop(i + 1))
            argv.pop(i)
        shot, demo, size = (taken.get("--shot"), taken.get("--demo"),
                            taken.get("--size"))

    if selftest:
        # On Windows Qt only writes warnings to stderr when it believes
        # stderr is a console; with a pipe it routes them to the debugger
        # instead. So a harness capturing stderr saw nothing, and an
        # assertion that no QML warnings occurred passed while six were being
        # emitted per run. Force them onto stderr where they can be checked.
        os.environ.setdefault("QT_ASSUME_STDERR_HAS_CONSOLE", "1")

    app = QGuiApplication(argv)
    app.setApplicationName("stereo360")
    app.setOrganizationName("stereo360")

    engine, controller = _make_engine(app)

    if selftest:
        window = engine.rootObjects()[0]

        if size:
            # Checking a layout against a small screen means actually giving
            # it one; the default window size says nothing about whether the
            # Convert button survives at 1366x768.
            w, _, h = size.partition("x")
            window.setWidth(int(w))
            window.setHeight(int(h))

        if demo:
            # Populate the view so the screenshot exercises the layout with
            # real content -- an empty panel hides exactly the problems
            # (elision, wrapping, image aspect) worth catching.
            # A real path, so the input probe actually runs and the source
            # readouts show real values rather than their empty state.
            probe_src = os.path.join(os.getcwd(), "input.mp4")
            window.setProperty("inputPath", probe_src)
            window.setProperty("outputPath",
                               probe_src.replace(".mp4", "_stereo.mp4"))
            window.setProperty("quality", "archival")
            controller._preview = QUrl.fromLocalFile(demo).toString()
            controller.previewChanged.emit()
            controller._status = "Frame 46 of 92"
            controller._detail = "38s left · 1.2s per frame · 55s elapsed"
            controller.statusChanged.emit()
            controller._progress = 0.5
            controller.progressChanged.emit()
            for level, text in (
                    ("info", "$ python -m stereo360 hallway_8k.mp4 -o "
                             "hallway_8k_stereo.mp4 --crf 13 --codec libx265 "
                             "--bitdepth 10 --preset slow --progress-json"),
                    ("info", "GPU accelerated: Depth Anything V2 on CUDA "
                             "(NVIDIA GeForce RTX 5070 Ti, fp16)"),
                    ("info", "Auto face size: 1920 (input 7680x3840)"),
                    ("info", "92 frames to render."),
                    ("warning", "Encoding is the bottleneck at this preset.")):
                controller.logged.emit(level, text)

        for item in overrides:
            name, _, value = item.partition("=")
            if value in ("true", "false"):
                window.setProperty(name, value == "true")
            elif value.lstrip("-").isdigit():
                window.setProperty(name, int(value))
            else:
                window.setProperty(name, value)

        def dump_rows():
            """Every labelled settings row and whether it is showing.

            Conditional rows (the Model / ONNX pair, the chunk controls) are
            the part of the layout a screenshot cannot check when they sit
            below the fold, so make the state assertable directly.
            """
            for child in window.findChildren(object):
                label = child.property("label")
                if (isinstance(label, str) and label
                        and child.property("labelWidth") is not None):
                    print(f"ROW\t{label}\t{bool(child.property('visible'))}")
            # Controls that are not labelled rows -- the VR180 direction
            # picker -- keyed by objectName. `startFrac` comes along because
            # it is the one number that has to agree with what the render
            # crops, and a picker pointing elsewhere is invisible here and
            # obvious in a headset.
            for child in window.findChildren(object):
                name = child.objectName()
                if name:
                    # startFrac for the direction picker, currentIndex for a
                    # dropdown -- the two things a screenshot cannot assert
                    # and that can silently disagree with the state behind
                    # them. A ComboBox that shows one row and renders another
                    # is exactly the bug this catches.
                    value = child.property("startFrac")
                    if value is None:
                        value = child.property("currentIndex")
                    print(f"ITEM\t{name}\t{bool(child.property('visible'))}\t"
                          f"{'' if value is None else value}")
            # Settings values too, so a state one control derives from another
            # can be asserted rather than eyeballed.
            for name in ("quality", "depthBackend", "depthModel", "onnxModel",
                         "sourceSubsampling", "spatialAudio", "depthTiles",
                         "codec", "outputMode", "yaw", "outputWidth", "resolutionIndex",
                         "spatialAudioHint", "photoMode",
                         "strength", "gradientLimit",
                         "faceAngularCorrection", "poleCompensation", "leftShare",
                         "livePreview", "livePreviewEvery"):
                print(f"PROP\t{name}\t{window.property(name)}")
            for entry in controller.encoders:
                print(f"ENC\t{entry['name']}\t{entry['available']}"
                      f"\t{entry['hardware']}")

        def capture():
            if dump:
                dump_rows()
            if shot:
                image = window.grabWindow()
                if image.isNull():
                    print("SELFTEST grab returned a null image")
                    app.exit(1)
                    return
                image.save(shot)
                print(f"SELFTEST ok -> {shot} ({image.width()}x"
                      f"{image.height()}) size={window.width()}x"
                      f"{window.height()}")
            else:
                print(f"SELFTEST ok (loaded) size={window.width()}x"
                      f"{window.height()}")
            app.quit()

        # One beat for the scene graph to render before grabbing it.
        # The encoder probe runs a real one-frame encode per candidate,
        # so give it time to land before dumping state.
        QTimer.singleShot(9000 if dump else 2500, capture)

    return app.exec()
