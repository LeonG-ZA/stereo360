"""The UI's command building and its child-process plumbing.

Rendering is checked by `python -m stereo360_ui --selftest`; this covers the
parts that can be wrong without looking wrong -- a flag translated
incorrectly, a cancel that does not arrive, a preview that never surfaces.

No QML and no window: Runner and Controller are plain QObjects, so a
QCoreApplication event loop is enough and these run on a headless machine.
"""

import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="UI extra not installed")

from PySide6.QtCore import QCoreApplication  # noqa: E402

from stereo360_ui import options  # noqa: E402
from stereo360_ui.controller import Controller  # noqa: E402
from stereo360_ui.runner import Runner  # noqa: E402
from test_end_to_end import make_test_video  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


def _pump(app, predicate, timeout=180):
    """Spin the event loop until `predicate` or the timeout expires."""
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    return predicate()


BASE = {"input": "in.mp4", "output": "out.mp4"}


# --------------------------------------------------------------- build_argv


def test_defaults_produce_a_minimal_command():
    """Anything left at its CLI default is omitted, so the command the UI
    shows stays short enough to read and to paste."""
    argv = options.build_argv(dict(BASE, quality="standard"))
    assert argv == ["-m", "stereo360", "in.mp4", "-o", "out.mp4",
                    "--progress-json"]


def test_quality_presets_map_to_encoder_flags():
    argv = options.build_argv(dict(BASE, quality="archival"))
    assert "--codec" in argv and argv[argv.index("--codec") + 1] == "libx265"
    assert argv[argv.index("--crf") + 1] == "13"
    assert argv[argv.index("--bitdepth") + 1] == "10"
    assert argv[argv.index("--preset") + 1] == "slow"


def test_slow_presets_warn_that_the_encoder_becomes_the_bottleneck():
    """The one cost of a quality preset that nothing else in the UI reveals."""
    assert options.preset_note("standard") == ""
    assert "bottleneck" in options.preset_note("archival")
    assert "bottleneck" in options.preset_note("vr")


def test_non_default_settings_are_emitted():
    argv = options.build_argv(dict(
        BASE, strength=1.5, gradientLimit=0.0, depthTiles=2,
        splitBaseline=True, spatialAudio=True, faceSizeAuto=False,
        faceSize=960, maxFrames=30, startFrame=15))
    assert argv[argv.index("--strength") + 1] == "1.5"
    assert argv[argv.index("--gradient-limit") + 1] == "0"
    assert argv[argv.index("--depth-tiles") + 1] == "2"
    assert argv[argv.index("--face-size") + 1] == "960"
    assert argv[argv.index("--max-frames") + 1] == "30"
    assert argv[argv.index("--start-frame") + 1] == "15"
    assert "--split-baseline" in argv and "--spatial-audio" in argv


def test_chunk_flags_only_for_the_temporal_backend():
    """They are accepted by the CLI regardless but do nothing elsewhere, and
    emitting them would imply the setting had an effect."""
    plain = options.build_argv(dict(BASE, chunkSize=4, chunkOverlap=1,
                                    temporalFill=False))
    assert "--chunk-size" not in plain and "--no-temporal-fill" not in plain

    temporal = options.build_argv(dict(
        BASE, depthBackend="video-depth-anything", chunkSize=4,
        chunkOverlap=1, temporalFill=False))
    assert temporal[temporal.index("--chunk-size") + 1] == "4"
    assert "--no-temporal-fill" in temporal


def test_preview_mode_drops_video_only_flags():
    """Encoder settings, frame range and audio metadata mean nothing for a
    still, and would only make the shown command look more complex."""
    argv = options.build_argv(
        dict(BASE, quality="archival", maxFrames=30, spatialAudio=True),
        preview_frame=42, preview_output="p.png", preview_width=1600)
    assert argv[argv.index("-o") + 1] == "p.png"
    assert argv[argv.index("--preview-frame") + 1] == "42"
    assert not {"--crf", "--codec", "--bitdepth", "--preset", "--max-frames",
                "--spatial-audio"} & set(argv)


def test_missing_paths_are_rejected_with_a_readable_message():
    with pytest.raises(ValueError, match="input video"):
        options.build_argv({"output": "out.mp4"})
    with pytest.raises(ValueError, match="where to save"):
        options.build_argv({"input": "in.mp4"})


def test_display_command_quotes_paths_with_spaces():
    cmd = options.display_command(
        options.build_argv({"input": r"C:\My Videos\a.mp4",
                            "output": "out.mp4"}))
    assert '"C:\\My Videos\\a.mp4"' in cmd and cmd.startswith("python -m")


# ------------------------------------------------------------------- runner


def test_runner_completes_a_conversion(qapp, tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=6, with_audio=False)

    runner = Runner()
    result, progress = [], []
    runner.finished.connect(lambda *a: result.append(a))
    runner.progressed.connect(lambda *a: progress.append(a))
    runner.start(options.build_argv(
        {"input": src, "output": dst, "faceSizeAuto": False,
         "faceSize": 32}) + ["--passthrough"])

    assert _pump(qapp, lambda: bool(result)), "runner never finished"
    ok, cancelled, output = result[0]
    assert ok and not cancelled
    assert output == dst
    assert Path(dst).exists()
    # Progress must actually arrive, not just a start and a done.
    assert progress and progress[-1][0] == 6


def test_runner_cancel_stops_and_keeps_the_partial_file(qapp, tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=512, h=256, frames=600, fps=30, with_audio=False)

    runner = Runner()
    result, progress = [], []
    runner.finished.connect(lambda *a: result.append(a))
    runner.progressed.connect(lambda *a: progress.append(a))
    runner.start(options.build_argv(
        {"input": src, "output": dst, "faceSizeAuto": False,
         "faceSize": 256}) + ["--passthrough"])

    assert _pump(qapp, lambda: bool(progress), timeout=90), "no progress"
    runner.cancel()

    assert _pump(qapp, lambda: bool(result), timeout=120), "cancel ignored"
    ok, cancelled, _ = result[0]
    assert cancelled and not ok
    assert Path(dst).exists(), "a cancelled render must keep its frames"


# --------------------------------------------------------------- controller


def test_controller_preview_publishes_an_image(qapp, tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=256, h=128, frames=8, with_audio=False)

    ctrl = Controller()
    done = []
    ctrl.completed.connect(lambda *a: done.append(a))
    ctrl.preview({"input": src, "output": str(tmp_path / "out.mp4"),
                  "faceSizeAuto": False, "faceSize": 64,
                  "depthBackend": "auto"}, 3)

    # No depth model needed: passthrough is not available through the preview
    # path, so this exercises the real backend selection. Allow for a model
    # load on a cold cache.
    assert _pump(qapp, lambda: bool(done), timeout=600), "preview never ended"
    ok, cancelled, _ = done[0]
    assert ok and not cancelled
    assert ctrl.previewSource.startswith("file:")
    assert "?t=" in ctrl.previewSource, "needs a cache-buster to refresh"


def test_controller_reports_missing_input_without_spawning(qapp):
    ctrl = Controller()
    logs, done = [], []
    ctrl.logged.connect(lambda level, text: logs.append((level, text)))
    ctrl.completed.connect(lambda *a: done.append(a))
    ctrl.convert({"input": "", "output": "out.mp4"})

    assert done and done[0][0] is False
    assert any(level == "error" for level, _ in logs)
    assert not ctrl.busy


# ------------------------------------------------------------------- render


def _selftest(*extra):
    """Run the UI selftest and return (stdout, warning lines)."""
    import subprocess
    import sys
    import tempfile

    shot = str(Path(tempfile.gettempdir()) / "stereo360_ui_selftest.png")
    proc = subprocess.run(
        [sys.executable, "-m", "stereo360_ui", "--selftest", "--shot", shot,
         *extra],
        capture_output=True, text=True, timeout=300,
        cwd=str(Path(__file__).resolve().parent.parent))

    if proc.returncode != 0 and "Could not connect to display" in proc.stderr:
        pytest.skip("no display available")
    assert proc.returncode == 0, proc.stderr
    assert Path(shot).exists()

    # A QML warning does not stop the engine from producing a window, so this
    # is the real check. It only works because --selftest sets
    # QT_ASSUME_STDERR_HAS_CONSOLE: without it Qt sends warnings to the
    # debugger when stderr is a pipe, and the assertion silently passes while
    # the app warns six times a run.
    #
    # setGeometry is excluded deliberately: it fires when the *test machine's*
    # screen is smaller than the requested window, which says nothing about
    # the layout and would make this flaky on exactly the small displays the
    # sizing is meant to support.
    warnings = [ln for ln in proc.stderr.splitlines()
                if ("QML" in ln or "Detected anchors" in ln)
                and "setGeometry" not in ln]
    return proc.stdout, warnings


def _size_of(stdout):
    field = next(t for t in stdout.split() if t.startswith("size="))
    w, _, h = field[len("size="):].partition("x")
    return int(w), int(h)


def test_qml_loads_and_renders():
    """Catches QML syntax and binding errors, which are runtime-only.

    `--selftest` loads the interface, renders one frame, writes a screenshot
    and exits, so a typo in Main.qml fails here rather than the first time
    someone opens the app.
    """
    stdout, warnings = _selftest()
    assert "SELFTEST ok" in stdout
    assert not warnings, "QML warnings:\n" + "\n".join(warnings)


def test_default_window_fits_a_768_high_screen():
    """1366x768 is still common, and leaves roughly 690 px of client area
    once the title bar and taskbar are gone. A default window taller than
    that opens with its Convert button off the bottom of the screen."""
    stdout, _ = _selftest()
    width, height = _size_of(stdout)
    assert height <= 690, f"default window is {height} px tall"
    assert width <= 1366, f"default window is {width} px wide"


def test_layout_survives_a_short_window():
    """The compact metrics are bindings, and a binding that throws still
    renders — so exercise the small sizes rather than trusting them."""
    for size in ("1366x690", "900x560"):
        stdout, warnings = _selftest("--size", size)
        assert "SELFTEST ok" in stdout
        assert not warnings, f"at {size}:\n" + "\n".join(warnings)


# ------------------------------------------------------ source subsampling


def test_source_subsampling_flag_is_emitted():
    argv = options.build_argv(dict(BASE, sourceSubsampling=True))
    assert "--source-subsampling" in argv
    assert "--source-subsampling" not in options.build_argv(dict(BASE))


def test_source_subsampling_is_omitted_from_a_preview():
    """A preview is a PNG; chroma subsampling has no meaning for it."""
    argv = options.build_argv(dict(BASE, sourceSubsampling=True),
                              preview_frame=0, preview_output="p.png")
    assert "--source-subsampling" not in argv


def test_controller_probes_the_input(qapp, tmp_path: Path):
    """The UI learns the source's chroma (and frame count) out of process, so
    it never imports the core -- or numpy -- into the window's process."""
    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=128, h=64, frames=7, with_audio=False)

    ctrl = Controller()
    seen = []
    ctrl.sourceInfoChanged.connect(lambda: seen.append(dict(ctrl.sourceInfo)))
    ctrl.probeInput(src)

    assert _pump(qapp, lambda: any(s.get("width") for s in seen), timeout=120)
    info = next(s for s in seen if s.get("width"))
    assert info["width"] == 128 and info["height"] == 64
    assert info["chroma"] == "4:2:0"
    assert info["frame_count"] == 7


def test_controller_probe_of_a_missing_file_stays_empty(qapp, tmp_path: Path):
    ctrl = Controller()
    ctrl.probeInput(str(tmp_path / "nope.mp4"))
    _pump(qapp, lambda: False, timeout=6)   # let it fail on its own
    assert dict(ctrl.sourceInfo) == {}


# ------------------------------------------------------------ depth models


def test_base_model_is_the_default_and_emits_nothing():
    assert "--depth-model" not in options.build_argv(dict(BASE))
    assert "--depth-model" not in options.build_argv(
        dict(BASE, depthModel="base"))


def test_choosing_small_actually_gets_small():
    """The trap this guards: while Small was the CLI default, its entry in
    DEPTH_MODELS was blank and "fell through" to whatever the backend picked.
    Moving the default to Base would then have turned a Small selection into
    Base silently -- the dropdown saying one thing and the run doing another,
    which is the exact bug class already fixed once for the temporal backend.
    """
    argv = options.build_argv(dict(BASE, depthModel="small"))
    assert (argv[argv.index("--depth-model") + 1]
            == "depth-anything/Depth-Anything-V2-Small-hf")


def test_the_ui_and_the_cli_agree_on_which_model_is_default():
    """`build_argv` omits the flag for the default, so if these two ever drift
    the UI would quietly run a model the user did not choose."""
    from stereo360.depth.depth_anything import DEFAULT_MODEL

    assert (options.DEPTH_MODELS[options.DEFAULT_DEPTH_MODEL]["depth-anything"]
            == DEFAULT_MODEL)


def test_every_model_choice_maps_explicitly():
    """No blank entries. A blank one means 'whatever the default is', which is
    only ever correct by coincidence."""
    for name, mapping in options.DEPTH_MODELS.items():
        assert mapping.get("depth-anything"), name


@pytest.mark.parametrize("backend,expected", [
    ("auto", "depth-anything/Depth-Anything-V2-Large-hf"),
    ("depth-anything", "depth-anything/Depth-Anything-V2-Large-hf"),
    ("video-depth-anything", "large"),
])
def test_large_model_maps_per_backend(backend, expected):
    """The same dropdown choice means a Hub id for one backend and a variant
    name for the other; the UI resolves that so the user need not know."""
    argv = options.build_argv(dict(BASE, depthBackend=backend,
                                   depthModel="large"))
    assert argv[argv.index("--depth-model") + 1] == expected


def test_custom_onnx_model_is_passed_through():
    """The fast-mode path: a graph exported with scripts/export_onnx.py."""
    argv = options.build_argv(dict(BASE, depthBackend="onnx",
                                   onnxModel="models/depth_fast.onnx"))
    assert argv[argv.index("--onnx-model") + 1] == "models/depth_fast.onnx"


def test_onnx_model_omitted_when_blank_or_default():
    """No point restating the path the CLI already defaults to."""
    blank = options.build_argv(dict(BASE, depthBackend="onnx", onnxModel=""))
    assert "--onnx-model" not in blank
    same = options.build_argv(dict(
        BASE, depthBackend="onnx",
        onnxModel=options.DEFAULT_ONNX_MODEL))
    assert "--onnx-model" not in same


def test_onnx_path_ignored_for_torch_backends():
    """It would be passed but unused, implying it had an effect."""
    argv = options.build_argv(dict(BASE, depthBackend="depth-anything",
                                   onnxModel="models/depth_fast.onnx"))
    assert "--onnx-model" not in argv


@pytest.mark.parametrize("backend,model_row,onnx_row", [
    ("auto", True, False),
    ("depth-anything", True, False),
    ("video-depth-anything", True, False),
    ("onnx", False, True),
])
def test_model_rows_follow_the_backend(backend, model_row, onnx_row):
    """Only the row that applies is shown, which is what teaches the user that
    a custom ONNX model needs the ONNX backend selected."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "stereo360_ui", "--selftest", "--dump-rows",
         "--set", f"depthBackend={backend}"],
        capture_output=True, text=True, timeout=300,
        cwd=str(Path(__file__).resolve().parent.parent))
    if proc.returncode != 0 and "display" in proc.stderr:
        pytest.skip("no display available")
    assert proc.returncode == 0, proc.stderr

    rows = dict(
        (parts[1], parts[2] == "True")
        for parts in (line.split("\t") for line in proc.stdout.splitlines())
        if parts[0] == "ROW")
    assert rows["Model"] is model_row
    assert rows["ONNX model"] is onnx_row


def test_large_model_maps_to_its_hub_id():
    """Base is the default and so emits nothing; Large has to be asked for."""
    argv = options.build_argv(dict(BASE, depthModel="large"))
    assert (argv[argv.index("--depth-model") + 1]
            == "depth-anything/Depth-Anything-V2-Large-hf")


def test_base_falls_back_for_the_temporal_backend():
    """Video Depth Anything ships small and large only, so asking it for Base
    would fail at model load rather than anywhere visible."""
    argv = options.build_argv(dict(BASE, depthBackend="video-depth-anything",
                                   depthModel="base"))
    assert "--depth-model" not in argv


def _dump(*sets):
    import subprocess
    import sys

    args = []
    for item in sets:
        args += ["--set", item]
    proc = subprocess.run(
        [sys.executable, "-m", "stereo360_ui", "--selftest", "--dump-rows",
         *args], capture_output=True, text=True, timeout=300,
        cwd=str(Path(__file__).resolve().parent.parent))
    if proc.returncode != 0 and "display" in proc.stderr:
        pytest.skip("no display available")
    assert proc.returncode == 0, proc.stderr
    props, rows = {}, {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "PROP":
            props[parts[1]] = parts[2]
        elif parts[0] == "ROW":
            rows[parts[1]] = parts[2] == "True"
    return props, rows


def test_ui_moves_off_base_for_the_temporal_backend():
    """The UI should not leave a selection showing that cannot be honoured."""
    props, _ = _dump("depthModel=base", "depthBackend=video-depth-anything")
    assert props["depthBackend"] == "video-depth-anything"
    assert props["depthModel"] == "small"

    props, _ = _dump("depthModel=base", "depthBackend=depth-anything")
    assert props["depthModel"] == "base"


def test_base_is_not_offered_for_the_temporal_backend():
    """Reported bug: picking the backend first and Base second left the
    dropdown showing Base while the render silently used Small. The earlier
    test only covered the other order, which is why it passed."""
    for sets in (("depthModel=base", "depthBackend=video-depth-anything"),
                 ("depthBackend=video-depth-anything", "depthModel=base")):
        props, _ = _dump(*sets)
        assert props["depthModel"] == "small", (
            f"{sets}: dropdown must not show a model that will not run")

    props, _ = _dump("depthBackend=depth-anything", "depthModel=base")
    assert props["depthModel"] == "base", "Base must still work where valid"


def test_controller_probes_available_backends(qapp):
    """The UI asks which backends can run before offering them."""
    ctrl = Controller()
    seen = []
    ctrl.backendsChanged.connect(lambda: seen.append(list(ctrl.backends)))
    ctrl.probeBackends()

    assert _pump(qapp, lambda: bool(seen), timeout=180), "probe never returned"
    entries = {e["name"]: e for e in seen[-1]}
    assert set(entries) == {"auto", "depth-anything",
                            "video-depth-anything", "onnx"}
    assert all(isinstance(e["available"], bool) and e["detail"]
               for e in entries.values())


# -------------------------------------------------------------- encoders


def test_encoder_choice_overrides_the_preset_codec():
    """The two controls compose: the preset still sets quality, speed and bit
    depth; only the encoder is taken from the explicit choice."""
    argv = options.build_argv(dict(BASE, quality="archival",
                                   codec="hevc_nvenc"))
    assert argv[argv.index("--codec") + 1] == "hevc_nvenc"
    assert argv[argv.index("--crf") + 1] == "13"
    assert argv[argv.index("--bitdepth") + 1] == "10"
    assert argv[argv.index("--preset") + 1] == "slow"


def test_blank_encoder_leaves_the_preset_alone():
    assert options.build_argv(dict(BASE, codec="")) == \
        options.build_argv(dict(BASE))
    # And choosing the default explicitly still emits nothing.
    assert "--codec" not in options.build_argv(dict(BASE, codec="libx264"))


def test_controller_probes_encoders_at_the_output_size(qapp, tmp_path: Path):
    """Availability depends on resolution, so the probe has to be told the
    *output* size -- the source height doubled for top-bottom."""
    ctrl = Controller()
    seen = []
    ctrl.encodersChanged.connect(lambda: seen.append(list(ctrl.encoders)))
    ctrl.probeEncoders(320, 240)

    assert _pump(qapp, lambda: bool(seen), timeout=300), "probe never returned"
    entries = {e["name"]: e for e in seen[-1]}
    assert entries["libx264"]["available"] is True
    assert entries["libx264"]["hardware"] is False
    assert entries["hevc_nvenc"]["hardware"] is True
    assert all(e["detail"] for e in entries.values())


def test_controller_does_not_reprobe_the_same_size(qapp):
    """Each probe runs a real encode per candidate; repeating it on every
    input change would stall the UI for seconds at a time."""
    ctrl = Controller()
    calls = []
    ctrl.encodersChanged.connect(lambda: calls.append(1))
    ctrl.probeEncoders(320, 240)
    assert _pump(qapp, lambda: bool(calls), timeout=300)
    ctrl.probeEncoders(320, 240)
    _pump(qapp, lambda: len(calls) > 1, timeout=5)
    assert len(calls) == 1, "same size should be answered from memory"
