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


def _items(*sets):
    """{objectName: (visible, startFrac)} from a selftest run.

    Controls that are not labelled rows -- the direction picker -- are
    invisible to `_dump`, and the picker is the one thing in the interface
    whose *geometry* has to agree with the core.
    """
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

    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "ITEM":
            out[parts[1]] = (parts[2] == "True",
                             float(parts[3]) if parts[3] else None)
    return out


def test_the_direction_picker_appears_only_for_vr180():
    """360 output keeps the whole sphere, so there is no direction to choose;
    showing the control would imply otherwise."""
    assert _items("outputMode=360")["directionPicker"][0] is False
    assert _items("outputMode=vr180")["directionPicker"][0] is True


@pytest.mark.parametrize("yaw", [0, 90, 150])
def test_the_picker_points_where_the_render_crops(yaw):
    """The bug this exists to catch: a band drawn somewhere other than the
    columns the render keeps. Nothing else would reveal it short of putting on
    a headset and finding the file pointing at the sky.

    150 is deliberate -- it is past the seam, so the band wraps and the naive
    formula gives a negative left edge.
    """
    from stereo360 import pipeline

    width = 7680
    expected = pipeline.vr180_crop(width, yaw)[0] / width
    shown = _items("outputMode=vr180", f"yaw={yaw}")["directionPicker"][1]
    assert shown == pytest.approx(expected, abs=1.0 / width)


def test_leaving_vr180_clears_the_yaw():
    """It would otherwise be dropped silently by the 360 render and reappear
    on switching back, having meant nothing in between."""
    props, _ = _dump("outputMode=vr180", "yaw=45", "outputMode=360")
    assert props["outputMode"] == "360"
    assert float(props["yaw"]) == 0.0


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
    """Availability depends on resolution, so the probe works out the *output*
    size from the source and the mode rather than being handed it."""
    ctrl = Controller()
    seen = []
    ctrl.encodersChanged.connect(lambda: seen.append(list(ctrl.encoders)))
    ctrl.probeEncoders(320, 240, "360")

    assert _pump(qapp, lambda: bool(seen), timeout=300), "probe never returned"
    entries = {e["name"]: e for e in seen[-1]}
    assert entries["libx264"]["available"] is True
    assert entries["libx264"]["hardware"] is False
    assert entries["hevc_nvenc"]["hardware"] is True
    assert all(e["detail"] for e in entries.values())


def test_write_thumbnail_reads_one_frame(tmp_path: Path):
    """What the direction picker drags on. It has to exist before any depth
    work, so it decodes a frame and touches no model."""
    from stereo360 import ffmpeg_io

    src = str(tmp_path / "in.mp4")
    out = str(tmp_path / "t.jpg")
    make_test_video(src, w=256, h=128, frames=20, with_audio=False)

    assert ffmpeg_io.write_thumbnail(src, out, frame_index=10, width=64)
    info = ffmpeg_io.probe(out)
    assert (info.width, info.height) == (64, 32), "must stay 2:1"


def test_an_odd_thumbnail_width_is_rounded(tmp_path: Path):
    """A JPEG cannot be 4:2:0 at an odd width, and ffmpeg fails rather than
    rounding for you."""
    from stereo360 import ffmpeg_io

    src = str(tmp_path / "in.mp4")
    out = str(tmp_path / "t.jpg")
    make_test_video(src, w=256, h=128, frames=4, with_audio=False)

    assert ffmpeg_io.write_thumbnail(src, out, width=65)
    assert ffmpeg_io.probe(out).width == 64


def test_a_thumbnail_past_the_end_falls_back_to_the_first_frame(
        tmp_path: Path):
    """Reachable from the interface, which lets the frame number run free when
    the source does not declare a count. The wrong frame is still the right
    sphere, and a picker with a picture in it beats an empty box."""
    from stereo360 import ffmpeg_io

    src = str(tmp_path / "in.mp4")
    out = str(tmp_path / "t.jpg")
    make_test_video(src, w=256, h=128, frames=5, with_audio=False)

    assert ffmpeg_io.write_thumbnail(src, out, frame_index=99999)
    assert Path(out).stat().st_size > 0


def test_controller_fetches_a_source_thumbnail(qapp, tmp_path: Path):
    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=256, h=128, frames=8, with_audio=False)

    ctrl = Controller()
    seen = []
    ctrl.thumbnailChanged.connect(lambda: seen.append(ctrl.thumbnailSource))
    ctrl.requestThumbnail(src, 2)

    assert _pump(qapp, lambda: any(seen), timeout=180), "no thumbnail arrived"
    assert ctrl.thumbnailSource.startswith("file:")
    assert "?t=" in ctrl.thumbnailSource, "needs a cache-buster to refresh"


def test_controller_does_not_refetch_the_same_frame(qapp, tmp_path: Path):
    """Every control that can change the picture asks for it, so the same
    request arrives several times over for one user action."""
    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=256, h=128, frames=8, with_audio=False)

    ctrl = Controller()
    calls = []
    ctrl.thumbnailChanged.connect(lambda: calls.append(ctrl.thumbnailSource))
    ctrl.requestThumbnail(src, 2)
    assert _pump(qapp, lambda: bool(calls), timeout=180)
    ctrl.requestThumbnail(src, 2)
    _pump(qapp, lambda: len(calls) > 1, timeout=4)
    assert len(calls) == 1, "same frame should be answered from memory"


def test_a_thumbnail_of_a_missing_file_stays_empty(qapp, tmp_path: Path):
    ctrl = Controller()
    ctrl.requestThumbnail(str(tmp_path / "nope.mp4"), 0)
    _pump(qapp, lambda: False, timeout=8)       # let it fail on its own
    assert ctrl.thumbnailSource == ""


def test_controller_does_not_reprobe_the_same_size(qapp):
    """Each probe runs a real encode per candidate; repeating it on every
    input change would stall the UI for seconds at a time."""
    ctrl = Controller()
    calls = []
    ctrl.encodersChanged.connect(lambda: calls.append(1))
    ctrl.probeEncoders(320, 240, "360")
    assert _pump(qapp, lambda: bool(calls), timeout=300)
    ctrl.probeEncoders(320, 240, "360")
    _pump(qapp, lambda: len(calls) > 1, timeout=5)
    assert len(calls) == 1, "same size should be answered from memory"


def test_changing_the_output_mode_reprobes_the_encoders(qapp):
    """The same source is a different frame in the other mode -- 7680x7680 in
    360, 7680x3840 in VR180 -- and there are encoders that take one and refuse
    the other. Answering the second from the first's memory would offer an
    encoder that cannot run."""
    ctrl = Controller()
    calls = []
    ctrl.encodersChanged.connect(lambda: calls.append(1))
    ctrl.probeEncoders(320, 240, "360")
    assert _pump(qapp, lambda: bool(calls), timeout=300)
    ctrl.probeEncoders(320, 240, "vr180")
    assert _pump(qapp, lambda: len(calls) > 1, timeout=300), \
        "the mode changed the output size, so the answer cannot be reused"


# ---------------------------------------------------------------- VR180 mode


def test_360_output_is_the_default_and_emits_nothing():
    assert "--output-mode" not in options.build_argv(dict(BASE))
    assert "--output-mode" not in options.build_argv(
        dict(BASE, outputMode="360"))


def test_vr180_output_is_emitted():
    argv = options.build_argv(dict(BASE, outputMode="vr180"))
    assert argv[argv.index("--output-mode") + 1] == "vr180"


def test_yaw_travels_only_with_vr180():
    """The CLI refuses a yaw in 360 mode rather than quietly ignoring it, so a
    control left over from a VR180 session would turn into a failed run. The
    UI clears it on the mode change too; this is the second line of defence."""
    assert "--yaw" not in options.build_argv(
        dict(BASE, outputMode="360", yaw=90))
    argv = options.build_argv(dict(BASE, outputMode="vr180", yaw=90))
    assert argv[argv.index("--yaw") + 1] == "90"


def test_a_zero_yaw_is_left_off():
    assert "--yaw" not in options.build_argv(
        dict(BASE, outputMode="vr180", yaw=0))


def test_the_output_mode_survives_into_a_preview():
    """Unlike the encoder settings, which a still has no use for: the point of
    previewing a VR180 frame is to see which half of the sphere you chose."""
    argv = options.build_argv(dict(BASE, outputMode="vr180", yaw=-45),
                              preview_frame=3, preview_output="p.png")
    assert argv[argv.index("--output-mode") + 1] == "vr180"
    assert argv[argv.index("--yaw") + 1] == "-45"


@pytest.mark.parametrize("mode", ["360", "vr180"])
@pytest.mark.parametrize("size", [(7680, 3840), (5760, 2880), (3840, 1920),
                                  (1920, 960), (322, 161), (2, 1)])
def test_the_ui_and_the_core_agree_on_the_output_size(size, mode):
    """`options.output_size` duplicates `pipeline.output_geometry` so that the
    window's process never imports numpy. The duplication is deliberate; drift
    is not, since this size decides which encoders get offered."""
    from stereo360 import pipeline

    assert options.output_size(*size, mode) == pipeline.output_geometry(*size,
                                                                        mode)


def test_vr180_is_what_brings_an_8k_source_under_the_level_cap():
    """Worth surfacing, and worth surfacing *carefully*: the 360 frame is over
    the cap and is still the correct YouTube master. See the note below."""
    assert options.output_size(7680, 3840, "360") == (7680, 7680)
    assert options.output_size(7680, 3840, "vr180") == (7680, 3840)
    assert 7680 * 7680 > options.MAX_LEVEL_LUMA
    assert 7680 * 3840 <= options.MAX_LEVEL_LUMA


def test_the_cap_is_not_an_hevc_number():
    """H.264 caps a frame at 139,264 macroblocks, HEVC at 35,651,584 luma
    samples, and 139264 x 256 is the same figure. It matters because the
    obvious reaction to "past HEVC's limit" is to switch to H.264, which buys
    nothing -- so the interface must not name one codec.

    Confirmed against x264 itself, which prints the macroblock number when
    handed a 7680x7680 frame: "frame MB size (480x480) > level limit (139264)".
    """
    assert options.MAX_LEVEL_LUMA == 139_264 * 256


# ------------------------------------------------------------ output width


def test_the_source_width_emits_no_flag():
    """Full size is the default and the commonest case; the flag appears only
    when a size was actually chosen."""
    assert "--output-width" not in options.build_argv(
        dict(BASE, sourceWidth=7680, outputWidth=7680))
    assert "--output-width" not in options.build_argv(
        dict(BASE, sourceWidth=7680))


def test_a_chosen_width_is_emitted():
    argv = options.build_argv(dict(BASE, sourceWidth=7680, outputWidth=5760))
    assert argv[argv.index("--output-width") + 1] == "5760"


def test_the_width_travels_with_the_mode():
    argv = options.build_argv(dict(BASE, sourceWidth=7680, outputWidth=5760,
                                   outputMode="vr180"))
    assert argv[argv.index("--output-mode") + 1] == "vr180"
    assert argv[argv.index("--output-width") + 1] == "5760"


@pytest.mark.parametrize("mode", ["360", "vr180"])
@pytest.mark.parametrize("width", [7680, 5760, 4096, 3840, 1920])
def test_the_ui_and_the_core_agree_on_the_scaled_size(mode, width):
    """Same duplication as `output_size`, same reason, same risk: this number
    decides which encoders get offered."""
    from stereo360 import pipeline

    assert (options.output_size(7680, 3840, mode, width)
            == pipeline.output_geometry(7680, 3840, mode, width))


def test_an_8k_source_is_offered_a_size_that_plays():
    """The whole point. 7680x7680 is over the decode cap -- black on a Quest 3
    in both codecs -- so there has to be something else on the list."""
    choices = options.resolution_choices(7680, 3840, "360")
    assert choices[0]["native"] and not choices[0]["fits"]
    playable = [c for c in choices if c["fits"]]
    assert playable and playable[0]["width"] == 5760
    assert playable[0]["label"] == "5760×5760"


def test_a_4k_source_has_nothing_to_warn_about():
    """The restriction appears exactly when it applies, and a 4K project never
    goes near it."""
    for c in options.resolution_choices(3840, 1920, "360"):
        assert c["fits"], c


def test_vr180_is_under_the_cap_at_full_width():
    choices = options.resolution_choices(7680, 3840, "vr180")
    assert choices[0]["native"] and choices[0]["fits"]
    assert choices[0]["label"] == "7680×3840"


def test_choices_are_largest_first_and_never_upscale():
    """The CLI refuses to scale up, so offering it would be offering a crash."""
    choices = options.resolution_choices(5760, 2880, "360")
    widths = [c["width"] for c in choices]
    assert widths == sorted(widths, reverse=True)
    assert max(widths) == 5760


def test_the_probe_is_told_the_size_actually_being_encoded(qapp):
    """hevc_amf takes 5760x5760 and refuses 7680x7680, so a list built for the
    full size would offer an encoder that cannot run the chosen one."""
    ctrl = Controller()
    calls = []
    ctrl.encodersChanged.connect(lambda: calls.append(1))
    ctrl.probeEncoders(320, 240, "360", 0)
    assert _pump(qapp, lambda: bool(calls), timeout=300)
    ctrl.probeEncoders(320, 240, "360", 160)
    assert _pump(qapp, lambda: len(calls) > 1, timeout=300), \
        "a different output width is a different probe"


@pytest.mark.parametrize("extra,mode,expected_width", [
    ((), "360", 7680),
    (("outputWidth=5760",), "360", 5760),
    # The reported bug: pick a reduced size, then change format. The box read
    # "full size" and the render used 5760.
    (("outputWidth=5760", "outputMode=vr180"), "vr180", 5760),
    (("outputMode=vr180",), "vr180", 7680),
])
def test_the_resolution_box_shows_the_size_that_will_render(extra, mode,
                                                            expected_width):
    """A control showing one thing and doing another is the worst way for this
    to fail: there is nothing to see and nothing to suspect.

    ComboBox severs its own currentIndex binding as soon as anything writes to
    it, and replacing the model resets the index to 0 without changing the
    value the binding watched -- so a hand-written re-sync that listens for
    one signal goes stale on every other one.
    """
    source = str(Path(__file__).resolve().parent.parent / "input.mp4")
    if not Path(source).exists():
        pytest.skip("needs the 8K sample to have several sizes to choose from")

    items = _items(f"inputPath={source}", *extra)
    assert "resolutionBox" in items, "the row should be showing for an 8K source"
    shown = int(items["resolutionBox"][1])

    choices = options.resolution_choices(7680, 3840, mode)
    assert choices[shown]["width"] == expected_width, (
        f"box points at {choices[shown]['label']}, "
        f"but {expected_width} is what would render")
