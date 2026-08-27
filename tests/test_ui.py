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

from stereo360 import backends  # noqa: E402
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
    # A preset saved before the share slider existed carries the old boolean.
    # It meant an even split, which is now the default, so it emits nothing --
    # and that is the right answer rather than a lost setting.
    assert "--left-share" not in argv
    assert "--spatial-audio" in argv


def test_the_baseline_share_reaches_the_command_line():
    """An even split is the default and must emit nothing, while every other
    share must emit. A missing flag and an explicit default mean the same
    thing to the CLI, but only one of them says the control was left alone.
    An explicit share also has to beat the legacy boolean, so a preset
    carrying both is not ambiguous."""
    assert "--left-share" not in options.build_argv(dict(BASE))
    assert "--left-share" not in options.build_argv(dict(BASE, leftShare=0.5))
    for share, want in ((0.15, "0.15"), (0.0, "0"), (1.0, "1")):
        argv = options.build_argv(dict(BASE, leftShare=share))
        assert argv[argv.index("--left-share") + 1] == want
    both = options.build_argv(dict(BASE, splitBaseline=True, leftShare=0.15))
    assert both[both.index("--left-share") + 1] == "0.15"


def test_the_detail_split_is_on_unless_switched_off():
    """On by default, so the default emits nothing and only "off" speaks.

    Leaving the flag out asks for a radius scaled to the frame, which is not
    the same as passing 0 -- 0 turns the split off entirely."""
    assert "--detail-sigma" not in options.build_argv(dict(BASE))
    assert "--detail-sigma" not in options.build_argv(
        dict(BASE, sharedDetail=True))
    argv = options.build_argv(dict(BASE, sharedDetail=False))
    assert argv[argv.index("--detail-sigma") + 1] == "0"


def test_the_angular_correction_reaches_the_command_line():
    """Off by default, so an untouched slider must not emit the flag at all --
    a zero on the command line and no flag mean the same thing to the CLI, but
    only one of them says the setting was left alone."""
    assert "--face-angular-correction" not in options.build_argv(dict(BASE))
    argv = options.build_argv(dict(BASE, faceAngularCorrection=0.55))
    assert argv[argv.index("--face-angular-correction") + 1] == "0.55"


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
    v2 = dict(BASE, depthBackend="depth-anything")
    assert "--depth-model" not in options.build_argv(v2)
    assert "--depth-model" not in options.build_argv(dict(v2, depthModel="base"))


def test_choosing_small_actually_gets_small():
    """The trap this guards: while Small was the CLI default, its entry in
    DEPTH_MODELS was blank and "fell through" to whatever the backend picked.
    Moving the default to Base would then have turned a Small selection into
    Base silently -- the dropdown saying one thing and the run doing another,
    which is the exact bug class already fixed once for the temporal backend.
    """
    argv = options.build_argv(dict(BASE, depthBackend="depth-anything",
                                   depthModel="small"))
    assert (argv[argv.index("--depth-model") + 1]
            == "depth-anything/Depth-Anything-V2-Small-hf")


def test_the_recommended_backend_names_no_model_at_all():
    """The whole point of the empty sentinel: the CLI picks both the backend
    and its model from the kind of job, so naming either here would pin one
    half of a pair and let them drift."""
    argv = options.build_argv(dict(BASE, depthModel="large"))
    assert "--depth-backend" not in argv
    assert "--depth-model" not in argv


def test_depth_pro_takes_no_model():
    """It ships one checkpoint. Passing the shared Base/Large selection
    through would name a model that does not exist."""
    argv = options.build_argv(dict(BASE, depthBackend="depth-pro",
                                   depthModel="large"))
    assert argv[argv.index("--depth-backend") + 1] == "depth-pro"
    assert "--depth-model" not in argv


def test_v3_defaults_to_small_not_base():
    """V2 measured best at Base and V3 at Small, so a single shared default
    is wrong for one of them. Small is what V3 omits the flag for; Base is a
    413 MB download and has to be asked for."""
    v3 = dict(BASE, depthBackend="depth-anything-v3")
    assert "--depth-model" not in options.build_argv(dict(v3, depthModel="small"))
    argv = options.build_argv(dict(v3, depthModel="base"))
    assert argv[argv.index("--depth-model") + 1] == "base"


def test_the_ui_and_the_cli_agree_on_which_model_is_default():
    """`build_argv` omits the flag for the default, so if these two ever drift
    the UI would quietly run a model the user did not choose."""
    from stereo360.depth.depth_anything import DEFAULT_MODEL

    assert (options.DEPTH_MODELS[options.DEFAULT_DEPTH_MODEL]["depth-anything"]
            == DEFAULT_MODEL)


def test_the_ui_and_the_cli_agree_on_every_backends_default_model():
    """Same guard as the V2 one above, once per backend that has a default.
    `build_argv` omits the flag for these, so a drift means the UI silently
    runs a model nobody chose."""
    from stereo360.depth import depth_anything, depth_anything_v3

    assert options.DEFAULT_MODEL_FOR["depth-anything-v3"] \
        == depth_anything_v3.DEFAULT_VARIANT
    assert (options.DEPTH_MODELS[options.DEFAULT_MODEL_FOR["depth-anything"]]
            ["depth-anything"] == depth_anything.DEFAULT_MODEL)


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


@pytest.mark.parametrize("backend,onnx_row", [
    ("auto", False),
    ("depth-anything", False),
    ("video-depth-anything", False),
    ("depth-pro", False),
    ("depth-anything-v3", False),
    ("onnx", True),
])
def test_the_onnx_path_row_follows_the_backend(backend, onnx_row):
    """Backend and variant are one control now, so there is no separate model
    row to appear and disappear. The ONNX path still is its own row, and still
    only when it applies -- which is what teaches the user that a custom graph
    needs the ONNX backend selected."""
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
    assert "Model" not in rows, "the separate Model row should be gone"
    assert rows["Depth model"] is True, "the combined control is always shown"
    assert rows["ONNX model"] is onnx_row


def test_large_model_maps_to_its_hub_id():
    """Base is the default and so emits nothing; Large has to be asked for."""
    argv = options.build_argv(dict(BASE, depthBackend="depth-anything",
                                   depthModel="large"))
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
    assert set(entries) == set(backends.BACKENDS)
    assert all(isinstance(e["available"], bool) and e["detail"]
               for e in entries.values())


def test_probing_backends_does_not_narrate_what_is_missing(qapp):
    """A healthy install should not open with a list of its own shortcomings.

    Most people never want the ONNX export or a clone of the Video Depth
    Anything repo, so naming both as unavailable before anyone has looked at
    the depth settings reads as something being wrong. The dropdown already
    carries the reason under each entry it disables.
    """
    ctrl = Controller()
    lines = []
    ctrl.logged.connect(lambda lvl, txt: lines.append(txt))
    seen = []
    ctrl.backendsChanged.connect(lambda: seen.append(True))
    ctrl.probeBackends()

    assert _pump(qapp, lambda: bool(seen), timeout=180), "probe never returned"
    assert not [t for t in lines if "unavailable" in t.lower()], lines


def test_an_unusable_backend_explains_itself_when_it_is_picked(qapp):
    """Choosing a disabled entry snaps the selection back, and silently doing
    nothing is the one outcome that needs explaining."""
    ctrl = Controller()
    ctrl._backends = [
        {"name": "onnx", "available": False,
         "detail": "No exported model at models/depth_anything_v2_small.onnx"},
        {"name": "depth-pro", "available": True, "detail": "fine"},
    ]
    lines = []
    ctrl.logged.connect(lambda lvl, txt: lines.append((lvl, txt)))

    ctrl.explainBackend("onnx")
    assert len(lines) == 1
    level, text = lines[0]
    assert level == "warning"
    assert "onnx" in text and "No exported model" in text

    ctrl.explainBackend("depth-pro")
    assert len(lines) == 1, "an available backend has nothing to explain"
    ctrl.explainBackend("not-a-backend")
    assert len(lines) == 1, "an unknown name should not invent a reason"


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
    # An 8K 360 source now starts reduced, because full size at that shape is
    # 7680x7680 and does not decode on a headset. This is that default seen
    # end to end -- a real window, a real probe -- rather than the rule on
    # its own.
    ((), "360", 5760),
    (("outputWidth=5760",), "360", 5760),
    # The reported bug: pick a reduced size, then change format. The box read
    # "full size" and the render used 5760.
    (("outputWidth=5760", "outputMode=vr180"), "vr180", 5760),
    # VR180 keeps full size: 7680x3840 is under the cap and plays.
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


# ------------------------------------------------ detecting ambisonic audio


def _ambix_source(tmp_path: Path, channels: int) -> str:
    """A clip whose audio has `channels` channels and says nothing else."""
    import subprocess

    out = str(tmp_path / f"src{channels}.mp4")
    layout = {4: "4.0", 2: "stereo", 1: "mono", 6: "5.1"}.get(
        channels, f"{channels}C")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=128x64:rate=10:duration=1",
         "-f", "lavfi", "-i", f"anoisesrc=d=1:r=48000",
         "-map", "0:v", "-map", "1:a", "-ac", str(channels),
         "-c:a", "aac", "-c:v", "libx264", "-y", out],
        check=True, capture_output=True)
    return out


@pytest.mark.parametrize("channels,expected", [
    (4, True), (2, False), (1, False), (6, False),
])
def test_the_probe_reports_the_count_the_ui_decides_from(tmp_path, qapp,
                                                          channels, expected):
    """The UI ticks its spatial-audio switch from this number, so it has to
    survive the probe. 6 is 5.1 -- a perfect trap, since it is multichannel
    and emphatically not a soundfield."""
    from stereo360 import ambisonics

    src = _ambix_source(tmp_path, channels)
    ctrl = Controller()
    seen = []
    ctrl.sourceInfoChanged.connect(lambda: seen.append(dict(ctrl.sourceInfo)))
    ctrl.probeInput(src)
    assert _pump(qapp, lambda: any(s.get("width") for s in seen), timeout=120)

    info = next(s for s in seen if s.get("width"))
    assert info["audio_channels"] == channels
    assert (ambisonics.order_for_channels(info["audio_channels"])
            is not None) is expected


def test_the_switch_sets_itself_from_the_source(tmp_path):
    """Forgetting the flag is not a small mistake -- with a yaw it leaves every
    sound at the wrong bearing, and there is nothing to hear that says so. So
    the interface reads the file instead of waiting to be told."""
    four = _ambix_source(tmp_path, 4)
    two = _ambix_source(tmp_path, 2)

    props, _ = _dump(f"inputPath={four}")
    assert props["spatialAudio"] == "True", "4 channels should tick it"

    props, _ = _dump(f"inputPath={two}")
    assert props["spatialAudio"] == "False", "stereo must not"


def test_the_switch_says_it_decided_and_on_what_evidence(tmp_path):
    """With no declaration in the file, a channel count is all there is -- and
    it cannot tell ambiX from four separate microphones. So the one thing this
    hint must not do is sound certain."""
    props, _ = _dump(f"inputPath={_ambix_source(tmp_path, 4)}")
    hint = props["spatialAudioHint"]
    assert "Set from the file" in hint, "does not say it was decided for you"
    assert "4 audio channels" in hint, "does not give the evidence"
    assert "does not say so outright" in hint, "sounds more certain than it is"
    assert "untick" in hint.lower(), "does not say it can be wrong"


def test_the_cli_still_requires_the_flag():
    """Guessing is defensible in front of a switch someone can see. It is not
    defensible in a batch run nobody is watching, so the CLI is unchanged."""
    argv = options.build_argv(dict(BASE, sourceWidth=7680))
    assert "--spatial-audio" not in argv
    argv = options.build_argv(dict(BASE, spatialAudio=True))
    assert "--spatial-audio" in argv


def test_a_declared_file_is_not_hedged_about(tmp_path):
    """When the file says so itself there is nothing to guess, and the hint
    should not pretend otherwise. This is the signal VLC uses -- it reports
    "Channels: Ambisonics" for a track ffprobe describes as plain 4.0."""
    from stereo360 import spherical

    src = _ambix_source(tmp_path, 4)
    spherical.inject_spherical_metadata(src, stereo_mode="top-bottom",
                                        spatial_audio=True)
    props, _ = _dump(f"inputPath={src}")
    assert props["spatialAudio"] == "True"
    hint = props["spatialAudioHint"]
    assert "declares its audio as ambiX" in hint
    assert "Untick" not in hint, "no hedging when the file is explicit"


def test_the_probe_reports_what_the_file_declares(tmp_path, qapp):
    """ffprobe does not surface SA3D, so the tool reads the box itself."""
    from stereo360 import spherical

    src = _ambix_source(tmp_path, 4)
    ctrl = Controller()
    seen = []
    ctrl.sourceInfoChanged.connect(lambda: seen.append(dict(ctrl.sourceInfo)))
    ctrl.probeInput(src)
    assert _pump(qapp, lambda: any(s.get("width") for s in seen), timeout=120)
    assert next(s for s in seen if s.get("width"))["declares_ambix"] is False

    spherical.inject_spherical_metadata(src, stereo_mode="top-bottom",
                                        spatial_audio=True)
    ctrl2 = Controller()
    seen2 = []
    ctrl2.sourceInfoChanged.connect(lambda: seen2.append(dict(ctrl2.sourceInfo)))
    ctrl2.probeInput(src)
    assert _pump(qapp, lambda: any(s.get("width") for s in seen2), timeout=120)
    assert next(s for s in seen2 if s.get("width"))["declares_ambix"] is True


# ------------------------------------------------------------- photo mode


def _still(tmp_path: Path, name="src.jpg", w=256, h=128) -> str:
    import subprocess

    out = str(tmp_path / name)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc2=size={w}x{h}", "-frames:v", "1", "-y", out],
        check=True, capture_output=True)
    return out


def test_the_ui_and_the_core_agree_on_what_a_still_is():
    """Mirrored rather than imported, for the same reason as output_size: the
    window's process must never pull numpy in. Drift would mean the UI
    building a video command for a file the CLI treats as a photo."""
    from stereo360 import ffmpeg_io

    assert options.IMAGE_SUFFIXES == ffmpeg_io.IMAGE_SUFFIXES
    assert options.VIDEO_SUFFIXES == ffmpeg_io.VIDEO_SUFFIXES
    for path in ("a.jpg", "a.JPG", "a.png", "a.mp4", "a", "a.jpeg.mp4",
                 "a.avif", "a.heic"):
        assert options.is_image(path) == ffmpeg_io.is_image_path(path), path


def test_the_open_dialog_offers_every_format_the_tool_accepts():
    """The bug this exists for: the dialog listed five video extensions and
    nothing else, so a photo could only be opened by typing its name in. A
    filter list kept by hand is a second answer to "what does this open", and
    it was the wrong one for three commits."""
    combined = options.open_filters()[0]
    for suffix in options.IMAGE_SUFFIXES + options.VIDEO_SUFFIXES:
        assert "*" + suffix in combined, suffix


def test_the_open_dialog_never_traps_anyone():
    """Whatever the filters say, an unlisted extension must still be
    reachable -- ffmpeg sniffs content and does not care about names."""
    assert "All files (*)" in options.open_filters()


@pytest.mark.parametrize("photo,expected", [(True, ".jpg"), (False, ".mp4")])
def test_the_save_dialog_leads_with_the_right_format(photo, expected):
    """A photo job offered "MP4 video (*.mp4)" and a default suffix of mp4,
    which names the output of a JPEG conversion out.mp4."""
    assert expected in options.save_filters(photo)[0]


def test_the_ui_and_the_cli_agree_on_the_photo_tile_count():
    """Mirrored rather than imported, like the rest. If they drift, the spin
    box shows one number and the render uses another -- which this control
    would have done the moment the CLI default changed."""
    from stereo360 import cli

    assert options.PHOTO_DEPTH_TILES == cli.PHOTO_DEPTH_TILES


@pytest.mark.parametrize("path,tiles,expected", [
    ("in.mp4", 1, None),        # video default -- omitted
    ("in.mp4", 3, "3"),         # video, not the default -- emitted
    ("pano.jpg", 1, None),      # photo default -- omitted
    ("pano.jpg", 3, "3"),       # photo, tiling asked for -- emitted
    ("pano.jpg", 4, "4"),
])
def test_the_tiles_flag_is_emitted_against_the_right_default(path, tiles,
                                                             expected):
    argv = options.build_argv({"input": path, "output": "out.jpg",
                               "depthTiles": tiles})
    if expected is None:
        assert "--depth-tiles" not in argv
    else:
        assert argv[argv.index("--depth-tiles") + 1] == expected


def test_the_photo_default_is_compared_against_and_not_assumed(monkeypatch):
    """The two job kinds want the same tile count today, which makes the case
    above unable to tell a real comparison from a hardcoded 1. So move the
    photo default and check the flag follows: it has to be *read*, not
    assumed. It was 3 until the depth models changed, and could move again."""
    monkeypatch.setattr(options, "PHOTO_DEPTH_TILES", 3)
    photo = {"input": "pano.jpg", "output": "out.jpg"}

    assert "--depth-tiles" not in options.build_argv(dict(photo, depthTiles=3))
    argv = options.build_argv(dict(photo, depthTiles=1))
    assert argv[argv.index("--depth-tiles") + 1] == "1"


# ------------------------------------------------ where the size picker starts

def test_a_big_360_video_starts_at_a_size_that_plays():
    """Full size is the better master, but it is black on a Quest 3 at 8K --
    so defaulting to it means the commonest outcome is a file that will not
    play, found out after a three-hour render."""
    assert options.default_output_width(7680, 3840, "360") == 5760
    assert options.default_output_width(11904, 5952, "360") == 5760


def test_full_size_is_still_offered():
    """Reduced by default, not removed. The ask was a default, and uploading
    still wants the full-size master."""
    choices = options.resolution_choices(7680, 3840, "360")
    native = [c for c in choices if c["native"]]
    assert native and native[0]["width"] == 7680
    assert native[0]["label"] == "7680×7680"


@pytest.mark.parametrize("width,height,mode,photo,why", [
    (7680, 3840, "vr180", False, "VR180 at 8K is 29.5 MP and plays"),
    (7680, 3840, "360", True, "the cap belongs to the video decoder"),
    (5760, 2880, "360", False, "already at the safe width"),
    (3840, 1920, "360", False, "a 4K source cannot reach the cap"),
])
def test_the_default_is_left_alone_where_the_cap_does_not_bite(
        width, height, mode, photo, why):
    """0 means the source's own width. Reducing where nothing required it
    would cost resolution for no reason."""
    assert options.default_output_width(width, height, mode, photo) == 0, why


def test_the_chosen_width_reaches_the_command_line():
    """The reduction has to be explicit in the argv, not implied. The CLI
    default is still full size, so an omitted flag renders 7680x7680 while
    the box says 5760 -- which is the failure this control has had before."""
    argv = options.build_argv({"input": "in.mp4", "output": "out.mp4",
                               "outputWidth": 5760, "sourceWidth": 7680})
    assert argv[argv.index("--output-width") + 1] == "5760"


# ------------------------------------------------- the output box goes stale

def test_a_photos_name_does_not_survive_into_a_video_job():
    """The reported bug. Opening a photo and then a video left
    `pano_360_TB.jpg` in Output, and the box only ever filled when empty."""
    out = options.resolve_output("C:/x/pano_360_TB.jpg",
                                 "C:/x/pano_360_TB.jpg",
                                 "C:/x/clip_stereo.mp4", input_is_image=False)
    assert out == "C:/x/clip_stereo.mp4"


def test_a_hand_picked_name_that_fits_the_job_is_left_alone():
    """Only names this program proposed are its to revise."""
    out = options.resolve_output("C:/mine/my_edit.mkv", "C:/x/clip_stereo.mp4",
                                 "C:/x/other_stereo.mp4", input_is_image=False)
    assert out == "C:/mine/my_edit.mkv"


def test_a_hand_picked_name_of_the_wrong_kind_is_replaced_anyway():
    """Deliberately overriding someone's choice, because keeping it preserves
    a render that cannot succeed -- a video muxed into a .jpg, or a photo
    written to a .mkv."""
    assert options.resolve_output(
        "C:/mine/my_edit.mkv", "", "C:/x/pano_360_TB.jpg",
        input_is_image=True) == "C:/x/pano_360_TB.jpg"
    assert options.resolve_output(
        "C:/mine/my_pic.jpg", "", "C:/x/clip_stereo.mp4",
        input_is_image=False) == "C:/x/clip_stereo.mp4"


def test_an_empty_box_is_filled():
    assert options.resolve_output("", "", "C:/x/clip_stereo.mp4",
                                  input_is_image=False) == "C:/x/clip_stereo.mp4"


def test_the_output_mode_reaches_python_from_qml(qapp):
    """`@Slot(str, result=str)` on a two-argument method does not fail when
    QML passes both -- Qt drops the extra and the Python default applies. So
    `suggestOutput(url, "vr180")` returned a `_360_TB` name, and that token is
    what the Quest gallery reads to decide the layout.

    Asserted against the meta-object, since that is what QML resolves against;
    calling the method from Python passes either way and proves nothing.
    """
    mo = Controller().metaObject()
    assert mo.indexOfMethod("suggestOutput(QString,QString)") >= 0, \
        "QML calls this with two arguments"
    assert mo.indexOfMethod("suggestOutput(QString)") >= 0, \
        "and the one-argument form must keep working"


def test_every_slot_accepts_as_many_arguments_as_it_takes(qapp):
    """The general form of the bug above, which is silent in both directions:
    Qt drops surplus arguments rather than raising, so the only symptom is a
    parameter mysteriously stuck at its default."""
    import inspect

    mo = Controller().metaObject()
    wrong = []
    for name, fn in inspect.getmembers(Controller, inspect.isfunction):
        params = [p for p in inspect.signature(fn).parameters if p != "self"]
        if not params or not any(
                mo.method(i).name().data().decode() == name
                for i in range(mo.methodCount())):
            continue
        registered = {mo.method(i).methodSignature().data().decode()
                      for i in range(mo.methodCount())
                      if mo.method(i).name().data().decode() == name}
        widest = max(s.count(",") + 1 if "()" not in s else 0
                     for s in registered)
        if widest < len(params):
            wrong.append(f"{name}: takes {len(params)} args "
                         f"({', '.join(params)}), widest slot accepts "
                         f"{widest} -- {sorted(registered)}")
    assert not wrong, "slots that silently drop arguments:\n" + "\n".join(wrong)


def test_a_photo_command_drops_the_flags_the_cli_would_refuse():
    """Not cosmetic. The CLI *refuses* --max-frames, --start-frame and
    --spatial-audio for an image rather than ignoring them, so emitting one --
    a spatial-audio switch left on from the last video, say -- would fail
    every photo conversion."""
    argv = options.build_argv({
        "input": "photo.jpg", "output": "out.jpg",
        "spatialAudio": True, "maxFrames": 30, "startFrame": 5})
    for flag in ("--spatial-audio", "--max-frames", "--start-frame"):
        assert flag not in argv, flag


def test_a_photo_command_drops_the_encoder_settings():
    """CRF and codec describe a video encoder. A photo is written by OpenCV at
    settings the pipeline chooses, so passing them would imply an effect."""
    argv = options.build_argv(dict(BASE, input="photo.jpg", output="out.jpg",
                                   quality="archival"))
    assert not {"--crf", "--codec", "--preset", "--bitdepth"} & set(argv)


def test_a_photo_command_keeps_what_still_applies():
    """Format, direction and the 3D controls all mean exactly what they mean
    for video."""
    argv = options.build_argv({
        "input": "photo.jpg", "output": "out.jpg", "outputMode": "vr180",
        "yaw": 30, "strength": 1.4, "sourceWidth": 7680, "outputWidth": 5760})
    assert argv[argv.index("--output-mode") + 1] == "vr180"
    assert argv[argv.index("--yaw") + 1] == "30"
    assert argv[argv.index("--strength") + 1] == "1.4"
    assert argv[argv.index("--output-width") + 1] == "5760"


def test_a_video_command_is_unchanged():
    """The photo branch must not leak into the path everything else uses."""
    argv = options.build_argv(dict(BASE, quality="archival", maxFrames=30,
                                   spatialAudio=True))
    assert "--spatial-audio" in argv and "--max-frames" in argv
    assert argv[argv.index("--crf") + 1] == "13"


def test_a_preview_of_a_video_is_not_treated_as_a_photo():
    """A preview writes a .png, but the *input* is a video, so it is still a
    preview and keeps its own flags."""
    argv = options.build_argv(dict(BASE), preview_frame=3,
                              preview_output="p.png")
    assert "--preview-frame" in argv


@pytest.mark.parametrize("mode,expected", [
    ("360", "holiday_360_TB.jpg"),
    ("vr180", "holiday_180x180_3dh.jpg"),
])
def test_a_photo_gets_an_output_name_a_player_can_read(qapp, mode, expected):
    """It knows the format and the convention; the person should not have to.
    Always .jpg whatever went in, since that is what headsets read."""
    ctrl = Controller()
    got = ctrl.suggestOutput("file:///photos/holiday.png", mode)
    assert Path(got).name == expected


def test_a_video_still_gets_the_old_name(qapp):
    ctrl = Controller()
    got = ctrl.suggestOutput("file:///clips/trip.mp4", "360")
    assert Path(got).name == "trip_stereo.mp4"


def test_the_controller_can_tell_a_photo_from_a_video(qapp):
    ctrl = Controller()
    assert ctrl.isImage("file:///a/b.jpg") is True
    assert ctrl.isImage("file:///a/b.mp4") is False


def test_photo_mode_hides_what_cannot_apply(tmp_path):
    """Showing a control that does nothing implies it does something."""
    props, rows = _dump(f"inputPath={_still(tmp_path)}")
    assert props["photoMode"] == "True"
    for label in ("Preset", "Encoder", "Start at", "Limit", "Spatial audio",
                  "Chunk size", "Chunk overlap", "Temporal fill"):
        assert rows[label] is False, f"{label} should be hidden for a photo"


def test_photo_mode_keeps_what_does_apply(tmp_path):
    props, rows = _dump(f"inputPath={_still(tmp_path)}")
    for label in ("Format", "Strength", "Gradient limit", "Depth model",
                  "Depth tiles", "Device"):
        assert rows[label] is True, f"{label} should still be shown"


def test_one_control_covers_every_backend(tmp_path):
    """Backend and variant used to be two rows, and the second vanished for
    the backends with nothing to choose -- Recommended picks its own, Depth
    Pro ships one checkpoint. One list of concrete pairs has no such gap, so
    the control is there whatever is selected and never offers a dropdown
    that changes nothing."""
    still = _still(tmp_path)
    for backend in ("", "depth-pro", "depth-anything-v3", "depth-anything",
                    "video-depth-anything", "onnx"):
        _, rows = _dump(f"inputPath={still}", f"depthBackend={backend}")
        assert rows["Depth model"] is True, f"{backend!r} lost the control"
        assert "Model" not in rows, f"{backend!r} still has the old Model row"
        assert "Backend" not in rows, f"{backend!r} still has the old Backend row"


def test_a_video_still_shows_the_video_controls(tmp_path):
    """The hiding must be scoped to photo mode, not applied everywhere."""
    from test_end_to_end import make_test_video

    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=128, h=64, frames=4, with_audio=False)
    props, rows = _dump(f"inputPath={src}")
    assert props["photoMode"] == "False"
    for label in ("Preset", "Encoder", "Start at", "Spatial audio"):
        assert rows[label] is True, label


def test_the_panel_shows_the_result_once_it_exists(qapp, tmp_path):
    """For a photo the panel is not a preview: the converted image *is* the
    deliverable, so it replaces the source in the same panel."""
    src = _still(tmp_path)
    dst = str(tmp_path / "out.jpg")

    ctrl = Controller()
    done = []
    ctrl.completed.connect(lambda *a: done.append(a))
    ctrl.convert({"input": src, "output": dst, "faceSizeAuto": False,
                  "faceSize": 64})

    assert _pump(qapp, lambda: bool(done), timeout=600), "never finished"
    ok, cancelled, output = done[0]
    assert ok and not cancelled
    assert Path(dst).exists()
    assert ctrl.previewSource.startswith("file:")
    assert "out.jpg" in ctrl.previewSource, "the panel should show the result"
    assert "?t=" in ctrl.previewSource, "needs a cache-buster to refresh"


def test_a_video_conversion_does_not_hijack_the_panel(qapp, tmp_path):
    """An MP4 is not something the panel can show, and claiming otherwise
    would leave it displaying the previous job's picture."""
    from test_end_to_end import make_test_video

    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=4, with_audio=False)

    ctrl = Controller()
    done = []
    ctrl.completed.connect(lambda *a: done.append(a))
    ctrl.convert({"input": src, "output": dst, "faceSizeAuto": False,
                  "faceSize": 32, "maxFrames": 2})
    assert _pump(qapp, lambda: bool(done), timeout=600)
    assert ctrl.previewSource == "", "a video render is not a picture"


# --------------------------------------- showing a finished photo in the panel


def test_a_full_size_photo_is_too_big_for_qt_to_decode(tmp_path):
    """Why the preview panel caps its decode, demonstrated rather than
    asserted from memory.

    A finished 360 photo is the full-size deliverable -- 11904x11904 from an
    Insta360 X5 -- and QImageReader refuses anything whose decoded form
    exceeds its allocation limit. 11904 squared at 4 bytes is 567 MB against a
    256 MB default, so the load fails, the panel stays empty, and the run that
    produced a perfectly good file looks like it produced nothing.

    Reproduced at 1/256th the size by moving the limit rather than by building
    a 567 MB image, so this costs milliseconds and does not depend on the
    machine having the memory to fail honestly.

    JPEG deliberately, and not only because it is what a photo job writes:
    libjpeg can decode straight to a reduced size, so the scaled read never
    allocates the full frame. PNG cannot, and a scaled read of one still
    allocates it all and still fails -- which is why the panel also reports a
    failed decode rather than relying on the cap always working.
    """
    import cv2
    import numpy as np
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QImageReader

    src = tmp_path / "big.jpg"
    cv2.imwrite(str(src), np.zeros((2048, 2048, 3), np.uint8))

    original = QImageReader.allocationLimit()
    try:
        QImageReader.setAllocationLimit(1)          # MB; 2048^2 x4 = 16 MB
        assert QImageReader(str(src)).read().isNull(), \
            "expected Qt to refuse the unscaled decode"

        capped = QImageReader(str(src))
        capped.setScaledSize(QSize(256, 256))
        assert not capped.read().isNull(), \
            "a scaled decode is what sourceSize asks Qt for, and it must work"
    finally:
        QImageReader.setAllocationLimit(original)


def test_the_preview_panel_caps_its_decode():
    """The fix for the above, pinned. Without sourceSize the Image element
    asks for the full decode and silently fails; this is not visible in any
    screenshot test, because the symptom *is* an empty panel."""
    qml = (Path(__file__).resolve().parent.parent
           / "stereo360_ui" / "qml" / "Main.qml").read_text(encoding="utf-8")
    block = qml[qml.index("id: previewImage"):]
    block = block[:block.index("\n                    }")]
    assert "sourceSize" in block, "previewImage must cap its decode size"
    assert "Image.Error" in block, "a failed decode must not be silent"


def test_the_controller_explains_a_preview_that_cannot_be_shown(qapp):
    """An empty panel reads as "no output", and the next move is to run the
    whole conversion again. The file is fine; say so."""
    ctrl = Controller()
    seen = []
    ctrl.logged.connect(lambda level, text: seen.append((level, text)))
    ctrl.reportPreviewFailure("file:///c:/x/out_360_TB.jpg?t=123.4")

    assert seen and seen[0][0] == "warning"
    assert "out_360_TB.jpg" in seen[0][1]
    assert "?t=" not in seen[0][1], "the cache-buster is noise to a reader"
    assert "written and fine" in seen[0][1]


# ------------------------------------------- what the window says while waiting


def test_model_preparation_reaches_the_status_line(qapp):
    """The 1.9 GB stills model downloads before frame one exists, and the
    progress bar is driven by frames. Without this the window says
    "Starting..." with an empty detail for the whole download, which is
    indistinguishable from a hang."""
    ctrl = Controller()
    ctrl._busy = True
    ctrl._set_status("Starting…")

    ctrl._runner.staged.emit("Loading Depth Pro 'apple/DepthPro-hf' on device "
                             "'auto' (downloads ~1.9 GB on first use)...")

    assert ctrl.status == "Starting…", "the heading was already right"
    assert "1.9 GB" in ctrl.detail


def test_only_the_backend_events_become_status(qapp):
    """Keyed off the structured `backend` field rather than the wording, so
    ordinary info lines stay in the log where they belong."""
    from stereo360_ui.runner import Runner

    runner = Runner()
    staged, logged = [], []
    runner.staged.connect(staged.append)
    runner.logged.connect(lambda level, text: logged.append(text))

    runner._dispatch({"type": "info", "message": "backend line",
                      "backend": "depth-pro"})
    runner._dispatch({"type": "info", "message": "ordinary line"})

    assert staged == ["backend line"]
    assert logged == ["backend line", "ordinary line"]


def test_pole_compensation_is_omitted_at_its_default():
    """1 is off, and off is what the CLI already does, so the command stays
    short. Same rule as every other control here."""
    assert "--pole-compensation" not in options.build_argv(
        dict(BASE, poleCompensation=1.0))


def test_pole_compensation_reaches_the_command():
    argv = options.build_argv(dict(BASE, poleCompensation=3.0))
    assert argv[argv.index("--pole-compensation") + 1] == "3"
