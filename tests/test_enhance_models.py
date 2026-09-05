"""The enhancement models: where they live, and who is told when they do not."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_every_spec_points_at_the_path_its_loader_reads():
    """The fetcher and the loader must not be able to disagree about where a
    model lives. Both read the one registry, so a move cannot leave the
    fetcher writing somewhere nothing looks."""
    from stereo360 import enhance_models as em
    from stereo360 import interpolate, upscalers

    want = {v.code: v.path for v in upscalers.VARIANTS}
    want["rife"] = interpolate.DEFAULT_MODEL
    assert {s.key: s.path for s in em.SPECS} == want


def test_every_spec_has_a_script_that_exists():
    """`fetch` shells out to these by path, so a rename would turn the
    download button into a no-op that reports success."""
    from stereo360 import enhance_models as em

    for s in em.SPECS:
        assert os.path.exists(os.path.join(em.repo_root(), s.script)), s.key


def test_paths_resolve_beside_the_package_not_the_caller(tmp_path, monkeypatch):
    """The installer runs from a temporary directory and the interface from
    wherever it was launched. A relative `models/...` would send the download
    somewhere neither of them reads back."""
    from stereo360 import enhance_models as em

    monkeypatch.chdir(tmp_path)
    for s in em.SPECS:
        assert os.path.isabs(s.full_path)
        assert s.full_path.startswith(em.repo_root())


def test_a_missing_script_fails_that_model_and_not_the_others(monkeypatch):
    """A partial result is the normal case, not an error: Real-ESRGAN is an
    export rather than a copy and can fail on a machine the other two suit."""
    from stereo360 import enhance_models as em

    monkeypatch.setattr(em, "SPECS", (
        em.Spec("gone", "Absent", os.path.join("models", "nope.bin"),
                os.path.join("scripts", "no_such_script.py"), 1, False,
                "upscale"),))
    monkeypatch.setattr(em, "BY_KEY", {s.key: s for s in em.SPECS})
    got = em.fetch()
    assert got["gone"]["ok"] is False
    assert "missing from this build" in got["gone"]["detail"]


def test_fetch_never_raises_when_the_script_fails(monkeypatch, tmp_path):
    """A download button that throws is worse than one that reports a
    failure, because the traceback lands in a log the user is not reading."""
    from stereo360 import enhance_models as em

    script = tmp_path / "boom.py"
    script.write_text("import sys; sys.exit(3)\n")
    monkeypatch.setattr(em, "repo_root", lambda: str(tmp_path))
    monkeypatch.setattr(em, "SPECS", (
        em.Spec("boom", "Boom", os.path.join("models", "nope.bin"),
                "boom.py", 1, False, "upscale"),))
    monkeypatch.setattr(em, "BY_KEY", {s.key: s for s in em.SPECS})
    got = em.fetch()
    assert got["boom"]["ok"] is False


def test_an_unknown_name_is_refused_rather_than_silently_ignored():
    """Asking for a model that does not exist has to fail loudly: quietly
    fetching nothing looks exactly like success."""
    p = subprocess.run([sys.executable, "-m", "stereo360",
                        "--fetch-enhancers", "nonsense"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode == 2
    assert "Unknown model" in p.stderr


@pytest.mark.parametrize("width,offered", [(3840, True), (7680, False)])
def test_the_width_gate_does_not_depend_on_topaz(monkeypatch, width, offered):
    """The regression this exists for. `offered` is the judgement that a
    source is below 8K and nothing else, but the no-Topaz branch used to
    hard-code it False -- and the interface gates the whole Enhance panel on
    it. So a machine without Topaz could never be shown the free shader or
    Real-ESRGAN however well installed they were, which is invisible while
    Topaz is the only upscaler and a bug the moment it is not."""
    from stereo360 import upscale

    monkeypatch.setattr(upscale, "find", lambda: None)
    got = upscale.describe(width)
    assert got["available"] is False        # Topaz is still reported absent
    assert got["offered"] is offered       # but the width still gets an answer


def test_the_probe_says_what_is_missing_and_what_it_costs():
    """The panel cannot offer a download it is not told about, and the reason
    a model is absent used to reach nobody -- it sat in this payload while the
    card hid itself."""
    p = subprocess.run([sys.executable, "-m", "stereo360",
                        "--probe-upscalers", "--probe-width", "3840",
                        "--probe-fps", "30"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0
    got = json.loads(p.stdout)
    assert "fetchable" in got and "fetchable_mb" in got
    for entry in got["fetchable"]:
        assert {"key", "label", "mb", "needs_torch"} <= set(entry)


# ------------------------------------------------- which downloads to offer
#
# Upscaling and interpolation are judged separately because they answer
# different questions about a source. Merging them hid the whole panel for an
# 8K 30 fps video: too wide to gain anything from upscaling, and squarely the
# case frame interpolation exists for.

_FETCH = ('"fetchable": ['
          '{"key":"fsrcnnx","label":"FSRCNNX upscaling shader",'
          '"kind":"upscale","mb":0.1,"needs_torch":false},'
          '{"key":"esrgan","label":"Real-ESRGAN photo upscaler",'
          '"kind":"upscale","mb":4.9,"needs_torch":true},'
          '{"key":"rife","label":"RIFE frame interpolation",'
          '"kind":"interpolate","mb":21.0,"needs_torch":false}]')


def _panel(offered, fps_offered):
    """(visible, offered keys) for a source with these two judgements."""
    import subprocess

    payload = ("topaz={"
               f'"available": false, "offered": {str(offered).lower()}, '
               f'"fps_offered": {str(fps_offered).lower()}, '
               '"needs_login": false, "interpolate_offered": false, '
               '"shader": {"available": false}, '
               '"photo_model": {"available": false}, '
               '"models": [], "interpolators": [], ' + _FETCH + "}")
    p = subprocess.run([sys.executable, "-m", "stereo360_ui", "--selftest",
                        "--dump-rows", "--set", payload],
                       capture_output=True, text=True, timeout=300, cwd=ROOT)
    if p.returncode != 0 and "display" in p.stderr:
        pytest.skip("no display available")
    assert p.returncode == 0, p.stderr
    visible, keys = False, ""
    for line in p.stdout.splitlines():
        part = line.split("\t")
        if part[0] == "ITEM" and part[1] == "enhanceFetch":
            visible = part[2] == "True"
        elif part[0] == "PROP" and part[1] == "fetchKeys":
            keys = part[2] if len(part) > 2 else ""
    return visible, keys


def test_a_source_that_suits_both_is_offered_both():
    visible, keys = _panel(offered=True, fps_offered=True)
    assert visible
    assert set(keys.split(",")) == {"fsrcnnx", "esrgan", "rife"}


def test_an_8k_30fps_source_is_offered_interpolation_only():
    """The bug this exists for. Both halves were gated on the *width*
    judgement, so an 8K source hid the entire panel -- including the
    interpolator, whose usefulness has nothing to do with width."""
    visible, keys = _panel(offered=False, fps_offered=True)
    assert visible
    assert keys == "rife"


def test_a_source_that_suits_neither_is_offered_nothing():
    """Not an empty card: no card. There is nothing to say."""
    visible, keys = _panel(offered=False, fps_offered=False)
    assert not visible
    assert keys == ""


def test_a_4k_60fps_source_is_offered_upscaling_only():
    """The mirror image, which the same merged gate would also have got
    wrong -- in the other direction, by offering an interpolator to a source
    already fast enough."""
    visible, keys = _panel(offered=True, fps_offered=False)
    assert visible
    assert set(keys.split(",")) == {"fsrcnnx", "esrgan"}


def test_the_probe_judges_frame_rate_without_asking_what_is_installed():
    """`interpolate_offered` answers "can this machine interpolate this now",
    which is false on a machine with no RIFE -- exactly the machine that has
    to be offered the download. The panel needs the source judged on its own."""
    out = {}
    for fps in (30, 60):
        p = subprocess.run([sys.executable, "-m", "stereo360",
                            "--probe-upscalers", "--probe-width", "7680",
                            "--probe-fps", str(fps)],
                           cwd=ROOT, capture_output=True, text=True,
                           timeout=600)
        assert p.returncode == 0
        out[fps] = json.loads(p.stdout)
    assert out[30]["fps_offered"] is True
    assert out[60]["fps_offered"] is False
    assert out[30]["offered"] is False          # 8K: too wide to upscale


def test_every_spec_declares_which_judgement_it_answers_to():
    """A new model with no `kind`, or a typo in one, would be filtered into
    neither half and silently never offered."""
    from stereo360 import enhance_models as em

    assert {s.kind for s in em.SPECS} <= {"upscale", "interpolate"}
    assert all(s.kind for s in em.SPECS)


# ------------------------------------------------ rendering at delivery size

def test_the_supersample_flag_only_appears_with_a_chosen_width():
    """At full size there is nothing to render smaller than, so the flag would
    promise a saving it cannot make."""
    from stereo360_ui import options

    base = dict(input="a.mp4", output="b.mp4", sourceWidth=7680)
    chosen = options.build_argv({**base, "outputWidth": 5760,
                                 "supersample": False})
    assert "--no-supersample" in chosen and "--output-width" in chosen
    full = options.build_argv({**base, "outputWidth": 0, "supersample": False})
    assert "--no-supersample" not in full


def test_supersampling_stays_the_default():
    """It is the better picture and the existing behaviour; the faster path is
    a choice, not a surprise."""
    from stereo360_ui import options

    base = dict(input="a.mp4", output="b.mp4", sourceWidth=7680,
                outputWidth=5760)
    assert "--no-supersample" not in options.build_argv(base)
    assert options._DEFAULTS["supersample"] is True


def test_an_explicit_face_size_survives_rendering_small(monkeypatch):
    """--face-size is a decision about the depth model, not about delivery.
    Rescaling it because the frame got smaller would answer a question the
    user already answered."""
    import inspect

    from stereo360 import pipeline

    src = inspect.getsource(pipeline.convert)
    assert "face_size_asked is None" in src, \
        "the auto face size must only be recomputed when it was auto"


# --------------------------------------------------- stopping a slow render

def test_cancel_is_checked_inside_the_chunk_not_only_between_chunks():
    """A chunk is one check, then depth for every frame in it, then a warp for
    every frame and every eye, and only then the first write. On a laptop that
    is over a hundred seconds with nothing looking at Stop -- while the
    interface gives up after a much shorter wait and kills the process. A
    killed run loses the spherical metadata, which is written after the last
    frame, so the output plays flat in a headset."""
    import inspect
    import re

    from stereo360 import pipeline

    src = inspect.getsource(pipeline._convert_chunked)
    warp_loop = src[src.index("dn_pres, rights, holes"):src.index("temporal_fill and")]
    assert "sink.check()" in warp_loop, \
        "the warp loop must notice a cancel between frames"
    assert len(re.findall(r"sink\.check\(\)", src)) >= 3, \
        "expected checks before the depth pass and inside both inner loops"


def test_the_kill_backstop_outlasts_an_uninterruptible_depth_pass():
    """The backstop is for a wedged child, not a slow one. A chunk's depth
    pass cannot be interrupted, so the wait has to exceed it or a healthy
    render gets killed for being slow."""
    from stereo360_ui import runner

    assert runner.KILL_AFTER_MS >= 120_000


# ------------------------------------------------------- the upscaler table

def test_the_two_defaults_are_different_models():
    """A still is judged on one frame, so the sharpest wins. A video is
    judged on how little the invented detail moves between frames, and the
    model that wins the first test loses the second by a distance -- Siax
    reads best frozen and amplifies a one-level change into four. One
    default cannot serve both."""
    from stereo360 import upscalers

    assert upscalers.VIDEO_DEFAULT != upscalers.PHOTO_DEFAULT
    assert upscalers.BY_CODE[upscalers.VIDEO_DEFAULT].stills_only is False
    assert upscalers.BY_CODE[upscalers.PHOTO_DEFAULT].stills_only is True


def test_the_video_default_is_a_shader():
    """A pre-pass runs once per frame for the length of the film, so the
    default has to be the thing that cannot crawl and costs a fraction of a
    second, not the thing that looks best on one frame."""
    from stereo360 import upscalers

    assert upscalers.BY_CODE[upscalers.VIDEO_DEFAULT].kind == "shader"


def test_every_variant_is_reachable_by_its_code():
    """The dispatch is one lookup now. It used to be a comparison per model,
    which is how a model ends up listed and unreachable."""
    from stereo360 import upscalers

    for v in upscalers.VARIANTS:
        assert upscalers.get(v.code) is v
        assert upscalers.get(v.code.upper()) is v, "codes come from a UI"
    assert upscalers.get("nonsense") is None
    assert upscalers.get(None) is None


def test_a_stills_only_model_is_never_a_video_fallback():
    """`for_job` falls forward when the preferred model is missing, and the
    one thing it must not do is fall forward onto the model that crawls."""
    from stereo360 import upscalers

    got = upscalers.for_job(is_photo=False)
    assert got is None or not got.stills_only


def test_the_old_shader_is_named_for_its_size():
    """It was "FSRCNNX" while it was the only one. With two, an unqualified
    name is the one that gets picked by accident."""
    from stereo360 import upscalers

    assert upscalers.BY_CODE["fsrcnnx8"].name == "FSRCNNX 8"
    assert upscalers.BY_CODE["fsrcnnx16"].name == "FSRCNNX 16"


def test_native_scale_is_preferred_over_a_bigger_graph():
    """A 4x graph asked for 2x computes sixteen times the source pixels to
    hand back four. The same ESRGAN architecture measured 129 s a frame at
    4x and 29 s at 2x, so everything that has a 2x checkpoint uses it."""
    from stereo360 import upscalers

    four = [v for v in upscalers.VARIANTS if v.scale != 2]
    assert [v.code for v in four] == ["siax"], \
        "only Siax has no 2x checkpoint"


# ------------------------------------------------------- audio through a pass

@pytest.mark.parametrize("module,why", [
    ("fsrcnnx", "the shader pass"),
    ("upscale", "the Topaz pass"),
    ("esrgan", "the ONNX pass"),
    ("interpolate", "the RIFE pass"),
])
def test_a_pre_pass_never_re_encodes_the_audio(module, why):
    """A pre-pass is a video stage, and the audio is a passenger.

    Left unmentioned, ffmpeg maps the source's track and encodes it with
    whatever the container defaults to. The working file is Matroska, whose
    default is Vorbis -- so an AAC source came out of the render as
    low-bitrate Vorbis, converted by a stage that never mentions audio, and
    the converter then copied that into the output believing it was the
    original. Three of the four passes had this; RIFE did not, and its
    `-c:a copy` is the shape the others now follow."""
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module(f"stereo360.{module}"))
    assert '"-c:a", "copy"' in src or '"-c:a","copy"' in src, \
        f"{why} does not say what to do with audio, so ffmpeg decides"


def test_the_onnx_pass_takes_audio_from_the_source_it_was_given():
    """Its frames arrive on stdin, so there is no audio on input 0 to carry.
    The source has to be opened a second time or the track is dropped
    silently -- and `?` on the map, because a clip with no audio is ordinary
    and must not fail."""
    import inspect

    from stereo360 import esrgan

    src = inspect.getsource(esrgan.run_video)
    assert '"-map", "1:a?"' in src
    assert '"-map", "0:v"' in src


def test_a_missing_export_dependency_is_reported_once(monkeypatch):
    """Five graphs are exported from checkpoints and all five need the same
    loader. Without it the run printed the same cryptic tail line five times
    -- `    pip install spandrel`, the last line of another script's stderr --
    and left the reader to notice they were one sentence."""
    from stereo360 import enhance_models as em

    monkeypatch.setattr(em, "missing_export_dependency",
                        lambda: "spandrel is not installed")
    monkeypatch.setattr(em, "_can_export", lambda: False)
    monkeypatch.setattr(em.Spec, "present", lambda self: False)
    said = []
    got = em.fetch(keys=["span", "spanldl", "siax"], on_line=said.append)

    assert sum("spandrel is not installed" in m for m in said) == 1
    assert any("pip install spandrel" in m for m in said)
    for key in ("span", "spanldl", "siax"):
        assert got[key]["ok"] is False
        assert got[key]["detail"] == "spandrel is not installed"


def test_the_shaders_are_still_fetched_without_the_exporter(monkeypatch):
    """They are a download, not a build. A machine that cannot export the
    graphs should still end up able to upscale."""
    from stereo360 import enhance_models as em

    monkeypatch.setattr(em, "_can_export", lambda: False)
    monkeypatch.setattr(em, "missing_export_dependency",
                        lambda: "spandrel is not installed")
    monkeypatch.setattr(em.Spec, "present", lambda self: False)
    tried = []

    def fake_run(argv, **kw):
        tried.append(argv[-1])

        class Done:
            returncode, stdout, stderr = 1, "", ""
        return Done()

    monkeypatch.setattr(em.subprocess, "run", fake_run)
    em.fetch(on_line=lambda m: None)
    assert "fsrcnnx16" in tried and "fsrcnnx8" in tried
    assert "span" not in tried, "an export that cannot run must not be tried"
