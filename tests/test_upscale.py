"""Optional Topaz upscaling.

Every test here runs on a machine that has never heard of Topaz. Detection is
pointed at temporary directories rather than the real install, so the suite
behaves identically on a build agent and on the one desktop that happens to
have a licence -- which is the same property the feature itself has to have.
"""

import json
import subprocess
import sys

import pytest

from stereo360 import upscale


def _model_json(path, short, name, model_type=1, pri=10, enabled=1):
    path.write_text(json.dumps({
        "shortName": short, "displayName": name, "modelType": model_type,
        "enabled": enabled,
        "gui": {"name": f"Family - {name}", "desc": f"about {short}",
                "minScale": 0.25, "maxScale": 6, "displayPri": pri},
    }), encoding="utf-8")


# ----------------------------------------------------------------- detection

def test_a_machine_without_topaz_gets_a_usable_no(tmp_path):
    """Not an exception. This is asked at startup on every machine, and a
    traceback there would take the interface down over a feature nobody on
    that machine asked for."""
    assert upscale.find(app_dirs=(str(tmp_path / "nope"),),
                        model_dirs=(str(tmp_path / "nope"),)) is None


def test_the_binary_alone_is_not_an_install(tmp_path):
    """Models live in ProgramData and the binary in Program Files, and one
    without the other cannot run anything."""
    app = tmp_path / "app"
    app.mkdir()
    (app / "ffmpeg.exe").write_bytes(b"")
    assert upscale.find(app_dirs=(str(app),),
                        model_dirs=(str(tmp_path / "gone"),)) is None


def test_describe_says_no_without_raising(monkeypatch):
    monkeypatch.setattr(upscale, "find", lambda *a, **k: None)
    d = upscale.describe(3840)
    assert d["available"] is False
    assert d["models"] == []
    assert d["offered"] is False
    assert d["reason"]


# -------------------------------------------------------------- when to offer

@pytest.mark.parametrize("width,expected", [
    (1920, True),
    (3840, True),          # the case this exists for
    (5760, True),
    (7679, True),
    (7680, False),         # already 8K
    (11904, False),
    (0, False),
])
def test_it_is_only_offered_below_8k(width, expected):
    """Above 8K a 2x lands at 16K, which no headset can decode and no part of
    this pipeline wants to hold in memory."""
    assert upscale.offered_for(width) is expected


# ----------------------------------------------------------------- the models

def test_only_upscalers_are_listed(tmp_path):
    """`tvai_up` takes upscalers. Frame interpolation is modelType 2 and
    belongs on a different filter, so offering it here would produce a run
    that fails after the model loads."""
    _model_json(tmp_path / "amq-13.json", "amq", "Medium Quality", model_type=1)
    _model_json(tmp_path / "chr-2.json", "chr", "Chronos", model_type=2)
    got = upscale.models(upscale.Install("ffmpeg.exe", str(tmp_path)))
    assert [m.short for m in got] == ["amq"]


def test_the_newest_version_of_each_model_wins(tmp_path):
    for v in (10, 12, 13):
        _model_json(tmp_path / f"amq-{v}.json", "amq", "Medium Quality")
    got = upscale.models(upscale.Install("ffmpeg.exe", str(tmp_path)))
    assert [m.code for m in got] == ["amq-13"]


def test_the_gui_name_is_used_because_display_name_is_ambiguous(tmp_path):
    """Artemis and Gaia both ship a model whose displayName is exactly
    "High Quality". Only `gui.name` says which is which, and a dropdown with
    two identical entries is worse than no dropdown."""
    _model_json(tmp_path / "ahq-12.json", "ahq", "High Quality")
    got = upscale.models(upscale.Install("ffmpeg.exe", str(tmp_path)))
    assert got[0].name == "Family - High Quality"
    assert got[0].desc and got[0].max_scale == 6


def test_a_disabled_model_is_not_offered(tmp_path):
    _model_json(tmp_path / "old-1.json", "old", "Retired", enabled=0)
    assert upscale.models(upscale.Install("ffmpeg.exe", str(tmp_path))) == []


def test_rubbish_in_the_model_directory_is_stepped_over(tmp_path):
    """The directory holds an exe, a zip and some json that is not a model."""
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    _model_json(tmp_path / "amq-13.json", "amq", "Medium Quality")
    got = upscale.models(upscale.Install("ffmpeg.exe", str(tmp_path)))
    assert [m.short for m in got] == ["amq"]


def test_an_unreadable_model_directory_is_not_fatal():
    assert upscale.models(upscale.Install("ffmpeg.exe", "/no/such/dir")) == []


# ------------------------------------------------------------------- the auth

def test_a_missing_token_is_a_login_not_a_probe(tmp_path):
    """No point starting an engine to be told what the absent file already
    says."""
    assert upscale.auth_state(
        upscale.Install("ffmpeg.exe", str(tmp_path))) == "login"


@pytest.mark.parametrize("line,expected", [
    ("INFO:  Successfully authenticated for user: someone@example.com", "ok"),
    ("INFO:  Failed to refresh auth: 401 reauthentication is required.", "login"),
    ("INFO:  Refresh failed. Watermark will be enabled", "login"),
])
def test_the_verdict_is_read_out_of_the_log(tmp_path, monkeypatch, line,
                                            expected):
    """Topaz writes an AES-encrypted auth.tpz, so an expired token is
    indistinguishable from a good one on disk. The verdict only exists in the
    log of a running job -- and arrives about a second into a run that takes
    nearly thirty, which is why this stops reading as soon as it has it.
    """
    (tmp_path / "auth.tpz").write_bytes(b"PK\x03\x04")

    class _Proc:
        def __init__(self):
            self.stderr = iter([f"{line}\n", "INFO: more lines\n"])
            self.killed = False

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            return 0

    made = []

    def fake_popen(*a, **k):
        p = _Proc()
        made.append(p)
        return p

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    got = upscale.auth_state(upscale.Install("ffmpeg.exe", str(tmp_path)))
    assert got == expected
    assert made[0].killed, "the probe must not be left running"


def test_a_binary_that_will_not_start_is_unknown_not_a_crash(tmp_path,
                                                             monkeypatch):
    (tmp_path / "auth.tpz").write_bytes(b"PK\x03\x04")

    def boom(*a, **k):
        raise OSError("not executable here")

    monkeypatch.setattr(subprocess, "Popen", boom)
    assert upscale.auth_state(
        upscale.Install("ffmpeg.exe", str(tmp_path))) == "unknown"


# -------------------------------------------------------------------- the CLI

def test_the_probe_prints_json_and_exits_zero_anywhere():
    """A GUI runs this at startup on every machine, so it must answer rather
    than fail whatever it finds."""
    proc = subprocess.run(
        [sys.executable, "-m", "stereo360", "--probe-upscalers",
         "--probe-width", "3840"],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    d = json.loads(proc.stdout)
    assert set(d) >= {"available", "auth", "models", "offered"}
    assert isinstance(d["models"], list)
    if d["available"]:
        assert d["offered"] is True, "3840 is below 8K and should be offered"


def test_the_probe_needs_no_input_file():
    """Same reason --probe-backends does not: it describes the machine."""
    proc = subprocess.run(
        [sys.executable, "-m", "stereo360", "--probe-upscalers"],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    json.loads(proc.stdout)


# ------------------------------------------------------------- the filter chain

def _fake_install(tmp_path, *specs):
    for short, name, mtype, ver in specs:
        _model_json(tmp_path / f"{short}-{ver}.json", short, name,
                    model_type=mtype)
    return upscale.Install("ffmpeg.exe", str(tmp_path))


def test_upscaling_comes_before_interpolation(tmp_path):
    """Both orders work and this one is much cheaper: the upscaler only sees
    the frames that were really shot rather than the doubled set, and it is
    the expensive half."""
    inst = _fake_install(tmp_path, ("amq", "MQ", 1, 13), ("chr", "Chronos", 2, 2))
    vf = upscale.chain(upscale.resolve(inst, "amq"), 2.0,
                       upscale.resolve(inst, "chr", "fi"), 60)
    assert vf.index("tvai_up") < vf.index("tvai_fi")
    assert "model=amq-13" in vf and "scale=2" in vf
    assert "model=chr-2" in vf and "fps=60" in vf


def test_either_half_can_stand_alone(tmp_path):
    inst = _fake_install(tmp_path, ("amq", "MQ", 1, 13), ("chr", "Chronos", 2, 2))
    assert upscale.chain(up=upscale.resolve(inst, "amq")).startswith("tvai_up")
    assert upscale.chain(fi=upscale.resolve(inst, "chr", "fi")).startswith("tvai_fi")
    assert upscale.chain() == ""


def test_a_model_resolves_by_short_name_or_exact_code(tmp_path):
    inst = _fake_install(tmp_path, ("amq", "MQ", 1, 13))
    assert upscale.resolve(inst, "amq").code == "amq-13"
    assert upscale.resolve(inst, "amq-13").code == "amq-13"
    assert upscale.resolve(inst, "AMQ").code == "amq-13"
    assert upscale.resolve(inst, "nope") is None


def test_upscalers_and_interpolators_are_kept_apart(tmp_path):
    """`tvai_up` will not run an interpolation model. Offering one there
    produces a run that fails after the engine has loaded."""
    inst = _fake_install(tmp_path, ("amq", "MQ", 1, 13), ("chr", "Chronos", 2, 2))
    assert [m.short for m in upscale.models(inst)] == ["amq"]
    assert [m.short for m in upscale.interpolators(inst)] == ["chr"]
    assert upscale.resolve(inst, "chr", "up") is None
    assert upscale.resolve(inst, "amq", "fi") is None


def test_the_default_upscaler_is_artemis_medium_quality():
    assert upscale.DEFAULT_UPSCALE == "amq"


# ------------------------------------------------------------------ the runner

def test_a_signed_out_topaz_is_reported_as_that_not_as_a_crash(tmp_path,
                                                               monkeypatch):
    """The run has already started by the time this is known, so it has to be
    turned back into the sentence a person can act on."""
    class _Proc:
        def __init__(self):
            self.stderr = iter(["INFO: Refresh failed. Watermark will be enabled\n"])
        def kill(self): pass
        def wait(self, timeout=None): return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Proc())
    monkeypatch.setattr(upscale, "_encoder",
                        lambda *a, **k: ["-c:v", "ffv1"])
    inst = _fake_install(tmp_path, ("amq", "MQ", 1, 13))
    with pytest.raises(upscale.UpscaleError, match="sign"):
        upscale.run(inst, "in.mp4", "out.mkv", up=upscale.resolve(inst, "amq"))


def test_asking_for_nothing_is_an_error_rather_than_a_silent_copy(tmp_path):
    inst = _fake_install(tmp_path, ("amq", "MQ", 1, 13))
    with pytest.raises(upscale.UpscaleError):
        upscale.run(inst, "in.mp4", "out.mkv")


def test_the_intermediate_is_yuv420p(tmp_path, monkeypatch):
    """Left to itself the chain hands back RGB, and the converter then passes
    `colorspace gbr` to libx264, which refuses it -- killing the run at the
    last step, after the upscale has been paid for."""
    from stereo360 import intermediate

    # Topaz hands back RGB and the converter's encoder then refuses the gbr
    # colour tags, so whatever else the working file keeps, it has to be YUV.
    monkeypatch.setattr(intermediate, "supported",
                        lambda ffmpeg, name: frozenset(
                            {"yuv420p", "yuv444p", "gbrp"}))
    for asked in ("gbrp", "rgb24", None):
        args, note = intermediate.encoder_args("ffmpeg.exe", pix_fmt=asked)
        assert "yuv420p" in args, f"{asked} should land on YUV"
        assert "gbr" not in note


# ------------------------------------------------------- how long a still wrap is

def test_a_one_frame_video_is_not_enough(tmp_path):
    """Measured, feeding Artemis Medium Quality copies of one frame: 1, 2 and
    3 all produce nothing and 4 produces four. So wrapping a still is not
    merely about getting it into a video container -- the filter wants a
    run-up, and asking for one frame would silently produce no output at all.
    """
    inst = _fake_install(tmp_path, ("amq", "MQ", 1, 13))
    assert upscale.wrap_frames(upscale.resolve(inst, "amq")) >= 4
    assert upscale.MIN_WRAP_FRAMES >= 4


def test_the_wrap_length_comes_from_the_model(tmp_path, monkeypatch):
    """Artemis declares preflight 5 and Gaia 2+2, so they need 6 and 5 -- not
    the flat 12 this used to send, where every extra frame is a full-size
    upscale."""
    (tmp_path / "amq-13.json").write_text(json.dumps({
        "shortName": "amq", "displayName": "MQ", "modelType": 1,
        "preflight": 5, "postflight": 0, "gui": {"name": "Artemis - MQ"}}),
        encoding="utf-8")
    (tmp_path / "ghq-5.json").write_text(json.dumps({
        "shortName": "ghq", "displayName": "HQ", "modelType": 1,
        "preflight": 2, "postflight": 2, "gui": {"name": "Gaia - HQ"}}),
        encoding="utf-8")
    inst = upscale.Install("ffmpeg.exe", str(tmp_path))
    assert upscale.wrap_frames(upscale.resolve(inst, "amq")) == 6
    assert upscale.wrap_frames(upscale.resolve(inst, "ghq")) == 5


def test_an_odd_declaration_cannot_turn_a_still_into_a_render(tmp_path):
    (tmp_path / "mad-1.json").write_text(json.dumps({
        "shortName": "mad", "displayName": "Mad", "modelType": 1,
        "preflight": 9000, "gui": {"name": "Mad"}}), encoding="utf-8")
    inst = upscale.Install("ffmpeg.exe", str(tmp_path))
    assert upscale.wrap_frames(upscale.resolve(inst, "mad")) \
        == upscale.MAX_WRAP_FRAMES


def test_no_model_still_gets_a_usable_length():
    assert upscale.wrap_frames(None) == upscale.MIN_WRAP_FRAMES


# ------------------------------------------------- what Topaz's ffmpeg reads
#
# Its bundled build lists an `av1` decoder that is a hardware-accelerated
# shell: asked for AV1 in software it answers "Function not implemented" and
# then dies with an access violation. The user sees a wall of ffmpeg noise
# ending in a complaint about a filter option, which is the last thing that
# happened rather than the thing that went wrong.


@pytest.mark.parametrize("codec", ["h264", "hevc", "HEVC", "prores", "vp9"])
def test_a_codec_it_can_read_needs_no_help(codec):
    """The common path costs nothing: no probe, no decoder flag."""
    assert upscale.input_args(None, "anything.mp4", codec) == []


def test_an_unknown_codec_is_settled_before_the_job_starts():
    """Not knowing is not the same as knowing it works -- but an unreadable
    source has to be found by asking, not assumed either way."""
    assert "av1" not in upscale._SOFTWARE_CODECS


def test_no_codec_at_all_is_left_alone():
    """A probe that failed should not turn into a refusal to run."""
    assert upscale.input_args(None, "anything.mp4", None) == []


needs_topaz = pytest.mark.skipif(upscale.find() is None,
                                 reason="Topaz Video AI is not installed here")


@needs_topaz
def test_av1_gets_a_decoder_that_works(tmp_path):
    """On a machine whose ffmpeg has no software AV1 decoder, the hardware one
    is found and used rather than crashed into."""
    src = tmp_path / "av1.mp4"
    made = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=320x160:rate=30", "-frames:v", "4",
         "-c:v", "libsvtav1", str(src)], capture_output=True)
    if made.returncode != 0 or not src.exists():
        pytest.skip("this ffmpeg cannot encode AV1 to test against")

    install = upscale.find()
    args = upscale.input_args(install, str(src), "av1")
    # Either the software decoder turned out to work here, or a hardware one
    # was chosen. What matters is that whatever came back actually decodes.
    assert upscale._can_decode(install, str(src), args)


def test_a_decode_failure_says_what_it_was():
    """The message has to name the cause and the way out, because the ffmpeg
    log ends by complaining about a filter option and buries the decoder."""
    for line in ("Error submitting packet to decoder: Function not implemented",
                 "Decode error rate 1 exceeds maximum 0.666667"):
        assert any(s in line.lower() for s in upscale._DECODE_FAILED)


# --------------------------------------------------------- the rate it reaches


def test_topaz_interpolation_always_names_a_rate():
    """tvai_fi defaults its own fps to 0, which means "leave the rate alone".
    A Chronos pass with no rate therefore ran the model over every frame and
    handed back a video at exactly the rate it came in at -- the switch was on
    and nothing happened."""
    fi = upscale.Model(code="chf-3", short="chf", name="Chronos Fast")
    assert "fps=" not in upscale.chain(fi=fi)          # the low-level default
    assert ":fps=60" in upscale.chain(fi=fi, fps=60)


def test_the_default_rate_is_double_the_source():
    from stereo360 import cli

    class _Info:
        fps = 29.97

    class _Args:
        interpolate_fps = 0.0

    fi = upscale.Model(code="chf-3", short="chf", name="Chronos Fast")
    assert cli._target_fps(_Args(), _Info(), fi) == pytest.approx(59.94)

    asked = _Args()
    asked.interpolate_fps = 48.0
    assert cli._target_fps(asked, _Info(), fi) == 48.0

    assert cli._target_fps(_Args(), _Info(), None) is None, "no pass, no rate"
    assert cli._target_fps(_Args(), None, fi) is None, "unread source, no guess"


# ------------------------------------------------------ one entry per name


def test_two_models_under_one_name_are_shown_once(tmp_path):
    """Topaz ships chf-3 and ifi-1 both named "Chronos Fast". Two identical
    lines in a dropdown read as a bug, and picking between them is a coin
    toss -- so the one Topaz itself labels wins."""
    (tmp_path / "chf-3.json").write_text(json.dumps({
        "shortName": "chf", "displayName": "Chronos Fast", "modelType": 2,
        "gui": {"name": "Chronos Fast", "displayPri": 489}}), encoding="utf-8")
    (tmp_path / "ifi-1.json").write_text(json.dumps({
        "shortName": "ifi", "modelType": 2,
        "gui": {"name": "Chronos Fast", "displayPri": 489}}), encoding="utf-8")

    got = upscale.interpolators(upscale.Install("ffmpeg.exe", str(tmp_path)))
    assert [m.short for m in got] == ["chf"]


def test_models_that_only_share_a_display_name_both_survive(tmp_path):
    """Aion and Aion Bottom both declare displayName "Aion" and are different
    models. The name shown is the gui name, so that is what has to be unique."""
    for code, short, gui in (("aion-1", "aion", "Aion"),
                             ("aiob-1", "aiob", "Aion Bottom")):
        (tmp_path / f"{code}.json").write_text(json.dumps({
            "shortName": short, "displayName": "Aion", "modelType": 2,
            "gui": {"name": gui, "displayPri": 490}}), encoding="utf-8")

    got = upscale.interpolators(upscale.Install("ffmpeg.exe", str(tmp_path)))
    assert sorted(m.short for m in got) == ["aiob", "aion"]


def test_the_newer_version_wins_when_neither_is_labelled(tmp_path):
    for code in ("zzz-1", "zzz-4"):
        (tmp_path / f"{code}.json").write_text(json.dumps({
            "shortName": code, "modelType": 2,
            "gui": {"name": "Same Name", "displayPri": 1}}), encoding="utf-8")

    got = upscale.interpolators(upscale.Install("ffmpeg.exe", str(tmp_path)))
    assert [m.code for m in got] == ["zzz-4"]


# ------------------------------------------------------------ the frame range


def test_the_range_is_cut_before_the_models_not_after():
    """`-frames:v` at the tail kills Topaz mid-stream -- it exits 0xC0000374,
    a corrupted heap, with nothing written. Trimming the input instead lets it
    reach the end of the stream the way it expects to."""
    assert upscale.trimmed("tvai_fi=x") == "tvai_fi=x", "no range, no filter"
    assert upscale.trimmed("tvai_fi=x", 0, 31) == \
        "trim=end_frame=31,setpts=PTS-STARTPTS,tvai_fi=x"
    assert upscale.trimmed("tvai_fi=x", 10, 15) == \
        "trim=start_frame=10:end_frame=15,setpts=PTS-STARTPTS,tvai_fi=x"
    # Before the models, so nothing is enhanced that nobody asked for.
    assert upscale.trimmed("tvai_up=y", 2).index("trim=") < \
        upscale.trimmed("tvai_up=y", 2).index("tvai_up")


@pytest.mark.parametrize("start,count,ratio,expected", [
    # Sixty frames from the start, doubling: read 31 source frames, which make
    # 61, and the renderer keeps 60.
    (0, 60, 2.0, (0, 31, 0, 61)),
    # Output 21 falls between source 10 and 11, so the pass starts at 10 and
    # its first frame is output 20 -- one to drop.
    (21, 8, 2.0, (10, 15, 1, 9)),
    # No interpolation: output frames are source frames.
    (0, 60, 1.0, (0, 60, 0, 60)),
    (100, 60, 1.0, (100, 160, 0, 60)),
    # No limit stays no limit.
    (0, None, 2.0, (0, None, 0, None)),
])
def test_the_prepass_window_turns_output_frames_into_source_frames(
        start, count, ratio, expected):
    """Without this the pass ran over the whole file and the renderer applied
    the range afterwards -- so a sixty-frame test render of an 8K video
    interpolated the entire thing first."""
    from stereo360 import cli

    assert cli._prepass_window(start, count, ratio) == expected


def test_the_window_never_asks_for_fewer_frames_than_wanted():
    """Rounding has to go outwards: one frame short is a range that stops
    before it was told to."""
    from stereo360 import cli

    for ratio in (1.0, 1.5, 2.0, 2.5, 4.0):
        for start in (0, 1, 7, 100):
            for count in (1, 2, 9, 60):
                skip, end, drop, produced = cli._prepass_window(
                    start, count, ratio)
                assert produced - drop >= count, (ratio, start, count)
                assert end > skip


def test_a_batch_gets_its_frame_range_back(tmp_path, monkeypatch):
    """The pre-pass re-states the range against the intermediate it writes.
    A batch runs every file through one `args`, so the second would inherit
    the first file's rewritten start -- the same trap `depth_backend` is
    already kept and restored for."""
    import argparse

    from stereo360 import cli

    args = argparse.Namespace(input=str(tmp_path / "in.mp4"),
                              output=str(tmp_path / "out.mp4"),
                              start_frame=21, max_frames=8,
                              upscale=None, interpolate=None)

    class _Pipeline:
        class ffmpeg_io:
            @staticmethod
            def is_image_path(path):
                return False

    def _boom(*a, **k):
        raise SystemExit("stopped after the pre-pass")

    monkeypatch.setattr(cli, "_run_converted", _boom)
    monkeypatch.setattr(cli, "_topaz_prepass",
                        lambda *a, **k: (setattr(args, "start_frame", 1)
                                         or args.input))
    with pytest.raises(SystemExit):
        cli._run(args, None, None, None, _Pipeline())

    assert (args.start_frame, args.max_frames) == (21, 8)
