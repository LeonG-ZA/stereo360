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
    monkeypatch.setattr(upscale, "_encoder", lambda i: ["-c:v", "ffv1"])
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
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: type(
        "R", (), {"stdout": "hevc_nvenc ffv1", "returncode": 0})())
    args = upscale._encoder(upscale.Install("ffmpeg.exe", str(tmp_path)))
    assert "yuv420p" in args


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
