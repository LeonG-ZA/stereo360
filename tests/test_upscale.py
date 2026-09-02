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
