"""Optional backends: actionable errors, and keeping their noise off stdout.

Both failures here were reported by a user rather than caught by a test:
selecting video-depth-anything died with a bare "No module named 'einops'",
and the same backend printed to stdout, putting non-JSON lines into a stream
documented as one JSON object per line.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from stereo360.depth import video_depth_anything as vda

ROOT = Path(__file__).resolve().parent.parent


class _Block:
    """Meta-path finder that makes one package look absent."""

    def __init__(self, name):
        self.name = name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name or fullname.startswith(self.name + "."):
            raise ModuleNotFoundError(f"No module named {self.name!r}",
                                      name=self.name)
        return None


def test_missing_dependency_names_what_to_install(monkeypatch):
    """A bare ModuleNotFoundError gives no clue that it came from an optional
    backend -- one that is offered in a dropdown, so it is easy to select
    without having read anything about it."""
    for name in list(sys.modules):
        if name.split(".")[0] in ("video_depth_anything", "einops"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Block("einops")] + sys.meta_path)

    with pytest.raises(ImportError) as excinfo:
        vda._load_model_class()

    message = str(excinfo.value)
    assert "requirements-vda.txt" in message
    assert "einops" in message


def test_deps_hint_is_specific():
    assert "{missing" in vda._DEPS_HINT
    assert "requirements-vda.txt" in vda._DEPS_HINT


def test_requirements_file_lists_the_real_dependencies():
    """einops and easydict are what the vendored tree actually needs; the rest
    of its imports belong to demos and eval harnesses we never load."""
    text = (ROOT / "requirements-vda.txt").read_text(encoding="utf-8")
    assert "einops" in text and "easydict" in text


def test_progress_json_stdout_is_reserved_for_events():
    """Third-party code prints, and does not know about the JSON contract.

    Checks the mechanism directly rather than through a backend, so it holds
    for whatever prints next -- transformers, tqdm, a vendored model.
    """
    code = (
        "import json, sys\n"
        "from stereo360.cli import _json_event_stream\n"
        "events = _json_event_stream()\n"
        "print('noise from some third-party module')\n"
        "sys.stdout.write('more noise\\n')\n"
        "events.write(json.dumps({'type': 'info', 'message': 'real'}) + '\\n')\n"
        "events.flush()\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=180, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr

    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"stdout should carry events only, got {lines}"
    assert json.loads(lines[0])["message"] == "real"
    # The prints are not lost -- they are diagnostics, and belong on stderr.
    assert "noise from some third-party module" in proc.stderr
    assert "more noise" in proc.stderr


# ------------------------------------------------------- backend probing


def test_probe_reports_every_backend():
    from stereo360 import backends

    names = [a.name for a in backends.probe_backends()]
    assert names == list(backends.BACKENDS)
    assert all(isinstance(a.detail, str) and a.detail
               for a in backends.probe_backends())


def test_probe_explains_each_way_a_backend_can_be_unusable(monkeypatch):
    """An unavailable backend has to say what to do about it -- that is the
    whole point of probing rather than letting the run fail."""
    from stereo360 import backends

    entry = {a.name: a for a in
             backends.probe_backends("models/definitely_absent.onnx")}
    assert not entry["onnx"].available
    assert "export_onnx.py" in entry["onnx"].detail

    real = backends._installed
    monkeypatch.setattr(backends, "_installed",
                        lambda m: False if m in ("einops", "easydict")
                        else real(m))
    entry = {a.name: a for a in backends.probe_backends()}
    assert not entry["video-depth-anything"].available
    assert "requirements-vda.txt" in entry["video-depth-anything"].detail

    monkeypatch.setattr(backends, "_installed",
                        lambda m: False if m in ("torch", "transformers")
                        else real(m))
    entry = {a.name: a for a in backends.probe_backends()}
    assert not entry["depth-anything"].available
    assert not entry["video-depth-anything"].available
    # ONNX exists precisely for machines without a usable torch.
    assert entry["onnx"].available

    monkeypatch.setattr(backends, "_installed", real)
    monkeypatch.setattr(backends, "_vda_clone", lambda: None)
    entry = {a.name: a for a in backends.probe_backends()}
    assert "clone" in entry["video-depth-anything"].detail


def test_probe_does_not_import_torch():
    """Probing runs in the UI process at startup; it must stay cheap."""
    code = (
        "import sys\n"
        "from stereo360.backends import probe_backends\n"
        "probe_backends()\n"
        "print('TORCH_IMPORTED' if 'torch' in sys.modules else 'CLEAN')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=180, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    assert "CLEAN" in proc.stdout, proc.stdout


def test_probe_backends_cli_emits_json():
    proc = subprocess.run(
        [sys.executable, "-m", "stereo360", "-", "--probe-backends"],
        capture_output=True, text=True, timeout=180, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert {e["name"] for e in data["backends"]} == set(
        __import__("stereo360.backends", fromlist=["x"]).BACKENDS)


# ------------------------------------------- the per-job-kind default backend


def test_the_default_backend_depends_on_the_kind_of_job(monkeypatch):
    """A video wants a model small and fast enough to run thousands of times;
    a still wants the best single frame available. One default cannot be both,
    which is why this is resolved from the input rather than from a constant."""
    from stereo360 import backends

    # Healthy machine: the torch preflight has nothing to report, so the
    # mapping is the only thing under test. Without this the assertions
    # below pass or fail on whether the test machine owns a GPU.
    monkeypatch.setattr(backends, "torch_backend_problem", lambda: None)

    assert backends.resolve_depth_backend(None, is_image=False) \
        == backends.VIDEO_BACKEND
    assert backends.resolve_depth_backend(None, is_image=True) \
        == backends.PHOTO_BACKEND


def test_an_explicit_backend_survives_the_default():
    """The None sentinel exists precisely so that asking for the video default
    on a photo is distinguishable from not asking. With a real default the two
    are the same string afterwards and the explicit choice is lost."""
    from stereo360 import backends

    for name in backends.BACKENDS:
        assert backends.resolve_depth_backend(name, is_image=True) == name
        assert backends.resolve_depth_backend(name, is_image=False) == name


def test_an_unavailable_default_falls_back_rather_than_failing(monkeypatch):
    """These two defaults carry dependencies the others do not -- onnxruntime
    for one, 3.6 GB of weights for the other. A machine without them should
    still convert something."""
    from stereo360 import backends

    def broken(*a, **kw):
        return [backends.Availability(n, False, "nope")
                for n in backends.BACKENDS]

    monkeypatch.setattr(backends, "probe_backends", broken)
    rec = []

    class Rec(backends.Reporter):
        def warning(self, msg, **kw):
            rec.append(msg)

    assert backends.resolve_depth_backend(None, True, Rec()) \
        == backends.FALLBACK_BACKEND
    assert rec and "unavailable" in rec[0]


def test_the_defaults_are_real_backends():
    from stereo360 import backends

    assert backends.VIDEO_BACKEND in backends.BACKENDS
    assert backends.PHOTO_BACKEND in backends.BACKENDS


def test_cli_choices_match_backends():
    """The parser spells the list out so `--help` stays import-free. That is
    only safe while something checks the copy against the original."""
    from stereo360 import backends, cli

    action = next(a for a in cli.build_parser()._actions
                  if a.dest == "depth_backend")
    assert list(action.choices) == list(backends.BACKENDS)
    # None, not "auto": the sentinel is what makes the per-job default work.
    assert action.default is None


def test_a_default_that_starts_but_would_not_finish_falls_back(monkeypatch):
    """Presence is not usability. Depth Pro on a machine with no torch GPU
    imports fine and then runs six 1536x1536 forward passes on the CPU for one
    photo, which reads as a hang rather than as a slow render. Whether the
    packages are installed is what `probe_backends` can see; whether the thing
    would finish is not."""
    from stereo360 import backends

    monkeypatch.setattr(backends, "probe_backends",
                        lambda *a, **kw: [backends.Availability(n, True, "ok")
                                          for n in backends.BACKENDS])
    monkeypatch.setattr(backends, "torch_backend_problem",
                        lambda: "torch 2.4.1+cpu has no GPU device")
    rec = []

    class Rec(backends.Reporter):
        def warning(self, msg, **kw):
            rec.append(msg)

    assert backends.resolve_depth_backend(None, True, Rec()) \
        == backends.FALLBACK_BACKEND
    assert rec and "not finish" in rec[0]
    # The warning has to name the escape hatch, or the only way back to the
    # better model is reading the source.
    assert "--depth-backend depth-pro" in rec[0]


def test_asking_for_a_torch_backend_outright_is_not_second_guessed(monkeypatch):
    """The preflight downgrades a *default*, never a choice. Depth Pro on the
    CPU is a legitimate thing to want for a single still, and refusing it would
    put the better model out of reach on exactly the machines that have no
    other way to run it."""
    from stereo360 import backends

    monkeypatch.setattr(backends, "torch_backend_problem", lambda: "no GPU")
    assert backends.resolve_depth_backend("depth-pro", True) == "depth-pro"


def test_the_video_default_never_imports_torch(monkeypatch):
    """The video default reaches onnxruntime and has no opinion about torch.
    Importing torch to check on it would put seconds and a GB of resident
    memory in front of every video job to answer a question it never asks."""
    from stereo360 import backends

    def boom():
        raise AssertionError("torch preflight ran on the video path")

    monkeypatch.setattr(backends, "probe_backends",
                        lambda *a, **kw: [backends.Availability(n, True, "ok")
                                          for n in backends.BACKENDS])
    monkeypatch.setattr(backends, "torch_backend_problem", boom)
    assert backends.resolve_depth_backend(None, False) \
        == backends.VIDEO_BACKEND


def test_the_fallback_is_reachable_without_torch_or_an_export():
    """Not "auto", which is the tempting answer and the wrong one:
    `autodetect.detect` chooses between torch and an exported ONNX model, so on
    a machine with neither it lands on Depth Anything V2 on the CPU -- the same
    absent GPU and the same torch that disqualified the default."""
    from stereo360 import backends

    assert backends.FALLBACK_BACKEND in backends.BACKENDS
    assert backends.FALLBACK_BACKEND not in backends._TORCH_BACKENDS
    assert backends.FALLBACK_BACKEND != "auto"
