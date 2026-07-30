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
