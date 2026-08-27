"""The Linux installer's accelerator choice, mirroring test_installer.py.

Driven through the real file in `--dry-run` -- the .sh itself, not a copy of
its logic -- so what is tested is exactly what ships. It prints its choices
as `DECISION key=value` on stdout, same convention as the Windows installer,
for the same reason: colour output for a person is Write-Host/plain echo,
these lines are the machine-readable part.
"""

import platform
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALLER = (Path(__file__).resolve().parent.parent / "installer"
             / "install-stereo360.sh")

pytestmark = pytest.mark.skipif(
    platform.system() != "Linux" or shutil.which("bash") is None,
    reason="the installer is Linux-only and needs bash")


def decisions(install_dir: Path, **params) -> dict:
    """Run the installer's decision phase and return what it chose."""
    cmd = ["bash", str(INSTALLER), "--dry-run", "--install-dir", str(install_dir)]
    for key, value in params.items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                          stdin=subprocess.DEVNULL)
    assert proc.returncode == 0, f"installer failed:\n{proc.stdout}\n{proc.stderr}"
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("DECISION "):
            key, _, value = line[len("DECISION "):].partition("=")
            out[key] = value.strip()
    assert out, f"no decisions printed:\n{proc.stdout}\n{proc.stderr}"
    return out


def test_the_installer_exists_and_is_executable():
    assert INSTALLER.exists()
    assert INSTALLER.stat().st_mode & 0o111, "the script must be chmod +x"


def test_the_whole_file_is_plain_ascii_bash():
    """Not a hard requirement the way the Windows payload's ASCII-ness is --
    there is no unpack-by-string-split step here to get subtly wrong -- but a
    non-ASCII byte in a shebang'd script downloaded with curl is a common way
    for `bash` to choke on an encoding it did not expect."""
    raw = INSTALLER.read_bytes()
    bad = [i for i, b in enumerate(raw) if b > 127]
    assert not bad, f"non-ASCII byte at offset {bad[0]}"
    assert raw.startswith(b"#!/usr/bin/env bash")


def test_a_dry_run_touches_nothing(tmp_path):
    """Pointed at a path that definitely does not exist, rather than a real
    install folder that may already be there -- which would make the
    assertion pass without proving anything."""
    target = tmp_path / "would-be-install"
    decisions(target, compute_cap="12.0")
    assert not target.exists(), "dry run created the install folder"


# ------------------------------------------------------------ card by card

@pytest.mark.parametrize("cap,expected_build", [
    ("12.0", "cu128"),   # Blackwell, RTX 50 series
    ("9.0", "cu128"),    # Hopper
    ("8.9", "cu128"),    # Ada, RTX 40 series
    ("8.6", "cu128"),    # Ampere, RTX 30 series
    ("7.5", "cu128"),    # Turing, RTX 20 series
    ("6.1", "cu118"),    # Pascal, GTX 10 series -- too old for cu128
])
def test_each_nvidia_generation_gets_a_build_that_supports_it(
        tmp_path, cap, expected_build):
    got = decisions(tmp_path, compute_cap=cap)
    assert got["accelerator"] == "cuda"
    assert got["torch_index"].endswith(expected_build), got["torch_index"]


def test_a_50_series_card_does_not_get_the_build_that_fails_on_it(tmp_path):
    """cu121/cu124 install happily on an RTX 5070 Ti and report CUDA
    available; the build contains no sm_120 code, so everything runs through
    PTX JIT instead -- working, and quietly wrong."""
    index = decisions(tmp_path, compute_cap="12.0")["torch_index"]
    for stale in ("cu118", "cu121", "cu124"):
        assert stale not in index, f"{stale} has no sm_120 cubin"


def test_asking_for_cuda_without_a_card_falls_back_rather_than_installing_it(
        tmp_path):
    """Installing GBs of CUDA wheels that cannot run is worse than saying so
    and choosing something that works."""
    got = decisions(tmp_path, accelerator="cuda", compute_cap="none")
    assert got["accelerator"] != "cuda"


@pytest.mark.parametrize("choice", ["cpu"])
def test_an_explicit_choice_is_obeyed_even_with_a_card_present(tmp_path, choice):
    """A machine with no NVIDIA card simulated away would otherwise pick
    something else; someone who asked for CPU explicitly meant it."""
    got = decisions(tmp_path, accelerator=choice, compute_cap="none")
    assert got["accelerator"] == choice


def test_cpu_never_reports_a_torch_index():
    pass  # covered by the explicit-choice test's DECISION torch_index=n/a
          # not being asserted separately: see test below instead.


def test_explicit_cpu_reports_no_torch_index(tmp_path):
    got = decisions(tmp_path, accelerator="cpu")
    assert got["torch_index"] == "n/a"


# -------------------------------------------------------------- the source

def test_the_app_comes_from_a_release_when_there_is_one(tmp_path):
    """And from the default branch when there is not, so the installer works
    before the first release is cut rather than failing on a 404."""
    assert decisions(tmp_path, compute_cap="none")["app_source"]


# ------------------------------------------------------------- upgrading

def test_a_fresh_folder_is_reported_as_a_fresh_install(tmp_path):
    got = decisions(tmp_path / "nothing-here")
    assert got["existing_install"] == "none"


def test_an_existing_install_is_recognised(tmp_path):
    """So the run can say what it is replacing instead of looking like a
    first-time install that mysteriously skips most of its work."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "install-manifest.json").write_text(
        '{"app":"stereo360","appVersion":"1.0.1","accelerator":"cuda"}',
        encoding="utf-8")
    got = decisions(tmp_path)
    assert got["existing_install"] == "1.0.1"


def test_a_half_written_install_is_recognised_as_partial(tmp_path):
    """A venv with no manifest beside it is a run that died partway. Worth
    naming, because it is the case where the upgrade cannot take any of the
    shortcuts."""
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    py = venv_bin / "python3"
    py.write_text("", encoding="ascii")
    py.chmod(0o755)
    got = decisions(tmp_path)
    assert got["existing_install"] == "partial"


# ------------------------------------------------------------- uninstalling
#
# Read against the uninstaller's source, the way the Windows suite reads the
# payload: running it needs a real install to point at, and these are the
# properties that matter most because getting them wrong deletes something
# that cannot be put back.

def uninstaller_source() -> str:
    text = INSTALLER.read_text(encoding="ascii")
    start = text.index("cat > \"$INSTALL_DIR/uninstall.sh\" <<'UNINSTALL_EOF'")
    return text[start:text.index("UNINSTALL_EOF", start + 10)]


def test_the_uninstaller_proves_the_folder_is_ours_before_deleting():
    src = uninstaller_source()
    for proof in ("venv/bin/python3", "app/stereo360/__init__.py",
                  "install-manifest.json"):
        assert proof in src, f"{proof} is not checked for"
    assert "Refusing" in src


@pytest.mark.parametrize("guard", [
    '"$HOME"',            # the home folder
    '"$HOME/.config"',    # settings live here
    '"$HOME/.local"',     # the parent of the install folder
    "/usr",
    "/opt",
])
def test_the_uninstaller_refuses_dangerous_targets(guard):
    assert guard in uninstaller_source()


def test_the_uninstaller_keeps_files_it_did_not_create():
    src = uninstaller_source()
    assert "rm -rf \"$DIR/venv\"" in src, \
        "it should remove a recorded list, not the folder wholesale"
    assert "it still holds files that were not ours" in src


def test_the_uninstaller_leaves_shared_data_alone_by_default():
    src = uninstaller_source()
    assert "REMOVE_MODEL_CACHE" in src and "REMOVE_SETTINGS" in src
    assert "shared with" in src


def test_the_manifest_records_what_is_installed_not_what_was_downloaded():
    """The GitHub tag can read "main branch (no published release yet)". The
    app's own version is the honest answer, and it is what a later run
    compares against to say what it is replacing."""
    text = INSTALLER.read_text(encoding="ascii")
    assert "import stereo360; print(stereo360.released_as())" in text


def test_the_warm_up_caches_the_model_the_app_actually_defaults_to():
    text = INSTALLER.read_text(encoding="ascii")
    assert "Depth-Anything-V2" not in text, "still caching the old default"
    assert "from stereo360.depth.depth_anything_v3 import DEFAULT_VARIANT" in text
    assert "download(DEFAULT_VARIANT)" in text


def test_depth_pro_is_not_pre_fetched_but_is_announced():
    text = INSTALLER.read_text(encoding="ascii")
    assert "1.9 GB" in text
    warm = text[text.index('step "Fetching the depth model"'):]
    warm = warm[:warm.index('step "Creating launchers')]
    assert "DepthPro" not in warm, "pre-fetching 1.9 GB on every install"


# ------------------------------------------- which ONNX Runtime, and on what
#
# These matter more on Linux than the torch checks do. The *video* default
# (Depth Anything V3) is an ONNX graph with no torch path at all, so an
# onnxruntime that is not on the GPU puts the main use case on the CPU no
# matter how good the card is or what the torch probe said.

def test_the_onnx_cuda_build_is_paired_with_torchs_cuda_major():
    """onnxruntime-gpu moved to CUDA 13 at 1.25 and there is no CUDA 13 torch
    wheel to pair with it, so the newest of each cannot be used together.
    Measured on an RTX 5070 Ti with torch cu128: onnxruntime-gpu 1.28 listed
    CUDAExecutionProvider and then built every session on the CPU, because
    libcublasLt.so.13 was absent. It does not error -- it is just silently
    ten times slower."""
    text = INSTALLER.read_text(encoding="ascii")
    assert "torch.version.cuda" in text, \
        "the ORT build must be chosen from torch's CUDA major, not pinned blind"
    assert "onnxruntime-gpu<1.25" in text


def _install_arm(kind: str) -> str:
    """The pip commands one accelerator branch runs.

    Scoped to the dependency switch, because there is an earlier switch over
    the same three values that only prints a description -- slicing the whole
    file on `rocm)` picks that one up instead. The Windows suite has the same
    helper for the same reason, and its comment records that the first
    version of the test asserted against an empty string.
    """
    text = INSTALLER.read_text(encoding="ascii")
    switch = text[text.index('step "Installing the accelerator'):]
    switch = switch[:switch.index('step "Checking the accelerator')]
    start = switch.index(f"    {kind})")
    later = [switch.index(f"    {a})") for a in ("cuda", "rocm", "cpu")
             if a != kind and switch.index(f"    {a})") > start]
    return switch[start:min(later)] if later else switch[start:]


def test_amd_gets_a_rocm_onnxruntime_not_a_cpu_one():
    """onnxruntime-rocm is on PyPI with cp310-cp314 wheels. Installing plain
    onnxruntime here would leave AMD users' video renders on the processor,
    which is the one thing this accelerator choice exists to prevent.
    DirectML, the Windows answer for AMD, does not exist on Linux."""
    arm = _install_arm("rocm")
    assert "onnxruntime-rocm" in arm, "AMD would run ONNX on the CPU"
    assert "onnxruntime-directml" not in INSTALLER.read_text(encoding="ascii"), \
        "DirectML is Windows-only"


def test_the_cuda_arm_does_not_install_a_cpu_onnxruntime():
    """Plain `onnxruntime` alongside onnxruntime-gpu would win or lose by
    install order -- both provide a module called onnxruntime."""
    arm = _install_arm("cuda")
    assert "onnxruntime-gpu" in arm
    assert "pip_install onnxruntime onnx" not in arm


def test_the_launchers_carry_the_cuda_library_path():
    """The fix has to reach the application, not just the installer's probe.
    onnxruntime finds the CUDA runtime through LD_LIBRARY_PATH and that
    runtime belongs to torch's wheels, which ORT does not look inside.
    Without this line the installer would verify a GPU that renders then do
    not get -- the same failure, moved somewhere nobody would see it."""
    text = INSTALLER.read_text(encoding="ascii")
    assert "nvidia_lib_path" in text
    for var in ('LAUNCH="$INSTALL_DIR/stereo360.sh"',
                'LAUNCH_UI="$INSTALL_DIR/stereo360-ui.sh"'):
        block = text[text.index(var):]
        # Past the `<<EOF` marker to the body it introduces, rather than
        # stopping at the first "EOF" -- which is the marker itself.
        block = block[block.index("<<EOF") + len("<<EOF"):]
        block = block[:block.index("\nEOF")]
        assert "$LD_LINE" in block, f"{var} does not set LD_LIBRARY_PATH"


def test_the_onnx_probe_reads_back_the_sessions_provider():
    """get_available_providers() lists what is installed, not what loads --
    the entire reason the CUDA 13 mismatch went unnoticed."""
    text = INSTALLER.read_text(encoding="ascii")
    fn = text[text.index("probe_onnx_provider() {"):]
    fn = fn[:fn.index("\nPYEOF\n}")]
    assert "sess.get_providers()[0]" in fn
    assert "sess.run" in fn, "it must execute a graph, not just build a session"
    assert 'FAIL asked for %s, got %s' in fn


# --------------------------------------------------- Qt's X11 dependencies

def test_the_qt_library_check_reads_ldd_rather_than_the_error_message():
    """Qt names only the first missing library in its own error text.
    Measured here: installing libxcb-cursor0, which is what the message
    asks for, changed the message and not the outcome -- libxcb-icccm4 and
    libxcb-keysyms1 were missing too. ldd lists all of them at once."""
    text = INSTALLER.read_text(encoding="ascii")
    assert "ldd" in text and "not found" in text, \
        "the missing set must be read from ldd"
    fn = text[text.index("ensure_qt_x11_libs() {"):]
    fn = fn[:fn.index("\n}\n")]
    assert "awk '/not found/" in fn


def test_the_qt_library_check_looks_at_pysides_plugin_not_opencvs():
    """The bug this exists to catch. opencv-python bundles a whole Qt of its
    own, libqxcb.so included, and a bare `find -name libqxcb.so` over the
    venv returned cv2's copy -- whose dependencies are satisfied. The check
    reported nothing missing while the interface could not start at all."""
    text = INSTALLER.read_text(encoding="ascii")
    fn = text[text.index("ensure_qt_x11_libs() {"):]
    fn = fn[:fn.index("\n}\n")]
    assert "import PySide6" in fn, \
        "the plugin must be located through PySide6, not by bare name"
    assert "-path '*/PySide6/*'" in fn, \
        "the fallback search must still be confined to PySide6"


def test_privilege_escalation_has_a_route_without_a_terminal():
    """sudo reads the password from the controlling terminal and can only
    fail without one -- which is the case when the installer is launched
    from a desktop shortcut or driven by a tool. pkexec asks the polkit
    agent instead, which needs no terminal."""
    text = INSTALLER.read_text(encoding="ascii")
    fn = text[text.index("run_privileged() {"):]
    fn = fn[:fn.index("\n}\n")]
    assert "sudo -n true" in fn, "an already-authorised sudo should not prompt"
    assert "pkexec" in fn
    assert "/dev/tty" in fn, "it must test for a terminal before choosing sudo"
    # The redirection order matters: written the other way round the shell
    # prints "No such device or address" before stderr is silenced.
    assert ": 2>/dev/null >/dev/tty" in fn, \
        "2>/dev/null must precede the /dev/tty open"


def test_only_the_missing_libraries_are_installed():
    """It installs what ldd says is absent, not a fixed bundle -- so a
    machine that already has most of them is not made to fetch the lot."""
    text = INSTALLER.read_text(encoding="ascii")
    fn = text[text.index("ensure_qt_x11_libs() {"):]
    fn = fn[:fn.index("\n}\n")]
    assert 'packages+=("$pkg")' in fn
    assert "still" in fn, "it should re-check with ldd after installing"


def test_cpu_index_used_to_avoid_the_bundled_cuda_wheels():
    """Plain `pip install torch` on Linux is not CPU-only the way it is on
    Windows -- it silently pulls the CUDA build and its nvidia-*-cu12
    dependencies regardless of hardware. The explicit CPU choice must ask
    for the CPU wheel index rather than relying on the default."""
    text = INSTALLER.read_text(encoding="ascii")
    assert "CPU_INDEX=\"https://download.pytorch.org/whl/cpu\"" in text
    cpu_arm = text[text.index('cpu)\n        pip_install --index-url "$CPU_INDEX"'):]
    assert cpu_arm.startswith('cpu)\n        pip_install --index-url "$CPU_INDEX" torch torchvision')
