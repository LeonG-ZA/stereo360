"""The installer's accelerator choice, which is the part that goes wrong.

Everything else the installer does is downloading and unzipping, and fails
loudly when it fails. Picking the PyTorch build does not. Measured on an RTX
5070 Ti against a cu118 build: it installs cleanly, reports
`torch.cuda.is_available() == True`, and kernels genuinely *run* -- CUDA
JIT-compiles the embedded PTX. Nothing looks wrong and nothing is using the
tuned code for the card. That decision is worth pinning.

Driven through the real file in `-DryRun` -- the .bat itself, not a copy of
its payload -- so what is tested is exactly what ships, unpacking included.
It prints its choices as `DECISION key=value` on stdout for this purpose.
"""

import platform
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALLER = (Path(__file__).resolve().parent.parent / "installer"
             / "Install stereo360.bat")
MARKER = "#@@ POWERSHELL PAYLOAD STARTS ON THE NEXT LINE @@"

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows" or shutil.which("powershell") is None,
    reason="the installer is Windows-only and needs powershell")


def payload() -> str:
    """The PowerShell out of the .bat, unpacked the way the .bat unpacks it."""
    return INSTALLER.read_text(encoding="ascii").split(MARKER)[-1]


def decisions(**params) -> dict:
    """Run the installer's decision phase and return what it chose."""
    cmd = [str(INSTALLER), "-DryRun"]
    for key, value in params.items():
        cmd += [f"-{key}", str(value)]
    # stdin closed, because the .bat ends in `pause` -- a person needs the
    # window to stay up long enough to read the result, and a test does not.
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                          stdin=subprocess.DEVNULL, shell=False)
    assert proc.returncode == 0, f"installer failed:\n{proc.stdout}\n{proc.stderr}"
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("DECISION "):
            key, _, value = line[len("DECISION "):].partition("=")
            out[key] = value.strip()
    assert out, f"no decisions printed:\n{proc.stdout}"
    return out


def test_the_installer_is_one_file():
    """One file, because two invited being separated -- someone downloads the
    .bat and not the .ps1, or moves only the one they were told to click."""
    assert INSTALLER.exists()
    assert sorted(p.name for p in INSTALLER.parent.iterdir()) == \
        ["Install stereo360.bat"]


def test_the_payload_is_readable_rather_than_encoded():
    """Deliberate. This is an unsigned installer that will trip SmartScreen,
    so someone wary enough to open it in Notepad should be able to read what
    it will do. Base64 would be smaller and would forfeit that."""
    text = INSTALLER.read_text(encoding="ascii")
    assert MARKER in text
    assert "-ExecutionPolicy Bypass" in text, \
        "a downloaded script will not run under the default policy"
    assert "param(" in payload() and "$InstallDir" in payload()


def test_the_whole_file_is_ascii():
    """The payload is unpacked by reading the .bat as text. Pure ASCII means
    that cannot go wrong however the encoding is guessed -- and the header
    promises as much, so it needs enforcing rather than hoping."""
    raw = INSTALLER.read_bytes()
    bad = [i for i, b in enumerate(raw) if b > 127]
    assert not bad, f"non-ASCII byte at offset {bad[0]}"
    assert not raw.startswith(b"\xef\xbb\xbf"), "a BOM would confuse cmd.exe"


def test_the_unpacked_payload_is_valid_powershell(tmp_path):
    """Unpacking is a string split, so a stray copy of the marker or a
    mangled line would produce something that only fails when a user runs it.
    Parse it here instead.

    Through a file rather than an argument: the payload is well past the
    command-line length limit, which this test discovered by failing with
    "The filename or extension is too long" the moment the uninstaller was
    added to it.
    """
    script = tmp_path / "payload.ps1"
    script.write_text(payload(), encoding="ascii")
    check = ("$e=$null; [void][System.Management.Automation.Language.Parser]"
             f"::ParseFile('{script}', [ref]$null, [ref]$e); "
             "if ($e) { $e[0].ToString(); exit 1 } else { 'ok' }")
    proc = subprocess.run(["powershell", "-NoProfile", "-Command", check],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"payload does not parse:\n{proc.stdout}"


# ------------------------------------------------------------ card by card

@pytest.mark.parametrize("cap,expected_build", [
    ("12.0", "cu128"),   # Blackwell, RTX 50 series -- the one that bit us
    ("9.0", "cu128"),    # Hopper
    ("8.9", "cu128"),    # Ada, RTX 40 series
    ("8.6", "cu128"),    # Ampere, RTX 30 series
    ("7.5", "cu128"),    # Turing, RTX 20 series
    ("6.1", "cu118"),    # Pascal, GTX 10 series -- too old for cu128
])
def test_each_nvidia_generation_gets_a_build_that_supports_it(cap,
                                                              expected_build):
    """cu128 ships cubins for sm_70 upward, which is Volta to Blackwell.
    Ada (sm_89) is absent from that list and still works, because CUDA is
    binary compatible within a major generation -- the sm_86 cubin runs on
    it. Pascal predates the whole range and needs the older build."""
    got = decisions(ComputeCap=cap)
    assert got["accelerator"] == "cuda"
    assert got["torch_index"].endswith(expected_build), got["torch_index"]


def test_a_50_series_card_does_not_get_the_build_that_fails_on_it():
    """The specific regression. cu121 installs happily on an RTX 5070 Ti and
    reports CUDA available; the build contains no sm_120 code, so everything
    runs through PTX JIT instead -- working, and quietly wrong."""
    index = decisions(ComputeCap="12.0")["torch_index"]
    for stale in ("cu118", "cu121", "cu124"):
        assert stale not in index, f"{stale} has no sm_120 cubin"


# ------------------------------------------------------------- no NVIDIA

def test_a_machine_without_an_nvidia_card_gets_directml():
    """Not a consolation prize: DirectML runs on any Direct3D 12 GPU, so AMD
    and Intel get GPU depth, and without the 2.5 GB PyTorch download."""
    got = decisions(ComputeCap="none")
    assert got["accelerator"] == "directml"
    assert got["torch_index"] == "n/a"


def test_asking_for_cuda_without_a_card_falls_back_rather_than_installing_it():
    """Installing 2.5 GB of CUDA wheels that cannot run is worse than saying
    so and choosing something that works."""
    assert decisions(Accelerator="cuda",
                     ComputeCap="none")["accelerator"] == "directml"


@pytest.mark.parametrize("choice", ["directml", "cpu"])
def test_an_explicit_choice_is_obeyed_even_with_a_card_present(choice):
    """This machine has an NVIDIA GPU, so `auto` would pick CUDA. Someone who
    asked for something else meant it."""
    assert decisions(Accelerator=choice)["accelerator"] == choice


# -------------------------------------------------------------- the source

def test_the_app_comes_from_a_release_when_there_is_one():
    """And from the default branch when there is not, so the installer works
    before the first release is cut rather than failing on a 404."""
    assert decisions(ComputeCap="none")["app_source"]


# ------------------------------------------------------------- uninstalling
#
# Read against the uninstaller's source rather than by running it: running it
# needs a real 5.7 GB install to point at, and these are the properties that
# matter most, because getting them wrong deletes something that cannot be
# put back. The behaviour itself was exercised against a real install --
# user files survived, settings and the model cache were kept, the shortcut
# and the registry entry went.

def uninstaller_source() -> str:
    """The uninstaller, as the installer will write it out."""
    text = payload()
    start = text.index("Set-Content -Path (Join-Path $InstallDir 'uninstall.ps1')")
    return text[start:text.index("'@", start)]


def test_the_uninstaller_proves_the_folder_is_ours_before_deleting():
    """The guard that matters. Everything after it deletes recursively, so
    the target has to be positively identified -- not merely 'not obviously
    dangerous'. All three of these exist only in an install we made."""
    src = uninstaller_source()
    for proof in ("python\\python.exe", "app\\stereo360\\__init__.py",
                  "install-manifest.json"):
        assert proof in src, f"{proof} is not checked for"
    assert "Refusing" in src


@pytest.mark.parametrize("guard", [
    "GetPathRoot",                       # a drive root
    "$env:USERPROFILE",                  # the home folder
    "$env:windir",                       # Windows itself
    "MyDocuments",                       # Documents
    "Programs",                          # the parent of the install folder
])
def test_the_uninstaller_refuses_dangerous_targets(guard):
    """Verified against the real thing too: pointed at C:\\, the profile,
    Documents, LOCALAPPDATA, its parent and Windows, it refused all six."""
    assert guard in uninstaller_source()


def test_the_uninstaller_keeps_files_it_did_not_create():
    """The install folder is removed non-recursively, so it goes only if
    removing the recorded contents left it empty. Anything of the user's in
    there keeps the folder alive instead of being swept up with it."""
    src = uninstaller_source()
    assert "createdDirs" in src and "createdFiles" in src, \
        "it should remove a recorded list, not the folder wholesale"
    assert "it still holds files that were not ours" in src


def test_the_uninstaller_leaves_shared_data_alone_by_default():
    """The model cache is shared with every other tool that uses Hugging
    Face -- gigabytes that were never ours to delete."""
    src = uninstaller_source()
    assert "RemoveModelCache" in src and "RemoveSettings" in src
    assert "shared with" in src


def test_the_uninstall_wrapper_survives_deleting_itself():
    """`exit`, not `exit /b`, and everything on one line.

    The wrapper sits in the folder it is about to remove. cmd.exe reads a
    batch file incrementally from disk, so after the uninstaller has run it
    goes back for the next line and finds the file gone: "The system cannot
    find the path specified", exit 1, after a completely successful
    uninstall. Settings > Apps reads that code and reports a failure.

    Measured on a self-deleting batch: `exit /b` gives exit 1 and the error,
    plain `exit` gives exit 0 and silence.
    """
    src = payload()
    start = src.index("Set-Content -Path (Join-Path $InstallDir 'Uninstall stereo360.bat')")
    wrapper = src[src.index('@"', start): src.index('"@', start)]
    run_line = [ln for ln in wrapper.splitlines()
                if "uninstall.ps1" in ln and "powershell" in ln]
    assert len(run_line) == 1, "the call should be on exactly one line"
    line = run_line[0]
    assert "exit /b" not in line, \
        "exit /b sends cmd back to a file that no longer exists"
    assert line.rstrip().endswith("exit !RC!"), line
    assert "enabledelayedexpansion" in wrapper, \
        "%ERRORLEVEL% on one line is substituted before there is one"


def test_the_installer_registers_with_add_remove_programs():
    """So nobody has to know where it went. Per-user, under HKCU, which
    needs no elevation."""
    text = payload()
    assert "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall" in text
    for value in ("DisplayName", "UninstallString", "QuietUninstallString",
                  "InstallLocation", "EstimatedSize"):
        assert value in text, f"{value} is not registered"


def test_a_dry_run_touches_nothing(tmp_path):
    """It is the mode the tests run in, dozens of times. If it wrote to the
    install folder this suite would be downloading PyTorch on every run.

    Pointed at a path that definitely does not exist, rather than the real
    install folder -- that one may already be there, which would make the
    assertion pass without proving anything."""
    target = tmp_path / "would-be-install"
    decisions(ComputeCap="12.0", InstallDir=str(target))
    assert not target.exists(), "dry run created the install folder"
