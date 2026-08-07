"""The installer's accelerator choice, which is the part that goes wrong.

Everything else the installer does is downloading and unzipping, and fails
loudly when it fails. Picking the PyTorch build does not. Measured on an RTX
5070 Ti against a cu118 build: it installs cleanly, reports
`torch.cuda.is_available() == True`, and kernels genuinely *run* -- CUDA
JIT-compiles the embedded PTX. Nothing looks wrong and nothing is using the
tuned code for the card. That decision is worth pinning.

Driven through the real script in `-DryRun`, so what is tested is the code
that ships rather than a Python transcription of it. The script prints its
choices as `DECISION key=value` on stdout for exactly this purpose.
"""

import platform
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parent.parent / "installer" / "install.ps1"

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows" or shutil.which("powershell") is None,
    reason="the installer is Windows-only and needs powershell")


def decisions(**params) -> dict:
    """Run the installer's decision phase and return what it chose."""
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
           "-File", str(INSTALLER), "-DryRun"]
    for key, value in params.items():
        cmd += [f"-{key}", str(value)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"installer failed:\n{proc.stdout}\n{proc.stderr}"
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("DECISION "):
            key, _, value = line[len("DECISION "):].partition("=")
            out[key] = value.strip()
    assert out, f"no decisions printed:\n{proc.stdout}"
    return out


def test_the_installer_exists_and_is_self_contained():
    """Both halves ship together: the .bat is what a person can double-click,
    and it refuses helpfully if separated from the script it calls."""
    assert INSTALLER.exists()
    bat = INSTALLER.parent / "Install stereo360.bat"
    assert bat.exists()
    text = bat.read_text(encoding="utf-8", errors="replace")
    assert "-ExecutionPolicy Bypass" in text, \
        "a downloaded .ps1 will not run under the default policy"
    assert "install.ps1 is missing" in text, "should say so rather than flash"


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


def test_a_dry_run_touches_nothing(tmp_path):
    """It is the mode the tests run in, dozens of times. If it wrote to the
    install folder this suite would be downloading PyTorch on every run.

    Pointed at a path that definitely does not exist, rather than the real
    install folder -- that one may already be there, which would make the
    assertion pass without proving anything."""
    target = tmp_path / "would-be-install"
    decisions(ComputeCap="12.0", InstallDir=str(target))
    assert not target.exists(), "dry run created the install folder"
