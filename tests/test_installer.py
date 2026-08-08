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


# ------------------------------------------------- which ONNX Runtime wheel


def onnx_probe() -> str:
    """The Test-OnnxProvider probe, lifted out of the .bat as it ships."""
    text = payload()
    body = text[text.index("function Test-OnnxProvider"):]
    start = body.index("$probe = @'") + len("$probe = @'")
    return body[start:body.index("'@")].strip()


def _install_arm(kind: str) -> str:
    """The pip commands one accelerator branch runs.

    Scoped to the dependency switch, because the payload has an earlier switch
    over the same three values that only prints a description. Slicing the
    whole file on `'cuda' {` picks that one up instead, and in the wrong
    order -- which is how the first version of this test managed to assert
    against an empty string.
    """
    text = payload()
    switch = text[text.index('Write-Step "Installing the accelerator'):]
    switch = switch[:switch.index("Write-Step 'Checking the accelerator")]
    start = switch.index(f"'{kind}' {{")
    later = [switch.index(f"'{a}' {{") for a in ("cuda", "directml", "cpu")
             if a != kind and switch.index(f"'{a}' {{") > start]
    return switch[start:min(later)] if later else switch[start:]


def test_no_path_installs_onnxruntime_gpu():
    """It is the obvious wheel for an NVIDIA card and the wrong one. The
    published build ships no sm_120 kernels, so on any RTX 50 series card the
    CUDA provider fails with "no kernel image is available for execution on
    the device" -- the same failure Test-CudaReally exists to catch for torch,
    one layer along. DirectML runs on the same card at 12x the CPU speed.

    Asserted against the code, not the whole payload: the comments name the
    package on purpose, so that nobody reaches for it again.

    Comments are stripped rather than the pip lines being singled out. The
    first version of this test looked for a line containing both `Invoke-Pip`
    and the package, which passes happily when the argument list wraps -- and
    it does wrap, so the check was decorative. Verified by reintroducing the
    regression and watching it stay green.
    """
    offenders = []
    for kind in ("cuda", "directml", "cpu"):
        code = "\n".join(line for line in _install_arm(kind).splitlines()
                         if not line.lstrip().startswith("#"))
        if "onnxruntime-gpu" in code:
            offenders.append(f"{kind}: {code.strip()}")
    assert not offenders, "\n".join(offenders)


def test_every_gpu_path_installs_directml():
    """Including the CUDA one, which looks wrong. The two defaults use two
    runtimes: Depth Pro takes CUDA through torch, Depth Anything V3 takes the
    GPU through DirectML, and both end up accelerated on one machine."""
    cuda = _install_arm("cuda")
    assert "onnxruntime-directml" in cuda, \
        "the CUDA path leaves the ONNX default on the CPU"
    assert "--index-url" in cuda and "torch" in cuda, \
        "the CUDA path must still install a CUDA torch for Depth Pro"
    assert "onnxruntime-directml" in _install_arm("directml")
    assert "onnxruntime-directml" not in _install_arm("cpu"), \
        "a machine with no GPU has no use for it"


def test_a_gpu_install_is_verified_by_running_a_graph():
    """`get_available_providers()` lists what is installed, not what works --
    that distinction is the entire reason onnxruntime-gpu is not used here."""
    text = payload()
    assert "function Test-OnnxProvider" in text
    assert "Test-OnnxProvider $py" in text, "the probe is defined but not run"
    assert "InferenceSession" in text and "sess.run" in text, \
        "the probe must execute a graph, not just enumerate providers"


def test_the_probe_pins_the_model_ir_version():
    """`onnx` emits the newest IR version it knows and onnxruntime rejects
    anything above its own maximum. Unpinned, the first run of this probe
    failed with "Unsupported model IR version: 13" against a DirectML runtime
    that was working perfectly -- a check that cries wolf gets switched off."""
    assert "ir_version" in onnx_probe()


def test_the_probe_answers_rather_than_crashes():
    """Whatever this machine has, the probe must report OK or FAIL and not
    fall over: the installer reads its exit code and prints its output."""
    import sys

    proc = subprocess.run([sys.executable, "-c", onnx_probe()],
                          capture_output=True, text=True, timeout=300)
    out = proc.stdout.strip()
    assert out.startswith(("OK ", "FAIL ")), f"unusable output: {out!r}"
    assert "Traceback" not in proc.stderr, proc.stderr
    assert (proc.returncode == 0) == out.startswith("OK "), \
        "the exit code disagrees with the message"


def test_the_probe_passes_where_a_gpu_provider_is_installed():
    """The half that catches a broken probe rather than a broken GPU. Skipped
    on a CPU-only machine, which is the honest thing to do -- there FAIL is
    the right answer and proves nothing about the probe."""
    import sys

    ort = pytest.importorskip("onnxruntime")
    gpu = {"DmlExecutionProvider", "CUDAExecutionProvider",
           "CoreMLExecutionProvider"} & set(ort.get_available_providers())
    if not gpu:
        pytest.skip("no GPU execution provider installed here")

    proc = subprocess.run([sys.executable, "-c", onnx_probe()],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        f"a GPU provider is installed ({sorted(gpu)}) but the probe says:\n"
        f"{proc.stdout}{proc.stderr}")


# ------------------------------------------------------------- upgrading


def run_ps(script: str, tmp_path: Path) -> str:
    """Run PowerShell from a file, and return its stdout.

    From a file because these scripts carry Windows paths and PowerShell
    quoting, and passing them through -Command means escaping the same
    backslashes for Python, for the shell, and for PowerShell in turn. Two
    tests were written that way first and both failed on their own quoting
    rather than on anything they were testing.
    """
    f = tmp_path / "probe.ps1"
    f.write_text(script, encoding="ascii")
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(f)],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


#: Lifts Expand-Zip out of the shipped .bat and defines it. The bug was in that
#: function, so a reimplementation here would test the wrong code.
_LOAD_EXPAND_ZIP = """
$ErrorActionPreference = 'Stop'
$text = Get-Content -Raw -LiteralPath $env:STEREO360_INSTALLER
$start = $text.IndexOf('function Expand-Zip')
$end = $text.IndexOf('function Get-ComputeCapability')
Invoke-Expression $text.Substring($start, $end - $start)
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
"""


def test_extracting_over_an_existing_install_works(tmp_path, monkeypatch):
    """Running the installer twice used to be impossible.

    `ZipFile::ExtractToDirectory` throws on the first file that already
    exists, so a second run died at the first step that writes anything --
    "Installing Python" -- with a raw .NET exception naming python.exe. There
    was no upgrade path at all; the only route to a new version was uninstall
    and reinstall, re-downloading 2.5 GB of PyTorch to replace some .py files.
    """
    monkeypatch.setenv("STEREO360_INSTALLER", str(INSTALLER))
    t = tmp_path.as_posix()
    out = run_ps(_LOAD_EXPAND_ZIP + f"""
$src = '{t}/src'; $null = New-Item -ItemType Directory $src
Set-Content "$src/python.exe" 'v1' -Encoding ascii
$null = New-Item -ItemType Directory "$src/lib"
Set-Content "$src/lib/core.pyd" 'x' -Encoding ascii
$zip = '{t}/a.zip'
[System.IO.Compression.ZipFile]::CreateFromDirectory($src, $zip)

$dest = '{t}/dest'
Expand-Zip -Path $zip -Destination $dest
# What pip put there, which the zip knows nothing about.
$null = New-Item -ItemType Directory "$dest/site-packages"
Set-Content "$dest/site-packages/torch.txt" 'big' -Encoding ascii

Set-Content "$src/python.exe" 'v2' -Encoding ascii
Remove-Item $zip
[System.IO.Compression.ZipFile]::CreateFromDirectory($src, $zip)
Expand-Zip -Path $zip -Destination $dest

Write-Output ('exe=' + (Get-Content "$dest/python.exe"))
Write-Output ('kept=' + (Test-Path "$dest/site-packages/torch.txt"))
Write-Output ('nested=' + (Test-Path "$dest/lib/core.pyd"))
""", tmp_path)
    assert "exe=v2" in out, "the second extract did not overwrite"
    assert "kept=True" in out, \
        "site-packages was destroyed -- that is a 2.5 GB re-download"
    assert "nested=True" in out


def test_a_zip_cannot_escape_the_destination(tmp_path, monkeypatch):
    """Extracting entry by entry means doing the path check ourselves, which
    ExtractToDirectory did for us."""
    monkeypatch.setenv("STEREO360_INSTALLER", str(INSTALLER))
    t = tmp_path.as_posix()
    out = run_ps(_LOAD_EXPAND_ZIP + f"""
$zip = '{t}/evil.zip'
$fs = [System.IO.File]::Open($zip, 'Create')
$archive = New-Object System.IO.Compression.ZipArchive($fs, 'Create')
$entry = $archive.CreateEntry('../escaped.txt')
$w = New-Object System.IO.StreamWriter($entry.Open())
$w.Write('nope'); $w.Dispose(); $archive.Dispose(); $fs.Dispose()

try {{
    Expand-Zip -Path $zip -Destination '{t}/out'
    Write-Output 'ESCAPED'
}} catch {{ Write-Output 'REFUSED' }}
""", tmp_path)
    assert "REFUSED" in out, out
    assert not (tmp_path / "escaped.txt").exists()



def test_a_fresh_folder_is_reported_as_a_fresh_install(tmp_path):
    got = decisions(InstallDir=str(tmp_path / "nothing-here"))
    assert got["existing_install"] == "none"


def test_an_existing_install_is_recognised(tmp_path):
    """So the run can say what it is replacing instead of looking like a
    first-time install that mysteriously skips most of its work."""
    (tmp_path / "install-manifest.json").write_text(
        '{"app":"stereo360","installed":"2026-01-01T00:00:00",'
        '"accelerator":"cuda"}', encoding="utf-8")
    got = decisions(InstallDir=str(tmp_path))
    assert got["existing_install"] != "none"


def test_a_half_written_install_is_recognised_as_partial(tmp_path):
    """An interpreter with no manifest beside it is a run that died partway.
    Worth naming, because it is the case where the upgrade cannot take any of
    the shortcuts."""
    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "python.exe").write_text("", encoding="ascii")
    got = decisions(InstallDir=str(tmp_path))
    assert got["existing_install"] == "partial"


def test_an_upgrade_keeps_python_and_ffmpeg_when_they_are_good():
    """The two largest downloads after PyTorch. Kept on the strength of what
    they report when run, not on the file being present -- an interrupted
    download leaves a plausible-looking exe behind."""
    text = payload()
    assert "already installed here; keeping it and its packages" in text
    assert "already installed here; keeping it" in text
    # Both gates ask the binary, rather than testing for the path alone.
    assert "import sys; print('%d.%d.%d' % sys.version_info[:3])" in text
    assert r"ffmpeg version (\S+)" in text


def test_the_warm_up_caches_the_model_the_app_actually_defaults_to():
    """It cached Depth Anything V2 Base for a while after the defaults moved
    to V3 -- 400 MB per install for a model nothing would load, and the one
    that would was left to download on first use.

    Asserted as a coupling, not a name: the script imports DEFAULT_VARIANT
    from the app rather than spelling a model out, so the two cannot drift
    again."""
    text = payload()
    assert "Depth-Anything-V2" not in text, "still caching the old default"
    assert "from stereo360.depth.depth_anything_v3 import DEFAULT_VARIANT" in text
    assert "download(DEFAULT_VARIANT)" in text


def test_depth_pro_is_not_pre_fetched_but_is_announced():
    """3.6 GB is too much to put on someone who may only convert video. That
    makes the first still conversion slow, so the install says so."""
    text = payload()
    assert "3.6 GB" in text
    warm = text[text.index("Write-Step 'Fetching the depth model'"):]
    warm = warm[:warm.index("Write-Step 'Creating shortcuts'")]
    assert "DepthPro" not in warm, "pre-fetching 3.6 GB on every install"
