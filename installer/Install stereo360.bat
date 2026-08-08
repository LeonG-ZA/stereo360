@echo off
rem ===========================================================================
rem  stereo360 installer  --  monoscopic 360 video to stereoscopic 3D for VR
rem
rem  Double-click this. It does not matter which folder it is in: your
rem  Downloads folder is fine and is what this expects. It picks its own
rem  install location and downloads everything it needs.
rem
rem  ONE file, deliberately. A .bat beside a .ps1 invites being separated --
rem  people download one of the two, or move only the one they were told to
rem  double-click. Everything below the marker at the bottom is PowerShell,
rem  kept as plain readable text rather than encoded, so anyone wary of an
rem  unsigned installer can open this in Notepad and see exactly what it does
rem  before running it.
rem
rem  The wrapper exists because a .ps1 cannot be double-clicked at all:
rem  Explorer opens it in Notepad, and a file downloaded from the internet
rem  carries a Mark-of-the-Web that the default execution policy refuses.
rem ===========================================================================

setlocal
title Installing stereo360

rem Paths go through the environment rather than into the command text, so a
rem folder name containing a quote or an ampersand cannot break the command.
set "SELF=%~f0"
set "PSFILE=%TEMP%\stereo360_install_%RANDOM%%RANDOM%.ps1"

rem PowerShell does the unpacking, not `more` or `findstr`: both expand tabs,
rem cut long lines and can leave a stray control character behind, and this
rem payload has to come out byte for byte. Splitting on the marker and taking
rem the LAST piece steps neatly over the copy of the marker in this command.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$raw = Get-Content -LiteralPath $env:SELF -Raw; $tail = ($raw -split '#@@ POWERSHELL PAYLOAD STARTS ON THE NEXT LINE @@')[-1]; [IO.File]::WriteAllText($env:PSFILE, $tail, (New-Object Text.UTF8Encoding $false))"

if not exist "%PSFILE%" (
    echo.
    echo   Could not unpack the installer.
    echo   This usually means PowerShell is blocked on this machine.
    echo.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PSFILE%" %*
set "RC=%ERRORLEVEL%"
del "%PSFILE%" >nul 2>&1

echo.
if "%RC%"=="0" (
    echo   Finished. You can close this window.
) else (
    echo   The installer stopped with error %RC%.
    echo   The messages above say where. Nothing outside the install folder
    echo   was changed, so it is safe to fix the problem and run this again.
)
echo.
pause
exit /b %RC%

rem  Nothing below this line is ever read by cmd.exe -- it stops at the
rem  `exit /b` above -- so the payload may contain % and other characters
rem  that would otherwise need escaping. It must stay pure ASCII though;
rem  a test enforces that.
#@@ POWERSHELL PAYLOAD STARTS ON THE NEXT LINE @@
<#
.SYNOPSIS
    Installs stereo360 and everything it needs, on a machine with nothing.

.DESCRIPTION
    Written because the manual install asks too much: a Python, a PyTorch
    build matched to your GPU, ffmpeg on PATH, and model weights. Getting the
    PyTorch build wrong is the worst of these, because it does not announce
    itself. `torch.cuda.is_available()` returns True on a mismatched build,
    and -- measured on an RTX 5070 Ti against cu118 -- kernels still *run*,
    JIT-compiled from the embedded PTX. Everything looks fine and nothing is
    using the tuned code paths for the card. So this checks that the build
    actually contains an architecture the GPU can use, rather than trusting
    either flag or a smoke test.

    Nothing here touches the system. No admin, no PATH changes, no existing
    Python: an embeddable interpreter goes in the install folder and stays
    there. Uninstalling is deleting the folder.

    This script does not care where it is run from -- people double-click
    installers in their Downloads folder, and that is the correct thing to
    expect. It chooses its own location and fetches the application itself.

.PARAMETER InstallDir
    Where to put it. Defaults to %LOCALAPPDATA%\Programs\stereo360, which
    needs no elevation.

.PARAMETER Accelerator
    auto (default), cuda, directml or cpu. `auto` looks for an NVIDIA card
    and falls back to directml, which runs on any Direct3D 12 GPU -- AMD and
    Intel included -- without the 2.5 GB PyTorch download.

.PARAMETER DryRun
    Decide everything and print it, download nothing. The decisions are
    printed as `DECISION key=value` lines so they can be tested.

.PARAMETER ComputeCap
    Pretend the GPU has this compute capability, e.g. 12.0. For testing the
    selection logic against cards that are not in this machine.
#>
[CmdletBinding()]
param(
    # ...\Programs\stereo360, not ...\stereo360. Qt derives its own cache and
    # settings path from the application and organisation names, both of
    # which are "stereo360", so it already owns %LOCALAPPDATA%\stereo360.
    # Installing there would mix program files into the user's data and make
    # "uninstall by deleting the folder" throw away their settings too.
    # %LOCALAPPDATA%\Programs is where per-user installs belong anyway.
    [string] $InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs\stereo360'),
    [ValidateSet('auto', 'cuda', 'directml', 'cpu')]
    [string] $Accelerator = 'auto',
    [switch] $DryRun,
    [string] $ComputeCap = '',
    [string] $Repo = 'LeonG-ZA/stereo360'
)

$ErrorActionPreference = 'Stop'

# Many of the hosts below refuse anything older, and Windows PowerShell 5.1
# still negotiates TLS 1.0 by default. Without this the first download fails
# with an unhelpful "connection was closed" error.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# ---------------------------------------------------------------- constants

# 3.12 rather than the newest: every wheel this needs has been published for
# it for years. The README asks for 3.10+, so this is comfortably inside.
$PythonVersion = '3.12.8'
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$GetPipUrl = 'https://bootstrap.pypa.io/get-pip.py'

# A GPL build, with libx264 and libx265 and without libfdk_aac. That is not an
# oversight: ffmpeg built with libfdk_aac requires --enable-nonfree, and its
# own licence says the result may not be redistributed. stereo360 detects
# libfdk_aac when it is present and says so when it is not, so anyone who
# wants it can drop their own ffmpeg.exe in here.
#
# Gyan's numbered *releases*, not a nightly, and the reason is Smart App
# Control -- on by default in Windows 11, and it blocks unsigned programs it
# has no reputation for. Every ffmpeg build is unsigned, so reputation is all
# there is, and a nightly gets a brand new hash every day and never earns
# any. Measured here: a freshly downloaded nightly came back "An Application
# Control policy has blocked this file" while a Gyan release ran fine.
#
# Trust follows the file, not how it arrived -- the same bytes copied out of
# winget's package folder into a temp directory still ran -- so downloading
# the widely used build directly is enough. winget is not needed, and would
# install outside this folder and put ffmpeg on PATH, which this deliberately
# does not do.
#
# "essentials" rather than "full": 106 MB against 240 MB, and it still has
# libx264, libx265 and libopus, which is everything stereo360 asks for.
$FfmpegRepo = 'GyanD/codexffmpeg'
$FfmpegAssetPattern = '*essentials_build.zip'
# Used only if the release API cannot be reached.
$FfmpegUrlFallback = 'https://github.com/GyanD/codexffmpeg/releases/download/9.0/ffmpeg-9.0-essentials_build.zip'

# cu128 carries cubins for sm_70 through sm_120 -- Volta to Blackwell, which
# is every consumer card since 2017 and includes the 50 series. Ada (sm_89)
# is not listed explicitly and does not need to be: CUDA guarantees binary
# compatibility within a major generation, so the sm_86 cubin runs on it.
$CudaIndexModern = 'https://download.pytorch.org/whl/cu128'
# Pascal and older. Anything this old is slow enough that CPU is not far off.
$CudaIndexLegacy = 'https://download.pytorch.org/whl/cu118'

$MinFreeGb = 8

# Per-user, under HKCU, so listing the app in Settings > Apps needs no
# elevation. Without this entry stereo360 does not appear there at all, and
# removing it means knowing where it was installed -- which is exactly what
# nobody remembers.
$ArpKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\stereo360'

# ------------------------------------------------------------------ display

$script:StepNo = 0
$script:StepTotal = 15

function Write-Step {
    param([string] $Text)
    $script:StepNo++
    $pct = [int](100 * ($script:StepNo - 1) / $script:StepTotal)
    Write-Progress -Activity 'Installing stereo360' -Status $Text -PercentComplete $pct
    Write-Host ''
    Write-Host ("[{0}/{1}] {2}" -f $script:StepNo, $script:StepTotal, $Text) -ForegroundColor Cyan
}

function Write-Detail { param([string] $Text) Write-Host "      $Text" -ForegroundColor DarkGray }
function Write-Good   { param([string] $Text) Write-Host "      $Text" -ForegroundColor Green }
function Write-Warn   { param([string] $Text) Write-Host "      $Text" -ForegroundColor Yellow }
function Write-Decision {
    <#
        Write-Output, deliberately, not Write-Host. Everything else here
        writes to the host for colour, which is invisible to a pipeline --
        these lines are the machine-readable part, and a test that could not
        capture them would be testing nothing.
    #>
    param([string] $Key, [string] $Value)
    Write-Output ("DECISION {0}={1}" -f $Key, $Value)
}

# ---------------------------------------------------------------- downloads

function Get-Download {
    <#
        Streams to disk with a real byte count. The PyTorch wheel is about
        2.5 GB, and a spinner against a download that size reads as a hang.
    #>
    param([string] $Url, [string] $Destination, [string] $Label)

    Add-Type -AssemblyName System.Net.Http
    $client = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromMinutes(60)
    $client.DefaultRequestHeaders.Add('User-Agent', 'stereo360-installer')
    try {
        $resp = $client.GetAsync($Url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
        if (-not $resp.IsSuccessStatusCode) {
            throw "$Label download failed: HTTP $([int]$resp.StatusCode) from $Url"
        }
        $total = $resp.Content.Headers.ContentLength
        $in = $resp.Content.ReadAsStreamAsync().Result
        $out = [System.IO.File]::Create($Destination)
        try {
            $buf = New-Object byte[] 1048576
            $done = 0L
            $lastShown = -1
            while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
                $out.Write($buf, 0, $n)
                $done += $n
                if ($total) {
                    $pct = [int](100 * $done / $total)
                    if ($pct -ne $lastShown) {
                        $lastShown = $pct
                        Write-Progress -Activity "Downloading $Label" `
                            -Status ("{0:N0} MB of {1:N0} MB" -f ($done / 1MB), ($total / 1MB)) `
                            -PercentComplete $pct -Id 1
                    }
                } else {
                    Write-Progress -Activity "Downloading $Label" `
                        -Status ("{0:N0} MB" -f ($done / 1MB)) -Id 1
                }
            }
        } finally {
            $out.Close(); $in.Close()
        }
        Write-Progress -Activity "Downloading $Label" -Completed -Id 1
        Write-Detail ("downloaded {0:N0} MB" -f ((Get-Item $Destination).Length / 1MB))
    } finally {
        $client.Dispose()
    }
}

function Expand-Zip {
    param([string] $Path, [string] $Destination)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (-not (Test-Path $Destination)) { New-Item -ItemType Directory -Path $Destination -Force | Out-Null }
    [System.IO.Compression.ZipFile]::ExtractToDirectory($Path, $Destination)
}

# ------------------------------------------------------------ gpu detection

function Get-ComputeCapability {
    <#
        Straight from nvidia-smi, so it works before any Python exists.
        Returns '' when there is no NVIDIA card.
    #>
    # 'none' simulates a machine with no NVIDIA card, which is otherwise
    # untestable on a machine that has one.
    if ($ComputeCap -eq 'none') { return '' }
    if ($ComputeCap) { return $ComputeCap }
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return '' }
    try {
        $out = Invoke-Native { & $smi.Source --query-gpu=compute_cap --format=csv,noheader }
    } catch {
        return ''
    }
    if ($LASTEXITCODE -ne 0 -or -not $out) { return '' }
    return ($out | Select-Object -First 1).ToString().Trim()
}

function Select-CudaIndex {
    param([string] $Cap)
    if (-not $Cap) { return '' }
    $major = 0
    [void][int]::TryParse(($Cap -split '\.')[0], [ref]$major)
    if ($major -ge 7) { return $CudaIndexModern }
    return $CudaIndexLegacy
}

function Resolve-Accelerator {
    <#
        Returns a hashtable: kind, cap, index.

        `auto` prefers CUDA when an NVIDIA card is present and DirectML
        otherwise. DirectML is not a consolation prize -- it runs on any
        Direct3D 12 GPU, so AMD and Intel users get GPU depth without the
        2.5 GB PyTorch download.
    #>
    $cap = ''
    if ($Accelerator -ne 'cpu' -and $Accelerator -ne 'directml') {
        $cap = Get-ComputeCapability
    }
    $kind = $Accelerator
    if ($kind -eq 'auto') {
        if ($cap) { $kind = 'cuda' } else { $kind = 'directml' }
    }
    if ($kind -eq 'cuda' -and -not $cap) {
        # Asked for explicitly but there is nothing to run it on. Say so
        # rather than installing 2.5 GB of CUDA wheels that cannot be used.
        Write-Warn 'CUDA was requested but there is no NVIDIA GPU here'
        $kind = 'directml'
    }
    return @{ kind = $kind; cap = $cap; index = (Select-CudaIndex $cap) }
}

# ------------------------------------------------------------------ the run

function Invoke-Native {
    <#
        Runs an external program without letting its stderr kill the script.

        With $ErrorActionPreference = 'Stop', Windows PowerShell turns
        anything a native program writes to stderr into a *terminating*
        error, whatever its exit code. pip writes ordinary progress and retry
        warnings there, so a single "Retrying... Read timed out" -- which pip
        then recovers from by itself -- aborted the whole install.

        Exit codes are the only reliable signal from a native program, so
        that is what every caller checks.
    #>
    param([scriptblock] $Command)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Command } finally { $ErrorActionPreference = $previous }
}

function Invoke-Pip {
    param([string] $Python, [string[]] $PipArgs, [string] $Label)
    # Generous retries and timeout: the people this installer is for are not
    # all on a developer's connection, and the wheels are large. pip's
    # defaults give up after 15 seconds, which a slow link reaches easily.
    $full = @('--disable-pip-version-check', '--retries', '5', '--timeout', '60') + $PipArgs
    Write-Detail "pip $($PipArgs -join ' ')"
    Invoke-Native { & $Python -m pip @full }
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (pip exit $LASTEXITCODE)" }
}

function Test-CudaReally {
    <#
        The check that matters, and the reason this installer exists.

        `torch.cuda.is_available()` reports that a driver and a device are
        present, not that this build contains code for that device. On an
        RTX 50 series card with a pre-cu128 build it returns True and then
        every kernel launch fails with "no kernel image is available for
        execution on the device" -- hours into a render.

        So: launch an actual kernel and read the result back.
    #>
    param([string] $Python)
    $probe = @'
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("FAIL torch.cuda.is_available() is False"); sys.exit(1)

    major, minor = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()

    # Running a kernel is NOT enough on its own. Measured on an RTX 5070 Ti
    # (sm_120) against a cu118 build: the matmul below *succeeds*, because
    # CUDA JIT-compiles the embedded compute_37 PTX at runtime. It works and
    # it is wrong -- no tuned kernel for the card, and PyTorch prints its own
    # "if you want to use the ... GPU" warning while doing it.
    #
    # So check the build actually contains code for this architecture. CUDA
    # is binary compatible within a major generation, so any sm_{major}{n}
    # with n <= this card's minor will do -- which is why Ada (sm_89) is
    # happy on a build that only lists sm_86.
    native = [a for a in archs
              if a.startswith("sm_%d" % major)
              and a[len("sm_%d" % major):].isdigit()
              and int(a[len("sm_%d" % major):]) <= minor]
    if not native:
        print("FAIL this build has no code for sm_%d%d -- it lists %s. "
              "It would run through PTX JIT, slowly."
              % (major, minor, ",".join(archs)))
        sys.exit(1)

    x = torch.randn(512, 512, device="cuda")
    val = float((x @ x).sum().item())          # a real kernel, read back
    torch.cuda.synchronize()
    if val != val:                              # NaN means it did not run
        print("FAIL kernel produced NaN"); sys.exit(1)

    print("OK %s on %s (sm_%d%d, using %s)"
          % (torch.__version__, torch.cuda.get_device_name(0),
             major, minor, native[-1]))
except Exception as exc:
    print("FAIL %s: %s" % (type(exc).__name__, exc)); sys.exit(1)
'@
    $file = Join-Path $env:TEMP 'stereo360_cuda_probe.py'
    Set-Content -Path $file -Value $probe -Encoding utf8
    $result = (Invoke-Native { & $Python $file 2>&1 }) | Out-String
    Remove-Item $file -ErrorAction SilentlyContinue
    return @{ ok = ($LASTEXITCODE -eq 0); detail = $result.Trim() }
}

function Test-OnnxProvider {
    <#
        The same question as Test-CudaReally, asked of the other runtime.

        The default video model is an ONNX graph, so "is there a GPU provider"
        is only half of it -- onnxruntime-gpu on a Blackwell card *offers*
        CUDAExecutionProvider and then fails on the first kernel, because the
        published wheel carries no sm_120 code. Listing providers would report
        success. So build a session on the best one and run it.

        A tiny hand-written graph rather than the real depth model: this runs
        before any model has been downloaded, and 105 MB is not a thing to
        fetch just to answer a yes/no question.
    #>
    param([string] $Python)
    $probe = @'
import sys
try:
    import numpy as np, onnxruntime as ort
    from onnx import TensorProto, helper

    order = ["DmlExecutionProvider", "CUDAExecutionProvider",
             "CoreMLExecutionProvider"]
    have = ort.get_available_providers()
    gpu = next((p for p in order if p in have), None)
    if gpu is None:
        print("FAIL no GPU provider installed; have %s" % ",".join(have))
        sys.exit(1)

    # One convolution: enough to need a real kernel for the device, which is
    # precisely what a mismatched build does not have.
    node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3],
                            pads=[1, 1, 1, 1])
    graph = helper.make_graph(
        [node], "probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 32, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 32, 32])],
        [helper.make_tensor("w", TensorProto.FLOAT, [8, 4, 3, 3],
                            np.full(8 * 4 * 3 * 3, 0.05, np.float32))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    # Pin the IR version. `onnx` defaults to the newest it knows, and
    # onnxruntime rejects anything above its own maximum -- the first run of
    # this probe died on "Unsupported model IR version: 13, max supported
    # 11" against a DirectML runtime that was working perfectly. A probe that
    # fails when the thing it tests is fine is worse than no probe.
    model.ir_version = 10

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=[gpu])
    used = sess.get_providers()[0]
    if used != gpu:
        print("FAIL asked for %s, got %s" % (gpu, used)); sys.exit(1)

    out = sess.run(None, {"x": np.ones((1, 4, 32, 32), np.float32)})[0]
    if not np.isfinite(out).all():
        print("FAIL %s produced non-finite output" % gpu); sys.exit(1)
    print("OK onnxruntime %s ran on %s" % (ort.__version__, used))
except Exception as exc:
    print("FAIL %s: %s" % (type(exc).__name__, exc)); sys.exit(1)
'@
    $file = Join-Path $env:TEMP 'stereo360_onnx_probe.py'
    Set-Content -Path $file -Value $probe -Encoding utf8
    $result = ((Invoke-Native { & $Python $file 2>&1 }) | Out-String).Trim()
    Remove-Item $file -ErrorAction SilentlyContinue
    $ok = ($LASTEXITCODE -eq 0)
    $provider = 'none'
    if ($ok -and $result -match 'ran on (\w+)') { $provider = $Matches[1] }
    return @{ ok = $ok; detail = $result; provider = $provider }
}

function Get-AppRelease {
    <#
        The latest published release, or the default branch if there is not
        one yet. Prefers a named .zip asset over the source zipball, so a
        future release can ship something trimmed without changing this.
    #>
    param([string] $Repo)
    $api = "https://api.github.com/repos/$Repo/releases/latest"
    try {
        $rel = Invoke-RestMethod -Uri $api -Headers @{ 'User-Agent' = 'stereo360-installer' }
    } catch {
        return @{ url = "https://github.com/$Repo/archive/refs/heads/main.zip"
                  name = 'main branch (no published release yet)' }
    }
    $asset = $rel.assets | Where-Object { $_.name -like '*.zip' } | Select-Object -First 1
    if ($asset) { return @{ url = $asset.browser_download_url; name = $rel.tag_name } }
    return @{ url = $rel.zipball_url; name = $rel.tag_name }
}

# ============================================================== main

Write-Host ''
Write-Host '  stereo360 installer' -ForegroundColor White
Write-Host '  monoscopic 360 video -> stereoscopic 3D for VR' -ForegroundColor DarkGray
Write-Host ''

Write-Step 'Checking this machine'
if ([IntPtr]::Size -ne 8) { throw '64-bit Windows is required.' }
$drive = (Split-Path -Qualifier $InstallDir)
$free = (Get-PSDrive -Name $drive.TrimEnd(':')).Free / 1GB
Write-Detail ("Windows {0}, {1:N0} GB free on {2}" -f [Environment]::OSVersion.Version, $free, $drive)
if ($free -lt $MinFreeGb) {
    throw ("Need about $MinFreeGb GB free on $drive, found {0:N1} GB." -f $free)
}
Write-Detail "install folder: $InstallDir"

Write-Step 'Choosing the accelerator'
$acc = Resolve-Accelerator
# One step more for DirectML, which has to build its own depth model.
if ($acc.kind -eq 'directml') { $script:StepTotal = 16 }
if ($acc.cap) { Write-Detail "NVIDIA GPU, compute capability $($acc.cap)" }
else { Write-Detail 'no NVIDIA GPU detected' }
Write-Decision 'accelerator' $acc.kind
Write-Decision 'compute_cap' $(if ($acc.cap) { $acc.cap } else { 'none' })
Write-Decision 'torch_index' $(if ($acc.kind -eq 'cuda') { $acc.index } else { 'n/a' })
switch ($acc.kind) {
    'cuda'     { Write-Detail 'PyTorch with CUDA -- about 2.5 GB, the long step' }
    'directml' { Write-Detail 'ONNX Runtime with DirectML -- runs on any Direct3D 12 GPU, about 200 MB' }
    'cpu'      { Write-Detail 'CPU only -- depth is roughly ten times slower' }
}

$release = Get-AppRelease -Repo $Repo
Write-Decision 'app_source' $release.name

if ($DryRun) {
    Write-Progress -Activity 'Installing stereo360' -Completed
    Write-Host ''
    Write-Host '  Dry run: nothing was downloaded or installed.' -ForegroundColor Yellow
    Write-Host ''
    exit 0
}

# ---- python -------------------------------------------------------------
Write-Step "Installing Python $PythonVersion (private to this folder)"
$py = Join-Path $InstallDir 'python\python.exe'
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
$tmp = Join-Path $InstallDir 'tmp'
New-Item -ItemType Directory -Path $tmp -Force | Out-Null
$zip = Join-Path $tmp 'python.zip'
Get-Download -Url $PythonUrl -Destination $zip -Label 'Python'
Expand-Zip -Path $zip -Destination (Join-Path $InstallDir 'python')
Remove-Item $zip
# The embeddable build ships with site-packages disabled, so pip installs
# would land somewhere nothing imports from. One commented line in the ._pth
# is the whole difference.
$pth = Get-ChildItem (Join-Path $InstallDir 'python') -Filter '*._pth' | Select-Object -First 1
(Get-Content $pth.FullName) -replace '^#\s*import site', 'import site' |
    Set-Content $pth.FullName -Encoding ascii
Write-Good 'Python ready'

Write-Step 'Bootstrapping pip'
$getpip = Join-Path $tmp 'get-pip.py'
Get-Download -Url $GetPipUrl -Destination $getpip -Label 'get-pip'
Invoke-Native { & $py $getpip --no-warn-script-location }
if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed' }
Remove-Item $getpip
Write-Good 'pip ready'

# ---- the app ------------------------------------------------------------
Write-Step "Downloading stereo360 ($($release.name))"
$appzip = Join-Path $tmp 'app.zip'
Get-Download -Url $release.url -Destination $appzip -Label 'stereo360'
$unpack = Join-Path $tmp 'app'
Expand-Zip -Path $appzip -Destination $unpack
# A GitHub archive wraps everything in one directory whose name carries the
# commit; the release asset form may not. Handle both by looking for the
# package rather than assuming a shape.
$appRoot = Get-ChildItem $unpack -Recurse -Directory -Filter 'stereo360' |
    Where-Object { Test-Path (Join-Path $_.FullName '__init__.py') } |
    Select-Object -First 1
if (-not $appRoot) { throw 'could not find the stereo360 package in the download' }
$src = Split-Path $appRoot.FullName -Parent
$dest = Join-Path $InstallDir 'app'
if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
Move-Item $src $dest
Remove-Item $appzip; Remove-Item $unpack -Recurse -Force -ErrorAction SilentlyContinue

# Put the app on the interpreter's path, permanently.
#
# An embeddable Python's ._pth *replaces* the normal sys.path computation
# rather than adding to it, and the current directory is not part of what it
# builds -- the "." line means the folder holding python.exe. So `-m
# stereo360` fails with "No module named stereo360" no matter what directory
# it is run from, which would have broken both launchers and not just the
# self-test. One relative line fixes it for every entry point at once.
$appLine = '..\app'
if ((Get-Content $pth.FullName) -notcontains $appLine) {
    (Get-Content $pth.FullName) + $appLine | Set-Content $pth.FullName -Encoding ascii
}
Write-Good "stereo360 in $dest"

# ---- dependencies -------------------------------------------------------
Write-Step "Installing the accelerator ($($acc.kind))"
switch ($acc.kind) {
    'cuda' {
        Invoke-Pip $py @('install', '--no-warn-script-location', '--index-url', $acc.index, 'torch', 'torchvision') 'PyTorch'
        # DirectML on an NVIDIA card, which looks wrong and is not.
        #
        # The two defaults use different runtimes: Depth Pro is torch, and gets
        # CUDA from the line above. Depth Anything V3 is an ONNX graph, and the
        # obvious wheel for it here -- onnxruntime-gpu -- ships no sm_120
        # kernels, so on any RTX 50 series card its CUDA provider dies with
        # "no kernel image is available for execution on the device". Same
        # failure Test-CudaReally exists to catch for torch, one layer along.
        #
        # DirectML goes through Direct3D 12 and does not care whose GPU it is.
        # Measured on an RTX 5070 Ti, six cubemap faces in one call: 1.91 s on
        # the CPU provider against 0.15 s on DirectML, and bit-identical output
        # -- max absolute difference 0.00000 against the CPU result, which
        # matters because every score in findings.md was measured on CPU.
        #
        # From PyPI rather than $acc.index, which carries no ORT wheels.
        # `onnx` comes along for Test-OnnxProvider, which builds a throwaway
        # graph to check the GPU can actually execute one.
        Invoke-Pip $py @('install', '--no-warn-script-location',
                         'onnxruntime-directml', 'onnx') 'ONNX Runtime (DirectML)'
    }
    'directml' {
        # onnx and onnxscript are for the *exporter*, not for running: the
        # DirectML path needs a model of its own, built a few steps below.
        #
        # Deliberately not `-r requirements-onnx.txt`, which asks for plain
        # onnxruntime. That and onnxruntime-directml both install a module
        # called onnxruntime, and whichever lands second wins -- a CPU-only
        # runtime silently replacing the GPU one is exactly the failure this
        # whole path exists to avoid.
        Invoke-Pip $py @('install', '--no-warn-script-location', 'torch',
                         'torchvision', 'onnxruntime-directml', 'onnx',
                         'onnxscript') 'ONNX Runtime (DirectML)'
    }
    'cpu' {
        Invoke-Pip $py @('install', '--no-warn-script-location', 'torch', 'torchvision', 'onnxruntime') 'PyTorch (CPU)'
    }
}

Write-Step 'Checking the accelerator actually runs'
if ($acc.kind -eq 'cuda') {
    $check = Test-CudaReally $py
    if ($check.ok) {
        Write-Good $check.detail
    } else {
        Write-Warn 'CUDA did not run a test kernel:'
        Write-Warn $check.detail
        if ($acc.index -ne $CudaIndexLegacy) {
            Write-Warn 'retrying with the older CUDA build'
            Invoke-Pip $py @('install', '--no-warn-script-location', '--force-reinstall',
                             '--index-url', $CudaIndexLegacy, 'torch', 'torchvision') 'PyTorch (cu118)'
            $check = Test-CudaReally $py
        }
        if (-not $check.ok) {
            Write-Warn 'falling back to CPU. Depth will be roughly ten times slower.'
            Invoke-Pip $py @('install', '--no-warn-script-location', '--force-reinstall', 'torch', 'torchvision') 'PyTorch (CPU)'
            $acc.kind = 'cpu'
        } else {
            Write-Good $check.detail
        }
    }
} else {
    Write-Detail 'no CUDA build to verify for this accelerator'
}

# Both GPU paths now install an ONNX runtime, and an installed provider is not
# a working one -- that is the whole lesson of the CUDA check above, and the
# reason onnxruntime-gpu is not used here. So run the graph.
if ($acc.kind -ne 'cpu') {
    $ort = Test-OnnxProvider $py
    if ($ort.ok) { Write-Good $ort.detail }
    else {
        Write-Warn 'the GPU could not run an ONNX graph:'
        Write-Warn $ort.detail
        Write-Warn 'video depth will run on the processor. Everything works,'
        Write-Warn 'it is just slower.'
    }
    Write-Decision 'onnx_provider' $ort.provider
}
Write-Decision 'accelerator_final' $acc.kind

Write-Step 'Installing core dependencies'
Invoke-Pip $py @('install', '--no-warn-script-location', '-r', (Join-Path $dest 'requirements.txt')) 'core dependencies'
Write-Good 'numpy, opencv, transformers, Pillow'

Write-Step 'Installing the desktop interface'
Invoke-Pip $py @('install', '--no-warn-script-location', '-r', (Join-Path $dest 'requirements-ui.txt')) 'PySide6'
Write-Good 'PySide6'

# ---- ffmpeg -------------------------------------------------------------
Write-Step 'Installing ffmpeg'
$ffzip = Join-Path $tmp 'ffmpeg.zip'
# Resolve the newest numbered release rather than pinning a version that
# would quietly rot. Falls back to a known-good one if the API is unreachable.
$ffUrl = $FfmpegUrlFallback
try {
    $ffRel = Invoke-RestMethod "https://api.github.com/repos/$FfmpegRepo/releases/latest" `
        -Headers @{ 'User-Agent' = 'stereo360-installer' }
    $asset = $ffRel.assets | Where-Object { $_.name -like $FfmpegAssetPattern } |
        Select-Object -First 1
    if ($asset) {
        $ffUrl = $asset.browser_download_url
        Write-Detail "ffmpeg $($ffRel.tag_name) (Gyan release build)"
    }
} catch {
    Write-Detail 'could not reach the release API; using the known-good build'
}
Get-Download -Url $ffUrl -Destination $ffzip -Label 'ffmpeg'
$ffdir = Join-Path $tmp 'ffmpeg'
Expand-Zip -Path $ffzip -Destination $ffdir
$bin = Get-ChildItem $ffdir -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1
$ffDest = Join-Path $InstallDir 'ffmpeg'
New-Item -ItemType Directory -Path $ffDest -Force | Out-Null
Copy-Item (Join-Path $bin.DirectoryName '*.exe') $ffDest -Force
Remove-Item $ffzip; Remove-Item $ffdir -Recurse -Force -ErrorAction SilentlyContinue
Write-Good 'ffmpeg and ffprobe (GPL build, no libfdk_aac -- see the README)'

# ---- model --------------------------------------------------------------
Write-Step 'Fetching the depth model'
$env:PATH = "$ffDest;$env:PATH"
$warm = Join-Path $tmp 'warm.py'
Set-Content -Path $warm -Encoding utf8 -Value @'
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
name = "depth-anything/Depth-Anything-V2-Base-hf"
AutoImageProcessor.from_pretrained(name)
AutoModelForDepthEstimation.from_pretrained(name)
print("cached", name)
'@
Invoke-Native { & $py $warm }
if ($LASTEXITCODE -ne 0) {
    Write-Warn 'could not pre-fetch the model; it will download on first use'
} else {
    Write-Good 'Depth Anything V2 Base cached'
}
Remove-Item $warm -ErrorAction SilentlyContinue

# ---- launchers ----------------------------------------------------------
Write-Step 'Creating shortcuts'
$launch = Join-Path $InstallDir 'stereo360.bat'
Set-Content -Path $launch -Encoding ascii -Value @"
@echo off
set "PATH=%~dp0ffmpeg;%PATH%"
cd /d "%~dp0app"
"%~dp0python\python.exe" -m stereo360 %*
"@
$launchUi = Join-Path $InstallDir 'stereo360-ui.bat'
Set-Content -Path $launchUi -Encoding ascii -Value @"
@echo off
set "PATH=%~dp0ffmpeg;%PATH%"
cd /d "%~dp0app"
start "" "%~dp0python\pythonw.exe" -m stereo360_ui %*
"@
$menu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
$sh = New-Object -ComObject WScript.Shell
$lnk = $sh.CreateShortcut((Join-Path $menu 'stereo360.lnk'))
$lnk.TargetPath = $launchUi
$lnk.WorkingDirectory = $InstallDir
$lnk.Description = 'Convert 360 video to stereoscopic 3D for VR'
$lnk.Save()
Write-Good 'Start Menu shortcut created'

# ---- the DirectML model --------------------------------------------------
if ($acc.kind -eq 'directml') {
    Write-Step 'Building the GPU depth model'

    # Without this, choosing DirectML accelerates nothing.
    #
    # torch on Windows from PyPI is CPU-only, so the torch backend runs on the
    # processor. The ONNX backend is the one that reaches an AMD or Intel GPU
    # -- but it needs an exported model, and the repository ships none
    # (models/ is git-ignored). So the runtime was installed and nothing could
    # use it: the machine reported "DirectML" and quietly ran depth on the CPU,
    # ten times slower, which is worse than not offering the option.
    #
    # --static-batch is required rather than preferred: DirectML rejects the
    # graph's Reshape once the batch axis is dynamic, even at batch 1.
    #
    # Ten seconds and about 100 MB, measured. The weights land in a companion
    # .onnx.data file beside the graph, which is why this exports straight
    # into place rather than building elsewhere and copying one file.
    $exporter = Join-Path $dest 'scripts\export_onnx.py'
    $modelOut = Join-Path $dest 'models\depth_anything_v2_small.onnx'
    if (-not (Test-Path $exporter)) {
        Write-Warn 'the exporter is missing from this build; skipping'
    } else {
        Invoke-Native { & $py $exporter --static-batch --out $modelOut }
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $modelOut)) {
            Write-Warn 'could not build the GPU model. Depth will run on the'
            Write-Warn 'processor, which works but is roughly ten times'
            Write-Warn 'slower. Everything else is installed and usable.'
            $acc.kind = 'cpu (the GPU model could not be built)'
        } else {
            $mb = [int](((Get-ChildItem (Split-Path $modelOut) -Filter '*.onnx*' |
                          Measure-Object Length -Sum).Sum) / 1MB)
            Write-Good "depth model built for DirectML ($mb MB)"
        }
    }
}

# ---- uninstaller ---------------------------------------------------------
Write-Step 'Writing the uninstaller'

# Recorded, not inferred. The uninstaller removes what this file says it
# created and nothing else, so a file dropped in the folder later, or a Start
# Menu entry belonging to something else, is never in scope.
$manifest = [ordered]@{
    app          = 'stereo360'
    installed    = (Get-Date).ToString('s')
    installDir   = $InstallDir
    accelerator  = $acc.kind
    createdDirs  = @('python', 'app', 'ffmpeg')
    createdFiles = @('stereo360.bat', 'stereo360-ui.bat', 'uninstall.ps1',
                     'Uninstall stereo360.bat', 'install-manifest.json')
    shortcuts    = @((Join-Path $menu 'stereo360.lnk'))
    registryKey  = $ArpKey
    # Written by the application as it runs, not by this installer, and in
    # the model cache's case shared with every other tool that uses Hugging
    # Face. Recorded so the uninstaller can offer them, never assume them.
    userData     = [ordered]@{
        settings   = (Join-Path $env:LOCALAPPDATA 'stereo360')
        modelCache = (Join-Path $env:USERPROFILE '.cache\huggingface')
    }
}
$manifest | ConvertTo-Json -Depth 5 |
    Set-Content -Path (Join-Path $InstallDir 'install-manifest.json') -Encoding utf8

# A literal here-string: nothing inside is expanded now, so the uninstaller's
# own variables survive intact. Everything specific to this install it reads
# back from the manifest beside it.
Set-Content -Path (Join-Path $InstallDir 'uninstall.ps1') -Encoding ascii -Value @'
<#
.SYNOPSIS
    Removes stereo360, and only stereo360.

.DESCRIPTION
    Deletes what the installer recorded creating in install-manifest.json,
    rather than anything it works out for itself. Two things follow.

    Files you put in the install folder are never touched. The recorded
    folders go, then the folder itself goes only if that left it empty; if
    anything of yours is still there, the folder stays and this says so.

    Your settings and the downloaded models are kept unless you ask for them.
    The model cache especially is shared with any other tool that uses
    Hugging Face, so removing it silently could take gigabytes that were
    never ours to take.
#>
[CmdletBinding()]
param(
    [switch] $RemoveSettings,
    [switch] $RemoveModelCache,
    [switch] $Silent,
    [switch] $DryRun,
    [string] $Target,      # set when this has relaunched itself out of TEMP
    [switch] $Stage2
)

$ErrorActionPreference = 'Stop'

function Say  { param($t) Write-Host "      $t" -ForegroundColor DarkGray }
function Good { param($t) Write-Host "      $t" -ForegroundColor Green }
function Warn { param($t) Write-Host "      $t" -ForegroundColor Yellow }
function Head { param($t) Write-Host ''; Write-Host "  $t" -ForegroundColor Cyan }

function Get-FolderMb {
    param([string] $Path)
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return 0 }
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -Force `
                -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    if (-not $sum) { return 0 }
    return [math]::Round($sum / 1MB, 1)
}

function Assert-LooksLikeOurInstall {
    <#
        The guard that makes everything after it safe.

        What follows deletes recursively, so before any of it runs the target
        has to prove it is a stereo360 install -- not an empty string that
        resolves to a drive root, and not a home folder somebody pointed this
        at by accident.
    #>
    param([string] $Dir)

    if ([string]::IsNullOrWhiteSpace($Dir)) {
        throw 'No install folder given. Refusing to delete anything.'
    }
    $full = [IO.Path]::GetFullPath($Dir).TrimEnd('\')
    if ($full -eq [IO.Path]::GetPathRoot($full).TrimEnd('\')) {
        throw "$full is a drive root. Refusing."
    }
    if ($full.Length -lt 12) {
        throw "$full is too short to be an install folder. Refusing."
    }
    $protected = @(
        $env:USERPROFILE, $env:LOCALAPPDATA, $env:APPDATA, $env:windir,
        $env:SystemRoot, $env:ProgramFiles, ${env:ProgramFiles(x86)},
        $env:ProgramData, (Join-Path $env:LOCALAPPDATA 'Programs'),
        [Environment]::GetFolderPath('Desktop'),
        [Environment]::GetFolderPath('MyDocuments'),
        [Environment]::GetFolderPath('MyPictures'),
        [Environment]::GetFolderPath('MyVideos')
    )
    foreach ($p in $protected) {
        if ($p -and $full -eq [IO.Path]::GetFullPath($p).TrimEnd('\')) {
            throw "$full is a system or user folder. Refusing."
        }
    }
    # Positive proof rather than merely the absence of danger: all three of
    # these exist only inside an install this program made.
    foreach ($proof in @('python\python.exe', 'app\stereo360\__init__.py',
                         'install-manifest.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $full $proof))) {
            throw "$full is not a stereo360 install ($proof is missing). Refusing."
        }
    }
    return $full
}

function Remove-Thing {
    param([string] $Path, [switch] $Recurse)
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return $false }
    if ($DryRun) { Say "would remove $Path"; return $true }
    Remove-Item -LiteralPath $Path -Force -Recurse:$Recurse
    return $true
}

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$dir = Assert-LooksLikeOurInstall $(if ($Target) { $Target } else { $here })

$m = Get-Content (Join-Path $dir 'install-manifest.json') -Raw | ConvertFrom-Json
if ($m.installDir -and
        ([IO.Path]::GetFullPath($m.installDir).TrimEnd('\') -ne $dir)) {
    throw ("The manifest describes $($m.installDir), but this is $dir. " +
           'Refusing to delete a folder it does not describe.')
}

# ------------------------------------------------------------------ stage 1

if (-not $Stage2) {
    Write-Host ''
    Write-Host '  Uninstall stereo360' -ForegroundColor White

    Head 'This will remove'
    Say ("{0}   ({1:N0} MB)" -f $dir, (Get-FolderMb $dir))
    foreach ($s in @($m.shortcuts)) {
        if (Test-Path -LiteralPath $s) { Say $s }
    }
    Say 'the entry in Settings > Apps'

    Head 'This will be left alone'
    $settings = $m.userData.settings
    $cache = $m.userData.modelCache
    if (Test-Path -LiteralPath $settings) {
        Say ("your settings -- {0:N1} MB in {1}" -f (Get-FolderMb $settings), $settings)
        if ($RemoveSettings) { Warn '   ...except you asked for those too' }
    }
    if (Test-Path -LiteralPath $cache) {
        Say ("downloaded models -- {0:N0} MB in {1}" -f (Get-FolderMb $cache), $cache)
        Say '   shared with any other tool that uses Hugging Face'
        if ($RemoveModelCache) { Warn '   ...except you asked for those too' }
    }
    Say 'anything of your own inside the install folder'

    if ($DryRun) {
        Write-Host ''
        Write-Host '  Dry run: nothing was removed.' -ForegroundColor Yellow
        exit 0
    }
    if (-not $Silent) {
        Write-Host ''
        if ((Read-Host '  Remove stereo360? [y/N]') -notmatch '^(y|yes)$') {
            Write-Host '  Cancelled. Nothing was changed.' -ForegroundColor Yellow
            exit 2
        }
    }

    # A script cannot delete the folder it is running from, so step outside
    # and finish from there.
    $stage = Join-Path $env:TEMP ('stereo360_uninstall_' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    Copy-Item (Join-Path $dir 'uninstall.ps1') $stage
    Copy-Item (Join-Path $dir 'install-manifest.json') $stage
    $argv = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
              (Join-Path $stage 'uninstall.ps1'), '-Stage2', '-Target', $dir)
    if ($RemoveSettings)   { $argv += '-RemoveSettings' }
    if ($RemoveModelCache) { $argv += '-RemoveModelCache' }
    # Called directly rather than through Start-Process: with -Wait and
    # -NoNewWindow its ExitCode is not dependable, and Settings > Apps reads
    # this code -- a stray non-zero makes Windows report a failed uninstall
    # after a perfectly good one.
    & powershell @argv
    exit $LASTEXITCODE
}

# ------------------------------------------------------------------ stage 2

Head 'Removing'

foreach ($s in @($m.shortcuts)) {
    if (-not (Test-Path -LiteralPath $s)) { continue }
    # Only if it still points into this install. A shortcut with the same
    # name put there by something else is not ours to delete.
    try {
        $target = (New-Object -ComObject WScript.Shell).CreateShortcut($s).TargetPath
    } catch { $target = '' }
    if ($target -and $target.StartsWith($dir, [StringComparison]::OrdinalIgnoreCase)) {
        [void](Remove-Thing $s)
        Good 'Start Menu shortcut removed'
    } else {
        Warn "left $s alone -- it points at '$target', not at this install"
    }
}

if ($m.registryKey -and (Test-Path $m.registryKey)) {
    [void](Remove-Thing $m.registryKey -Recurse)
    Good 'Settings > Apps entry removed'
}

foreach ($d in @($m.createdDirs))  { [void](Remove-Thing (Join-Path $dir $d) -Recurse) }
foreach ($f in @($m.createdFiles)) { [void](Remove-Thing (Join-Path $dir $f)) }
Good 'program files removed'

# Non-recursive, deliberately: this succeeds only if nothing of yours is left
# in there. If something is, the folder stays and you get told what.
$left = @(Get-ChildItem -LiteralPath $dir -Force -ErrorAction SilentlyContinue)
if ($left.Count -eq 0) {
    Remove-Item -LiteralPath $dir -Force
    Good "$dir removed"
} else {
    Warn "kept $dir -- it still holds files that were not ours:"
    foreach ($item in ($left | Select-Object -First 10)) { Say $item.Name }
}

if ($RemoveSettings   -and (Remove-Thing $m.userData.settings   -Recurse)) { Good 'settings removed' }
if ($RemoveModelCache -and (Remove-Thing $m.userData.modelCache -Recurse)) { Good 'model cache removed' }

Write-Host ''
Write-Host '  stereo360 has been removed.' -ForegroundColor Green
if (-not $RemoveModelCache -and (Test-Path -LiteralPath $m.userData.modelCache)) {
    Write-Host ("  The downloaded models are still in {0}." -f $m.userData.modelCache) -ForegroundColor Gray
    Write-Host '  They are shared with other tools, so they were left in place.' -ForegroundColor Gray
}
Write-Host ''
exit 0
'@

Set-Content -Path (Join-Path $InstallDir 'Uninstall stereo360.bat') -Encoding ascii -Value @"
@echo off
setlocal enabledelayedexpansion
title Uninstalling stereo360
rem Everything after the call is on ONE line on purpose. This file lives in
rem the folder being deleted, and cmd.exe reads a batch file incrementally
rem from disk -- so once the uninstaller has removed the folder, cmd cannot
rem read any further lines and stops with "The system cannot find the path
rem specified" and exit code 1, after a perfectly successful uninstall.
rem Settings ^> Apps reads that code and reports a failure.
rem
rem Keeping the rest on the same line means cmd has already parsed it before
rem powershell runs. Delayed expansion because %%ERRORLEVEL%% on one line
rem would be substituted at parse time, which is before there is one.
rem
rem `exit`, not `exit /b`. Measured, because the difference is the whole bug:
rem `exit /b` returns to the batch context and cmd goes back to the file for
rem the next line, which is when it discovers the file is gone. Plain `exit`
rem ends cmd.exe there and then. Same script, exit 1 versus exit 0.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1" %* & set "RC=!ERRORLEVEL!" & pause & exit !RC!
"@
Write-Good 'uninstaller written, with a manifest of exactly what to remove'

# ---- add/remove programs -------------------------------------------------
Write-Step 'Listing it in Settings > Apps'
$size = [int]((Get-ChildItem $InstallDir -Recurse -File -Force `
                   -ErrorAction SilentlyContinue |
               Measure-Object Length -Sum).Sum / 1KB)
$uninstaller = Join-Path $InstallDir 'Uninstall stereo360.bat'
New-Item -Path $ArpKey -Force | Out-Null
$props = [ordered]@{
    DisplayName          = 'stereo360'
    DisplayVersion       = "$($release.name)"
    Publisher            = 'stereo360'
    InstallLocation      = $InstallDir
    UninstallString      = "`"$uninstaller`""
    QuietUninstallString = "`"$uninstaller`" -Silent"
    DisplayIcon          = (Join-Path $InstallDir 'python\pythonw.exe')
    URLInfoAbout         = "https://github.com/$Repo"
    NoModify             = 1
    NoRepair             = 1
    EstimatedSize        = $size
}
foreach ($k in $props.Keys) {
    $type = if ($props[$k] -is [int]) { 'DWord' } else { 'String' }
    New-ItemProperty -Path $ArpKey -Name $k -Value $props[$k] `
        -PropertyType $type -Force | Out-Null
}
Write-Good 'it can now be removed from Settings > Apps like anything else'

# ---- self test ----------------------------------------------------------
Write-Step 'Testing the installation'
Push-Location (Join-Path $InstallDir 'app')
try {
    $probe = Invoke-Native { & $py -m stereo360 --probe-backends - } | Out-String
    if ($LASTEXITCODE -ne 0) { throw 'backend probe failed' }
    Write-Good 'the pipeline starts and can see its backends'

    # For DirectML this is the point of the whole exercise, so check it rather
    # than assume it: the ONNX backend is the only one that reaches an AMD or
    # Intel GPU, and it counts as available only once the model exists.
    if ($acc.kind -eq 'directml') {
        if ($probe -match '"name":\s*"onnx",\s*"available":\s*true') {
            Write-Good 'the ONNX backend is available, so depth runs on the GPU'
        } else {
            Write-Warn 'the ONNX backend is still unavailable, so depth will'
            Write-Warn 'run on the processor. The line above from --probe-'
            Write-Warn 'backends says why.'
        }
    }

    # The interface is what most people will actually open, and it depends on
    # PySide6 and a working QML runtime -- neither of which the pipeline
    # touches. --selftest loads the window, renders one frame and exits, so a
    # broken interface is caught here rather than by someone double-clicking
    # the shortcut and getting nothing.
    $ui = Invoke-Native { & $py -m stereo360_ui --selftest } | Out-String
    if ($LASTEXITCODE -ne 0 -or $ui -notmatch 'SELFTEST ok') {
        throw "the interface did not start:`n$ui"
    }
    Write-Good ('the interface opens and renders  (' + ($ui.Trim() -split "`n")[-1].Trim() + ')')
    # Not decoration -- nothing works without ffmpeg, and there is a specific
    # way it fails that is worth naming. Windows 11 turns Smart App Control on
    # by default, and it blocks unsigned executables it has never seen before.
    # ffmpeg builds are unsigned, and a nightly build has a brand new hash
    # every day, so it has no reputation to be judged on. Measured on this
    # machine: the same command that printed a version earlier came back with
    # "An Application Control policy has blocked this file".
    try {
        $ffOut = Invoke-Native { & (Join-Path $ffDest 'ffmpeg.exe') -hide_banner -version } |
            Select-Object -First 1
    } catch {
        $ffOut = $null
    }
    if ($ffOut) {
        Write-Detail $ffOut
    } else {
        Write-Warn 'ffmpeg is installed but Windows will not run it.'
        Write-Warn 'That is usually Smart App Control, which blocks unsigned'
        Write-Warn 'programs it does not recognise. stereo360 cannot read or'
        Write-Warn 'write video until it is allowed. Either turn Smart App'
        Write-Warn 'Control off in Windows Security > App & browser control,'
        Write-Warn 'or install ffmpeg yourself (winget install Gyan.FFmpeg)'
        Write-Warn "and delete $ffDest so the one on PATH is used."
    }
} finally {
    Pop-Location
}
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue

Write-Progress -Activity 'Installing stereo360' -Completed
Write-Host ''
Write-Host '  Done.' -ForegroundColor Green
Write-Host "  Installed to     $InstallDir" -ForegroundColor Gray
Write-Host "  Running on       $($acc.kind)" -ForegroundColor Gray
Write-Host '  Start it from    the Start Menu, or stereo360-ui.bat' -ForegroundColor Gray
Write-Host '  Uninstall by     deleting the folder above' -ForegroundColor Gray
Write-Host ''
