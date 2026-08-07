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
$FfmpegUrl = 'https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip'

# cu128 carries cubins for sm_70 through sm_120 -- Volta to Blackwell, which
# is every consumer card since 2017 and includes the 50 series. Ada (sm_89)
# is not listed explicitly and does not need to be: CUDA guarantees binary
# compatibility within a major generation, so the sm_86 cubin runs on it.
$CudaIndexModern = 'https://download.pytorch.org/whl/cu128'
# Pascal and older. Anything this old is slow enough that CPU is not far off.
$CudaIndexLegacy = 'https://download.pytorch.org/whl/cu118'

$MinFreeGb = 8

# ------------------------------------------------------------------ display

$script:StepNo = 0
$script:StepTotal = 13

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
    }
    'directml' {
        Invoke-Pip $py @('install', '--no-warn-script-location', 'torch', 'torchvision', 'onnxruntime-directml') 'ONNX Runtime (DirectML)'
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
    Write-Detail 'nothing to verify for this accelerator'
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
Get-Download -Url $FfmpegUrl -Destination $ffzip -Label 'ffmpeg'
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

# ---- self test ----------------------------------------------------------
Write-Step 'Testing the installation'
Push-Location (Join-Path $InstallDir 'app')
try {
    Invoke-Native { & $py -m stereo360 --probe-backends - } | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'backend probe failed' }
    Write-Good 'the pipeline starts and can see its backends'

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
    Invoke-Native { & (Join-Path $ffDest 'ffmpeg.exe') -hide_banner -version } |
        Select-Object -First 1 | ForEach-Object { Write-Detail $_ }
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
