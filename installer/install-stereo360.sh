#!/usr/bin/env bash
# ============================================================================
#  stereo360 installer for Linux -- monoscopic 360 video to stereoscopic 3D
#
#  Run it straight from the shell:
#      curl -fsSL https://raw.githubusercontent.com/LeonG-ZA/stereo360/main/installer/install-stereo360.sh | bash
#  or download it first and run it locally:
#      chmod +x install-stereo360.sh && ./install-stereo360.sh
#
#  Flags (put them after `-s --` when piping through bash):
#      --install-dir DIR    where to put it (default: see below)
#      --accelerator MODE   auto (default), cuda, rocm, cpu
#      --dry-run            decide everything, print it, download nothing
#      --compute-cap X.Y    pretend the GPU has this compute capability
#      --repo OWNER/NAME    default: LeonG-ZA/stereo360
#
#  This is the Linux sibling of "Install stereo360.bat". That file is a
#  single .bat specifically because a .ps1 cannot be double-clicked and a
#  downloaded one is blocked by the default execution policy -- neither
#  problem exists here. `curl | bash` is the standard way software gets
#  installed on Linux, and a plain .sh downloaded in a browser has no
#  execution-policy block to route around, so one plain, readable script is
#  the whole thing, same as the .bat, without needing a wrapper.
#
#  Otherwise this does not touch the system: no PATH changes to your shell rc
#  files, no existing Python -- it creates a venv inside the install folder
#  and leaves any system Python alone. Uninstalling is running
#  <install dir>/uninstall.sh, which removes exactly what this script
#  recorded creating -- see the manifest it writes at the end.
#
#  The one exception is root, used only to install the X11 libraries Qt's
#  xcb platform plugin links but PySide6's wheel does not ship -- so no pip
#  package can supply them, and without them the interface does not fail
#  cleanly, it aborts with SIGABRT. Which ones are missing is read out of
#  `ldd` rather than assumed, and only those are installed. It asks through
#  sudo at a terminal, or the desktop's polkit dialog when there is not one;
#  failing both it warns and prints the command to run by hand.
# ============================================================================

set -uo pipefail
# Deliberately not `set -e`. Several things here are expected to sometimes
# fail -- a GPU probe on a machine with no GPU, a self-test with no display
# -- and treating those as fatal would make an honest fallback impossible to
# tell apart from a bug. Every command whose failure *should* stop the
# install is checked explicitly and sent through `die`.

# ---------------------------------------------------------------- constants

DEFAULT_REPO="LeonG-ZA/stereo360"
MIN_FREE_GB=8

# cu128 carries cubins for sm_70 through sm_120 -- Volta to Blackwell, which
# is every consumer NVIDIA card since 2017. Ada (sm_89) is not listed
# explicitly and does not need to be: CUDA is binary compatible within a
# major generation, so the sm_86 cubin runs on it.
CUDA_INDEX_MODERN="https://download.pytorch.org/whl/cu128"
# Pascal and older. Anything this old is slow enough that CPU is not far off.
CUDA_INDEX_LEGACY="https://download.pytorch.org/whl/cu118"
# Plain `pip install torch` on Linux is NOT cpu-only the way it is on
# Windows -- the default PyPI Linux wheel links CUDA and pulls several
# hundred MB of nvidia-*-cu12 packages as dependencies whether or not a GPU
# is present. Asking for this index explicitly is what actually gets a small,
# CPU-only install.
CPU_INDEX="https://download.pytorch.org/whl/cpu"
# ROCm wheels move faster than CUDA's and this index will need bumping as
# PyTorch drops old ones -- check https://download.pytorch.org/whl/rocm* if
# this stops resolving.
ROCM_INDEX="https://download.pytorch.org/whl/rocm6.2"

# A static GPL build with libx264, libx265 and libopus -- BtbN's Linux
# builds are the closest match to Gyan's Windows ones the Windows installer
# uses: widely used, plain numbered/rolling builds rather than something
# built here from source. "latest" is BtbN's own rolling tag, not GitHub's
# "most recent release" concept, and its asset names are stable, so the
# fallback below does not need the release API at all if it cannot be
# reached.
FFMPEG_REPO="BtbN/FFmpeg-Builds"
FFMPEG_TAG="latest"
FFMPEG_ASSET_SUFFIX="linux64-gpl.tar.xz"
FFMPEG_URL_FALLBACK="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"

# ------------------------------------------------------------------ display

USE_COLOR=0
[ -t 1 ] && USE_COLOR=1
if [ "$USE_COLOR" -eq 1 ]; then
    C_RESET=$'\033[0m'; C_CYAN=$'\033[36m'; C_GRAY=$'\033[90m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_WHITE=$'\033[97m'
else
    C_RESET=""; C_CYAN=""; C_GRAY=""; C_GREEN=""; C_YELLOW=""; C_WHITE=""
fi

STEP_NO=0
STEP_TOTAL=14

step()   { STEP_NO=$((STEP_NO + 1)); printf '\n%s[%d/%d] %s%s\n' "$C_CYAN" "$STEP_NO" "$STEP_TOTAL" "$1" "$C_RESET"; }
detail() { printf '%s      %s%s\n' "$C_GRAY" "$1" "$C_RESET"; }
good()   { printf '%s      %s%s\n' "$C_GREEN" "$1" "$C_RESET"; }
warn()   { printf '%s      %s%s\n' "$C_YELLOW" "$1" "$C_RESET"; }
# Write-Output, not Write-Host, deliberately: everything above is for a
# person watching, and this is the machine-readable line a test greps for.
decision() { printf 'DECISION %s=%s\n' "$1" "$2"; }
die() { printf '\n  %sERROR:%s %s\n\n' "$C_YELLOW" "$C_RESET" "$1" >&2; exit 1; }

usage() {
    cat <<'EOF'
stereo360 installer for Linux

  install-stereo360.sh [--install-dir DIR] [--accelerator auto|cuda|rocm|cpu]
                        [--dry-run] [--compute-cap X.Y] [--repo OWNER/NAME]

  --install-dir DIR   where to put it (default: XDG_DATA_HOME/Programs/stereo360)
  --accelerator MODE   auto (default) picks CUDA on an NVIDIA card, ROCm on an
                        AMD one, else CPU
  --dry-run            decide everything and print it; download nothing
  --compute-cap X.Y    pretend the GPU has this compute capability, or `none`
                        to pretend there is no NVIDIA GPU -- for testing
  --repo OWNER/NAME    where to fetch the app from
EOF
}

# ---------------------------------------------------------------- arguments

INSTALL_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/Programs/stereo360"
ACCELERATOR="auto"
DRY_RUN=0
COMPUTE_CAP=""
REPO="$DEFAULT_REPO"

while [ $# -gt 0 ]; do
    case "$1" in
        --install-dir)   [ $# -ge 2 ] || die "--install-dir needs a value"; INSTALL_DIR="$2"; shift 2 ;;
        --accelerator)   [ $# -ge 2 ] || die "--accelerator needs a value"; ACCELERATOR="$2"; shift 2 ;;
        --compute-cap)   [ $# -ge 2 ] || die "--compute-cap needs a value"; COMPUTE_CAP="$2"; shift 2 ;;
        --repo)          [ $# -ge 2 ] || die "--repo needs a value"; REPO="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) die "unknown option: $1 (--help for usage)" ;;
    esac
done
case "$ACCELERATOR" in
    auto|cuda|rocm|cpu) ;;
    *) die "--accelerator must be auto, cuda, rocm or cpu" ;;
esac

# Paths derived from the install dir, fixed for the rest of the run.
VENV="$INSTALL_DIR/venv"
PY="$VENV/bin/python3"
DEST="$INSTALL_DIR/app"
FFDIR="$INSTALL_DIR/ffmpeg"
TMP="$INSTALL_DIR/tmp"
BIN_DIR="$HOME/.local/bin"
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$DESKTOP_DIR/stereo360.desktop"
MANIFEST="$INSTALL_DIR/install-manifest.json"

# Where the app's Qt/XDG data would live -- deliberately NOT the install dir.
# Qt derives a per-user data path from the application and organisation
# names, both "stereo360", which resolves to $XDG_DATA_HOME/stereo360. If
# the install lived there too, program files and user data would be mixed
# together and "uninstall by deleting the folder" would take the user's
# settings with it -- exactly why the Windows installer uses
# ...\Programs\stereo360 rather than ...\stereo360. The /Programs/ segment
# here exists for the same reason.
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/stereo360"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/stereo360"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/stereo360"
MODEL_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/huggingface"

# ------------------------------------------------------------------ helpers

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$2"
}

download() {
    local url="$1" dest="$2" label="$3"
    detail "downloading $label..."
    curl -fL --retry 5 --retry-delay 2 -A 'stereo360-installer' -o "$dest" "$url" \
        || die "$label download failed: $url"
    detail "$(du -h "$dest" 2>/dev/null | cut -f1) downloaded"
}

pip_install() {
    detail "pip install $*"
    "$PY" -m pip install --disable-pip-version-check --retries 5 --timeout 60 \
        --no-warn-script-location "$@"
    [ $? -eq 0 ] || die "pip install $* failed"
}

# The same, for the packages this is allowed to do without. Returns the exit
# status rather than calling die, so a GPU runtime that will not install on
# this machine can fall back to the CPU one instead of ending the install.
pip_try() {
    detail "pip install $*"
    "$PY" -m pip install --disable-pip-version-check --retries 5 --timeout 60 \
        --no-warn-script-location "$@"
}

# The directories torch's nvidia-*-cu12 wheels drop their shared objects in.
#
# onnxruntime does not depend on those wheels and does not look inside them,
# so on a machine where torch supplies the only CUDA runtime, ORT's provider
# library fails to load and the session silently falls back to the CPU.
# Putting these on LD_LIBRARY_PATH is what lets the two share one CUDA
# installation. Measured: without it "FELL BACK -> CPUExecutionProvider",
# with it "CUDAExecutionProvider", same wheels either way.
nvidia_lib_path() {
    local sp
    sp="$("$PY" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)" || return 0
    [ -n "$sp" ] && [ -d "$sp/nvidia" ] || return 0
    find "$sp/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null \
        | sort | tr '\n' ':' | sed 's/:$//'
}

read_manifest_field() {
    python3 - "$1" "$2" <<'PYEOF' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get(sys.argv[2], ""))
except Exception:
    pass
PYEOF
}

run_privileged() {
    <<'DOC' :
Runs "$@" as root, asking for a password in whichever way this session can
actually answer.

Three routes, because a desktop installer is run in more than one way and
only one of these covers each:

  sudo -n      already authorised -- a cached timestamp or NOPASSWD. Costs
               nothing to try and skips prompting entirely when it works.
  sudo         a person at a shell. sudo reads the password from the
               controlling terminal, NOT from stdin, which is what makes it
               survive `curl | bash` where stdin is the download.
  pkexec       no controlling terminal at all -- launched from a file
               manager, a desktop shortcut, or a tool driving the shell.
               sudo can only fail here ("a terminal is required to
               authenticate"); pkexec asks the desktop's polkit agent
               instead, which is a GUI dialog and needs no terminal.

The `: >/dev/tty` test is what separates route two from route three, and it
has to be a real open of the terminal rather than `[ -t 0 ]`: under
`curl | bash` stdin is a pipe while /dev/tty is still perfectly usable, and
treating that as "no terminal" would send a shell user to a GUI dialog they
may not even have.
DOC
    if sudo -n true 2>/dev/null; then
        sudo -n "$@"
        return $?
    fi
    # `2>/dev/null` first, deliberately: redirections are applied left to
    # right, so with the open of /dev/tty written first the shell has already
    # printed "No such device or address" by the time stderr is silenced.
    if command -v sudo >/dev/null 2>&1 && : 2>/dev/null >/dev/tty; then
        sudo "$@"
        return $?
    fi
    if command -v pkexec >/dev/null 2>&1 \
            && { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; }; then
        detail "asking for permission through the desktop's password dialog..."
        # An absolute path, because pkexec refuses a bare program name.
        local prog
        prog="$(command -v "$1")" || return 127
        shift
        pkexec "$prog" "$@"
        return $?
    fi
    if command -v sudo >/dev/null 2>&1; then
        sudo "$@"
        return $?
    fi
    return 127
}

# Maps a missing soname onto the package that carries it, for one package
# manager. Empty output means "no idea", which the caller reports rather than
# guessing at.
#
# A table rather than a query, because the tools that could answer this
# properly are not there to ask: apt-file is not installed by default and
# needs its own index built, `dnf provides` needs the filelists metadata, and
# both cost a download before they can answer. This set is small, changes
# about never, and covers what Qt's xcb plugin links.
package_for_soname() {
    local soname="$1" mgr="$2"
    local apt="" dnf="" pacman="" zypper="" apk=""
    case "$soname" in
        libxcb-cursor.so.*)
            apt=libxcb-cursor0; dnf=xcb-util-cursor; pacman=xcb-util-cursor
            zypper=libxcb-cursor0; apk=xcb-util-cursor ;;
        libxcb-icccm.so.*)
            apt=libxcb-icccm4; dnf=xcb-util-wm; pacman=xcb-util-wm
            zypper=libxcb-icccm4; apk=xcb-util-wm ;;
        libxcb-keysyms.so.*)
            apt=libxcb-keysyms1; dnf=xcb-util-keysyms; pacman=xcb-util-keysyms
            zypper=libxcb-keysyms1; apk=xcb-util-keysyms ;;
        libxcb-image.so.*)
            apt=libxcb-image0; dnf=xcb-util-image; pacman=xcb-util-image
            zypper=libxcb-image0; apk=xcb-util-image ;;
        libxcb-render-util.so.*)
            apt=libxcb-render-util0; dnf=xcb-util-renderutil
            pacman=xcb-util-renderutil; zypper=libxcb-render-util0
            apk=xcb-util-renderutil ;;
        libxkbcommon-x11.so.*)
            apt=libxkbcommon-x11-0; dnf=libxkbcommon-x11
            pacman=libxkbcommon-x11; zypper=libxkbcommon-x11-0
            apk=libxkbcommon-x11 ;;
        libxkbcommon.so.*)
            apt=libxkbcommon0; dnf=libxkbcommon; pacman=libxkbcommon
            zypper=libxkbcommon0; apk=libxkbcommon ;;
        # The rest all live in the base libxcb package everywhere except
        # Debian, which splits one shared object per binary package.
        libxcb-xkb.so.*)
            apt=libxcb-xkb1; dnf=libxcb; pacman=libxcb; zypper=libxcb-xkb1
            apk=libxcb ;;
        libxcb-shape.so.*)
            apt=libxcb-shape0; dnf=libxcb; pacman=libxcb; zypper=libxcb-shape0
            apk=libxcb ;;
        libxcb-randr.so.*)
            apt=libxcb-randr0; dnf=libxcb; pacman=libxcb; zypper=libxcb-randr0
            apk=libxcb ;;
        libxcb-xinerama.so.*)
            apt=libxcb-xinerama0; dnf=libxcb; pacman=libxcb
            zypper=libxcb-xinerama0; apk=libxcb ;;
        libxcb-sync.so.*)
            apt=libxcb-sync1; dnf=libxcb; pacman=libxcb; zypper=libxcb-sync1
            apk=libxcb ;;
        libxcb-xfixes.so.*)
            apt=libxcb-xfixes0; dnf=libxcb; pacman=libxcb
            zypper=libxcb-xfixes0; apk=libxcb ;;
        libxcb-util.so.*)
            apt=libxcb-util1; dnf=xcb-util; pacman=xcb-util; zypper=libxcb-util1
            apk=xcb-util ;;
        *) return 0 ;;
    esac
    case "$mgr" in
        apt-get) printf '%s' "$apt" ;;
        dnf)     printf '%s' "$dnf" ;;
        pacman)  printf '%s' "$pacman" ;;
        zypper)  printf '%s' "$zypper" ;;
        apk)     printf '%s' "$apk" ;;
    esac
}

# The only system packages this installer ever touches, and the one class of
# dependency that genuinely cannot live in the venv: Qt's xcb platform plugin
# links these, and PySide6's wheel does not bundle them. Without them opening
# the interface does not fail cleanly -- Qt calls qFatal() and the process
# dies with SIGABRT, "no Qt platform plugin could be initialized", even with
# a perfectly good X server running.
#
# Asked of ldd rather than hardcoded, which is the whole point. The first
# version of this installed libxcb-cursor0 because that is what Qt's error
# message names -- and on this machine that fixed the message and not the
# problem, because libxcb-icccm4 and libxcb-keysyms1 were missing too and Qt
# only ever complains about the first. ldd lists all of them at once, and
# transitively, so what gets installed is what is actually absent rather than
# what the error text happened to mention.
ensure_qt_x11_libs() {
    command -v ldd >/dev/null 2>&1 || return 0

    # PySide6's plugin specifically, asked of the interpreter rather than
    # found by name. opencv-python bundles a complete Qt of its own,
    # libqxcb.so included, and a plain `find -name libqxcb.so` over the venv
    # returns whichever it reaches first -- which was cv2's, whose
    # dependencies are satisfied. The check then reported nothing missing
    # while the interface it was supposed to be vetting could not start.
    local plugin="" pyside_dir
    pyside_dir="$("$PY" -c 'import PySide6, os; print(os.path.dirname(PySide6.__file__))' 2>/dev/null)"
    if [ -n "$pyside_dir" ] && [ -d "$pyside_dir" ]; then
        plugin="$(find "$pyside_dir" -name 'libqxcb.so' -print -quit 2>/dev/null)"
    fi
    if [ -z "$plugin" ]; then
        plugin="$(find "$VENV" -path '*/PySide6/*' -name 'libqxcb.so' -print -quit 2>/dev/null)"
    fi
    [ -n "$plugin" ] || return 0

    local -a missing=()
    while IFS= read -r so; do
        [ -n "$so" ] && missing+=("$so")
    done < <(ldd "$plugin" 2>/dev/null | awk '/not found/ {print $1}' | sort -u)
    [ ${#missing[@]} -eq 0 ] && return 0

    local mgr=""
    for candidate in apt-get dnf pacman zypper apk; do
        if command -v "$candidate" >/dev/null 2>&1; then mgr="$candidate"; break; fi
    done
    if [ -z "$mgr" ]; then
        warn "the interface needs these system libraries, and this distro's package manager was not recognised:"
        warn "  ${missing[*]}"
        return 1
    fi

    local -a packages=() unknown=()
    local pkg
    for so in "${missing[@]}"; do
        pkg="$(package_for_soname "$so" "$mgr")"
        if [ -n "$pkg" ]; then
            case " ${packages[*]:-} " in *" $pkg "*) ;; *) packages+=("$pkg") ;; esac
        else
            unknown+=("$so")
        fi
    done
    if [ ${#unknown[@]} -gt 0 ]; then
        warn "the interface needs these libraries and this script does not know which package carries them:"
        warn "  ${unknown[*]}"
    fi
    [ ${#packages[@]} -eq 0 ] && return 1

    local -a install_cmd=()
    case "$mgr" in
        apt-get) install_cmd=(apt-get install -y "${packages[@]}") ;;
        dnf)     install_cmd=(dnf install -y "${packages[@]}") ;;
        pacman)  install_cmd=(pacman -S --noconfirm "${packages[@]}") ;;
        zypper)  install_cmd=(zypper install -y "${packages[@]}") ;;
        apk)     install_cmd=(apk add "${packages[@]}") ;;
    esac
    detail "the interface needs system libraries pip cannot install -- running:"
    detail "  ${install_cmd[*]}"
    if run_privileged "${install_cmd[@]}"; then
        # Confirmed rather than assumed: a package manager can exit 0 and
        # still leave the loader unable to find what it installed, and this
        # check is cheap next to finding out at the SIGABRT.
        local still
        still="$(ldd "$plugin" 2>/dev/null | awk '/not found/ {print $1}' | sort -u)"
        if [ -n "$still" ]; then
            warn "installed, but these are still missing:"
            warn "  $(printf '%s ' $still)"
            return 1
        fi
        good "Qt's X11 libraries installed (${packages[*]})"
        return 0
    fi
    warn "could not install them automatically. The interface will not open"
    warn "until they are present -- install them by hand with:"
    warn "  sudo ${install_cmd[*]}"
    return 1
}

# ------------------------------------------------------------ gpu detection

get_compute_cap() {
    # 'none' simulates a machine with no NVIDIA card, which is otherwise
    # untestable on a machine that has one.
    if [ "$COMPUTE_CAP" = "none" ]; then echo ""; return; fi
    if [ -n "$COMPUTE_CAP" ]; then echo "$COMPUTE_CAP"; return; fi
    command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return; }
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -n1 | tr -d '[:space:]'
}

select_cuda_index() {
    local cap="$1" major
    [ -z "$cap" ] && { echo ""; return; }
    major="${cap%%.*}"
    if [ "$major" -ge 7 ] 2>/dev/null; then echo "$CUDA_INDEX_MODERN"; else echo "$CUDA_INDEX_LEGACY"; fi
}

detect_amd_gpu() {
    if [ -d /opt/rocm ] || command -v rocminfo >/dev/null 2>&1; then echo 1; return; fi
    if command -v lspci >/dev/null 2>&1; then
        if lspci 2>/dev/null | grep -Ei 'vga|3d|display' | grep -Eqi 'amd|ati|radeon'; then
            echo 1; return
        fi
    fi
    echo 0
}

# Sets ACC_KIND, ACC_CAP, ACC_INDEX.
resolve_accelerator() {
    local cap="" amd=0
    if [ "$ACCELERATOR" != "cpu" ] && [ "$ACCELERATOR" != "rocm" ]; then
        cap="$(get_compute_cap)"
    fi
    ACC_KIND="$ACCELERATOR"
    if [ "$ACC_KIND" = "auto" ]; then
        if [ -n "$cap" ]; then
            ACC_KIND="cuda"
        else
            amd="$(detect_amd_gpu)"
            [ "$amd" = "1" ] && ACC_KIND="rocm" || ACC_KIND="cpu"
        fi
    fi
    if [ "$ACC_KIND" = "cuda" ] && [ -z "$cap" ]; then
        # Asked for explicitly but nvidia-smi finds nothing -- a reliable
        # negative signal, unlike AMD detection below. Say so rather than
        # installing GBs of CUDA wheels that cannot run.
        warn "CUDA was requested but there is no NVIDIA GPU here"
        amd="$(detect_amd_gpu)"
        [ "$amd" = "1" ] && ACC_KIND="rocm" || ACC_KIND="cpu"
    fi
    if [ "$ACC_KIND" = "rocm" ] && [ "$ACCELERATOR" = "rocm" ]; then
        amd="$(detect_amd_gpu)"
        if [ "$amd" != "1" ]; then
            # Unlike nvidia-smi, detection here depends on lspci existing
            # and naming the card in a recognisable way, so a negative is
            # not trusted the same way -- warn and honour the explicit
            # choice rather than silently downgrading it.
            warn "ROCm was requested but no AMD GPU was detected (needs lspci or rocminfo to check)"
            warn "proceeding anyway -- pass --accelerator cpu if this machine has no GPU"
        fi
    fi
    ACC_CAP="$cap"
    ACC_INDEX=""
    [ "$ACC_KIND" = "cuda" ] && ACC_INDEX="$(select_cuda_index "$cap")"
}

# --------------------------------------------------------------- app source

# Sets ASSET_URL, RELEASE_NAME.
get_app_release() {
    local api="https://api.github.com/repos/$REPO/releases/latest" json
    json="$(curl -fsSL -H 'User-Agent: stereo360-installer' "$api" 2>/dev/null)"
    if [ -z "$json" ]; then
        ASSET_URL="https://github.com/$REPO/archive/refs/heads/main.zip"
        RELEASE_NAME="main branch (no published release yet)"
        return
    fi
    ASSET_URL="$(printf '%s' "$json" | python3 -c '
import json, sys
d = json.load(sys.stdin)
asset = next((a["browser_download_url"] for a in d.get("assets", [])
              if a["name"].endswith(".zip")), None)
print(asset or d.get("zipball_url", ""))
' 2>/dev/null)"
    RELEASE_NAME="$(printf '%s' "$json" | python3 -c \
        'import json, sys; print(json.load(sys.stdin).get("tag_name", ""))' 2>/dev/null)"
    if [ -z "$ASSET_URL" ] || [ -z "$RELEASE_NAME" ]; then
        ASSET_URL="https://github.com/$REPO/archive/refs/heads/main.zip"
        RELEASE_NAME="main branch (no published release yet)"
    fi
}

# ================================================================== main

printf '\n%s  stereo360 installer%s\n' "$C_WHITE" "$C_RESET"
printf '%s  monoscopic 360 video -> stereoscopic 3D for VR%s\n\n' "$C_GRAY" "$C_RESET"

require_cmd curl "curl is required. Install it with your package manager (e.g. 'sudo apt install curl') and run this again."
require_cmd tar "tar is required and should already be on any Linux system; something unusual is going on here."
require_cmd python3 "python3 was not found. Install it with your package manager (e.g. 'sudo apt install python3 python3-venv') and run this again."

step "Checking this machine"
ARCH="$(uname -m)"
[ "$ARCH" = "x86_64" ] || die "64-bit x86 Linux is required (found $ARCH)."
ANCESTOR="$INSTALL_DIR"
while [ ! -d "$ANCESTOR" ]; do ANCESTOR="$(dirname "$ANCESTOR")"; done
AVAIL_GB=$(( $(df -Pk "$ANCESTOR" | awk 'NR==2{print $4}') / 1024 / 1024 ))
detail "$(uname -s) $(uname -r), ${AVAIL_GB} GB free"
[ "$AVAIL_GB" -ge "$MIN_FREE_GB" ] || die "Need about ${MIN_FREE_GB} GB free, found ${AVAIL_GB} GB."
detail "install folder: $INSTALL_DIR"

# An upgrade is the same run as a fresh install, deliberately -- see the
# Windows installer's comment on the same choice. What changes is what gets
# said, and that the venv and ffmpeg are kept if they already work.
EXISTING_FOUND=0; EXISTING_PARTIAL=0; EXISTING_VERSION=""
if [ -f "$MANIFEST" ]; then
    EXISTING_FOUND=1
    EXISTING_VERSION="$(read_manifest_field "$MANIFEST" appVersion)"
    [ -z "$EXISTING_VERSION" ] && EXISTING_PARTIAL=1
elif [ -x "$PY" ]; then
    EXISTING_FOUND=1
    EXISTING_PARTIAL=1
fi
if [ "$EXISTING_FOUND" -eq 1 ]; then
    if [ "$EXISTING_PARTIAL" -eq 1 ]; then
        warn "found a half-finished install here; replacing it"
    else
        detail "upgrading an existing install ($EXISTING_VERSION)"
    fi
    detail "your settings and downloaded models are kept"
fi
if [ "$EXISTING_FOUND" -eq 0 ]; then EX_DECISION="none"
elif [ "$EXISTING_PARTIAL" -eq 1 ]; then EX_DECISION="partial"
else EX_DECISION="$EXISTING_VERSION"; fi
decision existing_install "$EX_DECISION"

step "Choosing the accelerator"
resolve_accelerator
case "$ACC_KIND" in
    cuda) detail "NVIDIA GPU, compute capability $ACC_CAP" ;;
    rocm) detail "AMD GPU -- using ROCm" ;;
    cpu)  detail "no usable GPU found -- running on the CPU" ;;
esac
decision accelerator "$ACC_KIND"
decision compute_cap "${ACC_CAP:-none}"
if [ "$ACC_KIND" = "cuda" ]; then decision torch_index "$ACC_INDEX"; else decision torch_index "n/a"; fi
case "$ACC_KIND" in
    cuda) detail "PyTorch with CUDA -- a few GB, the long step" ;;
    rocm)
        detail "PyTorch with ROCm -- a few GB. The ONNX video-depth model"
        detail "still runs on the CPU: no ROCm-accelerated onnxruntime wheel ships on PyPI"
        ;;
    cpu)  detail "CPU only -- depth is roughly ten times slower" ;;
esac

get_app_release
decision app_source "$RELEASE_NAME"

if [ "$DRY_RUN" -eq 1 ]; then
    printf '\n  %sDry run: nothing was downloaded or installed.%s\n\n' "$C_YELLOW" "$C_RESET"
    exit 0
fi

mkdir -p "$INSTALL_DIR" "$TMP"

# ---- python ---------------------------------------------------------------
step "Setting up Python (a private virtual environment)"
PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PYMAJ="${PYVER%%.*}"; PYMIN="${PYVER##*.}"
if [ "$PYMAJ" -lt 3 ] || { [ "$PYMAJ" -eq 3 ] && [ "$PYMIN" -lt 10 ]; }; then
    die "Python 3.10+ is required; found $PYVER."
fi
detail "system Python $PYVER, isolated in a venv -- nothing outside this folder is touched"

HAVE_VENV=0
if [ -x "$PY" ] && "$PY" -c 'pass' >/dev/null 2>&1; then
    HAVE_VENV=1
    good "existing virtual environment found here; keeping it and its packages"
fi
if [ "$HAVE_VENV" -eq 0 ]; then
    # --without-pip, deliberately: venv's normal path bootstraps pip through
    # the standard library's ensurepip module, which several distros (Debian
    # and Ubuntu among them) package separately from python3 itself --
    # "python3.NN-venv", version-specific, and needing sudo to install. Doing
    # it ourselves with get-pip.py below sidesteps that distro packaging
    # split entirely, the same way the Windows installer avoids relying on
    # anything but the interpreter it downloads.
    VENV_ERR="$(mktemp)"
    if ! python3 -m venv --without-pip "$VENV" >"$VENV_ERR" 2>&1; then
        ERRTXT="$(cat "$VENV_ERR")"; rm -f "$VENV_ERR"
        die "could not create a virtual environment: $ERRTXT"
    fi
    rm -f "$VENV_ERR"
    good "virtual environment ready"
fi

step "Bootstrapping pip"
if "$PY" -m pip --version >/dev/null 2>&1; then
    good "pip already installed here; keeping it"
else
    GETPIP="$TMP/get-pip.py"
    download "https://bootstrap.pypa.io/get-pip.py" "$GETPIP" "get-pip"
    "$PY" "$GETPIP" --quiet || die "pip bootstrap failed"
    rm -f "$GETPIP"
    good "pip ready"
fi

# ---- the app ----------------------------------------------------------
step "Downloading stereo360 ($RELEASE_NAME)"
APPZIP="$TMP/app.zip"
download "$ASSET_URL" "$APPZIP" "stereo360"
UNPACK="$TMP/app"
rm -rf "$UNPACK"; mkdir -p "$UNPACK"
python3 -m zipfile -e "$APPZIP" "$UNPACK" || die "could not unpack the download"
# A GitHub archive wraps everything in one directory whose name carries the
# commit; a release asset may not. Look for the package rather than assume a
# shape, same as the Windows installer.
APP_PKG_DIR="$(python3 - "$UNPACK" <<'PYEOF'
import os, sys
root = sys.argv[1]
for dirpath, _dirnames, filenames in os.walk(root):
    if os.path.basename(dirpath) == "stereo360" and "__init__.py" in filenames:
        print(dirpath)
        break
PYEOF
)"
[ -n "$APP_PKG_DIR" ] || die "could not find the stereo360 package in the download"
SRC="$(dirname "$APP_PKG_DIR")"
rm -rf "$DEST"
mv "$SRC" "$DEST" || die "could not move the downloaded app into place"
rm -f "$APPZIP"; rm -rf "$UNPACK"

# Puts the app on the venv's import path permanently, the same trick as the
# `..\app` line the Windows installer adds to python's ._pth: one file
# naming an absolute path is enough, so nothing needs `cd` or PYTHONPATH.
SITE_PACKAGES="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
echo "$DEST" > "$SITE_PACKAGES/stereo360-app.pth"
good "stereo360 in $DEST"

# ---- dependencies -------------------------------------------------------
step "Installing the accelerator ($ACC_KIND)"
case "$ACC_KIND" in
    cuda)
        pip_install --index-url "$ACC_INDEX" torch torchvision
        # The ONNX runtime has to be built against the same CUDA major as
        # torch, because torch's wheels are where the CUDA runtime actually
        # comes from here -- nothing else installs one.
        #
        # Asked of torch rather than pinned blind: onnxruntime-gpu moved to
        # CUDA 13 at 1.25, and there is no CUDA 13 torch wheel to pair with
        # it yet, so the newest of each cannot be used together. Measured on
        # an RTX 5070 Ti with torch cu128: onnxruntime-gpu 1.28 listed
        # CUDAExecutionProvider, then built every session on the CPU because
        # libcublasLt.so.13 was not there. It does not error -- it just runs
        # the video default ten times slower and says nothing.
        TORCH_CUDA_MAJOR="$("$PY" -c "import torch; print((torch.version.cuda or '').split('.')[0])" 2>/dev/null)"
        if [ "$TORCH_CUDA_MAJOR" = "12" ]; then
            detail "torch brought CUDA 12; matching it with the CUDA 12 onnxruntime"
            pip_install 'onnxruntime-gpu<1.25' onnx
        else
            pip_install onnxruntime-gpu onnx
        fi
        ;;
    rocm)
        pip_install --index-url "$ROCM_INDEX" torch torchvision
        # onnxruntime-rocm, from PyPI, and this matters more than it looks:
        # the video default (Depth Anything V3) is an ONNX graph with no
        # torch path at all, so a plain `onnxruntime` here would leave the
        # main use case running on the processor no matter how good the GPU
        # is. It needs a ROCm install on the system to load; the probe below
        # checks that rather than assuming, and falls back if it cannot.
        #
        # From PyPI rather than $ROCM_INDEX, which carries no ORT wheels.
        if ! pip_try onnxruntime-rocm onnx; then
            warn "no onnxruntime-rocm wheel for this Python; the video depth"
            warn "model will run on the processor"
            pip_install onnxruntime onnx
        fi
        ;;
    cpu)
        pip_install --index-url "$CPU_INDEX" torch torchvision
        pip_install onnxruntime
        ;;
esac

probe_gpu_torch() {
    "$PY" - <<'PYEOF'
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("FAIL torch.cuda.is_available() is False"); sys.exit(1)
    is_rocm = bool(getattr(torch.version, "hip", None))
    x = torch.randn(512, 512, device="cuda")
    val = float((x @ x).sum().item())
    torch.cuda.synchronize()
    if val != val:
        print("FAIL kernel produced NaN"); sys.exit(1)
    if is_rocm:
        print("OK %s (ROCm) on %s"
              % (torch.__version__, torch.cuda.get_device_name(0)))
    else:
        # Not enough that a kernel ran -- CUDA JIT-compiles from embedded
        # PTX for an architecture the build has no native code for, which
        # works and uses none of the tuned kernels for the card. Check the
        # build actually contains code for this architecture.
        major, minor = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        native = [a for a in archs
                  if a.startswith("sm_%d" % major)
                  and a[len("sm_%d" % major):].isdigit()
                  and int(a[len("sm_%d" % major):]) <= minor]
        if not native:
            print("FAIL this build has no code for sm_%d%d -- it lists %s. "
                  "It would run through PTX JIT, slowly."
                  % (major, minor, ",".join(archs)))
            sys.exit(1)
        print("OK %s on %s (sm_%d%d, using %s)"
              % (torch.__version__, torch.cuda.get_device_name(0),
                 major, minor, native[-1]))
except Exception as exc:
    print("FAIL %s: %s" % (type(exc).__name__, exc)); sys.exit(1)
PYEOF
}

step "Checking the accelerator actually runs"
if [ "$ACC_KIND" = "cuda" ] || [ "$ACC_KIND" = "rocm" ]; then
    RESULT="$(probe_gpu_torch)"
    RC=$?
    if [ $RC -eq 0 ]; then
        good "$RESULT"
    else
        warn "the accelerator did not run a test kernel:"
        warn "$RESULT"
        if [ "$ACC_KIND" = "cuda" ] && [ "$ACC_INDEX" != "$CUDA_INDEX_LEGACY" ]; then
            warn "retrying with the older CUDA build"
            pip_install --force-reinstall --index-url "$CUDA_INDEX_LEGACY" torch torchvision
            RESULT="$(probe_gpu_torch)"
            RC=$?
        fi
        if [ $RC -ne 0 ]; then
            warn "falling back to CPU. Depth will be roughly ten times slower."
            pip_install --force-reinstall --index-url "$CPU_INDEX" torch torchvision
            ACC_KIND="cpu"
        else
            good "$RESULT"
        fi
    fi
else
    detail "no GPU build to verify for this accelerator"
fi

# The check that matters most for video, and the one that is easiest to get
# wrong: the video default (Depth Anything V3) is an ONNX graph with no torch
# path at all, so if the ONNX runtime is not on the GPU then the main use
# case is on the processor however good the card is and whatever the torch
# probe above said.
#
# `get_available_providers()` lists what is *installed*, not what loads.
# Measured on an RTX 5070 Ti: onnxruntime-gpu 1.28 cheerfully lists
# CUDAExecutionProvider and then silently builds the session on
# CPUExecutionProvider, because it wants CUDA 13 (libcublasLt.so.13) while
# torch's cu128 wheels bring CUDA 12. Nothing errors. So the session's own
# provider is read back and a kernel is run.
probe_onnx_provider() {
    "$PY" - "$1" <<'PYEOF'
import sys
want = sys.argv[1]
try:
    # Imported first, deliberately: torch's wheels carry the CUDA libraries
    # (libcublasLt.so.12, libcudnn.so.9) that a CUDA-12 onnxruntime needs,
    # and importing torch loads them into the process where onnxruntime's
    # provider library can then resolve them. Without this the provider can
    # fail to load even though everything needed is installed.
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    import numpy as np, onnxruntime as ort
    from onnx import TensorProto, helper

    have = ort.get_available_providers()
    if want not in have:
        print("FAIL %s is not installed; have %s" % (want, ",".join(have)))
        sys.exit(1)

    node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3],
                            pads=[1, 1, 1, 1])
    graph = helper.make_graph(
        [node], "probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 32, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 32, 32])],
        [helper.make_tensor("w", TensorProto.FLOAT, [8, 4, 3, 3],
                            np.full(8 * 4 * 3 * 3, 0.05, np.float32))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10  # onnxruntime rejects an IR version above its max

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=[want])
    used = sess.get_providers()[0]
    if used != want:
        print("FAIL asked for %s, got %s" % (want, used)); sys.exit(1)

    out = sess.run(None, {"x": np.ones((1, 4, 32, 32), np.float32)})[0]
    if not np.isfinite(out).all():
        print("FAIL %s produced non-finite output" % want); sys.exit(1)
    print("OK onnxruntime %s ran on %s" % (ort.__version__, used))
except Exception as exc:
    print("FAIL %s: %s" % (type(exc).__name__, exc)); sys.exit(1)
PYEOF
}

ONNX_PROVIDER="cpu"
case "$ACC_KIND" in
    cuda) WANT_EP="CUDAExecutionProvider" ;;
    rocm) WANT_EP="ROCMExecutionProvider" ;;
    *)    WANT_EP="" ;;
esac
# Exported before the probe rather than only written into the launchers,
# so that what is tested here is what will run later. A probe that passes
# under different library paths than the application gets is worth nothing.
NV_LIBS="$(nvidia_lib_path)"
if [ -n "$NV_LIBS" ]; then
    export LD_LIBRARY_PATH="$NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ -n "$WANT_EP" ]; then
    RESULT="$(probe_onnx_provider "$WANT_EP")"
    if printf '%s' "$RESULT" | grep -q '^OK '; then
        good "$RESULT"
        ONNX_PROVIDER="$ACC_KIND"
    else
        warn "the GPU could not run an ONNX graph:"
        warn "$RESULT"
        warn "VIDEO depth will run on the processor -- that is the default"
        warn "path, so this is the slow one to care about. Stills, which use"
        warn "torch, are unaffected."
    fi
else
    detail "the ONNX video-depth model runs on the CPU with this accelerator"
fi
decision onnx_provider "$ONNX_PROVIDER"
decision accelerator_final "$ACC_KIND"

step "Installing core dependencies"
pip_install -r "$DEST/requirements.txt"
good "numpy, opencv, transformers, Pillow"

step "Installing the desktop interface"
pip_install -r "$DEST/requirements-ui.txt"
good "PySide6"
ensure_qt_x11_libs

# ---- ffmpeg -------------------------------------------------------------
step "Installing ffmpeg"
USING_SYSTEM_FFMPEG=0
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1 \
        && ffmpeg -hide_banner -version >/dev/null 2>&1; then
    detail "using ffmpeg already on PATH ($(command -v ffmpeg))"
    USING_SYSTEM_FFMPEG=1
elif [ -x "$FFDIR/ffmpeg" ] && "$FFDIR/ffmpeg" -hide_banner -version >/dev/null 2>&1; then
    good "ffmpeg already installed here; keeping it"
else
    mkdir -p "$FFDIR"
    FFURL="$FFMPEG_URL_FALLBACK"
    FFJSON="$(curl -fsSL -H 'User-Agent: stereo360-installer' \
        "https://api.github.com/repos/$FFMPEG_REPO/releases/tags/$FFMPEG_TAG" 2>/dev/null)"
    if [ -n "$FFJSON" ]; then
        FFASSET="$(printf '%s' "$FFJSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for a in d.get('assets', []):
    n = a['name']
    if n.endswith('$FFMPEG_ASSET_SUFFIX') and 'shared' not in n:
        print(a['browser_download_url']); break
" 2>/dev/null)"
        [ -n "$FFASSET" ] && FFURL="$FFASSET"
    fi
    FFTAR="$TMP/ffmpeg.tar.xz"
    download "$FFURL" "$FFTAR" "ffmpeg"
    require_cmd xz "xz is required to unpack ffmpeg. Install it with 'sudo apt install xz-utils' (Debian/Ubuntu) or your distro's equivalent, and run this again."
    FFEXTRACT="$TMP/ffmpeg-extract"
    rm -rf "$FFEXTRACT"; mkdir -p "$FFEXTRACT"
    tar -xJf "$FFTAR" -C "$FFEXTRACT" || die "could not unpack ffmpeg"
    FFBINDIR="$(dirname "$(find "$FFEXTRACT" -type f -name ffmpeg | head -n1)")"
    [ -n "$FFBINDIR" ] || die "could not find ffmpeg in the downloaded archive"
    cp "$FFBINDIR/ffmpeg" "$FFBINDIR/ffprobe" "$FFDIR/" || die "could not copy ffmpeg into place"
    chmod +x "$FFDIR/ffmpeg" "$FFDIR/ffprobe"
    rm -f "$FFTAR"; rm -rf "$FFEXTRACT"
    good "ffmpeg and ffprobe (static GPL build, no libfdk_aac -- see the README)"
fi

# ---- model --------------------------------------------------------------
step "Fetching the depth model"
export PATH="$FFDIR:$PATH"
WARM_OUT="$("$PY" - <<'PYEOF' 2>&1
from stereo360.depth.depth_anything_v3 import DEFAULT_VARIANT, download
print("cached", download(DEFAULT_VARIANT))
PYEOF
)"
if [ $? -ne 0 ]; then
    warn "could not pre-fetch the model; it will download on first use"
else
    good "Depth Anything V3 small cached (the video default)"
    detail "Depth Pro, for stills, downloads on first use -- about 1.9 GB"
fi

# ---- launchers ------------------------------------------------------------
step "Creating launchers and a menu entry"
# The LD_LIBRARY_PATH line is not decoration and is why it is written into
# both launchers rather than only exported during the install: onnxruntime
# finds the CUDA runtime through it, and that runtime belongs to torch's
# wheels. Without it here, the installer would verify a GPU that the
# application then does not get -- which is the exact failure this whole
# script exists to catch, moved one step later where nobody would see it.
LD_LINE=""
if [ -n "$NV_LIBS" ]; then
    LD_LINE="export LD_LIBRARY_PATH=\"$NV_LIBS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
fi

LAUNCH="$INSTALL_DIR/stereo360.sh"
cat > "$LAUNCH" <<EOF
#!/usr/bin/env bash
export PATH="$FFDIR:\$PATH"
$LD_LINE
exec "$PY" -m stereo360 "\$@"
EOF
chmod +x "$LAUNCH"

LAUNCH_UI="$INSTALL_DIR/stereo360-ui.sh"
cat > "$LAUNCH_UI" <<EOF
#!/usr/bin/env bash
export PATH="$FFDIR:\$PATH"
$LD_LINE
exec "$PY" -m stereo360_ui "\$@"
EOF
chmod +x "$LAUNCH_UI"

mkdir -p "$BIN_DIR"
ln -sf "$LAUNCH" "$BIN_DIR/stereo360"
ln -sf "$LAUNCH_UI" "$BIN_DIR/stereo360-ui"
good "launchers: $BIN_DIR/stereo360 and $BIN_DIR/stereo360-ui"
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "$BIN_DIR is not on your PATH -- add this to your shell rc file:" \
       && warn "  export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac

mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=stereo360
Comment=Convert 360 video to stereoscopic 3D for VR
Exec=$LAUNCH_UI %f
Icon=video-x-generic
Terminal=false
Categories=AudioVideo;Video;
EOF
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1
good "menu entry created"

# ---- uninstaller and manifest ---------------------------------------------
step "Writing the uninstaller"
cat > "$INSTALL_DIR/uninstall.sh" <<'UNINSTALL_EOF'
#!/usr/bin/env bash
# Removes stereo360, and only stereo360.
#
# Deletes what the installer recorded creating in install-manifest.json
# rather than anything worked out fresh here. Files you put in the install
# folder are never touched; the folder itself goes only if that leaves it
# empty. Settings and the downloaded model cache are kept unless you pass
# --remove-settings / --remove-model-cache -- the model cache especially is
# shared with any other tool that uses Hugging Face.
set -uo pipefail

REMOVE_SETTINGS=0
REMOVE_MODEL_CACHE=0
SILENT=0
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --remove-settings)    REMOVE_SETTINGS=1; shift ;;
        --remove-model-cache) REMOVE_MODEL_CACHE=1; shift ;;
        --silent)             SILENT=1; shift ;;
        --dry-run)             DRY_RUN=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Positive proof this is a stereo360 install, not merely the absence of
# danger -- everything after this deletes recursively. All three of these
# exist only inside an install this script made.
for proof in venv/bin/python3 app/stereo360/__init__.py install-manifest.json; do
    if [ ! -e "$DIR/$proof" ]; then
        echo "$DIR is not a stereo360 install ($proof is missing). Refusing." >&2
        exit 1
    fi
done

for protected in "$HOME" / /usr /usr/local /opt \
        "$HOME/.local" "$HOME/.local/share" "$HOME/.local/bin" "$HOME/.config" \
        "$HOME/Desktop" "$HOME/Documents" "$HOME/Downloads" "$HOME/Pictures" "$HOME/Videos"; do
    if [ "$DIR" = "$protected" ]; then
        echo "$DIR is a system or user folder. Refusing." >&2
        exit 1
    fi
done

MANIFEST="$DIR/install-manifest.json"
get_field() {
    python3 -c "import json; d=json.load(open('$MANIFEST')); print(d.get('$1',''))" 2>/dev/null
}
CONFIG_DIR="$(get_field configDir)"
SETTINGS_DIR="$(get_field settingsDir)"
CACHE_DIR="$(get_field cacheDir)"
MODEL_CACHE="$(get_field modelCache)"
BIN_STEREO="$(get_field binStereo)"
BIN_UI="$(get_field binUi)"
DESKTOP_FILE="$(get_field desktopFile)"

folder_mb() { [ -d "$1" ] && du -sm "$1" 2>/dev/null | cut -f1 || echo 0; }

if [ "$SILENT" -ne 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "  Uninstall stereo360"
    echo ""
    echo "  This will remove"
    echo "      $DIR   ($(folder_mb "$DIR") MB)"
    [ -n "$BIN_STEREO" ] && [ -e "$BIN_STEREO" ] && echo "      $BIN_STEREO"
    [ -n "$BIN_UI" ] && [ -e "$BIN_UI" ] && echo "      $BIN_UI"
    [ -n "$DESKTOP_FILE" ] && [ -e "$DESKTOP_FILE" ] && echo "      $DESKTOP_FILE"
    echo ""
    echo "  This will be left alone"
    if [ -n "$CONFIG_DIR" ] && [ -d "$CONFIG_DIR" ]; then
        echo "      your settings -- $(folder_mb "$CONFIG_DIR") MB in $CONFIG_DIR"
        [ "$REMOVE_SETTINGS" -eq 1 ] && echo "         ...except you asked for those too"
    fi
    if [ -n "$MODEL_CACHE" ] && [ -d "$MODEL_CACHE" ]; then
        echo "      downloaded models -- $(folder_mb "$MODEL_CACHE") MB in $MODEL_CACHE"
        echo "         shared with any other tool that uses Hugging Face"
        [ "$REMOVE_MODEL_CACHE" -eq 1 ] && echo "         ...except you asked for those too"
    fi
    echo "      anything of your own inside the install folder"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "  Dry run: nothing was removed."
    exit 0
fi

if [ "$SILENT" -ne 1 ]; then
    echo ""
    read -r -p "  Remove stereo360? [y/N] " REPLY
    case "$REPLY" in
        y|Y|yes|YES) ;;
        *) echo "  Cancelled. Nothing was changed."; exit 2 ;;
    esac
fi

echo ""
echo "  Removing"
[ -n "$BIN_STEREO" ] && [ -L "$BIN_STEREO" ] && rm -f "$BIN_STEREO"
[ -n "$BIN_UI" ] && [ -L "$BIN_UI" ] && rm -f "$BIN_UI"
[ -n "$DESKTOP_FILE" ] && [ -f "$DESKTOP_FILE" ] && rm -f "$DESKTOP_FILE"
echo "      shortcuts removed"

rm -rf "$DIR/venv" "$DIR/app" "$DIR/ffmpeg" "$DIR/tmp"
rm -f "$DIR/stereo360.sh" "$DIR/stereo360-ui.sh" "$DIR/install-manifest.json"
echo "      program files removed"

# Non-recursive on purpose: this only succeeds if nothing of the user's is
# left in there. `uninstall.sh` itself -- the file currently executing --
# is removed first; unlinking a running script is fine on Linux, unlike
# Windows, because the shell already has it open and nothing needs to step
# outside the folder to finish deleting it.
rm -f "$DIR/uninstall.sh"
LEFT="$(find "$DIR" -mindepth 1 2>/dev/null)"
if [ -z "$LEFT" ]; then
    rmdir "$DIR" 2>/dev/null
    echo "      $DIR removed"
else
    echo "      kept $DIR -- it still holds files that were not ours:"
    printf '%s\n' "$LEFT" | head -n 10 | sed 's/^/          /'
fi

if [ "$REMOVE_SETTINGS" -eq 1 ]; then
    [ -n "$CONFIG_DIR" ] && rm -rf "$CONFIG_DIR"
    [ -n "$SETTINGS_DIR" ] && rm -rf "$SETTINGS_DIR"
    [ -n "$CACHE_DIR" ] && rm -rf "$CACHE_DIR"
    echo "      settings removed"
fi
if [ "$REMOVE_MODEL_CACHE" -eq 1 ] && [ -n "$MODEL_CACHE" ]; then
    rm -rf "$MODEL_CACHE"
    echo "      model cache removed"
fi

echo ""
echo "  stereo360 has been removed."
if [ "$REMOVE_MODEL_CACHE" -ne 1 ] && [ -n "$MODEL_CACHE" ] && [ -d "$MODEL_CACHE" ]; then
    echo "  The downloaded models are still in $MODEL_CACHE."
    echo "  They are shared with other tools, so they were left in place."
fi
echo ""
UNINSTALL_EOF
chmod +x "$INSTALL_DIR/uninstall.sh"
good "uninstaller written, with a manifest of exactly what to remove"

# Recorded, not inferred -- read of the app rather than $RELEASE_NAME, which
# is the GitHub tag and can read "main branch (no published release yet)".
APP_VERSION="$("$PY" -c "import stereo360; print(stereo360.released_as())" 2>/dev/null)"
[ -z "$APP_VERSION" ] && APP_VERSION="$RELEASE_NAME"

python3 - "$MANIFEST" "$APP_VERSION" "$INSTALL_DIR" "$ACC_KIND" \
    "$BIN_DIR/stereo360" "$BIN_DIR/stereo360-ui" "$DESKTOP_FILE" \
    "$CONFIG_DIR" "$DATA_DIR" "$CACHE_DIR" "$MODEL_CACHE" <<'PYEOF'
import datetime
import json
import sys

(manifest_path, app_version, install_dir, accelerator, bin_stereo, bin_ui,
 desktop_file, config_dir, data_dir, cache_dir, model_cache) = sys.argv[1:]

manifest = {
    "app": "stereo360",
    "appVersion": app_version,
    "installed": datetime.datetime.now().isoformat(timespec="seconds"),
    "installDir": install_dir,
    "accelerator": accelerator,
    "createdDirs": ["venv", "app", "ffmpeg", "tmp"],
    "createdFiles": ["stereo360.sh", "stereo360-ui.sh", "uninstall.sh",
                      "install-manifest.json"],
    "binStereo": bin_stereo,
    "binUi": bin_ui,
    "desktopFile": desktop_file,
    "userData": {
        "configDir": config_dir,
        "settingsDir": data_dir,
        "cacheDir": cache_dir,
        "modelCache": model_cache,
    },
}
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
PYEOF
good "manifest written"

# ---- self test ----------------------------------------------------------
step "Testing the installation"
PROBE_OUT="$("$PY" -m stereo360 --probe-backends - 2>&1)"
if [ $? -ne 0 ]; then
    warn "backend probe failed:"
    warn "$PROBE_OUT"
else
    good "the pipeline starts and can see its backends"
fi

# PySide6 needs a real display to create a window, even for --selftest,
# which a bare SSH session or a minimal server does not have. Skip straight
# to offscreen in that case rather than waiting on a doomed attempt.
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    detail "no display detected; testing the interface offscreen"
    UI_OUT="$(QT_QPA_PLATFORM=offscreen "$PY" -m stereo360_ui --selftest 2>&1)"
    UI_RC=$?
else
    UI_OUT="$("$PY" -m stereo360_ui --selftest 2>&1)"
    UI_RC=$?
    # $DISPLAY being set is not proof the window actually opened. Measured
    # here: a real, reachable X server plus a genuinely missing runtime
    # dependency (libxcb-cursor0, which Qt has required since 6.5) fails
    # with exactly this message and nothing about "no display" at all.
    # Retrying offscreen tells the two apart -- a QML bug fails both ways, a
    # missing system library only fails the first.
    if [ $UI_RC -ne 0 ] && printf '%s' "$UI_OUT" | grep -qi 'Qt platform plugin'; then
        RETRY_OUT="$(QT_QPA_PLATFORM=offscreen "$PY" -m stereo360_ui --selftest 2>&1)"
        if [ $? -eq 0 ] && printf '%s' "$RETRY_OUT" | grep -q 'SELFTEST ok'; then
            warn "the interface itself is fine, but this display would not open a window:"
            RELEVANT="$(printf '%s\n' "$UI_OUT" | grep -i 'xcb-cursor\|platform plugin')"
            [ -n "$RELEVANT" ] && warn "$RELEVANT"
            warn "on Debian/Ubuntu this is usually: sudo apt install libxcb-cursor0"
            UI_OUT="$RETRY_OUT"; UI_RC=0
        fi
    fi
fi
if [ $UI_RC -ne 0 ] || ! printf '%s' "$UI_OUT" | grep -q 'SELFTEST ok'; then
    warn "the interface did not start:"
    warn "$UI_OUT"
else
    good "the interface opens and renders  ($(printf '%s\n' "$UI_OUT" | tail -n1))"
fi

FF_EXE="ffmpeg"
[ "$USING_SYSTEM_FFMPEG" -eq 0 ] && FF_EXE="$FFDIR/ffmpeg"
FF_VER="$("$FF_EXE" -hide_banner -version 2>&1 | head -n1)"
if [ $? -eq 0 ]; then
    detail "$FF_VER"
else
    warn "ffmpeg did not run: $FF_VER"
fi

rm -rf "$TMP"

printf '\n%s  Done.%s\n' "$C_GREEN" "$C_RESET"
printf '%s  Installed to     %s%s\n' "$C_GRAY" "$INSTALL_DIR" "$C_RESET"
printf '%s  Running on       %s%s\n' "$C_GRAY" "$ACC_KIND" "$C_RESET"
printf '%s  Start it from    the applications menu, or %s%s\n' "$C_GRAY" "$BIN_DIR/stereo360-ui" "$C_RESET"
printf '%s  Command line     %s input.mp4 -o output.mp4%s\n' "$C_GRAY" "$BIN_DIR/stereo360" "$C_RESET"
printf '%s  Uninstall by     %s%s\n\n' "$C_GRAY" "$INSTALL_DIR/uninstall.sh" "$C_RESET"
