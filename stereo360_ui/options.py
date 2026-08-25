"""Translate UI state into a stereo360 command line.

The one place in the UI that knows flag names. Everything else passes a plain
dict around, so adding an option means touching this file and the QML control
that sets it -- not a chain of typed properties.

Only non-default flags are emitted. That keeps the command the UI shows the
user short enough to read, and short enough to paste into a terminal, which is
how someone graduates from the UI to the CLI.
"""

from __future__ import annotations

import os

from typing import Any, Dict, List, Optional

#: Stills, mirrored from `ffmpeg_io.IMAGE_SUFFIXES`. Two lines rather than an
#: import, for the same reason as `output_size`: this process must never pull
#: numpy in to draw a window. A test pins the two lists together.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
                  ".avif", ".heic", ".heif", ".hif")


#: Offered by the open dialog alongside the stills. Mirrored for the same
#: reason as the rest.
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mts",
                  ".m2ts", ".insv")


def is_image(path: Any) -> bool:
    """Whether this path names a still, and so a photo job rather than a
    video one."""
    return os.path.splitext(str(path or ""))[1].lower() in IMAGE_SUFFIXES


def _glob(suffixes) -> str:
    return " ".join("*" + s for s in suffixes)


def open_filters() -> List[str]:
    """Name filters for the open dialog, built from the accepted lists.

    Built rather than written out, because writing them out is how the dialog
    came to offer video only while the tool had accepted photos for three
    commits. A filter list that is maintained by hand is a second source of
    truth about what the tool opens, and it was already wrong.

    A dialog filter is only a filter -- ffmpeg sniffs content and ignores
    extensions entirely -- but one that hides the file someone wants makes the
    tool look as though it cannot open it, which is exactly what happened.
    """
    both = tuple(VIDEO_SUFFIXES) + tuple(IMAGE_SUFFIXES)
    return [f"Video or photo ({_glob(both)})",
            f"Video ({_glob(VIDEO_SUFFIXES)})",
            f"Photo ({_glob(IMAGE_SUFFIXES)})",
            "All files (*)"]


def resolve_output(current: str, previous_suggestion: str, suggestion: str,
                   input_is_image: bool) -> str:
    """What belongs in the Output box once the job has changed.

    `current` is what is there now, `previous_suggestion` the last name this
    program proposed -- the two being equal is how it knows the name is its
    own to revise rather than someone's choice to respect.

    Three cases:

    * Nothing there, or a name this program proposed -> take the suggestion.
    * A hand-picked name that suits the job -> leave it alone.
    * A hand-picked name that suits the *wrong kind* of job -> replace it.

    The third is the one that matters. Opening a photo and then a video left
    the photo's `..._360_TB.jpg` in the box, and nothing downstream caught it:
    the render ran every frame and only then died muxing a video into a JPEG,
    leaving a truncated file. Keeping that name is not respecting a choice,
    it is preserving a guaranteed failure -- and the core now refuses it up
    front regardless, so keeping it would only move the complaint.
    """
    if not current or current == previous_suggestion:
        return suggestion
    if is_image(current) != bool(input_is_image):
        return suggestion
    return current


def save_filters(photo: bool) -> List[str]:
    """Name filters for the save dialog.

    JPEG leads for photos, and not merely by convention: it is the only still
    format the headsets read the projection tags out of, so the others are
    offered as a way to get pixels, not as a way to get a VR photo.
    """
    if photo:
        return ["JPEG photo (*.jpg *.jpeg)", "PNG image (*.png)",
                "All files (*)"]
    return ["MP4 video (*.mp4)"]


#: Quality presets, and what each costs. The encoder timings are measured on
#: an 8K top-bottom render whose pipeline produces a frame every 0.80 s, so
#: anything above that number is the encoder becoming the bottleneck rather
#: than the conversion -- which is worth saying out loud in the UI, because
#: nothing else in the app would reveal it.
QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "draft": {
        "label": "Draft",
        "detail": "x264 crf 20 · smallest file",
        "crf": 20, "codec": "libx264", "preset": "medium", "bitdepth": 8,
        "encoder_cost": 0.28, "note": "",
    },
    "standard": {
        "label": "Standard",
        "detail": "x264 crf 18 · good for most viewing",
        "crf": 18, "codec": "libx264", "preset": "medium", "bitdepth": 8,
        "encoder_cost": 0.32, "note": "",
    },
    "vr": {
        "label": "VR final",
        "detail": "x265 crf 15 slow · higher bitrate",
        "crf": 15, "codec": "libx265", "preset": "slow", "bitdepth": 8,
        "encoder_cost": 1.26,
        "note": "Encoding becomes the bottleneck — expect roughly 1.6× "
                "the render time.",
    },
    "archival": {
        "label": "Archival",
        "detail": "x265 crf 13 slow · 10-bit master",
        "crf": 13, "codec": "libx265", "preset": "slow", "bitdepth": 10,
        "encoder_cost": 1.90,
        "note": "Encoding becomes the bottleneck — expect roughly 2.4× "
                "the render time.",
    },
}

#: Depth model choices, and what each maps to per backend. Base measured best
#: on this footage -- lowest depth noise, best sharpness per unit noise, and
#: 40% less frame-to-frame flicker than Small, for about +9% on total render
#: time since depth is only ~30% of the pipeline at 8K. Large cost twice that
#: again and measured no better. Every Hub id was checked to exist before being
#: offered: a dropdown entry that 404s on first use is worse than no entry.
#:
#: Every choice maps explicitly, including the one that matches the CLI
#: default. Leaving that one blank to "fall through" is what silently turns a
#: Small selection into whatever the default happens to be the day it moves.
DEPTH_MODELS: Dict[str, Dict[str, str]] = {
    "small": {"depth-anything": "depth-anything/Depth-Anything-V2-Small-hf",
              "depth-anything-v3": "small"},
    "base": {"depth-anything": "depth-anything/Depth-Anything-V2-Base-hf",
             "depth-anything-v3": "base"},
    "large": {"depth-anything": "depth-anything/Depth-Anything-V2-Large-hf",
              "video-depth-anything": "large",
              "depth-anything-v3": "large"},
}

#: What `--depth-model` is omitted for, because it is already the CLI default.
DEFAULT_DEPTH_MODEL = "base"

#: Per backend, since they no longer agree: V2 measured best at Base, while V3
#: measured best at Small -- Large was actually worse there. Keyed by backend
#: so a shared "base" selection does not quietly become the wrong default when
#: the backend changes underneath it.
DEFAULT_MODEL_FOR: Dict[str, str] = {
    "depth-anything": DEFAULT_DEPTH_MODEL,
    "video-depth-anything": "small",
    "depth-anything-v3": "small",
}

#: Models the temporal backend does not ship. It only has small and large, so
#: asking it for Base would fail at load time rather than in the UI.
_VDA_MODELS = ("small", "large")

#: Where the CLI looks for an exported ONNX model unless told otherwise.
DEFAULT_ONNX_MODEL = "models/depth_anything_v2_small.onnx"

#: Pipeline seconds per frame at 8K, used only to describe the presets above.
_PIPELINE_SPF = 0.80

#: The largest decoded picture either codec admits, in luma samples, reached
#: at the highest level each defines. Not an HEVC number: H.264 caps a frame at
#: 139,264 macroblocks, and 139264 x 256 is the same 35,651,584. Verified by
#: encoding 7680x7680 both ways -- x264 warns "frame MB size (480x480) > level
#: limit (139264)" and tags level 6.2 anyway; x265 says nothing and invents
#: "Level-6.3", which the spec does not define. So switching encoder does not
#: buy room, and neither tag can be believed.
#:
#: Width and height individually are fine to about 16,888 px; the area is what
#: is capped.
#:
#: This is a fact about *direct playback*, not about the file being wrong. Both
#: streams decode in software, and YouTube transcodes on upload, so 7680x7680
#: remains the right 8K 3D-360 master. It bites when a headset has to decode
#: the frame itself.
MAX_LEVEL_LUMA = 35_651_584


#: Widths offered as smaller alternatives to a source's own. Standard sizes
#: rather than arbitrary numbers, so the choice reads as a format ("5.7K")
#: instead of a slider position.
_STANDARD_WIDTHS = (7680, 5760, 4096, 3840, 2880, 1920)


def output_size(width: int, height: int, output_mode: str = "360",
                output_width: Optional[int] = None) -> tuple:
    """Encoded frame size for a `width` x `height` equirect source.

    Mirrors `pipeline.output_geometry`, which this process cannot call: the UI
    stays out of the core so that drawing a window never imports numpy. The
    duplication is pinned by a test that runs both over a range of sizes.
    """
    if output_width and output_width != width:
        scale = output_width / width
        width = int(round(width * scale)) // 2 * 2
        height = int(round(height * scale)) // 2 * 2
    if output_mode == "vr180":
        return (width // 2) // 2 * 2 * 2, height
    return width, height * 2


#: Tiles per cube face for a photo, mirrored from `cli.PHOTO_DEPTH_TILES`.
#: One, same as a video: tiling helped V2 and measurably hurts both models
#: that replaced it as defaults. The reasoning is with the CLI constant, and
#: a test pins this to its value.
PHOTO_DEPTH_TILES = 1


#: The largest 360 width whose stacked output a headset still decodes.
#: 5760x5760 is 33.2 MP against the 35.6 MP ceiling, and played on a Quest 3
#: where 7680x7680 showed nothing at all.
HEADSET_SAFE_WIDTH = 5760


def default_output_width(width: int, height: int, output_mode: str = "360",
                         is_photo: bool = False) -> int:
    """The width to start on for this source. 0 means the source's own.

    Full size is the better *master* -- it is what you upload, and YouTube
    transcodes anyway -- but it is not what most people are about to do with
    the file. An 8K 360 render at 7680x7680 is black on a Quest 3, so
    defaulting to it means the commonest outcome is a file that does not
    play, discovered after a three-hour render. Starting at a size that plays
    fails the other way, which is recoverable: the full size is still one
    click away and the dropdown says which is which.

    Only 360 video. VR180 at 8K is 7680x3840, comfortably under the ceiling,
    and stills are not subject to it at all -- a 59 MP stereo JPEG displays
    fine, because that limit belongs to the video decoder.
    """
    if is_photo or output_mode != "360" or width <= HEADSET_SAFE_WIDTH:
        return 0
    return HEADSET_SAFE_WIDTH


def resolution_choices(width: int, height: int,
                       output_mode: str = "360") -> List[Dict[str, Any]]:
    """Every output size worth offering for this source, largest first.

    The source's own width always comes first and is the default -- it is the
    right master for upload, whatever it measures. Smaller standard widths
    follow, and exist because an 8K 360 frame is 7680x7680 and no HEVC or
    H.264 level decodes that: confirmed black on a Quest 3 in both codecs,
    while 5760x5760 and 8K VR180 both play.

    From a 4K source nothing here is above the cap, so the list is just sizes
    and carries no warning. The restriction appears exactly when it applies.
    """
    seen = set()
    out = []
    for candidate in (width,) + _STANDARD_WIDTHS:
        if candidate > width or candidate in seen:
            continue
        seen.add(candidate)
        w, h = output_size(width, height, output_mode, candidate)
        out.append({
            "width": candidate,
            "label": f"{w}×{h}",
            "megapixels": round(w * h / 1e6, 1),
            "fits": w * h <= MAX_LEVEL_LUMA,
            "native": candidate == width,
        })
    return out


# CLI defaults. A value equal to its default is left off the command line.
_DEFAULTS = {
    "strength": 1.0,
    "gradientLimit": 1.0,
    "faceAngularCorrection": 0.0,
    "leftShare": 0.0,
    "depthTiles": 1,
    "fgErode": 2,
    "smooth": 0,
    "inpaint": "simple",
    "depthBackend": "",
    "device": "auto",
    "chunkSize": 8,
    "chunkOverlap": 2,
    "startFrame": 0,
    "crf": 18,
    "preset": "medium",
    "codec": "libx264",
    "bitdepth": 8,
    "outputMode": "360",
    "yaw": 0.0,
}


def preset_note(quality: str) -> str:
    """The warning line for a quality preset, or '' when it is free."""
    return QUALITY_PRESETS.get(quality, {}).get("note", "")


def _num(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def build_argv(
    opts: Dict[str, Any],
    *,
    preview_frame: Optional[int] = None,
    preview_output: Optional[str] = None,
    preview_width: int = 1600,
) -> List[str]:
    """Arguments after the interpreter: ['-m', 'stereo360', ...].

    `preview_frame` switches to single-frame mode, writing to
    `preview_output` instead of the configured video path.
    """
    if not opts.get("input"):
        raise ValueError("Choose an input video first.")

    is_preview = preview_frame is not None
    # A photo job. Not a preview -- the CLI decides that from the input's own
    # extension, exactly as this does, so the two cannot disagree about what
    # kind of job it is.
    photo = is_image(opts["input"]) and not is_preview
    out = preview_output if is_preview else opts.get("output")
    if not out:
        raise ValueError("Choose where to save the output first.")

    argv = ["-m", "stereo360", str(opts["input"]), "-o", str(out)]

    quality = QUALITY_PRESETS.get(opts.get("quality", "standard"),
                                  QUALITY_PRESETS["standard"])
    # Encoder settings are meaningless for a still, and passing them would
    # only make the displayed command look more complicated than it is.
    if not is_preview and not photo:
        # An explicit encoder choice overrides the preset's; everything else
        # about the preset (quality, speed, bit depth) still applies, so the
        # two controls compose instead of one shadowing the other.
        chosen = {"crf": quality["crf"], "preset": quality["preset"],
                  "bitdepth": quality["bitdepth"],
                  "codec": str(opts.get("codec") or "") or quality["codec"]}
        for flag, key in (("--crf", "crf"), ("--codec", "codec"),
                          ("--preset", "preset"), ("--bitdepth", "bitdepth")):
            if chosen[key] != _DEFAULTS[key]:
                argv += [flag, str(chosen[key])]

    # Applies to a preview as much as to a render: the whole point of
    # previewing a VR180 frame is to see which half of the sphere you chose.
    mode = str(opts.get("outputMode") or _DEFAULTS["outputMode"])
    if mode != _DEFAULTS["outputMode"]:
        argv += ["--output-mode", mode]
        # Only inside the mode that has a direction to choose. The CLI refuses
        # a yaw in 360 mode rather than ignore it, so emitting one here would
        # turn a stale control into a failed run.
        yaw = _num(opts.get("yaw"), 0.0)
        if abs(yaw) > 1e-9:
            argv += ["--yaw", f"{yaw:g}"]

    # Omitted when it matches the source, which is both the default and the
    # commonest case -- the flag only appears when a size was actually chosen.
    out_width = int(_num(opts.get("outputWidth"), 0))
    source_width = int(_num(opts.get("sourceWidth"), 0))
    if out_width > 0 and out_width != source_width:
        argv += ["--output-width", str(out_width)]

    if not opts.get("faceSizeAuto", True):
        size = int(_num(opts.get("faceSize"), 0))
        if size > 0:
            argv += ["--face-size", str(size)]

    strength = _num(opts.get("strength"), 1.0)
    if abs(strength - _DEFAULTS["strength"]) > 1e-9:
        argv += ["--strength", f"{strength:g}"]

    grad = _num(opts.get("gradientLimit"), 1.0)
    if abs(grad - _DEFAULTS["gradientLimit"]) > 1e-9:
        argv += ["--gradient-limit", f"{grad:g}"]

    angular = _num(opts.get("faceAngularCorrection"), 0.0)
    if abs(angular - _DEFAULTS["faceAngularCorrection"]) > 1e-9:
        argv += ["--face-angular-correction", f"{angular:g}"]

    # Against the default for *this kind of job*: the CLI uses 3 for a photo
    # and 1 for a video, so a fixed comparison here would emit the flag when
    # it is not needed and, worse, omit it when it is.
    tiles = int(_num(opts.get("depthTiles"), 1))
    if tiles != (PHOTO_DEPTH_TILES if photo else _DEFAULTS["depthTiles"]):
        argv += ["--depth-tiles", str(tiles)]

    # One number covers both halves of the control: which eye keeps the
    # source, and how far off it the other sits. 0 leaves the left eye
    # untouched, 1 the right, 0.5 splits evenly. `splitBaseline` is still
    # honoured so presets saved before the slider existed keep working.
    share = _num(opts.get("leftShare"), _DEFAULTS["leftShare"])
    if opts.get("leftShare") is None and opts.get("splitBaseline"):
        share = 0.5
    if abs(share - _DEFAULTS["leftShare"]) > 1e-9:
        argv += ["--left-share", f"{share:g}"]

    # Empty means the CLI's own per-job default -- V3 for video, Depth Pro for
    # a still. Passing nothing is what selects it, so the UI never has to know
    # which one that is.
    backend = str(opts.get("depthBackend") or "")
    if backend:
        argv += ["--depth-backend", backend]

    # Which "model" means depends on the backend: a Hub id or variant name for
    # the torch backends, a path to an exported graph for ONNX. The CLI keeps
    # those as two separate flags, so the UI works out which one applies rather
    # than making the user know. Depth Pro ships one checkpoint and the
    # recommended entry picks its own, so neither takes a model.
    if backend == "onnx":
        onnx = str(opts.get("onnxModel") or "").strip()
        if onnx and onnx != DEFAULT_ONNX_MODEL:
            argv += ["--onnx-model", onnx]
    elif backend not in ("", "depth-pro"):
        model = str(opts.get("depthModel", DEFAULT_DEPTH_MODEL))
        if backend == "video-depth-anything":
            # It ships small and large only; anything else would fail at load,
            # and its own default is small.
            key = "video-depth-anything"
            if model not in _VDA_MODELS:
                model = "small"
        elif backend == "depth-anything-v3":
            key = "depth-anything-v3"
        else:
            key = "depth-anything"
        mapped = DEPTH_MODELS.get(model, {}).get(key)
        # Emitted unless it is already what the CLI would pick, so the command
        # line stays short without depending on the two agreeing.
        if mapped and model != DEFAULT_MODEL_FOR.get(key):
            argv += ["--depth-model", mapped]

    device = opts.get("device", "auto")
    if device != _DEFAULTS["device"]:
        argv += ["--device", str(device)]

    erode = int(_num(opts.get("fgErode"), 2))
    if erode != _DEFAULTS["fgErode"]:
        argv += ["--fg-erode", str(erode)]

    smooth = int(_num(opts.get("smooth"), 0))
    if smooth != _DEFAULTS["smooth"]:
        argv += ["--smooth", str(smooth)]

    inpaint = opts.get("inpaint", "simple")
    if inpaint != _DEFAULTS["inpaint"]:
        argv += ["--inpaint", str(inpaint)]

    if backend == "video-depth-anything":
        chunk = int(_num(opts.get("chunkSize"), 8))
        if chunk != _DEFAULTS["chunkSize"]:
            argv += ["--chunk-size", str(chunk)]
        overlap = int(_num(opts.get("chunkOverlap"), 2))
        if overlap != _DEFAULTS["chunkOverlap"]:
            argv += ["--chunk-overlap", str(overlap)]
        if not opts.get("temporalFill", True):
            argv += ["--no-temporal-fill"]

    if is_preview:
        argv += ["--preview-frame", str(int(preview_frame)),
                 "--preview-width", str(int(preview_width))]
    elif photo:
        # Nothing to add, and several things not to. The CLI *refuses*
        # --max-frames, --start-frame and --spatial-audio for an image rather
        # than ignoring them, so emitting one -- a spatial-audio switch left
        # on from the last video, say -- would fail every photo conversion.
        pass
    else:
        # Frame range only applies to a real render; a preview names its own
        # frame and would be cut short by a max-frames cap.
        start = int(_num(opts.get("startFrame"), 0))
        if start != _DEFAULTS["startFrame"]:
            argv += ["--start-frame", str(start)]
        limit = int(_num(opts.get("maxFrames"), 0))
        if limit > 0:
            argv += ["--max-frames", str(limit)]
        if opts.get("spatialAudio"):
            argv += ["--spatial-audio"]
        # Only meaningful for a real encode; a preview is a PNG.
        if opts.get("sourceSubsampling"):
            argv += ["--source-subsampling"]

    argv += ["--progress-json"]
    return argv


def display_command(argv: List[str]) -> str:
    """The same command as a user could paste into a shell."""
    parts = ["python"] + list(argv)
    return " ".join(f'"{p}"' if " " in p else p for p in parts)
