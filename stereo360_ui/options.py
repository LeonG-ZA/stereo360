"""Translate UI state into a stereo360 command line.

The one place in the UI that knows flag names. Everything else passes a plain
dict around, so adding an option means touching this file and the QML control
that sets it -- not a chain of typed properties.

Only non-default flags are emitted. That keeps the command the UI shows the
user short enough to read, and short enough to paste into a terminal, which is
how someone graduates from the UI to the CLI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    "small": {"depth-anything": "depth-anything/Depth-Anything-V2-Small-hf"},
    "base": {"depth-anything": "depth-anything/Depth-Anything-V2-Base-hf"},
    "large": {"depth-anything": "depth-anything/Depth-Anything-V2-Large-hf",
              "video-depth-anything": "large"},
}

#: What `--depth-model` is omitted for, because it is already the CLI default.
DEFAULT_DEPTH_MODEL = "base"

#: Models the temporal backend does not ship. It only has small and large, so
#: asking it for Base would fail at load time rather than in the UI.
_VDA_MODELS = ("small", "large")

#: Where the CLI looks for an exported ONNX model unless told otherwise.
DEFAULT_ONNX_MODEL = "models/depth_anything_v2_small.onnx"

#: Pipeline seconds per frame at 8K, used only to describe the presets above.
_PIPELINE_SPF = 0.80

# CLI defaults. A value equal to its default is left off the command line.
_DEFAULTS = {
    "strength": 1.0,
    "gradientLimit": 1.0,
    "depthTiles": 1,
    "fgErode": 2,
    "smooth": 0,
    "inpaint": "simple",
    "depthBackend": "auto",
    "device": "auto",
    "chunkSize": 8,
    "chunkOverlap": 2,
    "startFrame": 0,
    "crf": 18,
    "preset": "medium",
    "codec": "libx264",
    "bitdepth": 8,
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
    out = preview_output if is_preview else opts.get("output")
    if not out:
        raise ValueError("Choose where to save the output first.")

    argv = ["-m", "stereo360", str(opts["input"]), "-o", str(out)]

    quality = QUALITY_PRESETS.get(opts.get("quality", "standard"),
                                  QUALITY_PRESETS["standard"])
    # Encoder settings are meaningless for a still, and passing them would
    # only make the displayed command look more complicated than it is.
    if not is_preview:
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

    tiles = int(_num(opts.get("depthTiles"), 1))
    if tiles != _DEFAULTS["depthTiles"]:
        argv += ["--depth-tiles", str(tiles)]

    if opts.get("splitBaseline"):
        argv += ["--split-baseline"]

    backend = opts.get("depthBackend", "auto")
    if backend != _DEFAULTS["depthBackend"]:
        argv += ["--depth-backend", str(backend)]

    # Which "model" means depends on the backend: a Hub id or variant name for
    # the torch backends, a path to an exported graph for ONNX. The CLI keeps
    # those as two separate flags, so the UI works out which one applies rather
    # than making the user know.
    if backend == "onnx":
        onnx = str(opts.get("onnxModel") or "").strip()
        if onnx and onnx != DEFAULT_ONNX_MODEL:
            argv += ["--onnx-model", onnx]
    else:
        model = str(opts.get("depthModel", DEFAULT_DEPTH_MODEL))
        if backend == "video-depth-anything":
            # It ships small and large only; anything else would fail at load,
            # and its own default is small.
            key = "video-depth-anything"
            if model not in _VDA_MODELS:
                model = "small"
        else:
            key = "depth-anything"
        mapped = DEPTH_MODELS.get(model, {}).get(key)
        # Emitted unless it is already what the CLI would pick, so the command
        # line stays short without depending on the two agreeing.
        if mapped and not (key == "depth-anything"
                           and model == DEFAULT_DEPTH_MODEL):
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
