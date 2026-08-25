"""Command-line interface for stereo360."""

from __future__ import annotations

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stereo360",
        description="Convert monoscopic 360 equirectangular video to "
                    "stereoscopic top-bottom 360 video.",
    )
    # The one import at parser-build time, and a cheap one: stereo360/__init__
    # is already imported by anything that reached this function. Worth having
    # because a version nothing can read is a version nothing keeps correct --
    # this constant said 0.1.0 through the whole of v1.0.0.
    from . import __version__, released_as

    p.add_argument("--version", action="version",
                   version=f"stereo360 {released_as(__version__)}")
    # Optional so the probes that describe the *machine* rather than a file --
    # --probe-backends and --probe-encoders -- can run without one. Required
    # for everything else, checked in main(). argparse cannot express that.
    p.add_argument("input", nargs="?",
                   help="Input monoscopic 360 video (equirectangular). Not "
                        "needed with --probe-backends or --probe-encoders.")
    p.add_argument("-o", "--output", default=None,
                   help="Output MP4 path (not needed with --probe-json)")
    p.add_argument("--face-size", type=int, default=None,
                   help="Cubemap face resolution in pixels. Default: auto "
                        "(input width / 4, the lossless value, e.g. 1920 for 7680 wide).")
    p.add_argument("--crf", type=int, default=18,
                   help="Encoder CRF, lower = better quality. Guide: 18 = quick tests, "
                        "14-16 = final VR output, 12-14 = archival/master. (default: 18)")
    p.add_argument("--preset", default="medium", help="Encoder preset (default: medium)")
    p.add_argument("--codec", default="libx264",
                   choices=[c[0] for c in __import__(
                       "stereo360.encoders", fromlist=["x"]).CANDIDATES],
                   help="Video codec. libx264 (default) / libx265 are CPU; "
                        "hevc_nvenc is NVIDIA hardware and ~5x faster at 8K "
                        "(needs driver 610+ for recent ffmpeg builds). "
                        "h264_nvenc cannot exceed 4096x4096, so it is unusable "
                        "for 8K top-bottom output. libx265 recommended with "
                        "--bitdepth 10")
    p.add_argument("--bitdepth", type=int, default=8, choices=[8, 10],
                   help="Output bit depth. 10-bit greatly reduces gradient banding in VR "
                        "headsets (sky, fog, walls). Use with libx265 for best results. "
                        "(default: 8)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Process only the first N frames (for testing)")
    p.add_argument("--no-cubemap", action="store_true",
                   help="Skip the cubemap round-trip (pure decode/encode test)")
    p.add_argument("--passthrough", action="store_true",
                   help="M1 mode: right eye = left eye (no depth, no stereo)")
    p.add_argument("--depth-backend", default=None,
                   # Spelled out rather than imported from backends, so
                   # --help stays import-free. test_cli_choices_match_backends
                   # keeps the two honest.
                   choices=["auto", "depth-anything-v3", "depth-pro",
                            "depth-anything", "video-depth-anything", "onnx"],
                   help="Default depends on the input: depth-anything-v3 for "
                        "video, depth-pro for a still image. "
                        "depth-anything-v3 = multi-view depth via ONNX, "
                        "flattest walls and floors of anything measured, runs "
                        "on the CPU; depth-pro = Apple Depth Pro, the sharpest "
                        "thin structures, 3.6 GB and GPU-hungry; auto = probe "
                        "the machine and use the fastest runtime available, "
                        "reporting which; depth-anything = per-frame V2 via "
                        "torch (M2); video-depth-anything = temporal video "
                        "depth model, flicker-free (M3); onnx = per-frame via "
                        "ONNX Runtime (M4: DirectML for AMD/Intel, CUDA, "
                        "CoreML; no PyTorch needed)")
    p.add_argument("--onnx-model", default="models/depth_anything_v2_small.onnx",
                   help="Path to the exported ONNX depth model (backend=onnx). "
                        "Export with: python scripts/export_onnx.py")
    p.add_argument("--ort-provider", default=None,
                   help="onnxruntime execution provider override, e.g. "
                        "DmlExecutionProvider (default: auto = best available)")
    p.add_argument("--fp16", action="store_true",
                   help="Half-precision inference for video-depth-anything "
                        "(faster on GPU, slight quality cost)")
    p.add_argument("--vda-input-size", type=int, default=518, metavar="N",
                   help="Inference resolution for video-depth-anything "
                        "(multiple of 14). Raise to 714-910 for scenes with "
                        "thin structures (railings, cables): at the default "
                        "518 the ViT's 14-px patches show up as blocky "
                        "squares around them. Costs time/VRAM (default: 518)")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Skip the first N input frames (resume a partial "
                        "conversion into a new output file; concatenate "
                        "segments afterwards with ffmpeg)")
    p.add_argument("--depth-model", default=None,
                   help="HuggingFace model id (depth-anything) or variant "
                        "'small'/'large' (video-depth-anything). Default for "
                        "depth-anything is Depth-Anything-V2-Base, which "
                        "measured the lowest depth noise and 40%% less "
                        "frame-to-frame flicker than Small for about +9%% "
                        "render time; pass the Small id for a faster run. The "
                        "temporal backend ships small and large only, and "
                        "defaults to small.")
    p.add_argument("--chunk-size", type=int, default=8,
                   help="Temporal chunk length for video-depth-anything; depth "
                        "is estimated with cross-frame context per chunk. "
                        "1 disables chunking. (default: 8)")
    p.add_argument("--chunk-overlap", type=int, default=2,
                   help="Frames re-estimated at chunk boundaries and "
                        "ramp-blended to hide seams (default: 2)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "mps", "cpu"],
                   help="Inference device (default: auto)")
    p.add_argument("--strength", type=float, default=1.0,
                   help="Stereo baseline strength; 1.0 = default comfort, "
                        "higher = stronger 3D effect (default: 1.0)")
    p.add_argument("--smooth", type=int, default=0, metavar="RADIUS",
                   help="Edge-aware depth smoothing radius (guided filter, "
                        "guided by the RGB frame). 0 disables smoothing. "
                        "Higher = smoother depth. Off by default: it was "
                        "measured at 66%% of total runtime, and with "
                        "--gradient-limit handling depth cliffs the output was "
                        "indistinguishable without it (default: 0)")
    p.add_argument("--smooth-eps", type=float, default=1e-3,
                   help="Guided-filter regularization; higher = smoother "
                        "(default: 0.001)")
    p.add_argument("--fg-erode", type=int, default=2, metavar="N",
                   help="Foreground erosion at depth edges (pixels): pulls "
                        "near-side boundary depth down to background level so "
                        "disocclusion holes fill from background instead of "
                        "smearing bright foreground. 0 disables (default: 2)")
    p.add_argument("--inpaint", default="simple",
                   choices=["simple", "learned"],
                   help="Hole filling: simple = OpenCV Telea (fast); learned "
                        "= LaMa neural inpainting, much better texture, "
                        "slower (default: simple)")
    p.add_argument("--no-temporal-fill", action="store_false",
                   dest="temporal_fill", default=True,
                   help="Disable filling holes from other frames in the chunk. "
                        "Temporal fill is on by default: it uses real pixels "
                        "another frame actually saw, in preference to anything "
                        "invented, so it strictly improves on --inpaint where "
                        "it applies. Needs --chunk-size > 1.")
    p.add_argument("--gradient-limit", type=float, default=1.0, metavar="X",
                   help="Clamp the depth gradient to X times the critical "
                        "slope at which the warp stops being injective. Holes "
                        "form exactly where that slope is exceeded, so this "
                        "prevents them rather than filling them. 1.0 is the "
                        "break-even point, lower is safer; 0 disables. Keeps "
                        "the full depth range and costs depth only on sharp "
                        "structures. On by default at 1.0, which measured zero "
                        "holes on 8K footage at full strength while keeping the "
                        "whole depth range; it costs roughly 9%% of the depth "
                        "amplitude on the very sharpest edges. Raise it if thin "
                        "structures look flat, or pass 0 to disable entirely.")
    p.add_argument("--spatial-audio", action="store_true",
                   help="Describe the audio track as ambiX ambisonics "
                        "(ACN ordering, SN3D normalization) by writing an SA3D "
                        "box, matching the third checkbox in Google's Spatial "
                        "Media Metadata Injector. The source audio must "
                        "actually be ambiX: 4, 9 or 16 channels for first, "
                        "second or third order. Spherical and stereoscopic "
                        "top-bottom are always written and need no flag. Also "
                        "what lets --yaw turn the soundfield with the view: "
                        "without this flag the audio is copied through and "
                        "every sound stays at its original bearing.")
    p.add_argument("--ambisonic-codec", default="auto",
                   choices=["auto", "libfdk_aac", "aac", "libopus",
                            "pcm_s24le"],
                   help="How to write the soundfield back when --yaw rotates "
                        "it. auto (default) takes the first your ffmpeg has, "
                        "in that order. libfdk_aac is the best AAC and the "
                        "format players expect, but its licence keeps it out "
                        "of most Windows builds; libopus handles any order; "
                        "pcm_s24le re-compresses nothing at all but "
                        "plays only on the desktop. Measured on a Quest 3: "
                        "AAC plays, PCM is silent, and Opus is silent in both "
                        "mapping families -- so 'aac' is the fallback when "
                        "libfdk_aac is absent, at the highest bitrate it "
                        "accepts, and it warns that it is the worse encoder. "
                        "Ignored without a yaw, when audio is copied "
                        "untouched.")
    p.add_argument("--left-share", type=float, default=None, metavar="F",
                   help="Fraction of the separation the LEFT eye carries, "
                        "0 to 1. 0.5 (default) splits it evenly; 0 leaves "
                        "the left eye untouched and puts the whole baseline "
                        "in the right; 1 leaves the right eye untouched "
                        "instead. The 3D effect is identical at every "
                        "setting - this chooses where the errors land, not "
                        "whether there are any. Sharing costs a second warp "
                        "per frame and roughly doubles chunk memory; "
                        "consider a smaller --chunk-size with it.")
    p.add_argument("--source-eye", choices=("left", "right"), default=None,
                   help="Which eye keeps the source frame when --left-share "
                        "is 0 or 1. Shorthand: --source-eye right is "
                        "--left-share 1. Which one to pick is scene "
                        "dependent: the warped eye HIDES what is behind an "
                        "occluder on one side and has to INVENT it on the "
                        "other, and which way that falls depends on where "
                        "each occluder sits.")
    p.add_argument("--split-baseline", action="store_true",
                   help="Deprecated. Means --left-share 0.5, which is now the default anyway.")
    p.add_argument("--depth-tiles", type=int, default=None, metavar="N",
                   help="Split each cubemap face into NxN overlapping tiles "
                        "for depth inference (feather-blended). Higher = "
                        "finer depth on thin structures (curtains, railings) "
                        "since tiles need little/no downsampling; N squared "
                        "times slower. 1 = whole faces. Not monotonic -- past "
                        "a point tiles are too small to hold a long edge in "
                        "context and it softens away. "
                        "(default: 1 for video, 3 for a photo)")
    # Default deliberately None rather than projection.FACE_OVERLAP: importing
    # projection here would pull numpy and cv2 into every --help.
    p.add_argument("--output-mode", default="360", choices=["360", "vr180"],
                   help="What to produce. 360 (default) is a full sphere per "
                        "eye, stacked top over bottom. vr180 keeps the middle "
                        "180 degrees and puts the eyes side by side, which is "
                        "what VR180 players expect and what Apple Vision Pro "
                        "content uses. Same pixel count either way, so vr180 "
                        "spends them on half the sphere at twice the angular "
                        "resolution -- and an 8K vr180 frame is inside HEVC's "
                        "decode limit where an 8K 360 frame is not. Input must "
                        "be 360 in both cases.")
    p.add_argument("--yaw", type=float, default=0.0, metavar="DEG",
                   help="Which way the VR180 field points, in degrees of "
                        "longitude, positive to the right. Wraps, so -200 and "
                        "+160 are the same. Free and lossless: it selects a "
                        "range of columns rather than rotating anything, so "
                        "nothing is resampled. Only valid with --output-mode "
                        "vr180, since 360 output keeps the whole sphere and "
                        "has no direction to choose. (default: 0)")
    p.add_argument("--output-width", type=int, default=None, metavar="W",
                   help="Deliver a frame W pixels wide instead of whatever "
                        "the source implies: 360 output becomes WxW, vr180 "
                        "Wx(W/2). Depth and warping still run at the source "
                        "resolution and only the finished eyes are resized, "
                        "so the result is supersampled rather than rendered "
                        "small -- and costs the same time as the full-size "
                        "render. The reason it exists: 8K 360 output is "
                        "7680x7680, which no HEVC or H.264 level decodes "
                        "(confirmed black on a Quest 3 in both codecs) while "
                        "still being the right master for YouTube, which "
                        "transcodes on ingest. 5760 is the largest square "
                        "that plays. Cannot scale up.")
    p.add_argument("--face-overlap", type=float, default=None, metavar="F",
                   help="How far each depth face reaches past its nominal 90 "
                        "degrees, in tangent units (0.15 = 98 degrees per "
                        "face). Neighbouring faces then share a band instead "
                        "of only an edge, so their depth scales are fitted on "
                        "real common ground and whatever disagreement is left "
                        "is cross-faded over degrees instead of stepping at "
                        "the seam. 0 restores exact faces, which creases the "
                        "ground where two of them meet. Going much wider "
                        "stretches the corners past what a depth model was "
                        "trained on and costs accuracy without helping the "
                        "seam. (default: 0.15)")
    p.add_argument("--face-angular-correction", type=float, default=None,
                   metavar="F",
                   help="Pull each depth face's edges back onto their true "
                        "rays. Depth Anything V3 estimates its own camera and "
                        "that estimate saturates around 65 degrees, so at the "
                        "98-degree faces used here it thinks it is looking "
                        "through a much longer lens than it is, and places "
                        "everything toward a face edge nearer than it really "
                        "is. On flat ground that reads as the floor bulging up "
                        "toward you at about one camera height out, which is "
                        "where the cube seam falls. F scales the fix: 0 is off, "
                        "0.55-0.7 measured best, 1.0 overshoots the other way. "
                        "It also lowers the depth range about 20%%, so pair it "
                        "with --strength 1.2 to keep the same parallax. "
                        "Measured on V3 only. (default: 0, off)")
    p.add_argument("--flatten-ground", type=float, default=0.0, metavar="F",
                   help="Pull the ground onto the flat plane it actually is. "
                        "For any plane, inverse depth is exactly linear in the "
                        "ray direction, so one plane is fitted to whatever is "
                        "below the horizon and the ground is blended toward "
                        "it. Fixes what --face-angular-correction cannot: that "
                        "correction straightens the dome, which is symmetric "
                        "about each face axis, and leaves the ground *tilted* "
                        "-- measured on a road, 0.13 camera heights too high "
                        "behind and 0.10 too low in front, which reads as the "
                        "road still bulging. A pixel's correction fades out as "
                        "it disagrees with the plane, so a kerb or a pothole "
                        "keeps its own depth while a bowed road does not. "
                        "Needs a dominant plane in view and refuses without "
                        "one. 0 = off. (default: 0)")
    p.add_argument("--temporal-depth", type=float, default=0.02,
                   metavar="TAU",
                   help="Hold static depth still between frames, so a "
                        "stationary object stops shifting slightly each frame. "
                        "Each pixel is pulled toward its previous value by an "
                        "amount that falls to zero as it starts moving, so "
                        "motion is not smeared. TAU is the normalised depth "
                        "change treated as 'not moving'; 0 disables. Applies "
                        "to the per-frame backends -- the temporal backend has "
                        "its own equivalent. (default: 0.02)")
    p.add_argument("--input-projection", default="auto",
                   choices=["auto", "equirectangular", "cubemap"],
                   help="How to read the input. auto (default) believes the "
                        "file's own sv3d/cbmp metadata and assumes "
                        "equirectangular when it declares nothing, which is "
                        "almost always right. cubemap reads a 3x2 cube layout "
                        "and feeds the file's own faces to depth estimation "
                        "instead of rebuilding them. Override only when a file "
                        "is tagged wrongly.")
    p.add_argument("--source-subsampling", action="store_true",
                   help="Encode with the source's chroma subsampling instead "
                        "of 4:2:0. Only worthwhile for a 4:2:2/4:4:4 source: "
                        "measured on 4:2:0 footage, 4:4:4 output was 4%% more "
                        "faithful for 25%% more encode time, because the "
                        "camera had already halved the chroma detail. 4:2:0 is "
                        "also the only layout headset hardware decoders accept "
                        "at 8K, so anything else is for masters and uploads, "
                        "not for direct playback. Falls back to 4:2:0 with a "
                        "warning where the codec cannot oblige.")
    p.add_argument("--probe-encoders", metavar="WxH", default=None,
                   help="Print which video encoders can encode WxH on this "
                        "machine as JSON, and exit. Availability is "
                        "resolution-dependent -- hevc_amf takes 3840x3840 and "
                        "refuses 7680x7680 -- so pass the *output* size, which "
                        "for top-bottom stereo is the source height doubled.")
    p.add_argument("--probe-backends", action="store_true",
                   help="Print which depth backends can actually run here as "
                        "JSON, with a reason for each that cannot, and exit. "
                        "Intended for a GUI so it does not offer a backend "
                        "that will fail seconds later.")
    p.add_argument("--probe-json", action="store_true",
                   help="Print what we can determine about the input as JSON "
                        "and exit: dimensions, fps, frame count, audio, "
                        "colour tags and chroma subsampling. Intended for a "
                        "GUI, so it does not have to reimplement any of it.")
    p.add_argument("--preview-frame", type=int, default=None, metavar="N",
                   help="Render source frame N as a single image to -o and "
                        "stop, instead of converting the whole video. The "
                        "output path must have an image extension (.png, "
                        ".jpg, ...) and receives the top-bottom stereo pair "
                        "exactly as the video would contain it. Use this to "
                        "judge --strength, --gradient-limit and --depth-tiles "
                        "in seconds rather than by sitting through a full "
                        "render. N is an absolute index into the source and "
                        "ignores --start-frame.")
    p.add_argument("--preview-width", type=int, default=2048, metavar="W",
                   help="Downscale the preview to W pixels wide; 0 keeps full "
                        "resolution. A full 8K preview is a 7680x7680 image "
                        "that costs seconds to encode for no visible benefit "
                        "(default: 2048)")
    p.add_argument("--thumbnail", metavar="PATH", default=None,
                   help="Write source frame --preview-frame to PATH as a "
                        "small JPEG and stop, converting nothing. This is the "
                        "picture the interface's VR180 direction picker drags "
                        "on, so it has to appear before any depth work has "
                        "been done -- it decodes one frame and touches no "
                        "model.")
    p.add_argument("--progress-json", action="store_true",
                   help="Emit machine-readable NDJSON events on stdout (one "
                        "JSON object per line: info, warning, start, "
                        "progress, done, error) instead of human-readable "
                        "text and a progress bar. Intended for a parent "
                        "process such as a GUI. In this mode a line reading "
                        "'cancel' on stdin stops the run cleanly, leaving a "
                        "playable file with the frames completed so far.")
    return p


def _json_event_stream():
    """Hand the event writer exclusive use of stdout.

    `--progress-json` promises one JSON object per line, and third-party code
    does not know that. The vendored Video Depth Anything tree prints
    "xFormers not available" straight to stdout, which put two unparseable
    lines into the stream; transformers and tqdm are equally free to print.

    Duplicating the descriptor and pointing fd 1 at stderr means everything
    anyone prints -- Python or native, ours or not -- lands on stderr, while
    only reporter events reach the parent. Done before any import that might
    print, so nothing escapes.
    """
    import io

    saved = os.dup(1)
    os.dup2(sys.stderr.fileno(), 1)
    return io.TextIOWrapper(io.FileIO(saved, "w"), encoding="utf-8",
                            line_buffering=True)


def _stdin_cancel_watcher():
    """Return a predicate that goes True when the parent asks us to stop.

    Signals are the usual way to do this and are awkward on Windows; a line on
    stdin behaves identically everywhere and, unlike a kill, lets the encoder
    finalize the partial file.

    EOF deliberately does not count as a cancel. A parent that spawns us with
    stdin closed -- or redirected from /dev/null -- would otherwise cancel the
    run the instant it started.

    MUST be called after numpy and friends are imported; see `main`.
    """
    import os
    import threading

    flag = threading.Event()

    def watch() -> None:
        buf = b""
        try:
            while True:
                chunk = os.read(0, 4096)
                if not chunk:
                    return          # EOF: no more commands, but not a cancel
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip().lower() in (b"cancel", b"stop"):
                        flag.set()
                        return
        except Exception:
            pass

    threading.Thread(target=watch, daemon=True).start()
    return flag.is_set


#: Flags that only mean something for a video. Given with an image input they
#: are a mistake worth naming: silently ignoring a --max-frames someone typed
#: is how a tool teaches people that its flags are decorative.
_VIDEO_ONLY_FLAGS = (
    ("max_frames", "--max-frames", "a still has one frame"),
    ("start_frame", "--start-frame", "there is nothing to skip"),
    ("preview_frame", "--preview-frame",
     "the output already is that one frame"),
    ("spatial_audio", "--spatial-audio", "an image has no audio track"),
)


def _refuse_video_only_flags(args) -> None:
    # Against the parser's own defaults, not truthiness. `--preview-frame 0`
    # is a perfectly ordinary thing to type and is falsy, so a truthiness test
    # lets exactly the most likely mistake through. Reading the defaults back
    # from the parser also means they cannot drift from the definitions above.
    #
    # It does mean `--start-frame 0` typed out is indistinguishable from not
    # typing it, since argparse does not record what was on the command line.
    # Harmless: it asks for the behaviour it would have got anyway.
    defaults = build_parser()
    given = [(flag, why) for attr, flag, why in _VIDEO_ONLY_FLAGS
             if getattr(args, attr, None) != defaults.get_default(attr)]
    if not given:
        return
    lines = ", ".join(f"{flag} ({why})" for flag, why in given)
    build_parser().error(
        f"the input is an image, so these do not apply: {lines}")


#: Tiles per cube face when nobody says otherwise.
#:
#: One everywhere now, and the photo case is the interesting one: it used to
#: be three. Tiling was worth its N-squared cost against V2, which could not
#: resolve a thin structure inside a whole face and gained real detail from
#: being shown a smaller crop -- judged on a Quest 3, and measured near-free
#: for a single frame at 12 s against 14 s for a 59 MP still.
#:
#: Both current defaults are hurt by it instead, measured on the same photo:
#:
#:   Depth Pro         wall wobble 53% -> 176%, 11 s -> 46 s
#:   Depth Anything V3 chair gap 1.42 -> 1.57, floor rms 28% -> 34%
#:
#: The mechanism is the same one that made 4x4 worse than 3x3 for V2 -- a tile
#: cannot see the context outside itself -- but it bites harder here because
#: what these two models are *good* at is global: V3 fuses six views, and
#: Depth Pro predicts metric depth, so tiles that disagree about scale do not
#: reconcile. Tiling no longer buys the detail either, since resolving thin
#: structure is exactly what Depth Pro already does better than tiled V2 did.
#:
#: Still worth `--depth-tiles 3` with `--depth-backend depth-anything`, where
#: it was measured to help. See "Choosing a depth model" in findings.md.
VIDEO_DEPTH_TILES = 1
PHOTO_DEPTH_TILES = 1


def _left_share(args) -> float:
    """The left eye's share of the separation, from three ways of saying it.

    `--left-share` is the real control. `--source-eye` names which eye keeps
    the source frame untouched, which is shorthand for the two ends of that
    range -- and it has to be handled explicitly now the default is an even
    split, since "left" is no longer what happens anyway. `--split-baseline`
    is the old boolean, kept working because it appears in saved presets and
    scripts. An explicit share always wins, so a preset carrying both is not
    ambiguous.
    """
    if getattr(args, "left_share", None) is not None:
        return float(min(max(args.left_share, 0.0), 1.0))
    if getattr(args, "source_eye", None) == "right":
        return 1.0
    if getattr(args, "source_eye", None) == "left":
        return 0.0
    if getattr(args, "split_baseline", False):
        return 0.5
    # Imported here, not at module scope: `pipeline` is deliberately loaded
    # late so `--help` and the probes do not pay for numpy and torch.
    from . import pipeline
    return pipeline.DEFAULT_LEFT_SHARE


def resolve_depth_tiles(requested, is_image: bool) -> int:
    """How many tiles to use, given what was asked for and what came in.

    `None` means nobody asked. The flag defaults to None rather than to 1 so
    that an explicit `--depth-tiles 1` on a photo is still honoured -- with a
    default of 1 the two are indistinguishable afterwards, and someone asking
    for whole faces would silently get three.
    """
    if requested is not None:
        return int(requested)
    return PHOTO_DEPTH_TILES if is_image else VIDEO_DEPTH_TILES


def _run(args, reporter, cancel, backends, pipeline):
    is_image = pipeline.ffmpeg_io.is_image_path(args.input)
    args.depth_tiles = resolve_depth_tiles(args.depth_tiles, is_image)
    args.depth_backend = backends.resolve_depth_backend(
        args.depth_backend, is_image, reporter)
    built = backends.build(
        passthrough=args.passthrough,
        depth_backend=args.depth_backend,
        onnx_model=args.onnx_model,
        ort_provider=args.ort_provider,
        device=args.device,
        depth_model=args.depth_model,
        fp16=args.fp16,
        vda_input_size=args.vda_input_size,
        smooth=args.smooth,
        smooth_eps=args.smooth_eps,
        chunk_size=args.chunk_size,
        reporter=reporter,
    )

    # Resolved here rather than in the parser so `--help` stays import-free.
    face_overlap = (pipeline.projection.FACE_OVERLAP
                    if args.face_overlap is None else args.face_overlap)
    angular_correction = (pipeline.projection.ANGULAR_CORRECTION
                          if args.face_angular_correction is None
                          else args.face_angular_correction)

    if pipeline.ffmpeg_io.is_image_path(args.input):
        _refuse_video_only_flags(args)
        return pipeline.convert_image(
            input_path=args.input,
            output_path=args.output,
            face_size=args.face_size,
            use_cubemap=not args.no_cubemap,
            depth_backend=built.backend,
            strength=args.strength,
            fg_erode=args.fg_erode,
            inpaint_mode=args.inpaint,
            depth_tiles=args.depth_tiles,
            left_share=_left_share(args),
            gradient_limit=args.gradient_limit,
            input_projection=args.input_projection,
            face_overlap=face_overlap,
            angular_correction=angular_correction,
            flatten_ground=args.flatten_ground,
            output_mode=args.output_mode,
            yaw=args.yaw,
            output_width=args.output_width,
            reporter=reporter,
        )

    if args.preview_frame is not None:
        if built.name == "video-depth-anything":
            # Worth saying rather than letting someone tune against a preview
            # that the render will not reproduce.
            reporter.warning(
                "video-depth-anything estimates depth from a chunk of "
                "consecutive frames. A single-frame preview has no temporal "
                "context, so it will differ slightly from the final render.",
                backend=built.name)
        return pipeline.preview_frame(
            input_path=args.input,
            output_path=args.output,
            frame_index=args.preview_frame,
            face_size=args.face_size,
            use_cubemap=not args.no_cubemap,
            depth_backend=built.backend,
            strength=args.strength,
            fg_erode=args.fg_erode,
            inpaint_mode=args.inpaint,
            depth_tiles=args.depth_tiles,
            left_share=_left_share(args),
            gradient_limit=args.gradient_limit,
            width=args.preview_width,
            input_projection=args.input_projection,
            face_overlap=face_overlap,
            angular_correction=angular_correction,
            flatten_ground=args.flatten_ground,
            output_mode=args.output_mode,
            yaw=args.yaw,
            reporter=reporter,
        )

    return pipeline.convert(
        input_path=args.input,
        output_path=args.output,
        face_size=args.face_size,
        crf=args.crf,
        preset=args.preset,
        codec=args.codec,
        bitdepth=args.bitdepth,
        max_frames=args.max_frames,
        use_cubemap=not args.no_cubemap,
        depth_backend=built.backend,
        strength=args.strength,
        chunk_size=built.chunk_size,
        chunk_overlap=args.chunk_overlap,
        fg_erode=args.fg_erode,
        start_frame=args.start_frame,
        inpaint_mode=args.inpaint,
        temporal_fill=args.temporal_fill,
        depth_tiles=args.depth_tiles,
        left_share=_left_share(args),
        gradient_limit=args.gradient_limit,
        spatial_audio=args.spatial_audio,
        ambisonic_codec=args.ambisonic_codec,
        source_subsampling=args.source_subsampling,
        input_projection=args.input_projection,
        temporal_depth=args.temporal_depth,
        face_overlap=face_overlap,
        angular_correction=angular_correction,
        flatten_ground=args.flatten_ground,
        output_mode=args.output_mode,
        yaw=args.yaw,
        output_width=args.output_width,
        reporter=reporter,
        cancel=cancel,
    )


def _probe_json(path: str) -> int:
    """Everything a GUI needs to describe the input, on stdout as JSON."""
    import json

    from .ffmpeg_io import probe
    from .spherical import declares_ambix

    info = probe(path)
    print(json.dumps({
        "width": info.width, "height": info.height, "fps": info.fps,
        "frame_count": info.frame_count, "duration": info.duration,
        "has_audio": info.has_audio, "pix_fmt": info.pix_fmt,
        "audio_channels": info.audio_channels,
        # From the file's own SA3D box, which ffprobe does not surface at all
        # -- it reports this camera's track as plain "4.0". VLC reads it, which
        # is why VLC says "Channels: Ambisonics" about the same file.
        "declares_ambix": declares_ambix(path),
        "chroma": info.chroma, "color_range": info.color.range,
        "color_space": info.color.space,
    }))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.probe_encoders:
        import json

        from . import encoders as _e

        try:
            w, _, h = args.probe_encoders.partition("x")
            size = (int(w), int(h))
        except ValueError:
            build_parser().error("--probe-encoders expects WxH, e.g. 7680x7680")
        infos = _e.probe(*size)
        print(json.dumps({"width": size[0], "height": size[1],
                          "recommended": _e.recommended(infos),
                          "encoders": [i.as_dict() for i in infos]}))
        return 0
    if args.probe_backends:
        import json

        from . import backends as _b

        print(json.dumps({"backends": [a.as_dict() for a in
                                       _b.probe_backends(args.onnx_model)]}))
        return 0
    # Past the machine-only probes, an input is required. Checked here rather
    # than by argparse, which cannot make a positional conditionally optional.
    if not args.input:
        build_parser().error("an input video is required")
    if args.probe_json:
        return _probe_json(args.input)
    if args.thumbnail:
        from .ffmpeg_io import write_thumbnail

        if not write_thumbnail(args.input, args.thumbnail,
                               frame_index=args.preview_frame or 0):
            print(f"could not read a frame from {args.input}", file=sys.stderr)
            return 1
        print(args.thumbnail)
        return 0
    if not args.output:
        build_parser().error("-o/--output is required")

    from .events import ConsoleReporter, JsonReporter

    reporter = (JsonReporter(_json_event_stream())
                if args.progress_json else ConsoleReporter())

    # Deferred so --help stays fast -- and, less obviously, so these finish
    # BEFORE the stdin watcher thread starts. On Windows a thread blocked
    # reading stdin deadlocks `import numpy`: measured on this machine, a run
    # with an open stdin pipe produced zero output in 20 s where the same run
    # with stdin closed finished in 0.3 s, and the freeze happens during the
    # import, before anything is written. Starting the thread afterwards
    # avoids it entirely. Do not move the watcher above this line.
    from . import backends, pipeline

    cancel = _stdin_cancel_watcher() if args.progress_json else None

    try:
        result = _run(args, reporter, cancel, backends, pipeline)
    except Exception as exc:
        # In text mode the traceback is the useful artefact and is left alone.
        # A parent process cannot do anything with a traceback on stderr, so
        # in JSON mode the failure becomes an event like everything else.
        if not args.progress_json:
            raise
        reporter.error(str(exc), kind=type(exc).__name__)
        return 1

    if pipeline.ffmpeg_io.is_image_path(args.input):
        # Both single-image paths return here. Falling through would reach the
        # video summary, which asks a PreviewResult for `cancelled` and frame
        # counts it does not have.
        reporter.info(f"Wrote {result.output_path} "
                      f"({result.width}x{result.height})",
                      output=result.output_path,
                      width=result.width, height=result.height)
        return 0
    if args.preview_frame is not None:
        reporter.info(f"Preview of frame {result.frame_index} -> "
                      f"{result.output_path} ({result.width}x{result.height})",
                      output=result.output_path, frame=result.frame_index,
                      width=result.width, height=result.height)
        return 0

    if result.cancelled:
        reporter.info(f"Cancelled after {result.frames_written} frames: "
                      f"{args.output}", output=args.output,
                      frames=result.frames_written)
        return 130
    reporter.info(f"Done: {args.output}", output=args.output,
                  frames=result.frames_written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
