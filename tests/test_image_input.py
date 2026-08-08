"""A single 360 photo in, a stereoscopic photo out.

Almost none of this is new machinery. ffmpeg reads a still as a one-frame
video and the preview path already writes an image, so depth, warp and stack
were converting photos before anyone asked them to. What is new is the front
door: recognising a still, defaulting to full resolution instead of a 2048-wide
preview, and refusing the flags that only mean something for a video.

The video paths must be untouched. Everything here that renders also checks
that the equivalent video invocation still behaves.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from stereo360 import ffmpeg_io, pipeline

ROOT = Path(__file__).resolve().parent.parent


def equirect(tmp_path: Path, name="src.jpg", w=512, h=256) -> str:
    """A small 2:1 still, which is what a 360 photo looks like."""
    out = str(tmp_path / name)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc2=size={w}x{h}", "-frames:v", "1", "-y", out],
        check=True, capture_output=True)
    return out


def run(*args, expect=0):
    proc = subprocess.run([sys.executable, "-m", "stereo360", *args],
                          capture_output=True, text=True, cwd=str(ROOT),
                          timeout=900)
    assert proc.returncode == expect, (
        f"exit {proc.returncode}, wanted {expect}\n{proc.stdout}\n{proc.stderr}")
    return proc


# ----------------------------------------------------------- what is a still

@pytest.mark.parametrize("path,expected", [
    ("a.jpg", True), ("a.JPG", True), ("a.jpeg", True), ("a.png", True),
    ("a.tif", True), ("a.webp", True),
    ("a.mp4", False), ("a.mov", False), ("a", False), ("a.jpeg.mp4", False),
])
def test_stills_are_recognised_by_extension(path, expected):
    """By extension, because probing cannot tell: ffmpeg reads a JPEG as a
    one-frame 25 fps video, which is indistinguishable from a short clip."""
    assert ffmpeg_io.is_image_path(path) is expected


def test_a_still_probes_as_a_video_which_is_why_extension_is_used(tmp_path):
    """The evidence for that decision, so it is not taken on trust."""
    info = ffmpeg_io.probe(equirect(tmp_path))
    assert info.fps > 0, "a still reports a frame rate"
    assert not info.has_audio


def test_everything_written_can_also_be_read():
    """A format the pipeline will write must be one it recognises as a still,
    or `-o out.png` from an image input would be refused by one and accepted
    by the other.

    Subset, not equality: the read set is deliberately the larger one. AVIF
    and HEIC go in and cannot come out, because `cv2.imencode` writes neither
    and the point of the feature is a JPEG anyway.
    """
    assert set(pipeline._PREVIEW_SUFFIXES) <= set(ffmpeg_io.IMAGE_SUFFIXES)
    assert pipeline._PREVIEW_SUFFIXES is ffmpeg_io.WRITABLE_IMAGE_SUFFIXES


def test_the_read_only_formats_are_the_phone_ones():
    """Named, so that widening the read set stays a decision rather than a
    drift. These are what a phone or a camera hands you."""
    read_only = set(ffmpeg_io.IMAGE_SUFFIXES) - set(
        ffmpeg_io.WRITABLE_IMAGE_SUFFIXES)
    assert read_only == {".avif", ".heic", ".heif", ".hif"}


def test_a_real_avif_is_accepted(tmp_path):
    """AVIF is not taken on trust: ffmpeg reads it here through the mp4
    demuxer, with no libheif involved, which is why it is in the set while
    HEIC's presence is a hope rather than a measurement."""
    src = tmp_path / "in.avif"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=512x256",
                    "-frames:v", "1", "-y", str(src)],
                   check=True, capture_output=True)
    assert ffmpeg_io.is_image_path(src)
    info = ffmpeg_io.probe(str(src))
    assert (info.width, info.height) == (512, 256)


def test_an_unreadable_still_says_so_in_words(tmp_path):
    """A build without libheif is the likely reason a .heic fails, and a
    CalledProcessError traceback does not say that to anyone."""
    bad = tmp_path / "broken.heic"
    bad.write_bytes(b"not a picture")
    with pytest.raises(ValueError) as excinfo:
        ffmpeg_io.probe(str(bad))
    assert "libheif" in str(excinfo.value)
    assert "broken.heic" in str(excinfo.value)


# ------------------------------------------------------------- the front door

def test_a_photo_converts_with_no_flags_at_all(tmp_path):
    """The whole point of step 1. This used to need
    `--preview-frame 0 --preview-width 0`, which describes the mechanism
    rather than the intent and silently ships a 2048-wide photo if the second
    half is forgotten."""
    src = equirect(tmp_path)
    dst = str(tmp_path / "out.jpg")
    proc = run(src, "-o", dst)

    assert Path(dst).exists()
    info = ffmpeg_io.probe(dst)
    assert (info.width, info.height) == (512, 512), "360 stacks top-bottom"
    assert "Wrote" in proc.stdout, "a photo is not a 'preview'"


def test_the_photo_comes_out_at_full_resolution(tmp_path):
    """Not capped at the 2048 a preview uses. A 7680x7680 stereo photo
    displays on a Quest 3 -- the 35.6 Mpx cap is the video decoder's, and a
    JPEG is a texture -- so there is nothing to gain by shrinking it."""
    src = equirect(tmp_path, w=4096, h=2048)
    dst = str(tmp_path / "big.jpg")
    run(src, "-o", dst)
    assert ffmpeg_io.probe(dst).width == 4096


def test_vr180_works_for_photos_too(tmp_path):
    src = equirect(tmp_path)
    dst = str(tmp_path / "half.jpg")
    run(src, "-o", dst, "--output-mode", "vr180")
    info = ffmpeg_io.probe(dst)
    assert (info.width, info.height) == (512, 256), "side by side, half sphere"


def test_the_exit_code_is_zero(tmp_path):
    """Caught a real bug: the image branch fell through to the video summary,
    which asks a PreviewResult for `cancelled`. The file was written and the
    process then died, so only the exit code told the truth."""
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "ok.jpg"))
    assert "Traceback" not in proc.stderr, proc.stderr


# ------------------------------------------------------------------ refusals

def test_an_image_will_not_write_a_video(tmp_path):
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "out.mp4"), expect=1)
    assert "must be an image this tool can write" in proc.stderr


# ------------------------------------------------------- tiles, by job kind

@pytest.mark.parametrize("requested,is_image,expected,why", [
    (None, True, 1, "a photo, nobody asked"),
    (None, False, 1, "a video, nobody asked"),
    (1, True, 1, "a photo, but 1 was asked for explicitly"),
    (4, True, 4, "a photo, 4 asked for"),
    (2, False, 2, "a video, 2 asked for"),
])
def test_the_tile_count_follows_the_kind_of_job(requested, is_image,
                                                expected, why):
    """A photo used to get three, which was right for V2 and measurably wrong
    for both models that replaced it -- Depth Pro's wall wobble more than
    triples under tiling. The resolver stays because an explicit count still
    has to be distinguishable from silence."""
    from stereo360 import cli

    assert cli.resolve_depth_tiles(requested, is_image) == expected, why


def test_asking_for_one_tile_on_a_photo_is_honoured():
    """The reason the flag defaults to None rather than to 1. With a default
    of 1 there is no way to tell "the user typed 1" from "nobody said
    anything", so someone asking for whole faces would silently get three."""
    from stereo360 import cli

    assert cli.build_parser().get_default("depth_tiles") is None
    assert cli.resolve_depth_tiles(1, True) == 1


def test_a_photo_actually_renders_with_the_photo_default(tmp_path):
    """End to end rather than by inspection, and in both directions: the
    default must *be* one tile, and tiling must still work when asked for.
    Checking only that a photo renders would pass with the tiling code
    deleted outright."""
    src = equirect(tmp_path, w=1024, h=512)
    default = tmp_path / "default.jpg"
    one, three = tmp_path / "one.jpg", tmp_path / "three.jpg"
    run(src, "-o", str(default), "--face-size", "128")
    run(src, "-o", str(one), "--face-size", "128", "--depth-tiles", "1")
    run(src, "-o", str(three), "--face-size", "128", "--depth-tiles", "3")
    assert default.read_bytes() == one.read_bytes(), \
        "the photo default is no longer one tile"
    assert default.read_bytes() != three.read_bytes(), \
        "--depth-tiles is being ignored"


def test_a_preview_of_a_video_is_tagged_too(tmp_path):
    """A stereo JPEG is a stereo JPEG, whatever asked for it.

    The tag used to be written only on the photo path, so previewing a frame
    of a *video* produced a file no headset would open -- identical pixels to
    one that opened fine, and nothing to say why. Checking settings before a
    long render is exactly when you want to look at the frame in a headset.
    """
    from stereo360 import gpano

    src = str(tmp_path / "clip.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f",
                    "lavfi", "-i", "testsrc2=size=256x128:d=1:r=4",
                    "-frames:v", "2", "-y", src], check=True,
                   capture_output=True)
    out = tmp_path / "preview.jpg"
    run(src, "-o", str(out), "--preview-frame", "0", "--face-size", "64")
    assert gpano.read_projection(str(out)), "no GPano in a video preview"


def test_a_png_preview_is_left_alone(tmp_path):
    """XMP goes in a JPEG APP1 segment. A PNG could carry it in an iTXt
    chunk, but that is not what players read, so writing one would be work
    toward a file nobody can view -- and PNG is the normal preview format."""
    src = str(tmp_path / "clip.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f",
                    "lavfi", "-i", "testsrc2=size=256x128:d=1:r=4",
                    "-frames:v", "2", "-y", src], check=True,
                   capture_output=True)
    out = tmp_path / "preview.png"
    proc = run(src, "-o", str(out), "--preview-frame", "0", "--face-size", "64")
    assert out.exists() and "Traceback" not in proc.stderr


def test_a_video_will_not_write_a_picture_and_says_so_before_rendering(
        tmp_path):
    """The counterpart, and the expensive one. ffmpeg accepts the job, renders
    every frame, and only then fails in `encoder.close()` with "exited with
    code 4294967274", leaving a truncated file. On an 8K render that is hours
    spent for a number nobody can read, so it is refused before the first
    frame -- which is also what makes this test fast enough to keep.
    """
    src = str(tmp_path / "clip.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f",
                    "lavfi", "-i", "testsrc2=size=256x128:d=1:r=4",
                    "-frames:v", "4", "-y", src], check=True,
                   capture_output=True)
    out = tmp_path / "stale.jpg"
    with pytest.raises(ValueError) as excinfo:
        pipeline.convert(input_path=src, output_path=str(out))
    assert "--preview-frame" in str(excinfo.value), \
        "should point at the flag that does write one frame of a video"
    assert not out.exists(), "refused before anything was written"


def test_an_image_will_not_write_a_format_opencv_cannot_encode(tmp_path):
    """`.heic` names a still, so the old check -- "is the output an image?" --
    waved it through and left the failure to cv2.imencode, which returns
    False and explains nothing."""
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "out.heic"), expect=1)
    assert "must be an image this tool can write" in proc.stderr
    assert ".jpg" in proc.stderr, "the message should name what does work"


@pytest.mark.parametrize("flag,value", [
    ("--max-frames", "10"),
    ("--start-frame", "5"),
    ("--preview-frame", "0"),
])
def test_video_only_flags_are_named_not_ignored(tmp_path, flag, value):
    """Silently ignoring a flag someone typed teaches them that the flags are
    decorative. Each refusal says why it does not apply."""
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "out.jpg"), flag, value, expect=2)
    assert flag in proc.stderr
    assert "the input is an image" in proc.stderr


def test_spatial_audio_is_refused_for_a_photo(tmp_path):
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "out.jpg"), "--spatial-audio",
               expect=2)
    assert "no audio track" in proc.stderr


def test_a_square_image_is_still_refused_as_a_half_equirect(tmp_path):
    """The input checks apply to photos exactly as they do to video -- a 1:1
    still is a 180 crop, and the tool wants the original sphere."""
    src = equirect(tmp_path, name="square.jpg", w=256, h=256)
    proc = run(src, "-o", str(tmp_path / "out.jpg"), expect=1)
    assert "180-degree footage" in proc.stderr


# ------------------------------------------------- the video paths still work

def test_video_conversion_is_unaffected(tmp_path):
    """The rule from vr180.md, kept here: nothing about adding photos may
    change what a video does."""
    from test_end_to_end import make_test_video

    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=4, with_audio=False)
    proc = run(src, "-o", dst, "--max-frames", "2", "--passthrough",
               "--face-size", "32")
    assert "Done:" in proc.stdout
    assert ffmpeg_io.probe(dst).height == 128


def test_a_video_preview_is_still_called_a_preview(tmp_path):
    """The two single-image paths report differently on purpose: one is the
    deliverable, the other is a look at what the deliverable will be."""
    from test_end_to_end import make_test_video

    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=128, h=64, frames=4, with_audio=False)
    proc = run(src, "-o", str(tmp_path / "p.png"), "--preview-frame", "1",
               "--preview-width", "64", "--face-size", "32", "--passthrough")
    assert "Preview of frame 1" in proc.stdout


def test_video_only_flags_are_fine_on_a_video(tmp_path):
    """The refusal must be scoped to image input, not applied everywhere."""
    from test_end_to_end import make_test_video

    src = str(tmp_path / "in.mp4")
    make_test_video(src, w=128, h=64, frames=6, with_audio=False)
    run(src, "-o", str(tmp_path / "out.mp4"), "--max-frames", "2",
        "--start-frame", "1", "--passthrough", "--face-size", "32")


# ---------------------------------------------------------------- the wrapper

def test_convert_image_defaults_to_full_size(tmp_path):
    """`width=0` means "do not downscale". Getting this wrong is invisible --
    the photo just quietly comes out at 2048."""
    import inspect

    src = equirect(tmp_path, w=1024, h=512)
    dst = str(tmp_path / "full.jpg")
    pipeline.convert_image(src, dst, depth_backend=None, face_size=128)
    assert ffmpeg_io.probe(dst).width == 1024
    assert "width" in inspect.getsource(pipeline.convert_image)


def test_convert_image_passes_settings_through(tmp_path):
    """It is a thin wrapper, so anything preview_frame understands must still
    arrive -- otherwise photos would silently ignore half the options."""
    src = equirect(tmp_path)
    dst = str(tmp_path / "yaw.jpg")
    pipeline.convert_image(src, dst, depth_backend=None, face_size=64,
                           output_mode="vr180", yaw=90.0)
    info = ffmpeg_io.probe(dst)
    assert (info.width, info.height) == (512, 256)


# ----------------------------------------------------------------- encoding

def test_jpeg_output_is_not_chroma_subsampled(tmp_path):
    """OpenCV defaults to 4:2:0, which is sensible for a web image and wrong
    for one that gets magnified across a headset's field of view. Measured on
    a real 7680x7680 frame, dropping it cut rms error 26% for 17% more bytes
    -- the cheapest win available here."""
    src = equirect(tmp_path)
    dst = str(tmp_path / "out.jpg")
    run(src, "-o", dst)
    assert ffmpeg_io.probe(dst).pix_fmt == "yuvj444p"


def test_the_jpeg_settings_are_the_measured_ones():
    """Spelled out so a later "tidy-up" cannot quietly drop one. Each is here
    for a reason recorded in image_encode_params."""
    import cv2

    params = pipeline.image_encode_params(".jpg")
    pairs = dict(zip(params[::2], params[1::2]))
    assert pairs[cv2.IMWRITE_JPEG_QUALITY] == 100
    assert (pairs[cv2.IMWRITE_JPEG_SAMPLING_FACTOR]
            == cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)
    assert pairs[cv2.IMWRITE_JPEG_OPTIMIZE] == 1


def test_the_jpeg_is_deliberately_not_progressive():
    """It would shave a little more, and a progressive 59-megapixel JPEG has
    to be decoded in several passes on a mobile GPU. Bytes are cheap there;
    decode time is not."""
    import cv2

    assert cv2.IMWRITE_JPEG_PROGRESSIVE not in pipeline.image_encode_params(".jpg")


@pytest.mark.parametrize("suffix", [".png", ".webp", ".tif", ".bmp"])
def test_lossless_formats_are_left_alone(suffix):
    """PNG and TIFF lose nothing already and WebP defaults to its own maximum,
    so there is nothing to improve and no setting to justify."""
    assert pipeline.image_encode_params(suffix) == []


def test_the_case_of_the_extension_does_not_matter(tmp_path):
    assert (pipeline.image_encode_params(".JPG")
            == pipeline.image_encode_params(".jpg"))


def test_quality_settings_apply_to_video_previews_too(tmp_path):
    """Deliberate. A preview exists so someone can judge --strength and
    --gradient-limit by eye, and it cannot do that while adding compression
    artifacts of its own that look like pipeline artifacts."""
    from test_end_to_end import make_test_video

    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "p.jpg")
    make_test_video(src, w=128, h=64, frames=4, with_audio=False)
    run(src, "-o", dst, "--preview-frame", "1", "--preview-width", "64",
        "--face-size", "32", "--passthrough")
    assert ffmpeg_io.probe(dst).pix_fmt == "yuvj444p"


def test_the_settings_actually_reduce_error(tmp_path):
    """The claim behind the choice, checked rather than cited: encoding the
    same picture with OpenCV's defaults is measurably further from it."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    # Smooth content with colour detail -- chroma subsampling shows up here,
    # which is the point. Pure noise would hide it.
    y, x = np.mgrid[0:256, 0:512]
    img = np.stack([(np.sin(x / 9.0) * 110 + 128),
                    (np.cos(y / 7.0) * 110 + 128),
                    ((x + y) % 255)], axis=-1).astype(np.uint8)

    def err(params):
        ok, buf = cv2.imencode(".jpg", img, params)
        assert ok
        back = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)
        return float(np.sqrt(((img.astype(np.float32) - back) ** 2).mean()))

    assert err(pipeline.image_encode_params(".jpg")) < err([]), \
        "the chosen settings should beat OpenCV's defaults"


# ------------------------------------------------------------- GPano metadata

def test_a_photo_is_tagged_as_a_sphere(tmp_path):
    """Without this a viewer sees a large flat JPEG. Measured on a Quest 3,
    GPano alone is enough to get a stacked frame read as stereo 360."""
    from stereo360 import gpano

    src = equirect(tmp_path)
    dst = str(tmp_path / "out.jpg")
    run(src, "-o", dst)

    tags = gpano.read_projection(dst)
    assert tags is not None, "no XMP was written"
    assert tags["ProjectionType"] == "equirectangular"
    assert tags["UsePanoramaViewer"] == "True"
    assert tags["StitchingSoftware"] == "stereo360"


def test_the_tag_describes_one_eye_not_the_stacked_frame(tmp_path):
    """GPano cannot describe a stacked pair, so it describes the panorama one
    eye covers. Device testing showed the dimensions are not read for layout
    at all -- two files disagreeing about them both worked -- so this is the
    description that happens to be true rather than the one that is required.
    """
    from stereo360 import gpano

    src = equirect(tmp_path)                       # 512x256 -> 512x512 output
    dst = str(tmp_path / "out.jpg")
    run(src, "-o", dst)

    tags = gpano.read_projection(dst)
    assert tags["CroppedAreaImageWidthPixels"] == "512"
    assert tags["CroppedAreaImageHeightPixels"] == "256", "one eye, not 512"
    assert tags["FullPanoWidthPixels"] == "512"
    assert tags["CroppedAreaLeftPixels"] == "0", "360 crops nothing"


def test_vr180_is_tagged_as_a_crop_from_the_middle(tmp_path):
    """A VR180 eye covers 180 degrees, which in GPano's terms is a crop from
    the middle of a sphere twice as wide. These are the exact numbers of the
    test file that displayed correctly on a Quest 3."""
    from stereo360 import gpano

    src = equirect(tmp_path)
    dst = str(tmp_path / "half.jpg")
    run(src, "-o", dst, "--output-mode", "vr180")

    tags = gpano.read_projection(dst)
    assert tags["CroppedAreaImageWidthPixels"] == "256", "half the frame"
    assert tags["FullPanoWidthPixels"] == "512", "of a sphere twice as wide"
    assert tags["CroppedAreaLeftPixels"] == "128", "from the middle"


@pytest.mark.parametrize("mode,size,expected", [
    ("360", (7680, 7680),
     {"crop_w": 7680, "crop_h": 3840, "full_w": 7680, "full_h": 3840,
      "left": 0, "top": 0}),
    ("vr180", (7680, 3840),
     {"crop_w": 3840, "crop_h": 3840, "full_w": 7680, "full_h": 3840,
      "left": 1920, "top": 0}),
])
def test_the_geometry_matches_what_was_tested_on_the_device(mode, size,
                                                            expected):
    """These exact numbers were in the files that worked, so they are pinned
    rather than left to be re-derived."""
    from stereo360 import gpano

    assert gpano.eye_geometry(*size, mode) == expected


def test_no_stereo_field_is_invented(tmp_path):
    """GPano defines none. Writing one anyway would be a plausible-looking
    property that no reader honours, and it would imply the file is described
    when the filename is doing that work."""
    from stereo360 import gpano

    src = equirect(tmp_path)
    dst = str(tmp_path / "out.jpg")
    run(src, "-o", dst)
    assert not any("stereo" in k.lower()
                   for k in gpano.read_projection(dst)), "invented a field"


def test_tagging_twice_leaves_one_packet(tmp_path):
    """Nothing writes a JPEG that already has XMP, so this is not a live bug.
    The equivalent guard on the MP4 side was quietly broken for months, which
    is reason enough to get it right where it costs ten lines."""
    from stereo360 import gpano

    src = equirect(tmp_path)
    dst = str(tmp_path / "twice.jpg")
    run(src, "-o", dst)

    gpano.inject_into_jpeg(dst, 512, 512, "360")
    once = Path(dst).read_bytes()
    gpano.inject_into_jpeg(dst, 512, 512, "360")
    assert Path(dst).read_bytes() == once, "a second pass changed the file"
    assert once.count(gpano.XMP_APP1_HEADER) == 1


def test_the_packet_is_found_by_walking_markers_not_searching(tmp_path):
    """Searching for the namespace string would also match those bytes inside
    compressed image data. Not hypothetical: the same shortcut on MP4 found an
    SA3D 187 MB into a file, inside mdat, with nonsense in every field."""
    from stereo360 import gpano

    src = equirect(tmp_path)
    dst = str(tmp_path / "planted.jpg")
    run(src, "-o", dst)

    data = Path(dst).read_bytes()
    # Bury a convincing decoy in the image data, past the real segment.
    planted = data + gpano.XMP_APP1_HEADER + b'GPano:ProjectionType="lies"'
    Path(dst).write_bytes(planted)
    assert gpano.read_projection(dst)["ProjectionType"] == "equirectangular"


def test_a_non_jpeg_says_it_carries_no_metadata(tmp_path):
    """PNG can hold XMP in an iTXt chunk, but that is not what players read,
    so writing it would be work in service of a file nobody can view."""
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "out.png"))
    assert "No 360 metadata" in proc.stdout or "No 360 metadata" in proc.stderr


def test_an_untagged_jpeg_reads_as_untagged(tmp_path):
    """The reader has to be able to say no, or the tests above prove nothing."""
    from stereo360 import gpano

    assert gpano.read_projection(equirect(tmp_path)) is None


def test_video_output_is_untouched_by_any_of_this(tmp_path):
    """GPano is for stills. An MP4 carries st3d/sv3d and must not grow an
    APP1 segment or lose what it already has."""
    from stereo360 import spherical
    from test_end_to_end import make_test_video

    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    make_test_video(src, w=128, h=64, frames=3, with_audio=False)
    run(src, "-o", dst, "--max-frames", "2", "--passthrough", "--face-size",
        "32")
    assert spherical.has_spherical_metadata(dst)
    assert b"GPano" not in Path(dst).read_bytes()


# ---------------------------------------------------------- filename tokens

@pytest.mark.parametrize("name,described", [
    ("garden_360_TB.jpg", True),
    ("garden_180x180_3dh.jpg", True),
    ("holiday_sbs.jpg", True),
    ("clip_OU.jpg", True),
    ("x_3DV.jpg", True),
    ("plain.jpg", False),
    # The two traps. A substring match would call both of these described.
    ("IMG_0180.jpg", False),
    ("artbook.jpg", False),
])
def test_a_filename_is_read_by_token_not_by_substring(name, described):
    """"tb" appears inside "artbook", and `IMG_0180.jpg` is a photo number
    rather than a projection. Splitting on separators avoids both; substring
    matching would claim each file already says what it is."""
    from stereo360 import vr_naming

    assert vr_naming.describes_stereo(name) is described


def test_only_the_stereo_token_counts_not_the_projection():
    """GPano carries the projection perfectly well, so the filename is needed
    for the thing GPano cannot say. Looking for `180` too is what would make
    `IMG_0180.jpg` look self-describing."""
    from stereo360 import vr_naming

    assert not vr_naming.describes_stereo("holiday_360.jpg")
    assert vr_naming.describes_stereo("holiday_TB.jpg")


@pytest.mark.parametrize("mode,expected", [
    ("360", "garden_360_TB.jpg"),
    ("vr180", "garden_180x180_3dh.jpg"),
])
def test_the_suggested_tokens_are_the_ones_that_were_tested(mode, expected):
    """Both spellings were measured displaying correctly on a Quest 3. Other
    documented spellings were not tested here, and picking an untried variant
    is how you end up debugging a filename."""
    from stereo360 import vr_naming

    assert vr_naming.suggest("garden.jpg", mode) == expected


def test_a_deliberate_name_is_never_argued_with():
    from stereo360 import vr_naming

    for name in ("holiday_sbs.jpg", "trip_360_TB.jpg"):
        assert vr_naming.suggest(name, "360") == name
        assert vr_naming.advice(name, "360") is None


def test_the_suggestion_keeps_the_directory_and_extension():
    from stereo360 import vr_naming

    got = vr_naming.suggest(str(Path("some") / "dir" / "my photo.jpeg"), "360")
    assert Path(got).parent == Path("some") / "dir"
    assert got.endswith("_360_TB.jpeg")


def test_a_plain_name_gets_advice_naming_the_players(tmp_path):
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "garden.jpg"))
    assert "garden_360_TB.jpg" in proc.stdout
    assert "SKYBOX" in proc.stdout, "should say who reads filenames"


def test_a_named_file_is_not_nagged(tmp_path):
    """Advice that fires when it is not needed stops being read."""
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "garden_360_TB.jpg"))
    assert "filename does not say" not in proc.stdout


def test_vr180_is_advised_its_own_tokens(tmp_path):
    src = equirect(tmp_path)
    proc = run(src, "-o", str(tmp_path / "half.jpg"), "--output-mode", "vr180")
    assert "half_180x180_3dh.jpg" in proc.stdout


def test_nothing_is_renamed_behind_your_back(tmp_path):
    """It suggests. The file goes exactly where it was asked to go."""
    src = equirect(tmp_path)
    dst = tmp_path / "garden.jpg"
    run(src, "-o", str(dst))
    assert dst.exists()
    assert not (tmp_path / "garden_360_TB.jpg").exists()
