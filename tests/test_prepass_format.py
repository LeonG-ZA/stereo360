"""What the shader pre-pass converts to before libplacebo sees the frame.

`chain` used to open with `format=yuv420p` for every source. That is the
source's own layout for all camera video and costs nothing there, but it was
a silent downgrade for anything better -- and the working file it feeds is
built to carry better: `intermediate.choose` asks for yuv444p and yuv420p10le
when the source has them, so the chroma and the bits were being thrown away
one filter before the container designed to keep them.

Measured on the outdoor pair, 4K to 8K against the 8K truth:

    forced 4:2:0, no shader    37.28 dB RGB    47.25 dB chroma
    the source's 4:4:4         37.51           48.76
    forced 4:2:0, C4F32        37.58           47.25
    the source's 4:4:4         37.85           48.71

and a 10-bit ramp carrying 878 distinct luma codes came out of the forced
graph with 440.

Something still has to be named, which is why this is a substitution rather
than a deletion. Every shader in `models/` hooks LUMA; handed RGB there is no
luma plane, the pass completes, and the result is a plain resample with
nothing to say the model never ran.
"""

import pytest

from stereo360 import fsrcnnx


class TestItKeepsWhatTheSourceHas:
    @pytest.mark.parametrize("pix_fmt,want", [
        ("yuv420p", "yuv420p"),
        ("yuvj420p", "yuv420p"),          # the range tag is not a layout
        ("yuv422p", "yuv422p"),
        ("yuv444p", "yuv444p"),
        ("yuvj444p", "yuv444p"),
        ("yuv420p10le", "yuv420p10le"),
        ("yuv422p10le", "yuv422p10le"),
        ("yuv444p10le", "yuv444p10le"),
    ])
    def test_a_planar_yuv_source_is_left_as_it_is(self, pix_fmt, want):
        assert fsrcnnx.filter_format(pix_fmt) == want

    def test_the_video_case_is_unchanged(self):
        """Every source measured this session is 4:2:0 -- two AV1 8K tours
        and the X5's HEVC -- so the common path must move not at all."""
        assert "format=yuv420p," in fsrcnnx.chain(3840, 1920,
                                                  pix_fmt="yuv420p")


class TestItFallsBackRatherThanGuessing:
    @pytest.mark.parametrize("pix_fmt", [
        "rgb24", "gbrp", "bgr0",          # no luma plane to hook
        "p010le", "nv12",                 # hardware surface layouts
        "yuv444p12le",                    # nothing here can carry 12-bit
        None,                             # an unprobed source
    ])
    def test_anything_unnameable_becomes_yuv420p(self, pix_fmt):
        assert fsrcnnx.filter_format(pix_fmt) == "yuv420p"

    def test_rgb_is_converted_rather_than_passed_through(self):
        """The load-bearing one. A LUMA hook on an RGB frame does not fail,
        it does nothing, and the output looks like a bad upscaler rather
        than like a missing one."""
        assert fsrcnnx.filter_format("rgb24") != "rgb24"


class TestItReachesTheFiltergraph:
    def test_chain_names_the_sources_format(self):
        got = fsrcnnx.chain(3840, 1920, 2.0, pix_fmt="yuv444p")
        assert got.startswith("format=yuv444p,libplacebo=")

    def test_ten_bit_survives_the_graph(self):
        assert fsrcnnx.chain(3840, 1920, pix_fmt="yuv420p10le").startswith(
            "format=yuv420p10le,")

    def test_the_default_is_still_the_safe_one(self):
        """Callers that predate this -- and there are several in the tests --
        must keep getting what they always got."""
        assert fsrcnnx.chain(3840, 1920).startswith("format=yuv420p,")


class TestAStillIsWrittenAsAStill:
    """The working file used to be a lossless HEVC bitstream in a file named
    `.png`. Nothing downstream complained, because ffmpeg reads by content --
    but `cli` re-derives photo-or-video from the working file's *extension*
    after a pre-pass, so the suffix decides which path the render takes and a
    video wearing it was one rename from routing a still down the video path.

    Writing it as a real image is also the better picture. Same crop, same
    shader, measured against the 8K truth:

        hevc in a .png, forced 4:2:0    36.46 dB RGB   46.24 dB chroma
        hevc in a .png, source 4:4:4    37.29          47.24
        a real .png, rgb24              37.78          48.31

    RGB carries full chroma and is what the depth pass reads anyway, so the
    YUV round-trip was pure loss.
    """

    IMAGES = ("out.png", "OUT.PNG", "photo.jpg", "photo.jpeg", "shot.webp")
    VIDEOS = ("work.mkv", "work.mp4", "work.mov")

    @pytest.mark.parametrize("where", IMAGES)
    def test_no_video_encoder_is_named_for_an_image(self, where):
        from stereo360 import intermediate

        args, _ = intermediate.choose(width=2048, height=1024, frames=1,
                                      where=where)
        assert "-c:v" not in args, (
            f"{where} would have been written as a video stream")

    @pytest.mark.parametrize("where", IMAGES)
    def test_it_says_one_file_rather_than_a_sequence(self, where):
        from stereo360 import intermediate

        args, _ = intermediate.choose(width=2048, height=1024, frames=1,
                                      where=where)
        assert "-update" in args, (
            "without it ffmpeg warns about a missing %03d pattern")
        assert args[args.index("-frames:v") + 1] == "1"

    @pytest.mark.parametrize("where", IMAGES)
    def test_no_pixel_format_is_forced_on_it(self, where):
        """The png encoder negotiates one from the frames it gets, which is
        how a 16-bit still would keep its depth if the path ever carried it."""
        from stereo360 import intermediate

        args, _ = intermediate.choose(pix_fmt="yuvj444p", width=2048,
                                      height=1024, frames=1, where=where)
        assert "-pix_fmt" not in args

    @pytest.mark.parametrize("where", VIDEOS)
    def test_a_video_still_gets_an_encoder(self, where):
        from stereo360 import intermediate

        args, _ = intermediate.choose(pix_fmt="yuv420p", width=3840,
                                      height=1920, frames=300, where=where)
        assert "-c:v" in args and "-pix_fmt" in args

    def test_both_runners_go_through_the_same_chooser(self):
        """fsrcnnx and nvvsr each write a still on the photo path, and the
        fix belongs in one place rather than in each of them."""
        import inspect

        from stereo360 import fsrcnnx, nvvsr

        for mod in (fsrcnnx, nvvsr):
            assert "intermediate.choose" in inspect.getsource(mod.run), (
                f"{mod.__name__}.run picks its own output arguments")
