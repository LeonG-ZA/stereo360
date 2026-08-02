"""VR180 output: the middle 180 degrees, eyes side by side.

Only the *output* changes. Input is still a full 360 equirect, and the depth
and warp stages are untouched — which is the whole reason this is a small
change rather than the large one the plan originally described. See
plans/vr180.md.

The property that ties it together: the encoder's frame size, the pixels
actually written, and the spherical metadata all derive from the same mode. If
any two of those disagree the file is silently wrong, and a headset is the only
place you would find out.
"""

import shutil
import struct
import subprocess

import numpy as np
import pytest

from stereo360 import pipeline, spherical


def eyes(w=64, h=32):
    """Two distinguishable eyes, with a column ramp so a crop is visible."""
    left = np.zeros((h, w, 3), np.uint8)
    right = np.zeros((h, w, 3), np.uint8)
    left[..., 0] = np.arange(w, dtype=np.uint8)[None, :]
    right[..., 1] = np.arange(w, dtype=np.uint8)[None, :]
    return left, right


# ------------------------------------------------------------------ geometry

def test_360_geometry_is_unchanged():
    assert pipeline.output_geometry(7680, 3840, "360") == (7680, 7680)


def test_vr180_keeps_the_pixel_count_and_spends_it_on_half_the_sphere():
    """An 8K source gives 7680x3840 either way. Same pixels, half the sphere,
    so twice the angular resolution — and, incidentally, inside the 35.65 Mpx
    decode cap that H.264 and HEVC share at their highest level, where the 360
    frame is not. That is about direct playback only: 7680x7680 is still the
    right master for YouTube, which transcodes."""
    w, h = pipeline.output_geometry(7680, 3840, "vr180")
    assert (w, h) == (7680, 3840)
    assert w * h <= 35_651_584
    assert pipeline.output_geometry(7680, 3840, "360")[0] * 7680 > 35_651_584


@pytest.mark.parametrize("w", [7680, 5760, 4096, 3840, 1920, 512])
def test_output_dimensions_are_always_even(w):
    """Odd dimensions are rejected by every yuv420p encoder."""
    for mode in pipeline.OUTPUT_MODES:
        ow, oh = pipeline.output_geometry(w, w // 2, mode)
        assert ow % 2 == 0 and oh % 2 == 0, (mode, w, ow, oh)


def test_an_unknown_mode_is_refused():
    for fn in (lambda m: pipeline.output_geometry(64, 32, m),
               lambda m: pipeline.stack_eyes(*eyes(), m)):
        with pytest.raises(ValueError, match="output_mode"):
            fn("180")          # plausible-looking, and not a mode


# ------------------------------------------------------------------ stacking

def test_360_stacks_top_over_bottom_with_the_left_eye_first():
    left, right = eyes()
    out = pipeline.stack_eyes(left, right, "360")
    assert out.shape == (64, 64, 3)
    np.testing.assert_array_equal(out[:32], left)
    np.testing.assert_array_equal(out[32:], right)


def test_vr180_stacks_left_beside_right():
    left, right = eyes()
    out = pipeline.stack_eyes(left, right, "vr180")
    assert out.shape == (32, 64, 3)
    # Each half is the middle 32 columns of its eye.
    np.testing.assert_array_equal(out[:, :32], left[:, 16:48])
    np.testing.assert_array_equal(out[:, 32:], right[:, 16:48])


def test_vr180_keeps_the_middle_of_the_sphere_not_an_end():
    """Cropping from column 0 would silently deliver the wrong 180 degrees."""
    left, right = eyes(w=64)
    out = pipeline.stack_eyes(left, right, "vr180")
    # The red ramp encodes the source column, so the first kept column is
    # readable straight off the output.
    assert out[0, 0, 0] == 16, "did not start at the middle-half boundary"
    assert out[0, 31, 0] == 47, "did not end at the middle-half boundary"


def test_the_crop_resamples_nothing():
    """It is a column range, which is why the direction will be free to move
    later without any loss of quality."""
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, (16, 64, 3), dtype=np.uint8)
    out = pipeline.stack_eyes(left, left, "vr180")
    np.testing.assert_array_equal(out[:, :32], left[:, 16:48])


def test_the_default_is_360():
    left, right = eyes()
    assert pipeline.DEFAULT_OUTPUT_MODE == "360"
    np.testing.assert_array_equal(pipeline.stack_eyes(left, right),
                                  pipeline.stack_eyes(left, right, "360"))


# ------------------------------------------------- the three must agree

def test_stack_size_matches_what_the_encoder_was_told():
    """The failure this prevents: an encoder configured for one frame size
    being fed another. ffmpeg would either error or silently misinterpret the
    raw stream, and neither is discovered until playback."""
    w, h = 64, 32
    left, right = eyes(w, h)
    for mode in pipeline.OUTPUT_MODES:
        out = pipeline.stack_eyes(left, right, mode)
        told = pipeline.output_geometry(w, h, mode)
        assert (out.shape[1], out.shape[0]) == told, mode


def test_the_metadata_matches_the_layout(tmp_path):
    """The pixels say side-by-side 180; the boxes must say the same."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    out = tmp_path / "m.mp4"
    if subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f",
                       "lavfi", "-i", "testsrc=size=320x160:rate=10:duration=1",
                       "-an", "-c:v", "libx264", "-movflags", "-faststart",
                       "-y", str(out)], capture_output=True).returncode:
        pytest.skip("ffmpeg could not build the fixture")

    spherical.inject_spherical_metadata(
        str(out), stereo_mode="left-right", horizontal_fov=pipeline.VR180_FOV)
    data = out.read_bytes()
    assert spherical._st3d("left-right") in data
    i = data.find(b"equi")
    _, _, left_b, right_b = struct.unpack(">IIII", data[i + 8:i + 24])
    assert left_b / 2 ** 32 == pytest.approx(0.25)
    assert right_b / 2 ** 32 == pytest.approx(0.25)


def test_convert_and_preview_both_take_the_mode():
    """A preview that ignored the mode would show a layout the render will not
    produce, which is worse than showing nothing."""
    import inspect

    for fn in (pipeline.convert, pipeline.preview_frame):
        p = inspect.signature(fn).parameters["output_mode"]
        assert p.default == pipeline.DEFAULT_OUTPUT_MODE, fn.__name__
    assert "stack_eyes" in inspect.getsource(pipeline.preview_frame)
    assert "stack_eyes" in inspect.getsource(pipeline._Sink.write)


def test_the_cli_exposes_both_modes():
    from stereo360 import cli

    parser = cli.build_parser()
    action = next(a for a in parser._actions if a.dest == "output_mode")
    assert set(action.choices) == set(pipeline.OUTPUT_MODES)
    assert action.default == pipeline.DEFAULT_OUTPUT_MODE
    src = inspect_source(cli._run)
    assert src.count("output_mode=args.output_mode") == 2, \
        "both convert and preview_frame must receive it"


def inspect_source(fn):
    import inspect

    return inspect.getsource(fn)


# ------------------------------------------------------------------ yaw

def test_yaw_zero_is_the_middle_of_the_sphere():
    assert pipeline.vr180_crop(64, 0.0) == (16, 32)


@pytest.mark.parametrize("yaw,expected_start", [
    (0.0, 16), (90.0, 32), (180.0, 48), (-90.0, 0), (-180.0, 48),
])
def test_yaw_moves_the_crop_by_the_right_amount(yaw, expected_start):
    """A quarter turn is a quarter of the width, and the field stays 180."""
    x0, half = pipeline.vr180_crop(64, yaw)
    assert (x0, half) == (expected_start, 32)


@pytest.mark.parametrize("a,b", [(-200.0, 160.0), (400.0, 40.0),
                                 (-360.0, 0.0), (540.0, 180.0)])
def test_yaw_wraps_so_any_value_is_legal(a, b):
    assert pipeline.vr180_crop(64, a) == pipeline.vr180_crop(64, b)


def test_a_yaw_across_the_seam_wraps_the_columns():
    """At yaw 180 the field straddles the +/-180 boundary, so the crop has to
    come from both ends of the frame rather than running off the end."""
    left, right = eyes(w=64)
    out = pipeline.stack_eyes(left, right, "vr180", yaw=180.0)
    assert out.shape == (32, 64, 3)
    # Columns 48..63 then 0..15 of the source, in that order.
    want = np.concatenate([left[:, 48:], left[:, :16]], axis=1)
    np.testing.assert_array_equal(out[:, :32], want)


def test_yaw_resamples_nothing():
    """Every output column is a source column, byte for byte. That is what
    makes the direction free to drag in a UI."""
    rng = np.random.default_rng(1)
    left = rng.integers(0, 255, (8, 64, 3), dtype=np.uint8)
    for yaw in (0.0, 37.0, 90.0, 175.0, -125.0):
        out = pipeline.stack_eyes(left, left, "vr180", yaw)
        for col in out[:, :32].transpose(1, 0, 2):
            assert any(np.array_equal(col, src)
                       for src in left.transpose(1, 0, 2)), yaw


def test_yaw_does_not_change_the_frame_size():
    left, right = eyes(w=64)
    sizes = {pipeline.stack_eyes(left, right, "vr180", y).shape
             for y in (0.0, 45.0, 180.0, -73.0)}
    assert len(sizes) == 1


def test_a_yaw_on_360_output_is_refused_rather_than_ignored():
    """Silently ignoring it would mean an hour of rendering pointing the wrong
    way. 360 keeps the whole sphere, so there is no direction to choose."""
    with pytest.raises(ValueError, match="vr180 output only"):
        pipeline.check_yaw("360", 45.0)
    pipeline.check_yaw("360", 0.0)          # the default must stay legal
    pipeline.check_yaw("vr180", 45.0)


def test_both_entry_points_check_the_yaw():
    import inspect

    for fn in (pipeline.convert, pipeline.preview_frame):
        assert inspect.signature(fn).parameters["yaw"].default == 0.0
        assert "check_yaw" in inspect.getsource(fn), fn.__name__


def test_the_cli_exposes_yaw_and_passes_it_to_both():
    from stereo360 import cli

    parser = cli.build_parser()
    assert parser.parse_args(["i.mp4", "-o", "o.mp4"]).yaw == 0.0
    assert parser.parse_args(["i.mp4", "-o", "o.mp4",
                              "--yaw", "-37.5"]).yaw == -37.5
    assert inspect_source(cli._run).count("yaw=args.yaw") == 2
