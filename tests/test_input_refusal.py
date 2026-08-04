"""Input this tool cannot use, refused with a reason rather than mangled.

The tool converts a monoscopic 360 sphere. Two plausible-looking inputs are not
that: a file that is already stereo, and a file that is already 180. Neither
would crash — both would run to completion and produce something wrong, which
is the failure mode worth spending code on.

The 180 case matters more than it sounds, because the answer is not "we don't
support that yet". Cropping a full sphere to 180 is *better* than processing a
180 source: whole cubemap faces instead of half-empty ones, and content beyond
the field edge for the warp to draw the second eye from. So the message asks
for the original rather than apologising. See plans/vr180.md.
"""

import pytest

from stereo360 import ffmpeg_io, pipeline
from stereo360.events import Reporter


class Recorder(Reporter):
    def __init__(self):
        super().__init__()
        self.warnings = []

    def warning(self, message, **fields):
        self.warnings.append(message)

    def info(self, message, **fields):
        pass


def info(width=7680, height=3840, projection="equirectangular",
         bound_left=0, bound_right=0, stereo_layout="2D"):
    return ffmpeg_io.VideoInfo(
        width=width, height=height, fps=30.0, frame_count=1, duration=1.0,
        has_audio=False, projection=projection, bound_left=bound_left,
        bound_right=bound_right, stereo_layout=stereo_layout)


def check(i, requested="auto"):
    rec = Recorder()
    pipeline.check_input_is_monoscopic_360(i, requested, rec)
    return rec


# ------------------------------------------------------- field of view maths

def test_a_full_sphere_reports_360():
    assert info().horizontal_fov == pytest.approx(360.0)


def test_bounds_give_the_field_of_view():
    """The bounds are pixels cropped from a notional full sphere, so the frame
    plus both bounds is the sphere."""
    assert info(width=7680, projection="tiled equirectangular",
                bound_left=3840, bound_right=3840
                ).horizontal_fov == pytest.approx(180.0)
    assert info(width=7680, projection="tiled equirectangular",
                bound_left=1280, bound_right=1280
                ).horizontal_fov == pytest.approx(270.0)


def test_an_undeclared_projection_has_no_field_of_view():
    """None means "did not say", not "not spherical" — untagged is the common
    case and has to fall through to the aspect-ratio hint instead."""
    assert info(projection=None).horizontal_fov is None


def test_only_a_positive_declaration_counts_as_stereo():
    assert not info(stereo_layout="2D").is_stereo
    assert not info(stereo_layout=None).is_stereo
    assert info(stereo_layout="side by side").is_stereo
    assert info(stereo_layout="top and bottom").is_stereo


# ------------------------------------------------------------------ accepted

def test_a_plain_360_equirect_is_accepted():
    assert check(info()).warnings == []


def test_an_untagged_2_to_1_file_is_accepted():
    """Untagged is the common case and is overwhelmingly a full equirect."""
    assert check(info(projection=None)).warnings == []


def test_a_cubemap_is_left_to_resolve_projection():
    """This check is about field of view and stereo, not projection kind."""
    assert check(info(width=5760, height=3840, projection="cubemap")).warnings == []


# ------------------------------------------------------------------ refused

def test_stereo_input_is_refused_as_already_3d():
    with pytest.raises(ValueError, match="already 3D"):
        check(info(stereo_layout="side by side"))


def test_stereo_is_checked_before_the_field_of_view():
    """A side-by-side 180 is 2:1, the same shape as a full equirect, so the
    aspect hint cannot see it. Whichever message comes out, it must not be the
    one that tells them to bring a 360 file they already have."""
    with pytest.raises(ValueError, match="already 3D"):
        check(info(width=7680, height=3840, projection="tiled equirectangular",
                   bound_left=3840, bound_right=3840,
                   stereo_layout="side by side"))


def test_a_declared_180_field_is_refused():
    with pytest.raises(ValueError, match="180-degree footage"):
        check(info(width=3840, height=3840, projection="tiled equirectangular",
                   bound_left=1920, bound_right=1920))


def test_an_untagged_square_file_is_refused_on_its_shape():
    with pytest.raises(ValueError, match="1:1 shape"):
        check(info(width=3840, height=3840, projection=None))


def test_the_refusal_says_what_to_do_instead():
    """A refusal that does not name the fix is just an obstacle."""
    with pytest.raises(ValueError) as e:
        check(info(width=3840, height=3840, projection=None))
    msg = str(e.value)
    assert "--output-mode vr180" in msg, "does not mention the feature wanted"
    assert "--input-projection" in msg, "does not mention the override"
    assert "better" in msg, "does not explain why 360 input is preferred"


@pytest.mark.parametrize("fov", [179.0, 90.0, 200.0, 358.0])
def test_anything_short_of_a_full_sphere_is_refused(fov):
    side = int((1.0 - fov / 360.0) / 2.0 * 7680 / (fov / 360.0))
    with pytest.raises(ValueError, match="180-degree footage"):
        check(info(width=7680, projection="tiled equirectangular",
                   bound_left=side, bound_right=side))


def test_rounding_in_the_bounds_does_not_trip_the_check():
    """The bounds are integers, so a genuine full sphere can measure a hair
    under 360. Refusing that would reject perfectly good input."""
    assert check(info(width=7680, projection="equirectangular",
                      bound_left=1, bound_right=1)).warnings == []


# ------------------------------------------------------------------ override

def test_an_explicit_projection_overrides_the_refusal_but_warns():
    """How a mistagged file gets through. The override wins, and says so."""
    rec = check(info(width=3840, height=3840, projection=None),
                requested="equirectangular")
    assert len(rec.warnings) == 1
    assert "180-degree footage" in rec.warnings[0]
    assert "equirectangular" in rec.warnings[0]


def test_the_override_does_not_rescue_stereo_input():
    """No projection setting turns two views into one."""
    with pytest.raises(ValueError, match="already 3D"):
        check(info(stereo_layout="side by side"),
              requested="equirectangular")


# ------------------------------------------------------------------ wiring

def test_both_entry_points_check_the_input():
    import inspect

    for fn in (pipeline.convert, pipeline.preview_frame):
        assert "check_input_is_monoscopic_360" in inspect.getsource(fn), \
            fn.__name__
