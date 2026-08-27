"""The angular correction: pulling each face's edges back onto their true rays.

The artifact this exists for: flat ground that lifts toward the camera as it
approaches a cube seam, reaching 0.18 camera heights of false elevation at
about one camera height out -- reported as certain patches of floor near the
camera sitting higher than they should.

The cause is not the seam. Depth Anything V3 predicts its own camera and that
prediction saturates at roughly 65 degrees whatever it is shown, so at the
98-degree faces this pipeline uses it under-reads the field of view by a third
and places everything toward a face edge too near. See
`projection.ANGULAR_CORRECTION`.
"""

import numpy as np
import pytest

from stereo360 import cli, pipeline, projection

OV = projection.FACE_OVERLAP


def test_table_is_one_at_the_centre_and_grows_to_the_corner():
    """The divisor is the ray's own foreshortening, so it is pinned at 1 on
    the axis -- a correction that moved the face centre would be changing
    depth the model got right."""
    div = projection.angular_correction_table(64, OV, 1.0)
    centre = div[31:33, 31:33]
    assert np.allclose(centre, 1.0, atol=2e-3), centre

    lim = 1.0 + OV
    assert div.max() == pytest.approx(np.sqrt(1 + 2 * lim * lim), rel=2e-2)
    # Monotone outward along a row from the centre.
    row = div[32, 32:]
    assert np.all(np.diff(row) > 0)


def test_strength_scales_between_off_and_the_full_conversion():
    full = projection.angular_correction_table(48, OV, 1.0)
    half = projection.angular_correction_table(48, OV, 0.5)
    off = projection.angular_correction_table(48, OV, 0.0)
    assert np.allclose(off, 1.0)
    assert np.allclose(half, 1.0 + 0.5 * (full - 1.0), atol=1e-6)


def test_zero_strength_leaves_the_faces_untouched():
    """The flag is off by default, so this is the path every existing render
    takes and it has to be bit-identical."""
    faces = {f: np.float32(1.0) + projection._face_dirs(f, 32)[..., 1]
             for f in projection.FACES}
    before = {f: v.copy() for f, v in faces.items()}
    projection.apply_angular_correction(faces, OV, 0.0)
    for f in projection.FACES:
        assert np.array_equal(faces[f], before[f])


def test_it_flattens_a_ground_plane_seen_through_a_narrowed_lens():
    """The whole point, on synthetic data with the real failure built in.

    A face is filled with the depth a model would report if it believed the
    view were narrower than it is: the true ray at angle theta gets the value
    belonging to a shallower ray, which is exactly what under-reading the
    field of view does. The correction has to undo it.
    """
    face_size = 96
    lim = 1.0 + OV
    t = ((np.arange(face_size) + 0.5) / face_size * 2.0 - 1.0) * lim
    a, b = np.meshgrid(t, t)
    sec = np.sqrt(1.0 + a * a + b * b)

    # True inverse depth on the ground for the down face is 1/R = 1/sec, with
    # the camera one unit up. A model that thinks every ray is closer to the
    # axis than it is reports the axial distance instead of the radial one.
    truth = (1.0 / sec).astype(np.float32)
    wrong = np.ones_like(truth, dtype=np.float32)      # z-depth: constant

    faces = {f: wrong.copy() for f in projection.FACES}
    projection.apply_angular_correction(faces, OV, 1.0)
    got = faces["-Y"]
    assert np.abs(got - truth).max() < 1e-6

    # And the error it removes is real: uncorrected, the corner reads 91%
    # nearer than the truth.
    assert wrong.max() / truth.min() == pytest.approx(sec.max(), rel=1e-6)


def test_assemble_depth_applies_it_before_fitting_the_faces_together():
    """Order matters: the correction moves the values in the shared band, so
    fitting first would fit them on numbers about to change."""
    import inspect

    src = inspect.getsource(pipeline.assemble_depth)
    assert src.index("apply_angular_correction") < src.index(
        "align_overlapping_faces")


def test_it_reaches_the_pipeline_from_every_entry_point():
    """A flag threaded through six signatures is a flag that can be dropped in
    one of them and silently do nothing."""
    import inspect

    for fn in (pipeline.assemble_depth, pipeline.depth_map_for_frame,
               pipeline.depth_maps_for_chunk, pipeline.right_eye_from_depth,
               pipeline.convert, pipeline.preview_frame,
               pipeline._convert_chunked):
        sig = inspect.signature(fn)
        assert "angular_correction" in sig.parameters, fn.__name__

    for fn in (pipeline.depth_map_for_frame, pipeline.depth_maps_for_chunk,
               pipeline.right_eye_from_depth, pipeline.convert,
               pipeline.preview_frame, pipeline._convert_chunked):
        default = inspect.signature(fn).parameters[
            "angular_correction"].default
        assert default == projection.ANGULAR_CORRECTION, fn.__name__


def test_the_cli_defers_its_default_without_letting_it_drift():
    """Same arrangement as --face-overlap, and for the same reason: the
    parser cannot import projection without dragging numpy into --help."""
    import inspect

    parser = cli.build_parser()
    args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
    assert args.face_angular_correction is None
    assert parser.parse_args(
        ["in.mp4", "-o", "o.mp4",
         "--face-angular-correction", "0.55"]).face_angular_correction == 0.55

    resolve = inspect.getsource(cli._render)
    assert "projection.ANGULAR_CORRECTION" in resolve
    assert "args.face_angular_correction is None" in resolve
    # Every entry point has to receive it, not just the video one.
    assert resolve.count("angular_correction=angular_correction") == 3


def test_it_is_off_by_default():
    """It measured well on two scenes and has never been judged in a headset,
    which is not enough to change what every existing render produces."""
    assert projection.ANGULAR_CORRECTION == 0.0
