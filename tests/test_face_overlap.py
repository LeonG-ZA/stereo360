"""Overlapping depth faces: the fix for the ground creasing at a cube seam.

The artifact these guard against, as reported from a Quest 3: a patch of
ground with a tree on it appearing to float in front of the rest of the
ground. Measured at 8K, the depth step across a cube seam was 259x the step
between ordinary neighbouring pixels; with overlapping faces it is 2.3x.
"""

import numpy as np
import pytest

from stereo360 import pipeline, projection
from stereo360.depth.base import DepthBackend

OV = projection.FACE_OVERLAP


EYE_HEIGHT = 1.7


def ground_plane_depth(dirs: np.ndarray) -> np.ndarray:
    """Inverse depth for a viewer standing on an infinite ground plane.

    Exactly the geometry that broke. Distance along a ray pointing `y` below
    the horizon is EYE_HEIGHT / -y, so inverse depth is just -y / EYE_HEIGHT,
    falling to zero at the horizon and above.

    What makes it a faithful reproduction is that the faces see very different
    slices of it: the down face only ever sees 0.34 to 0.59 (a few metres of
    ground) while the front face sees 0 to 0.42 (the same ground plus
    everything out to the horizon). Two faces that normalise those two ranges
    to the same output are no longer related by any affine map, which is the
    whole reason an edge-strip fit cannot rescue them.
    """
    return (np.clip(-dirs[..., 1], 0.0, None) / EYE_HEIGHT).astype(np.float32)


class PerFaceNormalising(DepthBackend):
    """A depth model that rescales each image to its own visible range.

    This is what real relative-depth backends do, and it is the whole source
    of the problem: the down face's range is spent on the few metres around
    your feet while the front face's is spent on everything out to the fog,
    so the same ground arrives at the seam with two different scales. A fixed
    affine per face would be reconcilable by construction and would prove
    nothing, so the mapping is deliberately non-affine.
    """

    def __init__(self, overlap: float) -> None:
        self.overlap = overlap

    def estimate(self, frame_rgb):
        return self.estimate_chunk([frame_rgb])[0]

    def estimate_chunk(self, frames_rgb):
        out = []
        for img in frames_rgb:
            # The face identity is carried in the image itself (see
            # _render_faces) so the fake model never has to be told which
            # face it is looking at.
            d = img[..., 0].astype(np.float32) / 255.0
            lo, hi = float(d.min()), float(d.max())
            n = (d - lo) / max(hi - lo, 1e-6)
            out.append((n ** 1.7).astype(np.float32) + 0.05)
        return out


def _render_faces(face_size: int, overlap: float) -> dict:
    """Ground-plane depth painted into the red channel of six face images."""
    faces = {}
    for face in projection.FACES:
        dirs = (projection._overlap_face_dirs(face, face_size, overlap)
                if overlap > 0 else projection._face_dirs(face, face_size))
        d = ground_plane_depth(dirs)
        img = np.zeros(d.shape + (3,), np.uint8)
        img[..., 0] = np.clip(d * 255, 0, 255).astype(np.uint8)
        faces[face] = img
    return faces


def _seam_step(depth: np.ndarray) -> float:
    """Median depth step across the front/down seam, over the ground's range.

    That one seam rather than all of them, because it is the one the artifact
    was reported on and the only one this scene puts content on: three
    quarters of the total seam length here runs through empty sky, where a
    median is a measure of nothing. The range is taken over the lower
    hemisphere for the same reason.
    """
    h, w = depth.shape
    dirs = projection.equirect_rows_to_dir(0, h, w, h)
    fi, _, _ = projection._face_local_coords(dirs)
    front, down = (projection.FACES.index(f) for f in ("+Z", "-Y"))
    ground = depth[h // 2:]
    span = float(np.percentile(ground, 99) - np.percentile(ground, 1))

    step = np.abs(np.diff(depth, axis=0))
    across = (((fi[:-1] == front) & (fi[1:] == down))
              | ((fi[:-1] == down) & (fi[1:] == front)))
    assert across.any(), "no front/down seam found to measure"
    return float(np.median(step[across]) / span)


def _depth_for(overlap: float, face_size: int = 256, w: int = 1024) -> np.ndarray:
    backend = PerFaceNormalising(overlap)
    faces = _render_faces(face_size, overlap)
    disp = dict(zip(projection.FACES,
                    backend.estimate_chunk([faces[f]
                                            for f in projection.FACES])))
    return pipeline.assemble_depth(disp, w, w // 2, overlap)


def test_exact_faces_crease_the_ground_at_a_seam():
    """The bug, reproduced. This one has to keep failing-if-fixed-wrongly: if
    the scene stops producing a step, the two tests below prove nothing.

    2.2% of the ground's depth range here, against 2.3% measured on the real
    8K clip the artifact was reported from -- close enough to trust that what
    the other tests improve is the same thing.
    """
    assert _seam_step(_depth_for(0.0)) > 0.015


def test_overlapping_faces_remove_the_crease():
    assert _seam_step(_depth_for(OV)) < 0.008


def test_overlap_beats_exact_faces_severalfold():
    """Stated as a ratio so the test says what actually improved, and cannot
    quietly pass if both paths degrade together."""
    exact = _seam_step(_depth_for(0.0))
    over = _seam_step(_depth_for(OV))
    assert over * 3 < exact, f"exact {exact:.5f} vs overlapped {over:.5f}"


def test_the_default_overlap_is_near_the_best_available_on_the_seam():
    """Guards the lower bound on the constant: it has to be wide enough that
    the seam step is within reach of the best any width achieves.

    Only the lower bound. What stops the default going wider is that the
    tangent projection stretches a face's corners past what a monocular depth
    model was trained on -- measured on real footage, the Large model loses
    13-20% of its ground-plane fidelity at 103 degrees. A fake model has no
    notion of projection distortion, so no synthetic scene can see that cost,
    and this test would happily wave through a 127-degree face.
    """
    steps = {ov: _seam_step(_depth_for(ov)) for ov in (0.0, OV, 0.25, 0.5)}
    best = min(steps.values())
    assert steps[OV] <= 1.15 * best, steps
    assert steps[OV] * 3 < steps[0.0], steps


def test_the_default_overlap_is_on():
    """A knob nobody sets is the one that matters. Guards against the default
    silently reverting to exact faces."""
    assert projection.FACE_OVERLAP > 0
    # Wide enough to share a band, narrow enough that the tangent projection
    # does not stretch the corners past what a depth model was trained on.
    assert 94 < projection.face_fov_degrees() < 106
    import inspect

    for fn in (pipeline.convert, pipeline.preview_frame,
               pipeline.depth_map_for_frame, pipeline.depth_maps_for_chunk):
        default = inspect.signature(fn).parameters["face_overlap"].default
        assert default == projection.FACE_OVERLAP, fn.__name__


def test_cli_defers_the_overlap_default_without_letting_it_drift():
    """`--face-overlap` cannot default to projection.FACE_OVERLAP directly --
    that would drag numpy and cv2 into every `--help` -- so it defaults to
    None and is resolved after the heavy imports. The cost of that is a number
    written twice, which is what this checks."""
    import inspect

    from stereo360 import cli

    parser = cli.build_parser()
    args = parser.parse_args(["in.mp4", "-o", "out.mp4"])
    assert args.face_overlap is None
    assert parser.parse_args(["in.mp4", "-o", "o.mp4",
                              "--face-overlap", "0"]).face_overlap == 0.0

    action = next(a for a in parser._actions
                  if a.dest == "face_overlap")
    assert str(projection.FACE_OVERLAP) in action.help
    assert f"{projection.face_fov_degrees():.0f} degrees" in action.help

    resolve = inspect.getsource(cli._run)
    assert "projection.FACE_OVERLAP" in resolve
    assert "args.face_overlap is None" in resolve


def test_zero_overlap_restores_the_exact_face_path():
    """The escape hatch has to be a real escape hatch."""
    face_size, w = 96, 384
    disp = {f: np.abs(projection._face_dirs(f, face_size)[..., 1]) + 0.1
            for f in projection.FACES}
    got = pipeline.assemble_depth({f: v.copy().astype(np.float32)
                                   for f, v in disp.items()}, w, w // 2, 0.0)
    faces = {f: v.copy().astype(np.float32) for f, v in disp.items()}
    projection.align_face_scales(faces)
    want = projection.cubemap_to_equirect(faces, w, w // 2)[..., 0]
    np.testing.assert_allclose(got, want)


def test_assembly_reproduces_a_field_the_faces_agree_on():
    """With no scale disagreement to fix, the cross-fade must be a no-op --
    otherwise it would be smoothing real depth."""
    face_size, w, h = 128, 512, 256
    faces = {f: ground_plane_depth(
        projection._overlap_face_dirs(f, face_size, OV))
        for f in projection.FACES}
    got = projection.overlapping_faces_to_equirect(faces, w, h, OV)
    want = ground_plane_depth(projection.equirect_rows_to_dir(0, h, w, h))
    rel = np.abs(got - want) / want.max()
    assert rel.max() < 0.02, f"max rel error {rel.max():.4f}"
    assert rel.mean() < 0.001, f"mean rel error {rel.mean():.5f}"


def test_every_direction_is_covered_by_at_least_one_face():
    """The weight sum is the divisor of the blend; a hole in it would be a
    divide-by-nothing, not a visible seam, so it is worth asserting directly."""
    w, h = 256, 128
    dirs = projection.equirect_rows_to_dir(0, h, w, h)
    total = np.zeros((h, w), np.float64)
    lim = 1.0 + OV
    for face in projection.FACES:
        origin, right, down = projection._FACE_BASIS[face]
        t = dirs @ origin.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (dirs @ right.astype(np.float64)) / t
            b = (dirs @ down.astype(np.float64)) / t
        keep = (t > 1e-9) & (np.abs(a) <= lim) & (np.abs(b) <= lim)
        total += (projection._overlap_weight(np.where(keep, a, 9.0), OV)
                  * projection._overlap_weight(np.where(keep, b, 9.0), OV)
                  * keep)
    # The face that owns a direction contributes at least 0.5 * 0.5.
    assert total.min() >= 0.25


def test_widening_source_faces_matches_widening_the_equirect():
    """Cubemap input keeps its promise not to round-trip through equirect, so
    the two ways of reaching a widened face must agree."""
    import cv2

    from tests.test_projection import make_test_equirect

    img = cv2.GaussianBlur(make_test_equirect(1024, 512).astype(np.uint8),
                           (9, 9), 0)
    exact = projection.equirect_to_cubemap(img, 128)
    from_faces = projection.cubemap_to_overlapping_faces(exact, OV)
    from_equi = projection.equirect_to_overlapping_faces(img, 128, OV)
    for f in projection.FACES:
        assert from_faces[f].shape == from_equi[f].shape
        err = np.abs(from_faces[f].astype(float) - from_equi[f].astype(float))
        assert err.mean() < 2.0, f"{f}: mean abs level error {err.mean():.2f}"


def test_widening_by_zero_returns_the_faces_unchanged():
    from tests.test_projection import make_test_equirect

    exact = projection.equirect_to_cubemap(
        make_test_equirect(512, 256).astype(np.uint8), 64)
    same = projection.cubemap_to_overlapping_faces(exact, 0.0)
    for f in projection.FACES:
        assert np.abs(same[f].astype(int) - exact[f].astype(int)).max() <= 1


def tilted_field(dirs: np.ndarray) -> np.ndarray:
    """A positive scalar field that varies across every face.

    Not the ground plane: that one is flat-topped, so the up face is constant
    and its scale is unidentifiable by construction -- correct behaviour, but
    it tells you nothing about whether the fit works.
    """
    n = np.array([0.3, -0.8, 0.5])
    n = n / np.linalg.norm(n)
    return (1.6 + dirs @ n).astype(np.float32)


def _disagreement(faces: dict, overlap: float) -> float:
    """RMS difference between what two faces report for the same direction,
    over every direction a pair of them share, relative to the depth range."""
    face_size = faces[projection.FACES[0]].shape[0]
    lim = 1.0 + overlap
    resid, span = [], np.ptp(np.concatenate(
        [faces[f].ravel() for f in projection.FACES]))
    for face in projection.FACES:
        i = projection.FACES.index(face)
        dirs = projection._overlap_face_dirs(face, face_size, overlap)
        fj, aj, bj = projection._face_local_coords(dirs)
        for j, other in enumerate(projection.FACES):
            sel = fj == j
            if j == i or sel.sum() < 16:
                continue
            here = faces[face][sel]
            # _sample_face reads local coords in [-1, 1]; the neighbour's map
            # covers [-lim, lim], so the coords rescale by lim.
            there = projection._sample_face(faces[other], aj[sel] / lim,
                                            bj[sel] / lim)
            resid.append((here - there) ** 2)
    return float(np.sqrt(np.concatenate(resid).mean()) / span)


def test_alignment_makes_the_faces_agree_where_they_overlap():
    """The fit's actual contract: two faces looking at the same direction
    should report the same depth. Not "recovers the exact affine I applied" --
    the solver deliberately picks between a full affine and a scale-only
    candidate by residual, so the parameters are its business and the
    agreement is ours.
    """
    face_size = 128
    truth = {f: tilted_field(
        projection._overlap_face_dirs(f, face_size, OV))
        for f in projection.FACES}
    scales = [0.6, 1.4, 0.8, 1.9, 1.1, 0.7]
    shifts = [0.05, -0.03, 0.10, -0.08, 0.02, 0.06]
    skewed = {f: (truth[f] * s + t).astype(np.float32)
              for f, s, t in zip(projection.FACES, scales, shifts)}

    before = _disagreement(skewed, OV)
    projection.align_overlapping_faces(skewed, OV)
    after = _disagreement(skewed, OV)
    assert after < 0.02, f"faces still disagree by {after:.4f} of the range"
    assert after * 5 < before, f"before {before:.4f} -> after {after:.4f}"


def test_the_blend_tables_are_built_once_not_per_frame():
    """The first version of the assembly rebuilt its coordinate tables on
    every call: 2.5 s a frame at 8K against 0.07 s for the exact-face
    assembly, and essentially all of it table construction rather than the six
    remaps the tables feed. It doubled the time of a whole render. Nothing
    about the output would show that coming back, so it is asserted here.
    """
    face_size, w, h = 64, 256, 128
    faces = {f: np.zeros((face_size, face_size), np.float32)
             for f in projection.FACES}
    projection.clear_map_caches()
    assert not projection._overlap_plan_cache

    projection.overlapping_faces_to_equirect(faces, w, h, OV)
    first = projection._overlap_blend_plan(w, h, face_size, OV)
    projection.overlapping_faces_to_equirect(faces, w, h, OV)
    second = projection._overlap_blend_plan(w, h, face_size, OV)

    assert first is second, "the plan was rebuilt"
    assert first[0][5] is second[0][5], "the coordinate tables were rebuilt"
    # One geometry at a time, like _face_to_equirect_maps: at 8K these tables
    # are ~450 MB and a run uses a single geometry throughout.
    projection._overlap_blend_plan(w * 2, h * 2, face_size, OV)
    assert len(projection._overlap_plan_cache) == 1


def test_the_blend_tables_cover_the_frame_without_covering_it_six_times():
    """Each face reaches over about a quarter of the sphere, so the six of
    them should span ~1.5 frames. Tabulating all six full-frame would be four
    times the memory and the work for pixels whose weight is zero."""
    face_size, w, h = 64, 512, 256
    projection.clear_map_caches()
    plan = projection._overlap_blend_plan(w, h, face_size, OV)
    covered = sum(e[5].size for e in plan) / (w * h)
    assert 1.0 < covered < 2.2, f"tables span {covered:.2f} frames"
    # The back face straddles the +/-180 degree wrap, so it needs two column
    # runs; a single bounding box for it would be the full width.
    assert sum(1 for e in plan if e[0] == "-Z") == 2


def test_the_folded_weights_sum_to_one_everywhere():
    """The plan pre-divides each weight by the total at that pixel, which
    removes a full-frame division per frame -- and would silently bias the
    depth if the totals were wrong."""
    face_size, w, h = 64, 256, 128
    projection.clear_map_caches()
    total = np.zeros((h, w), np.float64)
    for _, y0, y1, x0, x1, _, _, wt in projection._overlap_blend_plan(
            w, h, face_size, OV):
        total[y0:y1, x0:x1] += wt
    np.testing.assert_allclose(total, 1.0, atol=1e-5)


@pytest.mark.parametrize("overlap", [0.0, 0.15, 0.25, 0.5])
def test_assembly_never_produces_nan_or_negative_depth(overlap):
    face_size, w, h = 64, 256, 128
    rng = np.random.default_rng(0)
    faces = {f: rng.random((face_size, face_size)).astype(np.float32)
             for f in projection.FACES}
    out = pipeline.assemble_depth(faces, w, h, overlap)
    assert out.shape == (h, w)
    assert np.isfinite(out).all()
    assert (out >= 0).all()
