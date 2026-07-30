"""Regression tests for the geometric-correctness fixes:

1. The right eye is rendered by an INVERSE warp that samples the left image at
   sub-pixel coordinates (previously: forward colour splatting rounded to the
   nearest integer target, which re-rasterized thin structures every frame).
2. Per-face relative depth scales are reconciled before equirect assembly.
3. Depth assembly never interpolates across a cube seam.
"""

import numpy as np

from stereo360 import projection, warp


def test_subpixel_sampling_blends_source_pixels():
    """Sub-pixel proof: a 1-px stripe pattern must come back with blended
    intermediate values.

    Forward splatting copies whole source pixels, so the output can only ever
    contain the two source values {0, 255}. Backward bilinear sampling lands on
    fractional coordinates (the parallax shift varies continuously with
    longitude/latitude) and therefore produces genuine intermediate tones.
    """
    w, h = 512, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, ::2] = 255  # one-pixel-wide vertical stripes
    disp = np.full((h, w), 0.5, dtype=np.float32)

    right, hole = warp.right_eye_from_disparity(img, disp, strength=1.0,
                                                inpaint=False)

    valid = right[hole == 0]
    assert valid.size > 0
    # Any value strictly between the two source values can only come from a
    # fractional sample position. The *size* of the blend is not the point and
    # must not be asserted on: with the pole-tapered baseline the shift is a
    # uniform ~0.12 px everywhere instead of sweeping up to ~14 px in the
    # polar rows, so the blend is lopsided (values near 8 and 247) though no
    # less sub-pixel. An acceptance window of (40, 215) tested the old polar
    # fold, not the sampler.
    mid = ((valid[:, 0] > 0) & (valid[:, 0] < 255)).mean()
    assert mid > 0.1, (
        f"only {mid:.3%} of pixels carry intermediate (sub-pixel blended) "
        "values — sampling looks quantized to whole source pixels")


def test_thin_structure_responds_continuously_to_depth():
    """A thin bright bar must move *continuously* as depth changes.

    Sweeping depth over a range worth a couple of pixels of parallax, the bar's
    intensity-weighted centroid must trace many distinct sub-pixel positions.
    Nearest-neighbour forward splatting yields a staircase (a handful of
    plateaued integer positions), which is exactly the frame-to-frame
    shape-flipping artifact on railings.

    Depth is supplied pre-normalized (normalize=False): per-frame percentile
    normalization would rescale the bar to dn=1.0 in every iteration and hide
    the very variation being measured.
    """
    w, h = 512, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, 254:258] = 255  # thin vertical bar near lon=0 (max parallax)

    row = h // 2
    centroids = []
    for dnear in np.linspace(0.60, 0.90, 24):
        disp = np.full((h, w), 0.05, dtype=np.float32)
        disp[:, 250:262] = dnear  # the bar (plus margin) is the near structure
        right, _ = warp.right_eye_from_disparity(
            img, disp, strength=4.0, inpaint=False, fg_erode=0,
            normalize=False)
        line = right[row, :, 0].astype(np.float64)
        if line.sum() <= 0:
            continue
        centroids.append(float((line * np.arange(w)).sum() / line.sum()))

    centroids = np.asarray(centroids)
    assert centroids.size >= 12
    # Many distinct sub-pixel positions, not a few integer plateaus.
    assert np.unique(np.round(centroids, 3)).size >= 8, (
        f"bar position is quantized: {np.unique(np.round(centroids, 3))}")
    # And the motion is smooth: no single step jumps by a whole pixel.
    steps = np.abs(np.diff(centroids))
    assert steps.max() < 1.0, f"discontinuous jump of {steps.max():.3f} px"


def test_align_face_scales_removes_per_face_scale():
    """Six faces of a uniform-depth scene, each given a different arbitrary
    relative scale, must be brought back onto one common scale."""
    f = 32
    scales = {"+X": 1.0, "-X": 2.0, "+Y": 0.5,
              "-Y": 3.0, "+Z": 1.5, "-Z": 0.25}
    faces = {name: np.full((f, f), s, dtype=np.float32)
             for name, s in scales.items()}

    projection.align_face_scales(faces)

    means = np.array([faces[name].mean() for name in projection.FACES])
    assert means.min() > 0
    assert means.max() / means.min() < 1.02, (
        f"faces still disagree after alignment: {means}")


def test_align_face_scales_is_noop_when_already_consistent():
    f = 16
    faces = {name: np.full((f, f), 2.0, dtype=np.float32)
             for name in projection.FACES}
    projection.align_face_scales(faces)
    for name in projection.FACES:
        np.testing.assert_allclose(faces[name], 2.0, rtol=1e-4)


def test_depth_assembly_does_not_blend_across_seams():
    """With nearest sampling, assembling six distinct constant faces must
    reproduce only those exact values. Bilinear taps straddling an atlas block
    boundary would emit in-between values (phantom depth edges at cube seams).
    """
    f, w, h = 64, 256, 128
    vals = {name: float(i + 1) for i, name in enumerate(projection.FACES)}
    faces = {name: np.full((f, f), v, dtype=np.float32)
             for name, v in vals.items()}

    out = projection.cubemap_to_equirect(faces, w, h, nearest=True)[..., 0]

    stray = set(np.unique(out).tolist()) - set(vals.values())
    assert not stray, f"values interpolated across cube seams: {sorted(stray)}"


def test_depth_assembly_linear_path_would_blend():
    """Guards the premise of the previous test: the old bilinear path really
    does manufacture values that exist on no face."""
    f, w, h = 64, 256, 128
    vals = {name: float(i + 1) for i, name in enumerate(projection.FACES)}
    faces = {name: np.full((f, f), v, dtype=np.float32)
             for name, v in vals.items()}

    out = projection.cubemap_to_equirect(faces, w, h)[..., 0]

    stray = set(np.unique(out).tolist()) - set(vals.values())
    assert stray, "expected bilinear seam bleed in the legacy path"
