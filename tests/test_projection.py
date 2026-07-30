"""Unit tests for equirect <-> cubemap round-trip fidelity."""

import numpy as np

from stereo360 import projection


def make_test_equirect(w=1024, h=512) -> np.ndarray:
    """Smooth synthetic equirect image (low-frequency so bilinear error is small)."""
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = (np.sin(u / w * 2 * np.pi * 3) * 0.5 + 0.5) * 255
    g = (np.cos(v / h * np.pi * 2) * 0.5 + 0.5) * 255
    b = (u / w * 255)
    img = np.stack([r, g, b], axis=-1)
    return img.astype(np.float32)


def test_cubemap_face_shapes():
    img = make_test_equirect()
    faces = projection.equirect_to_cubemap(img, face_size=256)
    assert set(faces.keys()) == set(projection.FACES)
    for f in projection.FACES:
        assert faces[f].shape == (256, 256, 3)


def test_roundtrip_fidelity():
    w, h, face = 1024, 512, 512
    img = make_test_equirect(w, h)
    faces = projection.equirect_to_cubemap(img, face)
    back = projection.cubemap_to_equirect(faces, w, h)

    assert back.shape == img.shape

    # Ignore a thin border at the poles / edges where sampling error is expected.
    margin_x, margin_y = 4, 8
    a = img[margin_y:-margin_y, margin_x:-margin_x]
    b = back[margin_y:-margin_y, margin_x:-margin_x]
    err = np.abs(a.astype(np.float64) - b.astype(np.float64))
    assert err.mean() < 2.0, f"mean abs error too high: {err.mean():.3f}"
    assert np.percentile(err, 99) < 12.0, f"p99 error too high: {np.percentile(err, 99):.3f}"


def test_direction_convention():
    """Pixel at equirect center must land in the +Z face; right edge-of-center in +X."""
    w, h = 1024, 512
    # Direction at exact center (lon=0, lat=0) is +Z.
    d = projection._equirect_uv_to_dir(
        np.array([[w / 2 - 0.5]]), np.array([[h / 2 - 0.5]]), w, h
    )
    face, a, b = projection._face_local_coords(d)
    assert projection.FACES[face[0, 0]] == "+Z"
    assert abs(a[0, 0]) < 0.01 and abs(b[0, 0]) < 0.01

    # Direction at lon=+90deg (u at 3/4 width) is +X.
    d = projection._equirect_uv_to_dir(
        np.array([[w * 0.75 - 0.5]]), np.array([[h / 2 - 0.5]]), w, h
    )
    face, _, _ = projection._face_local_coords(d)
    assert projection.FACES[face[0, 0]] == "+X"
