"""The two backends that became the defaults, without downloading either.

Everything here drives the real batching, reciprocal and resize code through
a stand-in inference session. The models themselves are 105 MB and 1.9 GB, so
a test that fetched them would be a test nobody runs.
"""

import numpy as np
import pytest

from stereo360.depth import depth_anything_v3 as v3


# ------------------------------------------------------------- variant names


def test_every_variant_has_a_download_size():
    """The size is printed before the wait. A missing entry would be a
    KeyError at the worst moment -- after the user committed to the render."""
    assert set(v3.DOWNLOAD_MB) == set(v3._REPOS)
    assert v3.DEFAULT_VARIANT in v3._REPOS


@pytest.mark.parametrize("given,expected", [
    (None, "small"),
    ("small", "small"),
    ("LARGE", "large"),
    ("onnx-community/depth-anything-v3-base", "base"),
])
def test_variant_names_and_repo_ids_both_resolve(given, expected):
    assert v3.resolve_variant(given) == expected


def test_a_v2_model_id_is_refused_rather_than_ignored():
    """--depth-model is shared across backends, so a V2 id can arrive here.
    Loading V3 small while the command line says V2 Large is how a benchmark
    ends up measuring the wrong thing and nobody notices."""
    with pytest.raises(ValueError, match="not a Depth Anything V3 variant"):
        v3.resolve_variant("depth-anything/Depth-Anything-V2-Large-hf")


# ----------------------------------------------------------- the V3 batching


class FakeSession:
    """Records the shape it was handed and returns a known depth per view."""

    def __init__(self, depths):
        self.depths = depths          # one constant metric depth per view
        self.calls = []

    def run(self, outputs, feed):
        batch = next(iter(feed.values()))
        self.calls.append(batch.shape)
        n = batch.shape[1]
        d = np.array(self.depths[:n], dtype=np.float32)
        self.depths = self.depths[n:]
        # [batch, num_images, H, W], matching the exported graph.
        return [np.full((1, n, 40, 40), 1.0, np.float32) * d[None, :, None, None]]


def _backend(depths, max_views=6):
    b = object.__new__(v3.DepthAnythingV3Backend)
    b.input_size = 42
    b.max_views = max_views
    b._session = FakeSession(list(depths))
    b._input = "pixel_values"
    b._output = "predicted_depth"
    return b


def test_the_six_faces_go_through_as_one_multi_view_call():
    """The whole reason this backend is ONNX rather than torch. V3 reasons
    across the images in the num_images axis, so handing it the six faces one
    at a time would throw away the thing that makes it better than V2."""
    b = _backend([2.0] * 6)
    views = [np.zeros((80, 80, 3), np.uint8) for _ in range(6)]
    b.estimate_chunk(views)
    assert len(b._session.calls) == 1
    batch, num_images = b._session.calls[0][:2]
    assert (batch, num_images) == (1, 6)


def test_more_views_than_fit_are_split_rather_than_batched_whole():
    """The tiled path can ask for far more than six. Activation memory grows
    with the axis, so this caps it instead of letting the call fail."""
    b = _backend([1.0] * 14, max_views=6)
    views = [np.zeros((20, 20, 3), np.uint8) for _ in range(14)]
    out = b.estimate_chunk(views)
    assert len(out) == 14
    assert [c[1] for c in b._session.calls] == [6, 6, 2]


def test_depth_comes_back_as_inverse_depth_at_the_input_size():
    """The pipeline's contract is inverse depth, larger = closer. V3 predicts
    depth, so a near view must come back with the *larger* value."""
    b = _backend([2.0, 8.0])
    near, far = b.estimate_chunk([np.zeros((60, 30, 3), np.uint8),
                                  np.zeros((60, 30, 3), np.uint8)])
    assert near.shape == (60, 30) and far.shape == (60, 30)
    assert near.mean() > far.mean()
    assert near.mean() == pytest.approx(0.5, rel=1e-3)
    assert far.mean() == pytest.approx(0.125, rel=1e-3)


def test_estimate_is_estimate_chunk_of_one():
    b = _backend([4.0])
    one = b.estimate(np.zeros((10, 10, 3), np.uint8))
    assert one.shape == (10, 10)
    assert one.mean() == pytest.approx(0.25, rel=1e-3)


# -------------------------------------------------------------- Depth Pro


def test_depth_pro_is_declared_as_the_download_it_is():
    """1.9 GB arriving unannounced mid-render is the failure this guards.

    Bounded on both sides rather than just "large". The figure read 3600 for
    a long time, which is not the download at all -- apple/DepthPro-hf is one
    1904 MB safetensors file -- but is close to the model's peak VRAM, so the
    two had been conflated. An upper bound is what would have caught that.
    """
    from stereo360.depth import depth_pro

    assert 1500 < depth_pro.DOWNLOAD_MB < 2500, (
        f"{depth_pro.DOWNLOAD_MB} MB does not match the 1904 MB the repo holds")
    assert depth_pro.DEFAULT_MODEL == "apple/DepthPro-hf"


class _FakeTensor(np.ndarray):
    """Enough of a torch tensor for the two lines that touch one."""

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class _FakeProcessor:
    """Stands in for the HuggingFace image processor.

    Records the target size it was asked for, because resizing back to the
    caller's resolution is this class's job rather than the model's.
    """

    def __init__(self, metres):
        self.metres = metres
        self.target_sizes = None

    def __call__(self, images, return_tensors):
        import types
        return types.SimpleNamespace(to=lambda device: {})

    def post_process_depth_estimation(self, out, target_sizes):
        self.target_sizes = target_sizes
        d = np.full(target_sizes[0], self.metres, np.float32)
        return [{"predicted_depth": d.view(_FakeTensor)}]


def test_depth_pro_inverts_metric_depth():
    """It predicts metres, the pipeline wants inverse depth. Getting this
    backwards renders every scene inside out, which reads as pseudoscopic --
    near looking far while occlusion and perspective still say near -- rather
    than as obviously broken, so it is the hardest kind of bug to catch."""
    import contextlib
    import types

    from stereo360.depth.depth_pro import DepthProBackend

    b = object.__new__(DepthProBackend)
    b.device = "cpu"
    b._torch = types.SimpleNamespace(no_grad=contextlib.nullcontext)
    b._model = lambda **kw: None
    b._proc = _FakeProcessor(metres=4.0)

    inv = b.estimate(np.zeros((12, 20, 3), np.uint8))

    assert b._proc.target_sizes == [(12, 20)], "did not resize to the input"
    assert inv.shape == (12, 20)
    assert inv.dtype == np.float32
    assert inv.mean() == pytest.approx(0.25, rel=1e-3), "4 m should be 0.25"


def test_depth_pro_reads_nearer_as_larger():
    """The contract every backend here shares, stated as the comparison the
    warp actually makes rather than as a formula."""
    import contextlib
    import types

    from stereo360.depth.depth_pro import DepthProBackend

    def at(metres):
        b = object.__new__(DepthProBackend)
        b.device = "cpu"
        b._torch = types.SimpleNamespace(no_grad=contextlib.nullcontext)
        b._model = lambda **kw: None
        b._proc = _FakeProcessor(metres=metres)
        return b.estimate(np.zeros((6, 6, 3), np.uint8)).mean()

    assert at(1.5) > at(30.0)
