"""Depth Anything V3 via ONNX Runtime.

V3 is multi-view. Its graph takes `[batch, num_images, 3, H, W]` and reasons
across the images in one call rather than running them as an independent
batch, and the pipeline happens to hand this backend exactly the right thing:
`estimate_chunk` receives the six cubemap faces of one frame together, which
is six views from a single point covering the whole sphere. That is why V3
measures so much flatter on walls and floors than V2 did -- 20% wall wobble
against 103% -- see "Choosing a depth model" in findings.md.

ONNX rather than torch because the exported small graph runs the six faces in
under two seconds on a *CPU*, which makes the video default work on machines
with no usable GPU at all. There is no torch path here for that reason.

The graph outputs metric-ish depth; the pipeline wants inverse depth, so the
reciprocal is taken here.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .base import DepthBackend

#: Small rather than base or large, and this is the surprise: capacity does
#: not help here. Large scored *worse* than small on the chair gaps (1.66
#: against 1.42) for 13x the download. Small also runs on the CPU in about the
#: time base takes on a GPU. Measured in findings.md.
DEFAULT_VARIANT = "small"

_REPOS = {
    "small": "onnx-community/depth-anything-v3-small",
    "base": "onnx-community/depth-anything-v3-base",
    "large": "onnx-community/depth-anything-v3-large",
}

#: Roughly what each export costs on first use, for a message printed before
#: the wait rather than after it. Small is the default; the other two are a
#: real download to have chosen by accident.
DOWNLOAD_MB = {"small": 105, "base": 413, "large": 1383}

#: The native training resolution, and a multiple of the ViT patch size (14).
#: Feeding it more does not help: small at 1036 improved the chair gaps but
#: lost ground on both the wall and the floor, for four times the pixels.
_INPUT_SIZE = 518

#: Views per forward pass. Six is one whole frame, so the common case is a
#: single call and the multi-view fusion sees the entire sphere at once.
#: Bigger requests -- tiles -- are split rather than allowed to blow up
#: activation memory quadratically.
_MAX_VIEWS = 6


def resolve_variant(depth_model: Optional[str]) -> str:
    """Map a `--depth-model` value onto a V3 variant name.

    Accepts the bare variant (`small`), the ONNX repo id, or None. A V2
    HuggingFace id is *not* accepted: silently loading a different model than
    the one named is how a benchmark ends up measuring the wrong thing.
    """
    if not depth_model:
        return DEFAULT_VARIANT
    name = depth_model.strip().lower()
    if name in _REPOS:
        return name
    for variant, repo in _REPOS.items():
        if name == repo.lower():
            return variant
    raise ValueError(
        f"--depth-model {depth_model!r} is not a Depth Anything V3 variant; "
        f"expected one of {sorted(_REPOS)}. To run a V2 model use "
        f"--depth-backend depth-anything.")


def download(variant: str = DEFAULT_VARIANT) -> str:
    """Fetch the exported graph, returning the path to model.onnx.

    Only the fp32 graph, not the whole `onnx/` directory: that repo also
    carries quantized exports we do not use, and pulling them turns a 105 MB
    download into several hundred.
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        _REPOS[variant],
        allow_patterns=["onnx/model.onnx", "onnx/model.onnx_data",
                        "config.json"])
    return os.path.join(path, "onnx", "model.onnx")


class DepthAnythingV3Backend(DepthBackend):
    """Multi-view relative depth via an ONNX-exported Depth Anything V3."""

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        provider: Optional[str] = None,
        model_path: Optional[str] = None,
        input_size: int = _INPUT_SIZE,
        max_views: int = _MAX_VIEWS,
    ) -> None:
        import onnxruntime as ort

        from .onnx_backend import pick_provider

        self.variant = variant
        self.input_size = int(input_size)
        self.max_views = max(1, int(max_views))
        self.provider = pick_provider(provider)
        self.model_path = model_path or download(variant)

        opts = ort.SessionOptions()
        opts.log_severity_level = 3      # the exporter's warnings are not ours
        self._session = ort.InferenceSession(
            self.model_path, opts, providers=[self.provider])
        self._input = self._session.get_inputs()[0].name
        # Ask for the depth output by name. The graph also returns confidence
        # and camera parameters, and their order is the exporter's business
        # rather than something to hard-code an index against.
        names = [o.name for o in self._session.get_outputs()]
        self._output = "predicted_depth" if "predicted_depth" in names \
            else names[0]

    def estimate(self, frame_rgb: np.ndarray) -> np.ndarray:
        return self.estimate_chunk([frame_rgb])[0]

    def estimate_chunk(self, frames_rgb: List[np.ndarray]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for i in range(0, len(frames_rgb), self.max_views):
            out.extend(self._run(frames_rgb[i:i + self.max_views]))
        return out

    def _run(self, views: List[np.ndarray]) -> List[np.ndarray]:
        import cv2

        # Scaled to 0..1 and no further. Unlike the V2 export in
        # onnx_backend.py, this graph carries the ImageNet mean/std inside it,
        # so normalizing here would apply it twice. Confirmed by measurement
        # rather than by reading the export: the scores through this path
        # reproduce the standalone experiment to within 2%, and a double
        # normalization does not land 2% from the right answer.
        sz = self.input_size
        batch = np.stack([
            cv2.resize(v, (sz, sz), interpolation=cv2.INTER_AREA)
            .astype(np.float32) / 255.0
            for v in views]).transpose(0, 3, 1, 2)[None]
        depth = self._session.run([self._output], {self._input: batch})[0][0]

        maps = []
        for view, d in zip(views, depth):
            inv = 1.0 / np.maximum(d.astype(np.float32), 1e-3)
            h, w = view.shape[:2]
            maps.append(cv2.resize(inv, (w, h),
                                   interpolation=cv2.INTER_LINEAR))
        return maps

    def close(self) -> None:
        self._session = None
