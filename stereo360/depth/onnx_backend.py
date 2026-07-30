"""ONNX Runtime depth backend (M4) — runs an exported Depth Anything V2 graph
on any onnxruntime execution provider: DirectML (AMD/Intel on Windows), CUDA,
CoreML (Apple Silicon), or CPU. No PyTorch required at inference time.

Export the model first:
    python scripts/export_onnx.py
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .base import DepthBackend

_INPUT_SIZE = 518

# Cap on images per inference call, so a chunk cannot blow up device memory.
_MAX_BATCH = 8
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Provider preference order; first available wins.
_PROVIDERS = [
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]


def pick_provider(preference: Optional[str] = None) -> str:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if preference:
        if preference not in available:
            raise RuntimeError(
                f"onnxruntime provider '{preference}' not available; "
                f"have: {sorted(available)}")
        return preference
    for p in _PROVIDERS:
        if p in available:
            return p
    raise RuntimeError("No onnxruntime execution providers available")


class OnnxDepthBackend(DepthBackend):
    """Per-frame monocular depth via an ONNX-exported Depth Anything V2."""

    def __init__(self, model_path: str,
                 provider: Optional[str] = None) -> None:
        import onnxruntime as ort

        if not os.path.exists(model_path):
            raise FileNotFoundError("\n".join((
                f"ONNX depth model not found: {model_path}",
                "The graph is not shipped with the repository; export it once:",
                "    python scripts/export_onnx.py",
                "(prefix PYTHONIOENCODING=utf-8 on a Windows console). That "
                "needs torch, which is already a requirement.",
                "Or use --depth-backend depth-anything, which downloads its "
                "weights automatically and uses CUDA/MPS when torch has them.",
            )))
        self.provider = pick_provider(provider)
        self._model_path = model_path
        self._session = ort.InferenceSession(
            model_path, providers=[self.provider])
        # The exported graph bakes its input resolution in, and a "fast" export
        # deliberately uses a smaller one, so take it from the model rather
        # than assuming the 518 native size.
        shape = self._session.get_inputs()[0].shape
        self.input_size = shape[2] if isinstance(shape[2], int) else _INPUT_SIZE
        self.batched = self._probe_batching()

    def _probe_batching(self) -> bool:
        """Can this graph, on this provider, actually run batch > 1?

        A dynamic batch axis is necessary but not sufficient: DirectML rejects
        the graph's Reshape with "the parameter is incorrect" -- and does so
        even at batch 1, so a batch-enabled export is simply unusable there,
        while the identical graph runs fine on CPU and CUDA. Provider op
        coverage is not introspectable, so the only reliable test is to try.

        A failed run leaves the session unusable, so a failure here rebuilds
        it before falling back. If even batch 1 then fails, the model and the
        provider are incompatible and that is worth saying plainly rather than
        surfacing as a Reshape error on the first frame.
        """
        import onnxruntime as ort

        if isinstance(self._session.get_inputs()[0].shape[0], int):
            return False                      # exported with batch pinned to 1

        probe = np.zeros((2, 3, self.input_size, self.input_size),
                         dtype=np.float32)
        try:
            self._session.run(["depth"], {"pixel_values": probe})
            return True
        except Exception:
            pass

        self._session = ort.InferenceSession(self._model_path,
                                             providers=[self.provider])
        try:
            self._session.run(["depth"], {"pixel_values": probe[:1]})
        except Exception as exc:
            raise RuntimeError(
                f"'{self._model_path}' cannot run on {self.provider}. It was "
                "exported with a dynamic batch axis, which DirectML does not "
                "support; re-export without batching (or use the stock "
                "models/depth_anything_v2_small.onnx) for this provider."
            ) from exc
        return False

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        import cv2

        # INTER_AREA when shrinking: a 1920 face reaching a 518 input is a
        # 3.7x reduction, and INTER_CUBIC has a fixed 4x4 kernel that does not
        # antialias, so it aliases the very thin structures whose depth matters
        # most. Measured against PIL's scaled-support bicubic on real faces,
        # INTER_AREA matches to corr 0.99996 versus 0.99956.
        shrink = self.input_size ** 2 < frame_rgb.shape[0] * frame_rgb.shape[1]
        img = cv2.resize(frame_rgb, (self.input_size, self.input_size),
                         interpolation=cv2.INTER_AREA if shrink
                         else cv2.INTER_CUBIC)
        x = img.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        return x.transpose(2, 0, 1)

    def estimate_chunk(self, frames_rgb: list) -> list:
        """Depth for several images in as few inference calls as possible.

        Batching matters most on a discrete GPU: a single 518x518 forward pass
        is far too small to saturate one, so throughput is set by launch
        overhead rather than by arithmetic. Falls back to one call per image
        when the graph was exported with a fixed batch (re-run
        scripts/export_onnx.py to get the dynamic axis).
        """
        import cv2

        if not self.batched or len(frames_rgb) == 1:
            return [self.estimate(f) for f in frames_rgb]

        out = []
        for i in range(0, len(frames_rgb), _MAX_BATCH):
            group = frames_rgb[i:i + _MAX_BATCH]
            x = np.stack([self._preprocess(f) for f in group])
            depths = self._session.run(["depth"], {"pixel_values": x})[0]
            for f, d in zip(group, depths):
                h, w = f.shape[:2]
                out.append(cv2.resize(d.astype(np.float32), (w, h),
                                      interpolation=cv2.INTER_LINEAR))
        return out

    def estimate(self, frame_rgb: np.ndarray) -> np.ndarray:
        import cv2

        h, w = frame_rgb.shape[:2]
        # Square resizing distorts aspect slightly, which is acceptable for
        # relative depth (the HF processor's own resize is similarly
        # approximate).
        x = self._preprocess(frame_rgb)[None]        # (1, 3, S, S)
        depth = self._session.run(["depth"], {"pixel_values": x})[0][0]
        depth = cv2.resize(depth.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_LINEAR)
        return depth

    def close(self) -> None:
        self._session = None
