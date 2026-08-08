"""Apple Depth Pro via HuggingFace transformers (PyTorch).

The stills default, and it earns that on one axis the scorecard cannot see.
All three geometry scores measure large-scale shape, where V3's multi-view
design wins comfortably. None of them measures thin structure -- and Depth Pro
is the only model tried here that keeps the metallic trim along the kitchen
wall edge, which is the artifact that started the whole investigation. Judged
in a headset it came first; judged by the numbers it comes second. See
"Choosing a depth model" in findings.md.

The cost is a 3.6 GB download and roughly 2 s per frame on a GPU, which is why
this is the *stills* default and not the video one. Six faces of one photo is
a fine trade; eight thousand frames is not.

Depth Pro predicts metric depth in metres. The pipeline wants inverse depth,
so the reciprocal is taken here -- and that is also what makes it behave: a
flat floor's inverse depth is linear in sin(latitude) by construction.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import DepthBackend

DEFAULT_MODEL = "apple/DepthPro-hf"

#: Roughly what the weights cost on first use, for a message worth printing
#: before a user waits on it rather than after.
DOWNLOAD_MB = 3600


class DepthProBackend(DepthBackend):
    """Per-image metric depth via Apple Depth Pro, returned as inverse depth.

    One image per forward pass. Depth Pro runs a multi-scale patch pyramid
    internally and is already wide enough to keep a GPU busy, so batching
    faces buys nothing and costs activation memory a 3.6 GB model can ill
    afford.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        from .depth_anything import pick_device

        self._torch = torch
        self.device = pick_device(device)
        self.model_id = model_id
        self._proc = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            model_id).to(self.device).eval()

    def estimate(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        inputs = self._proc(images=frame_rgb, return_tensors="pt").to(
            self.device)
        with self._torch.no_grad():
            out = self._model(**inputs)
        # post_process_depth_estimation resizes back for us and applies the
        # focal-length scaling that makes the output metric.
        depth = self._proc.post_process_depth_estimation(
            out, target_sizes=[(h, w)])[0]["predicted_depth"]
        depth = depth.detach().float().cpu().numpy()
        return (1.0 / np.maximum(depth, 1e-3)).astype(np.float32)

    def estimate_chunk(self, frames_rgb: List[np.ndarray]) -> List[np.ndarray]:
        return [self.estimate(f) for f in frames_rgb]

    def close(self) -> None:
        self._model = None
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
