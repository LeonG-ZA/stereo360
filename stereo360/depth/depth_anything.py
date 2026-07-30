"""Depth Anything V2 backend via HuggingFace transformers (PyTorch).

Supports CUDA, Apple Silicon (MPS), and CPU. Weights are downloaded from the
HuggingFace hub on first use (~100-400 MB depending on variant).

Calls the model directly rather than through `pipeline("depth-estimation")`,
for two reasons that both showed up in measurement:

* the pipeline's `depth` output is a *PIL image* -- uint8, and min-max
  normalized per image. That quantizes depth to 256 levels and hands every
  cubemap face its own independent scale for `align_face_scales` to undo.
  `predicted_depth` is the raw float32 tensor: 1114 distinct values against
  256 on the same input.
* the pipeline takes one image at a time, and transformers says so out loud
  ("You seem to be using the pipelines sequentially on GPU"). One 518x518
  forward pass cannot saturate a discrete GPU, so throughput ends up set by
  launch overhead rather than by arithmetic -- 6 calls per frame, or 30 with
  --depth-tiles 2.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import DepthBackend

#: Base rather than Small. Measured on 8K footage (see "Which depth model?" in
#: the README): the lowest depth noise of the three, the best sharpness per
#: unit of noise, and 40% less frame-to-frame depth flicker than Small -- which
#: is the mechanism behind thin structures changing shape between frames. Depth
#: is only about 30% of the pipeline at 8K, so it costs roughly +9% of total
#: render time rather than the +28% the depth stage alone suggests. Large costs
#: twice that again and measured no better on any axis.
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"

# Images per forward pass. Caps activation memory on small GPUs while still
# giving a large one enough work to be worth the launch.
_MAX_BATCH = 8


def pick_device(preference: Optional[str] = None) -> str:
    """Resolve 'auto' to the best available torch device."""
    if preference and preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DepthAnythingBackend(DepthBackend):
    """Per-frame monocular depth via Depth Anything V2.

    Output is relative inverse depth (larger = closer), resized to the input
    resolution. Temporal consistency across frames is NOT provided by this
    backend — that arrives with the video depth model in M3.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_batch: int = _MAX_BATCH,
    ) -> None:
        self.device = pick_device(device)
        self.max_batch = max(1, max_batch)
        # Deferred imports so the CLI stays usable without torch installed.
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._torch = torch
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._proc = AutoImageProcessor.from_pretrained(model_id)
        try:      # transformers >= 5 renamed torch_dtype -> dtype
            model = AutoModelForDepthEstimation.from_pretrained(
                model_id, dtype=self.dtype)
        except TypeError:
            model = AutoModelForDepthEstimation.from_pretrained(
                model_id, torch_dtype=self.dtype)
        self._model = model.to(self.device).eval()

        cfg = self._proc.to_dict()
        size = cfg.get("size") or {}
        self._size = (size.get("height", 518), size.get("width", 518))
        self._keep_aspect = bool(cfg.get("keep_aspect_ratio", False))
        self._multiple = int(cfg.get("ensure_multiple_of", 1)) or 1
        self._rescale = float(cfg.get("rescale_factor", 1.0 / 255.0))
        self._mean = np.asarray(cfg.get("image_mean", (0.485, 0.456, 0.406)),
                                dtype=np.float32)
        self._std = np.asarray(cfg.get("image_std", (0.229, 0.224, 0.225)),
                               dtype=np.float32)

    def _target_size(self, h: int, w: int) -> tuple:
        """DPT's resize target: keep aspect, snap each side to a multiple."""
        m = self._multiple
        sh, sw = self._size[0] / h, self._size[1] / w
        if self._keep_aspect:
            if abs(1 - sw) < abs(1 - sh):
                sh = sw
            else:
                sw = sh

        def snap(v: float) -> int:
            return max(m, int(round(v / m) * m))

        return snap(sh * h), snap(sw * w)

    def _preprocess(self, frames_rgb: List[np.ndarray]) -> np.ndarray:
        """The image processor's job, done with cv2 instead.

        `DPTImageProcessor` costs 0.076 s for six 1920x1920 faces against
        0.002 s for the equivalent cv2 resize -- 73% of the whole depth stage
        on a 5070 Ti, where the GPU forward pass itself is only 0.020 s. It
        resizes (bicubic, aspect preserved, sides snapped to a multiple of 14),
        rescales by 1/255 and applies the ImageNet mean/std, all of which is
        reproduced here from the processor's own config rather than hardcoded.
        """
        import cv2

        th, tw = self._target_size(*frames_rgb[0].shape[:2])
        # INTER_AREA when shrinking, INTER_CUBIC when growing. This is not a
        # detail: a 1920 face reaching a 518 input is a 3.7x reduction, and
        # INTER_CUBIC applies a fixed 4x4 kernel with no antialiasing, so it
        # aliases. Against the HF processor (PIL bicubic, which does scale its
        # support) on real cubemap faces, INTER_AREA matches to corr 0.99996
        # and max 0.19, against 0.99956 and 0.84 for INTER_CUBIC.
        shrink = th * tw < frames_rgb[0].shape[0] * frames_rgb[0].shape[1]
        interp = cv2.INTER_AREA if shrink else cv2.INTER_CUBIC
        arr = np.empty((len(frames_rgb), th, tw, 3), np.float32)
        for i, f in enumerate(frames_rgb):
            arr[i] = cv2.resize(f, (tw, th), interpolation=interp)
        arr *= np.float32(self._rescale)
        arr -= self._mean
        arr /= self._std
        return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))

    def _infer(self, frames_rgb: List[np.ndarray]) -> np.ndarray:
        """Raw float32 predictions for a batch, at the model's resolution."""
        torch = self._torch
        shapes = {f.shape[:2] for f in frames_rgb}
        if len(shapes) == 1:
            pixel_values = torch.from_numpy(self._preprocess(frames_rgb))
        else:
            # Mixed sizes would resize to different targets and so could not be
            # stacked; let the processor handle that rare case.
            pixel_values = self._proc(images=frames_rgb,
                                      return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            out = self._model(pixel_values=pixel_values).predicted_depth
        if out.ndim == 2:                     # a batch of one can come back 2D
            out = out[None]
        return out.detach().float().cpu().numpy()

    def estimate(self, frame_rgb: np.ndarray) -> np.ndarray:
        return self.estimate_chunk([frame_rgb])[0]

    def estimate_chunk(self, frames_rgb: list) -> list:
        import cv2

        out: list = []
        for i in range(0, len(frames_rgb), self.max_batch):
            group = frames_rgb[i:i + self.max_batch]
            preds = self._infer(group)
            for frame, depth in zip(group, preds):
                h, w = frame.shape[:2]
                if depth.shape != (h, w):
                    depth = cv2.resize(depth, (w, h),
                                       interpolation=cv2.INTER_LINEAR)
                out.append(np.ascontiguousarray(depth, dtype=np.float32))
        return out

    def close(self) -> None:
        self._model = None
        self._proc = None
