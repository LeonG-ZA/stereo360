"""Video Depth Anything backend — temporally consistent depth (M3).

Uses the official Video Depth Anything model, which conditions each frame's
depth on preceding frames via a temporal keyframe mechanism, eliminating the
per-frame flicker of monocular backends.

The upstream repo is NOT pip-installable (it has no setup.py/pyproject.toml).
Clone it and either set VIDEO_DEPTH_ANYTHING_PATH or place it in
third_party/Video-Depth-Anything under the project root:

    git clone https://github.com/DepthAnything/Video-Depth-Anything third_party/Video-Depth-Anything

Checkpoint weights are downloaded from the HuggingFace hub on first use.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .base import DepthBackend
from .depth_anything import pick_device

# encoder -> (HF repo, checkpoint filename, model config)
_MODELS = {
    "small": (
        "depth-anything/Video-Depth-Anything-Small",
        "video_depth_anything_vits.pth",
        {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    ),
    "large": (
        "depth-anything/Video-Depth-Anything-Large",
        "video_depth_anything_vitl.pth",
        {"encoder": "vitl", "features": 256,
         "out_channels": [256, 512, 1024, 1024]},
    ),
}

_CLONE_HINT = (
    "Video Depth Anything repo not found. It has no pip package; clone it:\n"
    "  git clone https://github.com/DepthAnything/Video-Depth-Anything "
    "third_party/Video-Depth-Anything\n"
    "or set the VIDEO_DEPTH_ANYTHING_PATH environment variable to an "
    "existing clone.")

_DEPS_HINT = (
    "The video-depth-anything backend needs a package stereo360 does not "
    "install by default: {missing!r} is missing.\n"
    "  pip install -r requirements-vda.txt\n"
    "(That backend is the only thing needing them, so they are kept out of "
    "requirements.txt.)")


def _load_model_class():
    """Import VideoDepthAnything from a local clone of the upstream repo."""
    candidates = []
    env = os.environ.get("VIDEO_DEPTH_ANYTHING_PATH")
    if env:
        candidates.append(Path(env))
    # project root = stereo360/depth/ -> up two levels
    root = Path(__file__).resolve().parents[2]
    candidates.append(root / "third_party" / "Video-Depth-Anything")

    for path in candidates:
        if (path / "video_depth_anything" / "video_depth.py").is_file():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            try:
                from video_depth_anything.video_depth import (
                    VideoDepthAnything)
            except ModuleNotFoundError as exc:
                # The clone is present but needs packages the rest of
                # stereo360 does not. A bare "No module named 'einops'" gives
                # no clue that it came from an optional backend, or what to
                # install -- and this backend is offered in a dropdown, so it
                # is easy to select without ever having read about it.
                raise ImportError(_DEPS_HINT.format(missing=exc.name)) from exc

            return VideoDepthAnything
    raise ImportError(_CLONE_HINT)


class VideoDepthAnythingBackend(DepthBackend):
    """Temporal depth over frame chunks via Video Depth Anything.

    `estimate()` falls back to single-frame inference (no temporal context);
    use `estimate_chunk()` — the pipeline does — for consistency.
    """

    def __init__(self, variant: str = "small",
                 device: Optional[str] = None,
                 fp16: bool = False,
                 input_size: int = 518) -> None:
        if variant not in _MODELS:
            raise ValueError(
                f"Unknown variant '{variant}'; choose from {sorted(_MODELS)}")
        if input_size % 14:
            raise ValueError("input_size must be a multiple of 14 "
                             "(ViT patch size)")
        self._input_size = input_size
        self.device = pick_device(device)
        repo_id, ckpt_name, cfg = _MODELS[variant]

        model_cls = _load_model_class()

        import torch
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
        self._torch = torch
        self._model = model_cls(**cfg)
        self._model.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=True),
            strict=True)
        self._fp16 = fp16 and self.device == "cuda"
        self._model = self._model.to(self.device).eval()

    def estimate(self, frame_rgb: np.ndarray) -> np.ndarray:
        return self.estimate_chunk([frame_rgb])[0]

    def estimate_chunk(self, frames_rgb: list) -> list:
        """frames_rgb: list of (H, W, 3) uint8. Returns list of (H, W) float32
        inverse depth (larger = closer)."""
        if not frames_rgb:
            return []
        h, w = frames_rgb[0].shape[:2]
        # Upstream infer_video_depth expects (T, H, W, C) float32 in [0, 255];
        # it resizes to its internal input size itself.
        batch = np.stack(frames_rgb).astype(np.float32)
        depths, _ = self._model.infer_video_depth(
            batch, target_fps=-1, input_size=self._input_size,
            device=self.device, fp32=not self._fp16)
        out = []
        for d in depths:
            d = np.asarray(d, dtype=np.float32)
            if d.shape != (h, w):
                import cv2

                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            # VDA already returns *inverse* depth -- larger = closer -- the
            # same convention as Depth Anything V2, so it is used as-is.
            #
            # This used to invert it, on the belief that the model emitted
            # metric depth. Measured against Depth Anything V2 on the same
            # frames, the raw output correlates +0.99 while the inverted
            # output correlates -0.67, and on a landscape the raw sky reads
            # 0.019 against 6.05 for ground a metre away. So the inversion
            # was flipping near and far on every frame.
            #
            # It was also catastrophic rather than merely wrong outdoors: the
            # model reports exactly 0 for sky (43% of the pixels in one face
            # here), and 1/max(0, 1e-4) turned every sky pixel into 10000 --
            # four orders of magnitude nearer than anything real. Normalising
            # across that range squeezed the entire scene into an
            # interquartile width of 0.0003, which is why outdoor footage came
            # out with no visible stereo at all.
            np.maximum(d, 0.0, out=d)
            out.append(d)
        return out

    def close(self) -> None:
        self._model = None
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
