"""Depth backend abstraction.

All backends return *relative inverse depth* (a.k.a. disparity-like values):
larger values = closer to the camera, smaller = farther. Absolute scale is
undefined; the stereo-strength parameter in the warp stage absorbs it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DepthBackend(ABC):
    """Estimate relative inverse depth for RGB images."""

    @abstractmethod
    def estimate(self, frame_rgb: np.ndarray) -> np.ndarray:
        """frame_rgb: (H, W, 3) uint8. Returns (H, W) float32 inverse depth,
        where larger = closer."""

    def estimate_chunk(self, frames_rgb: list) -> list:
        """Estimate depth for a sequence of consecutive frames.

        Temporal backends override this to enforce cross-frame consistency.
        Default: plain per-frame estimation (M2 behavior).
        """
        return [self.estimate(f) for f in frames_rgb]

    def close(self) -> None:
        """Release any resources. Optional override."""

    def __enter__(self) -> "DepthBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
