"""Depth estimation backends."""

from .base import DepthBackend
from .smoothing import EdgeAwareSmoothingBackend

__all__ = ["DepthBackend", "EdgeAwareSmoothingBackend",
           "VideoDepthAnythingBackend", "OnnxDepthBackend"]


def __getattr__(name):  # lazy: these require optional dependencies
    if name == "VideoDepthAnythingBackend":
        from .video_depth_anything import VideoDepthAnythingBackend

        return VideoDepthAnythingBackend
    if name == "OnnxDepthBackend":
        from .onnx_backend import OnnxDepthBackend

        return OnnxDepthBackend
    raise AttributeError(name)
