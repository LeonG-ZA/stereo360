"""Pick the fastest depth runtime actually available on this machine.

There is no single best backend. Which one wins depends on hardware that the
user should not have to reason about:

  * an NVIDIA card with a CUDA torch build runs `depth-anything` on the GPU in
    fp16, and keeps `--depth-model` available for larger, more accurate
    variants;
  * an AMD or Intel GPU on Windows has no ROCm torch at all, so ONNX Runtime
    on DirectML is the *only* GPU path;
  * Apple Silicon has torch MPS;
  * and ONNX needs both an extra package and a one-time export, so it cannot
    simply be assumed present.

So the choice is probed rather than guessed, and the result is stated plainly
-- silently falling back to the CPU is the worst outcome, because it is an
order of magnitude slower and nothing on screen says so.
"""

from __future__ import annotations

import os
from typing import NamedTuple, Optional


class Runtime(NamedTuple):
    backend: str          # "depth-anything" or "onnx"
    device: str           # torch device, or the ORT provider name
    label: str            # human-readable, for the startup banner
    gpu: bool


def _torch_gpu() -> Optional[Runtime]:
    try:
        import torch
    except ImportError:
        return None
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return Runtime("depth-anything", "cuda",
                           f"Depth Anything V2 on CUDA ({name}, fp16)", True)
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return Runtime("depth-anything", "mps",
                           "Depth Anything V2 on Apple MPS", True)
    except Exception:
        return None
    return None


_GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider",
                  "CoreMLExecutionProvider", "ROCMExecutionProvider")


def _onnx_gpu(model_path: str) -> Optional[Runtime]:
    if not os.path.exists(model_path):
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    available = set(ort.get_available_providers())
    for prov in _GPU_PROVIDERS:
        if prov in available:
            return Runtime("onnx", prov,
                           f"ONNX Runtime on {prov}", True)
    return None


def _cpu_fallback(model_path: str) -> Runtime:
    try:
        import torch  # noqa: F401
        return Runtime("depth-anything", "cpu",
                       "Depth Anything V2 on CPU (torch)", False)
    except ImportError:
        pass
    return Runtime("onnx", "CPUExecutionProvider",
                   "ONNX Runtime on CPU", False)


def detect(model_path: str) -> Runtime:
    """Best available runtime. Torch GPU first, then an ONNX GPU provider.

    Torch leads because where it has a GPU it is at least as fast and keeps
    `--depth-model` usable. ONNX comes next because it covers the case torch
    cannot reach at all -- AMD and Intel GPUs on Windows.
    """
    return (_torch_gpu()
            or _onnx_gpu(model_path)
            or _cpu_fallback(model_path))


def banner(rt: Runtime) -> str:
    if rt.gpu:
        return f"GPU accelerated: {rt.label}"
    return (f"**WARNING** GPU not being utilized. CPU accelerated: "
            f"{rt.label}. Please read the README file.")
