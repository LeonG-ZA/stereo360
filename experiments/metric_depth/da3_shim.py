"""Import depth_anything_3.api without its export-only dependencies.

The package imports moviepy, matplotlib, pycolmap, trimesh and evo at module
level, for one purpose each: writing the prediction out as a Gaussian-splat
video, a COLMAP reconstruction, a GLB, or aligning poses against a ground-truth
trajectory. We want the depth tensor. Installing those into the same
interpreter that runs the repo is not free -- the dependency solver for this
package backtracked through every xformers release and started downloading a
source tarball, because no prebuilt wheel matches this torch -- so instead
stand in empty modules for whatever is genuinely missing.

`import_api` shims by discovery rather than from a fixed list, and returns what
it faked so the caller can check that nothing load-bearing (torch, numpy,
safetensors, huggingface_hub) is in it. Everything legitimately shimmed here is
used inside a function we never call, so an empty module gets the import
through and no attribute is ever read off it.
"""
from __future__ import annotations

import sys
import types

#: Anything in here being missing is a real problem, not something to paper
#: over: inference cannot run without it.
LOAD_BEARING = {"torch", "numpy", "safetensors", "huggingface_hub", "PIL",
                "cv2", "omegaconf", "addict", "yaml"}

MAX_ROUNDS = 24


def _fake(name: str) -> None:
    m = types.ModuleType(name)
    m.__getattr__ = lambda k: None              # type: ignore[method-assign]
    m.__path__ = []                              # allow `from x.y import z`
    sys.modules[name] = m


def import_api():
    """Return (DepthAnything3, list of module names stood in for)."""
    faked: list[str] = []
    for _ in range(MAX_ROUNDS):
        try:
            from depth_anything_3.api import DepthAnything3
            return DepthAnything3, faked
        except ModuleNotFoundError as exc:
            name = exc.name or ""
            root = name.split(".")[0]
            if not name or root in LOAD_BEARING or name in faked:
                raise
            # `from a.b import c` needs a.b to exist, which needs a to exist.
            parts = name.split(".")
            for i in range(len(parts)):
                sub = ".".join(parts[: i + 1])
                if sub not in sys.modules:
                    _fake(sub)
                    faked.append(sub)
    raise ImportError(f"still missing modules after {MAX_ROUNDS} rounds")


def patch_for_cpu(DA3) -> bool:
    """Make `DepthAnything3.forward` runnable without CUDA.

    The published forward opens with

        autocast_dtype = bfloat16 if torch.cuda.is_bf16_supported() else float16

    and `torch.cuda.is_bf16_supported()` does not answer False on a CPU-only
    build -- it initialises CUDA and raises. So the model cannot run at all
    here, autocast or no autocast.

    Replace it with the same call in plain fp32. Dropping autocast rather than
    forcing a half dtype is deliberate: CPU autocast wants bfloat16, this
    machine has no bfloat16 instructions to make that a win, and fp16 on CPU
    is emulated per-op. Full precision is both the fastest and the only one
    whose numerics we can trust, which matters when the whole point of the run
    is to read distances in metres off the output.

    Returns True if it patched, False if CUDA is really available.
    """
    import torch

    if torch.cuda.is_available():
        return False

    def forward(self, image, extrinsics=None, intrinsics=None,
                export_feat_layers=None, infer_gs=False, use_ray_pose=False,
                ref_view_strategy="saddle_balanced"):
        with torch.no_grad():
            return self.model(image, extrinsics, intrinsics,
                              export_feat_layers, infer_gs, use_ray_pose,
                              ref_view_strategy)

    DA3.forward = forward
    return True
