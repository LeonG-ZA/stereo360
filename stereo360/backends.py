"""Build a depth backend from the options the CLI exposes.

This lived inline in `cli.py`, which meant anything that was not the CLI --
a GUI, a benchmark script, a test -- had to reproduce the auto-detect wiring,
the provider/device hand-off and the chunk-size rule to get a working backend.
That is precisely the kind of duplication that drifts: the CLI gains a probe
step, the GUI does not, and the two disagree about which runtime is in use.

Imports stay inside the branches. Constructing an ONNX backend must not drag
in torch, and vice versa -- that is the whole point of having both, since the
ONNX path exists for machines with no usable torch GPU at all.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

from .depth.base import DepthBackend
from .events import Reporter

DEFAULT_ONNX_MODEL = "models/depth_anything_v2_small.onnx"

#: Backend names accepted by `build`, in the order the CLI lists them.
BACKENDS = ("auto", "depth-anything", "video-depth-anything", "onnx")


class Availability(NamedTuple):
    """Whether a backend can actually run here, and what to do if not."""

    name: str
    available: bool
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "available": self.available,
                "detail": self.detail}


def _installed(module: str) -> bool:
    """True if `module` can be imported, without importing it.

    find_spec rather than import: probing must not pull torch into a process
    that only wanted to draw a dropdown.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _vda_clone() -> Optional[str]:
    """Path to a Video Depth Anything checkout, if one is where we look."""
    import os

    candidates = []
    env = os.environ.get("VIDEO_DEPTH_ANYTHING_PATH")
    if env:
        candidates.append(env)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(root, "third_party",
                                   "Video-Depth-Anything"))
    for path in candidates:
        if os.path.isfile(os.path.join(path, "video_depth_anything",
                                       "video_depth.py")):
            return path
    return None


def probe_backends(onnx_model: str = DEFAULT_ONNX_MODEL) -> List[Availability]:
    """Which depth backends this machine can actually run.

    Offering a backend that cannot start is a bad trade: the failure arrives
    seconds later, out of context, and historically as a bare
    ModuleNotFoundError. Cheap enough to run at UI startup -- nothing here
    imports a model or a framework.
    """
    import os

    out = [Availability("auto", True,
                        "Picks the fastest runtime available here")]

    torch_ok = _installed("torch") and _installed("transformers")
    out.append(Availability(
        "depth-anything", torch_ok,
        "Per-frame depth via torch" if torch_ok
        else "Needs torch and transformers (requirements.txt)"))

    clone = _vda_clone()
    missing = [m for m in ("einops", "easydict") if not _installed(m)]
    if clone is None:
        vda = (False, "Needs a clone of the Video Depth Anything repo in "
                      "third_party/")
    elif not torch_ok:
        vda = (False, "Needs torch and transformers (requirements.txt)")
    elif missing:
        vda = (False, f"Needs {', '.join(missing)} "
                      f"(pip install -r requirements-vda.txt)")
    else:
        vda = (True, "Temporal depth, steadier across frames")
    out.append(Availability("video-depth-anything", *vda))

    if not _installed("onnxruntime"):
        onnx = (False, "Needs onnxruntime (requirements-onnx.txt)")
    elif not os.path.exists(onnx_model):
        onnx = (False, f"No exported model at {onnx_model} "
                       f"(python scripts/export_onnx.py)")
    else:
        onnx = (True, "ONNX Runtime; also the path for a custom fast model")
    out.append(Availability("onnx", *onnx))
    return out


class Built(NamedTuple):
    """A constructed backend plus what was actually resolved to build it.

    `name` and `device` are the *resolved* values, not what was asked for, so
    a caller that passed "auto" can display what it got.
    """

    backend: Optional[DepthBackend]
    name: str
    device: str
    chunk_size: int


def build(
    *,
    passthrough: bool = False,
    depth_backend: str = "auto",
    onnx_model: str = DEFAULT_ONNX_MODEL,
    ort_provider: Optional[str] = None,
    device: str = "auto",
    depth_model: Optional[str] = None,
    fp16: bool = False,
    vda_input_size: int = 518,
    smooth: int = 0,
    smooth_eps: float = 1e-3,
    chunk_size: int = 8,
    reporter: Optional[Reporter] = None,
) -> Built:
    """Construct the depth backend, resolving "auto" against this machine."""
    reporter = reporter or Reporter()

    if passthrough:
        return Built(None, "passthrough", "none", 1)

    if depth_backend not in BACKENDS:
        raise ValueError(f"Unknown depth backend {depth_backend!r}; "
                         f"expected one of {list(BACKENDS)}")

    name = depth_backend
    if name == "auto":
        from .depth import autodetect

        rt = autodetect.detect(onnx_model)
        # Silently running on the CPU is the worst outcome -- an order of
        # magnitude slower with nothing on screen saying so -- hence a
        # warning rather than an info line when no GPU was found.
        say = reporter.info if rt.gpu else reporter.warning
        say(autodetect.banner(rt), backend=rt.backend, device=rt.device,
            gpu=rt.gpu, runtime=rt.label)
        name = rt.backend
        if rt.backend == "onnx":
            ort_provider = ort_provider or rt.device
        elif device == "auto":
            device = rt.device

    if name == "video-depth-anything":
        from .depth.video_depth_anything import VideoDepthAnythingBackend

        variant = depth_model or "small"
        reporter.info(f"Loading Video Depth Anything '{variant}' on device "
                      f"'{device}'...", backend=name, model=variant,
                      device=device)
        backend: DepthBackend = VideoDepthAnythingBackend(
            variant=variant, device=device, fp16=fp16,
            input_size=vda_input_size)
        resolved_device = device
    elif name == "onnx":
        from .depth.onnx_backend import OnnxDepthBackend

        backend = OnnxDepthBackend(onnx_model, provider=ort_provider)
        resolved_device = backend.provider
        reporter.info(f"ONNX backend on provider '{backend.provider}'",
                      backend=name, model=onnx_model,
                      device=backend.provider)
    else:
        from .depth.depth_anything import DEFAULT_MODEL, DepthAnythingBackend

        model_id = depth_model or DEFAULT_MODEL
        reporter.info(f"Loading depth model '{model_id}' on device "
                      f"'{device}'...", backend=name, model=model_id,
                      device=device)
        backend = DepthAnythingBackend(model_id=model_id, device=device)
        resolved_device = device

    if smooth > 0:
        from .depth.smoothing import EdgeAwareSmoothingBackend

        backend = EdgeAwareSmoothingBackend(backend, radius=smooth,
                                            eps=smooth_eps)

    # Only the temporal model has cross-frame context to exploit, so chunking
    # is pointless -- and costs memory -- for every other backend. Decided
    # against the *resolved* name: "auto" never selects the temporal model
    # today, but keying off the raw request would silently break if it ever
    # did.
    chunks = chunk_size if name == "video-depth-anything" else 1
    return Built(backend, name, resolved_device, chunks)
