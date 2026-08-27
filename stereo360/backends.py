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
BACKENDS = ("auto", "depth-anything-v3", "depth-pro", "depth-anything",
            "video-depth-anything", "onnx")

#: What a run gets when nobody asked for a backend, by kind of input. The two
#: differ because the right trade differs: video wants a model small and fast
#: enough to run thousands of times, stills want the best single frame anyone
#: can get. Measured in "Choosing a depth model" in findings.md.
VIDEO_BACKEND = "depth-anything-v3"
PHOTO_BACKEND = "depth-pro"

#: Where a default falls back to when it cannot run *usefully* here. Not
#: "auto": `autodetect.detect` chooses between torch and an exported ONNX
#: model, and on a machine with neither it lands on Depth Anything V2 on the
#: CPU -- which shares the very dependency and the very lack of a GPU that
#: disqualified the default. V3 is the one GPU path that needs no torch and no
#: export: onnxruntime-directml covers AMD and Intel, and it fetches its own
#: 105 MB model.
FALLBACK_BACKEND = "depth-anything-v3"

#: Backends that run through torch, and so are only as fast as torch's device.
_TORCH_BACKENDS = ("depth-pro", "depth-anything", "video-depth-anything")


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

    if _installed("onnxruntime") and _installed("huggingface_hub"):
        v3 = (True, "Multi-view depth; the video default. Downloads 105 MB "
                    "on first use")
    else:
        v3 = (False, "Needs onnxruntime and huggingface_hub "
                     "(requirements-onnx.txt)")
    out.append(Availability("depth-anything-v3", *v3))

    out.append(Availability(
        "depth-pro", torch_ok,
        "Sharpest thin structures; the stills default. Downloads 1.9 GB on "
        "first use, and wants a GPU: ~3.7 GB of VRAM, against 9.4 GB of RAM "
        "and half an hour a frame on a processor" if torch_ok
        else "Needs torch and transformers (requirements.txt)"))

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


def torch_backend_problem() -> Optional[str]:
    """Why a torch depth backend would not finish here, or None if it would.

    Distinct from `probe_backends`, which asks whether the packages exist.
    Both failures this catches are ones presence cannot see:

    *Torch imports but has no GPU.* `pick_device` resolves to CUDA, then MPS,
    then the CPU, and there is no fourth option -- an AMD card on Windows has
    no ROCm build, and torch-directml is deliberately unused. Depth Pro is
    budgeted at roughly 2 s a frame on a GPU and runs six times for one photo,
    once per cube face, each face resized to 1536x1536 by its own processor
    regardless of `--face-size`. On the CPU that is not slow, it is a
    different order of magnitude, and it looks exactly like a hang.

    *Torch and transformers are both installed but disagree.* torch-directml
    pins torch==2.4.1, so installing it downgrades torch; a transformers new
    enough to want `DTensor` then fails at import. `find_spec` sees two
    healthy packages and reports the backend available.

    Imports for real, which is the point, so this is only called once a torch
    backend is actually about to be chosen -- never on the video path, whose
    default reaches onnxruntime instead.
    """
    try:
        import torch
    except Exception as e:                       # noqa: BLE001
        return f"torch does not import ({type(e).__name__}: {e})"
    try:
        # The names the backends actually construct, not the bare package.
        # transformers resolves its exports lazily, so `import transformers`
        # succeeds even when the module behind a name cannot load -- which is
        # exactly the torch-version failure this is here to catch.
        from transformers import (AutoImageProcessor,  # noqa: F401
                                  AutoModelForDepthEstimation)
    except Exception as e:                       # noqa: BLE001
        return (f"transformers does not import against torch "
                f"{getattr(torch, '__version__', '?')} "
                f"({type(e).__name__}: {e})")
    try:
        if torch.cuda.is_available():
            return None
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return None
    except Exception:                            # noqa: BLE001
        pass
    return (f"torch {getattr(torch, '__version__', '?')} has no GPU device, "
            f"so it would run on the CPU")


def resolve_depth_backend(requested: Optional[str], is_image: bool,
                          reporter: Optional[Reporter] = None) -> str:
    """Which backend to use, given what was asked for and what came in.

    `None` means nobody asked -- the flag defaults to None rather than to a
    name so that an explicit `--depth-backend depth-anything-v3` on a photo is
    still honoured. Same sentinel as `resolve_depth_tiles`, and for the same
    reason: with a real default the two cases are indistinguishable afterwards.

    Falls back when the preferred model cannot run here, because these two
    defaults carry dependencies the others do not -- onnxruntime for one,
    1.9 GB of weights for the other -- and a machine without them should still
    convert something rather than stop.

    Two checks, because "installed" and "usable" are not the same thing.
    `probe_backends` asks whether the packages are present, which is all it
    can afford to ask -- it runs behind a dropdown and must not import torch.
    `torch_backend_problem` then actually imports, and asks whether the thing
    would *finish*. A stills default that starts and then runs for half an
    hour is worse than one that says so and picks something else.
    """
    if requested:
        return requested
    want = PHOTO_BACKEND if is_image else VIDEO_BACKEND
    rep = reporter or Reporter()
    for a in probe_backends():
        if a.name == want and not a.available:
            rep.warning(
                f"Default depth backend '{want}' unavailable here "
                f"({a.detail}); falling back to {FALLBACK_BACKEND}",
                backend=want, detail=a.detail, fallback=FALLBACK_BACKEND)
            return FALLBACK_BACKEND
    if want in _TORCH_BACKENDS:
        problem = torch_backend_problem()
        if problem is not None:
            rep.warning(
                f"Default depth backend '{want}' would run but not finish "
                f"here ({problem}); falling back to {FALLBACK_BACKEND}. "
                f"Pass --depth-backend {want} to insist.",
                backend=want, detail=problem, fallback=FALLBACK_BACKEND)
            return FALLBACK_BACKEND
    return want


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

    if name == "depth-anything-v3":
        from .depth.depth_anything_v3 import (DOWNLOAD_MB,
                                              DepthAnythingV3Backend,
                                              resolve_variant)

        variant = resolve_variant(depth_model)
        reporter.info(f"Loading Depth Anything V3 '{variant}' (downloads "
                      f"~{DOWNLOAD_MB[variant]} MB on first use)...",
                      backend=name, model=variant)
        backend: DepthBackend = DepthAnythingV3Backend(
            variant=variant, provider=ort_provider)
        resolved_device = backend.provider
        reporter.info(f"Depth Anything V3 on provider '{backend.provider}'",
                      backend=name, model=variant, device=backend.provider)
    elif name == "depth-pro":
        from .depth.depth_pro import DEFAULT_MODEL as DEPTH_PRO_MODEL
        from .depth.depth_pro import DOWNLOAD_MB, DepthProBackend

        model_id = depth_model or DEPTH_PRO_MODEL
        reporter.info(f"Loading Depth Pro '{model_id}' on device '{device}' "
                      f"(downloads ~{DOWNLOAD_MB / 1000:.1f} GB on first "
                      f"use)...", backend=name, model=model_id, device=device)
        backend = DepthProBackend(model_id=model_id, device=device)
        resolved_device = backend.device
    elif name == "video-depth-anything":
        from .depth.video_depth_anything import VideoDepthAnythingBackend

        variant = depth_model or "small"
        reporter.info(f"Loading Video Depth Anything '{variant}' on device "
                      f"'{device}'...", backend=name, model=variant,
                      device=device)
        backend = VideoDepthAnythingBackend(
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
