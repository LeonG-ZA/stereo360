"""Which upscalers exist, what they cost, and which one to reach for.

One table rather than a constant per module, because there are now six of
them across two runtimes and the interface, the command line and the fetcher
all have to agree about the same list.

Everything in the notes below was measured on a Radeon 780M against
`outdoor.jpg`, upscaling 2x. Three numbers decide a model's place here:

*Seconds for a 4K to 8K frame*, which is what a video pre-pass actually asks.
The spread is not subtle -- 0.3 s to 129 s -- and it is what separates a
model that can touch video from one that cannot.

*Amplification*, the one that matters most and is least obvious. Each of
these is a single-image model run per frame with no memory, so the question
is not whether it remembers but how hard it reacts: perturb a frame by 0.86
levels, less than the change between two consecutive frames of a static
scene, and see how far the output moves. At 1.0 the change passes straight
through. Above it the model is re-deciding what to synthesise, and detail
re-invented differently each frame is what a headset shows as crawling.

*Fidelity*, in dB against a known original, which is the least useful of the
three and is recorded to stop it being reached for again: it ranks plain
Lanczos above every generative model here, because inventing plausible detail
is exactly what it penalises. It is a check that a model works, not a ranking.

    model            s/frame   amplify      dB
    FSRCNNX 8           0.21     1.12x    37.59
    FSRCNNX 16          0.31     1.13x    37.50
    SPAN                   5     0.99x    31.41
    SPAN ldl               6     1.05x    32.94
    compact ldl            5     1.22x    35.41
    Siax                 129     4.24x    26.21

Siax is stills-only on that evidence and not on preference: it is the one
judged best by eye on a frozen frame and it amplifies a one-level wobble into
four, which is the crawling the whole table exists to avoid.
"""

from __future__ import annotations

import os
from typing import Dict, NamedTuple, Optional, Sequence


class Variant(NamedTuple):
    """One upscaler: what it is, where it lives, and what it can be used for."""

    code: str
    name: str
    desc: str
    #: "shader" runs through ffmpeg and libplacebo; "onnx" through
    #: onnxruntime. The two have different runners and different failure
    #: modes, and nothing else in this table depends on which.
    kind: str
    filename: str            # under models/
    scale: int               # what the graph natively does
    stills_only: bool
    url: str
    mb: float                # download size, to say before the wait

    @property
    def path(self) -> str:
        return os.path.join("models", self.filename)


#: Native 2x wherever possible. A 4x graph asked for 2x computes sixteen
#: times the source pixels to hand back four, and the difference is not
#: academic: the same ESRGAN architecture is 129 s a frame at 4x and 29 s at
#: 2x. Siax stays 4x because that is the checkpoint that exists.
VARIANTS: Sequence[Variant] = (
    Variant("fsrcnnx16", "FSRCNNX 16",
            "The video default. A shader, not a generator -- it resamples "
            "rather than inventing, so nothing it draws can fail to hold "
            "still. Runs on any GPU with a Vulkan driver.",
            "shader", "FSRCNNX_x2_16-0-4-1.glsl", 2, False,
            "https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/"
            "FSRCNNX_x2_16-0-4-1.glsl", 0.24),
    Variant("fsrcnnx8", "FSRCNNX 8",
            "The smaller shader, and a third faster. It measured within "
            "0.1 dB of FSRCNNX 16 on grass and differed from it by about a "
            "third of a level on edges, so the choice is close.",
            "shader", "FSRCNNX_x2_8-0-4-1.glsl", 2, False,
            "https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/"
            "FSRCNNX_x2_8-0-4-1.glsl", 0.07),
    Variant("span", "SPAN",
            "The steadiest model measured, and the first fast enough to put "
            "in front of video: it passes a small change straight through "
            "rather than re-deciding what to draw.",
            "onnx", "2xNomosUni_span_multijpg.onnx", 2, False,
            "https://huggingface.co/Phips/2xNomosUni_span_multijpg/resolve/"
            "main/2xNomosUni_span_multijpg.safetensors", 4.5),
    Variant("spanldl", "SPAN ldl",
            "SPAN trained with Locally Discriminative Learning, which "
            "targets the artifacts a GAN leaves behind. Slightly closer to "
            "the source than plain SPAN for the same cost.",
            "onnx", "2xNomosUni_span_multijpg_ldl.onnx", 2, False,
            "https://huggingface.co/Phips/2xNomosUni_span_multijpg_ldl/"
            "resolve/main/2xNomosUni_span_multijpg_ldl.safetensors", 8.9),
    Variant("compactldl", "Compact ldl",
            "The smallest network here and the closest to the source of any "
            "model measured. Slightly livelier frame to frame than SPAN.",
            "onnx", "2xNomosUni_compact_multijpg_ldl.onnx", 2, False,
            "https://huggingface.co/Phips/2xNomosUni_compact_multijpg_ldl/"
            "resolve/main/2xNomosUni_compact_multijpg_ldl.safetensors", 2.4),
    Variant("siax", "Siax",
            "For stills. The best-looking of these on a frozen frame, and "
            "the worst on video by a distance -- it turns a one-level wobble "
            "into four, which reads as crawling. Two minutes a frame.",
            "onnx", "4x_NMKD-Siax.onnx", 4, True,
            "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/"
            "4x_NMKD-Siax_200k.pth", 67.0),
)

BY_CODE: Dict[str, Variant] = {v.code: v for v in VARIANTS}

#: A shader, because a video pre-pass runs thousands of times and the only
#: thing here that cannot crawl is the one that does not invent.
VIDEO_DEFAULT = "fsrcnnx16"

#: A still has no next frame to disagree with, so the model that reads best
#: on a frozen frame wins -- which is the opposite of the video answer, and
#: the reason these are two constants rather than one.
PHOTO_DEFAULT = "siax"


def get(code: Optional[str]) -> Optional[Variant]:
    return BY_CODE.get((code or "").strip().lower())


def present(v: Variant, root: str = ".") -> bool:
    return os.path.exists(os.path.join(root, v.path))


def for_job(is_photo: bool, root: str = ".") -> Optional[Variant]:
    """The best default actually installed here, or None.

    Falls forward rather than failing: a machine with the video default
    missing and a photo model present should be offered the photo model for a
    photo, and anything usable for a video, rather than nothing.
    """
    want = PHOTO_DEFAULT if is_photo else VIDEO_DEFAULT
    first = BY_CODE.get(want)
    if first is not None and present(first, root):
        return first
    for v in VARIANTS:
        if present(v, root) and (is_photo or not v.stills_only):
            return v
    return None
