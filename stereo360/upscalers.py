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

Amplification is a ratio and so travels between machines, but it does *not*
travel between scenes, which this table originally implied it did. Every row
above was nudged on a frame of `outdoor.jpg`. Re-measured on a frame of real
video the shaders held and every learned model rose:

    model          on outdoor.jpg   on video   temporal
    FSRCNNX 8              1.12x      1.11x       101%
    FSRCNNX 16             1.13x      1.10x       102%
    SPAN                   0.99x      1.43x       107%
    SPAN ldl               1.05x      1.63x       108%
    compact ldl            1.22x      1.87x       119%

The harness is the same one; the material is not. A resampler computes the
same thing whatever it is shown, so the shader rows reproduce. A generative
model re-decides more when there is more to decide about, and a photograph
gives it less than a video frame does. Read the left column as one scene's
and the right as the one that governs a pre-pass.

Siax is stills-only on that evidence and not on preference: it is the one
judged best by eye on a frozen frame and it amplifies a one-level wobble into
four, which is the crawling the whole table exists to avoid.

`ESRGAN 2x uni` was added later and measured on a different machine -- an
RTX 5070 Ti through DirectML, onnxruntime having no CUDA provider there -- so
its seconds do not belong in the column above and are given as a ratio
instead. Its dB does not either: run there, FSRCNNX 8 reads 39.27 against the
37.59 recorded above, a constant offset that applies to every row, while the
FSRCNNX 8-to-16 gap reproduces at 0.09 dB exactly. The column is comparable
within one run and not between two, which is worth knowing before anyone adds
a row from a third machine.

    model            vs Siax   amplify   temporal
    Compact ldl        15x        1.22x       --
    ESRGAN 2x uni     3.7x        1.82x     142%
    Siax                1x        4.24x       --

`spline36_SSSR` is the odd one out: it has no scale factor of its own. igv's
SSimSuperRes corrects whatever the main scaler did rather than doing the
scaling, so `scale` is 2 here to match the range it was measured over and not
because the shader is limited to it -- above 2x it does *more*, because the
kernel has more error to correct. The corollary is that pairing it with
FSRCNNX is pointless: at exactly 2x the prescaler lands on the output size,
the kernel does nothing, and SSSR finds nothing to fix -- 0.03 levels of
difference, against 0.47 at 3x where the kernel has real work.

Measured on the same twelve frames as the rows above:

    method            luma dB   SSIM   temporal   amplify   crawl   s/frame
    FSRCNNX 16          36.58  0.980       102%     1.10x   1.24x      0.32
    ArtCNN C4F16        37.10  0.981       101%     1.15x   1.26x      0.32
    spline36 + SSSR     36.66  0.981       102%     1.15x   1.22x      0.29

It beats the current default on every column and is a 6 KB file with no model
behind it. It is still not the default, because sharper measured is not the
same as better watched: FSRCNNX was preferred by eye on the same footage, and
one clip does not overturn that. `crawl` is the newer measure -- two
consecutive frames, masked to where the source did not move, against Lanczos,
which cannot invent and so is the unit.

`ArtCNN C4F16` was measured the same way and stays an option rather than the
default. Against FSRCNNX 16 on the same twelve frames it read 36.58 dB to
36.13, 0.981 SSIM to 0.980, 101% temporal to 102% and 1.15x amplification to
1.10x, for 0.34 s a frame against 0.30 -- ahead everywhere and dear nowhere.
What stopped it becoming the default is that the margin did not survive a
second scene: on the low-contrast stone of `indoor_4k.jpg` the two are hard
to tell apart, and ArtCNN is trained on anime, which is not what this
converts. One favourable scene and one neutral one is not enough to move a
default that thousands of frames depend on.

Amplification is a ratio of two differences and so is portable; it is the
number to trust across machines. ESRGAN 2x uni is the answer to "Siax is
nearly right but too slow": the same architecture at 2x rather than 4x, which
computes four times the source pixels instead of sixteen. It is stills-only
on measurement rather than on suspicion -- 142% temporal on real footage,
worse than Real-ESRGAN's 135%, which this project already rejected for video.
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
    #: Refuses the model for video outright, which is heavier than the
    #: evidence now supports and is meant to become advice rather than a bar
    #: -- "recommended for still images only", with the choice left to
    #: whoever is looking at the footage.
    #:
    #: Watched at 1:1 over 120 frames of dry grass, every model in this
    #: table looked stable, this flag's two included. The numbers that set
    #: the flag disagree with each other as well: Siax is barred on 4.24x
    #: amplification, yet in that clip it moved *less* between frames than
    #: Compact ldl, which is not barred. A 4x graph asked for 2x is
    #: downsampled by area on the way out (see esrgan.upscale), and that
    #: averaging may be quietly steadying it -- which would mean the flag
    #: rests on a measurement of something else. Worth re-testing before
    #: anyone relies on it either way.
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
    Variant("artcnn16", "ArtCNN C4F16",
            "A shader like FSRCNNX and within a hair of its cost. It "
            "measured ahead on brickwork and level on low-contrast stone, "
            "so it is offered rather than chosen: it is trained for anime, "
            "which is not what this converts.",
            "shader", "ArtCNN_C4F16.glsl", 2, False,
            "https://github.com/Artoriuz/ArtCNN/releases/download/v1.6.2/"
            "ArtCNN_C4F16.glsl", 0.21),
    Variant("artcnn32", "ArtCNN C4F32",
            "The larger shader. Half a decibel ahead of C4F16 on brickwork "
            "for twice the time, and still under a second a frame -- worth "
            "it when the source has fine repeating detail and not when it "
            "does not.",
            "shader", "ArtCNN_C4F32.glsl", 2, False,
            "https://github.com/Artoriuz/ArtCNN/releases/download/v1.6.2/"
            "ArtCNN_C4F32.glsl", 0.73),
    Variant("artcnnr8", "ArtCNN R8F64",
            "The sharpest thing here that is still safe in front of video: "
            "it resolves more than the shader and amplifies less, which "
            "nothing else in this table manages. Twelve times the shader's "
            "cost, so it is the choice when the wait is acceptable.",
            "onnx", "ArtCNN_R8F64.onnx", 2, False,
            "https://github.com/Artoriuz/ArtCNN/releases/download/v1.6.2/"
            "ArtCNN_R8F64.onnx", 3.5),
    Variant("spline36_sssr", "spline36_SSSR",
            "Not an upscaler but a corrector: libplacebo scales, then the "
            "shader downscales its own result, compares that against the "
            "source and fixes the difference. Crisper edges than the "
            "learned shaders and the steadiest thing measured -- though "
            "sharper is not the same as better, and FSRCNNX may still read "
            "well against it by eye.",
            "shader", "SSimSuperRes.glsl", 2, False,
            "https://gist.githubusercontent.com/igv/"
            "2364ffa6e81540f29cb7ab4c9bc05b6b/raw/"
            "15d93440d0a24fc4b8770070be6a9fa2af6f200b/SSimSuperRes.glsl",
            0.01),
    Variant("span", "SPAN",
            "The steadiest of the learned models and fast enough for video, "
            "though not as steady as its first measurement suggested: a "
            "small change comes back 1.4 times larger on real footage, "
            "against 1.0 on a photograph. The shaders are steadier.",
            "onnx", "2xNomosUni_span_multijpg.onnx", 2, False,
            "https://huggingface.co/Phips/2xNomosUni_span_multijpg/resolve/"
            "main/2xNomosUni_span_multijpg.safetensors", 4.5),
    Variant("spanldl", "SPAN ldl",
            "SPAN trained with Locally Discriminative Learning, which "
            "targets the artifacts a GAN leaves behind. Sharper than plain "
            "SPAN for the same cost, and slightly less settled with it -- "
            "1.6 times a small change against SPAN's 1.4.",
            "onnx", "2xNomosUni_span_multijpg_ldl.onnx", 2, False,
            "https://huggingface.co/Phips/2xNomosUni_span_multijpg_ldl/"
            "resolve/main/2xNomosUni_span_multijpg_ldl.safetensors", 8.9),
    Variant("compactldl", "Compact ldl",
            "The stills default. Sharper than the shader without inventing, "
            "because JPEG degradation was in its training -- it reads a "
            "compression artifact as damage where a faithful model sharpens "
            "it and a GAN paints over it. Slightly livelier frame to frame "
            "than SPAN.",
            "onnx", "2xNomosUni_compact_multijpg_ldl.onnx", 2, False,
            "https://huggingface.co/Phips/2xNomosUni_compact_multijpg_ldl/"
            "resolve/main/2xNomosUni_compact_multijpg_ldl.safetensors", 2.4),
    Variant("esrgan2x", "ESRGAN 2x uni",
            "For stills, when Siax is close but too slow: the same ESRGAN "
            "architecture at 2x rather than 4x, so it computes four times "
            "the source pixels instead of sixteen and finishes in a quarter "
            "the time. Sharper than Compact ldl without Siax's invention.",
            "onnx", "2xNomosUni_esrgan_multijpg.onnx", 2, True,
            "https://huggingface.co/Phips/2xNomosUni_esrgan_multijpg/"
            "resolve/main/2xNomosUni_esrgan_multijpg.safetensors", 33.5),
    Variant("siax", "Siax",
            "For stills, and the sharpest of these -- but it invents to get "
            "there, which shows as detail that was not in the scene. Worst "
            "on video by a distance: it turns a one-level wobble into four, "
            "which reads as crawling. Two minutes a frame.",
            "onnx", "4x_NMKD-Siax.onnx", 4, True,
            "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/"
            "4x_NMKD-Siax_200k.pth", 67.0),
)

BY_CODE: Dict[str, Variant] = {v.code: v for v in VARIANTS}

#: A shader, because a video pre-pass runs thousands of times and the only
#: thing here that cannot crawl is the one that does not invent.
VIDEO_DEFAULT = "fsrcnnx16"

#: Judged by eye on real 360 stills rather than by the scorecard, and the
#: scorecard would have chosen differently: Siax reads sharpest on a frozen
#: frame and invents visibly to get there, which a headset shows as detail
#: that was never in the scene.
#:
#: What separates this one is what it was trained on. The sources here are
#: compressed -- an ordinary outdoor frame measured 1.71 bits a pixel with a
#: blockiness ratio of 1.137 -- and `multijpg` means JPEG degradation was in
#: its training, so it treats a compression artifact as damage rather than as
#: detail. Models trained without that sharpen the artifacts faithfully and
#: read as noisy; GAN-trained ones paint over them and read as invented. The
#: `ldl` half is the loss that suppresses what a GAN would otherwise add.
#:
#: Also thirty times faster than Siax, which is the smaller reason.
PHOTO_DEFAULT = "compactldl"


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
