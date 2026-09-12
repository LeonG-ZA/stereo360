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

`LSDIR Compact v2` is the only entry here measured against a real answer.
Every other row scores an upscaler against an 8K frame it was asked to
reconstruct from a 4K one; this one was tested at 4x from a 2K source whose
true 8K we still had, so the invented detail could be seen rather than
inferred. On one 256px patch taken to 1024:

    method                 PSNR   contrast
    ArtCNN R8F64          35.44       1.28
    ArtCNN C4F16          34.77       1.39
    Lanczos 4x            34.62       1.13
    LSDIR Compact v2      31.73       2.30

Twice Lanczos's local contrast and three decibels below it. Beside the true
frame the difference is legible: the window frames it draws are crisper than
Lanczos and about right, while the brick grows a mottled streakiness the real
8K does not have and the foliage hardens into strands. That is a good trade
for a photograph and a bad one for a depth pass, which would read the
invented texture as geometry -- hence stills only. It costs 0.57 s for a
2K to 8K frame, the cheapest sharp option measured.

`LiveActionV1 SPAN` came off OpenModelDB, where its author says the existing
video models "all denoise or cause colour shifts" -- the fault this footage
already arrives with. Measured against the two onnx models it sits between:

    model                luma dB   SSIM   temporal   amplify   crawl
    LiveActionV1 SPAN      34.35  0.968       103%     0.85x   1.31x
    Compact ldl            34.83  0.968       119%     1.87x   1.96x
    SPAN                   31.65  0.966       107%     1.43x   1.54x

Half a decibel behind Compact ldl and far steadier: 1.31x of the Lanczos
crawl floor against 1.96x, which puts it beside the shaders (1.22-1.28x)
rather than beside the generators. The amplification is the odd number --
0.85x is *below* one, and below Lanczos at 0.96x, so it damps a small change
rather than passing it on. Nothing else measured does that, and it would
normally mean smoothing; it out-resolved Compact ldl on brickwork instead.

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
    """One upscaler: what it is, where it lives, and what it can be asked for."""

    code: str
    name: str
    desc: str
    #: "shader" runs through ffmpeg and libplacebo; "onnx" through
    #: onnxruntime; "resampler" is a libplacebo preset with no file of its
    #: own; "nvvsr" is NVIDIA's wheel. Different runners and different
    #: failure modes, and nothing else in this table depends on which.
    kind: str
    filename: str            # under models/, "" for anything with no file
    #: The factor it natively produces. **0 means it produces whatever it is
    #: asked for** -- true of every resampler, and of NVIDIA VSR, which takes
    #: output dimensions rather than a factor. Anything else hands the
    #: remainder to a resampler; see `residual`.
    scale: int
    #: What it does about detail that is not in the source, which is the axis
    #: that decides when to reach for it. Measured as the share of spectral
    #: energy above the ceiling a 4K source can carry on an 8K grid, where
    #: the real 8K holds 4.68%:
    #:
    #:     resampler   0.83 - 1.24%   adds nothing
    #:     predictor   1.79 - 2.36%   extrapolates from the real pixels
    #:     generator   4.03 - 4.58%   invents plausible detail
    #:
    #: Three bands that do not overlap, and not a restatement of `kind`:
    #: ArtCNN R8F64 is an onnx file and a predictor, and the generators
    #: include a shader-speed one in NVIDIA VSR.
    category: str
    url: str
    mb: float                # download size, to say before the wait
    #: Cost relative to plain lanczos over the same 150 frames of 4K to 8K.
    #: Architecture fixes the ordering and hardware only scales it, which is
    #: why a relative figure can be shipped where a millisecond count cannot
    #: -- the same reasoning that makes the "fastest" marker safe.
    #:
    #: Comparable *within* a category and not across one: the shader and
    #: resampler rows carry ffmpeg's decode and encode, which is most of
    #: lanczos's own 20.6 ms; the onnx rows were timed in process around
    #: `esrgan.upscale` alone; the VSR rows in a torch loop.
    cost: float = 1.0
    #: Largest factor it will produce when `scale` is 0. 0 means no ceiling.
    max_scale: float = 0.0
    #: The libplacebo preset, or the VSR quality level -- whatever the runner
    #: needs naming that is not a filename.
    option: str = ""
    #: Which way this resampler is for. A 4x graph asked for 2x leaves a
    #: *downscale* to finish, and that is a different filter set: an
    #: area filter averages the samples it is discarding, where a sharpening
    #: upscaler used backwards just aliases. Only ever "down" on a resampler.
    direction: str = "up"

    @property
    def path(self) -> str:
        return os.path.join("models", self.filename) if self.filename else ""

    @property
    def any_scale(self) -> bool:
        """Whether it covers the whole factor by itself, at any factor."""
        return self.scale == 0


RESAMPLER, PREDICTOR, GENERATOR = "resampler", "predictor", "generator"

#: Ordered for the interface, and the order is chosen per category rather
#: than globally, because a single ranking would assert something the
#: measurements do not support.
#:
#: The predictors *are* ordered by quality: C4F32 >= C4F16 > FSRCNNX 16 >=
#: FSRCNNX 8 > R8F64 held on all four sources tested, with SSSR apart as the
#: one that wins when the source is soft. C4F32 DN is the exception to the
#: ordering rather than to the rule: it sits next to its sibling rather than
#: at its own score, because the 0.35 dB it gives up is what it is for.
#: The generators are not ordered by quality, because
#: theirs flipped -- LiveActionV1 was the best of them at 4x on outdoor.jpg
#: and last on the Giethoorn blossom -- so they run by scale and name. The
#: resamplers run by how much acutance they add, which is a fact about the
#: filters rather than a verdict on them.
VARIANTS: Sequence[Variant] = (
    # ---- resamplers: no file, no download, built into libplacebo ---------
    Variant("lanczos", "Lanczos",
            "A plain resampler. Adds nothing that was not in the source, "
            "which is what makes it the safest choice on footage that is "
            "already sharpened or compressed -- it won every such rung "
            "tested.",
            "resampler", "", 0, RESAMPLER, "", 0.0, 1.0),
    Variant("spline36", "spline36",
            "libplacebo's default. Indistinguishable from Lanczos here: "
            "0.08 dB apart at 4K to 8K.",
            "resampler", "", 0, RESAMPLER, "", 0.0, 1.0, 0.0, "spline36"),
    Variant("ewa_lanczos", "ewa_lanczos",
            "Jinc rather than sinc, and applied over a disc rather than "
            "along the axes, so it has no preferred direction and treats a "
            "diagonal like anything else. About three times the taps.",
            "resampler", "", 0, RESAMPLER, "", 0.0, 1.1, 0.0, "ewa_lanczos"),
    Variant("ewa_lanczossharp", "ewa_lanczossharp",
            "ewa_lanczos with the kernel narrowed 1.9%, which is a mild "
            "sharpening. The constant is Nicolas Robidoux's.",
            "resampler", "", 0, RESAMPLER, "", 0.0, 1.1, 0.0,
            "ewa_lanczossharp"),
    Variant("ewa_lanczos4sharpest", "ewa_lanczos4sharpest",
            "The one to reach for on camera footage. Despite the name it "
            "behaves as a de-ringer, because it is the only preset here "
            "with anti-ringing -- 0.8, which is what makes its 11.5% "
            "narrower kernel usable. Best of anything measured on the "
            "oversharpened and haloed rungs.",
            "resampler", "", 0, RESAMPLER, "", 0.0, 1.2, 0.0,
            "ewa_lanczos4sharpest"),
    Variant("area", "INTER_AREA",
            "The way back down, for a 4x graph asked for less. It averages "
            "the samples it discards rather than dropping them, which is "
            "also why the overshoot is not simply waste: on the matched "
            "pair tested, four samples averaged came out 1.5 dB ahead of "
            "the same architecture's native 2x.",
            "resampler", "", 0, RESAMPLER, "", 0.0, 1.0, 0.0, "area", "down"),

    # ---- predictors: extrapolate from the pixels that are there ----------
    Variant("artcnn32", "ArtCNN C4F32",
            "The best predictor measured, and the one to reach for on a "
            "clean source: it won on 360 stills, on broadcast video and on "
            "an 8K walking tour. Four convolutions, 32 filters, and the "
            "neutral build -- Artoriuz ships denoising ones too, and DN is "
            "below.",
            "shader", "ArtCNN_C4F32.glsl", 2, PREDICTOR,
            "https://github.com/Artoriuz/ArtCNN/releases/download/v1.6.2/"
            "ArtCNN_C4F32.glsl", 0.73, 2.6),
    Variant("artcnn32dn", "ArtCNN C4F32 DN",
            "The same network trained to denoise and soften instead of to "
            "reproduce, and the one predictor that behaves like a "
            "resampler -- 1.10% of its energy above the source's ceiling, "
            "inside the resampler band and below every other predictor "
            "here. On degraded footage it lands on ewa_lanczos4sharpest's "
            "texture almost exactly. On a clean source it gives up 0.35 dB "
            "to plain C4F32, which is the trade it exists to make.",
            "shader", "ArtCNN_C4F32_DN.glsl", 2, PREDICTOR,
            "https://github.com/Artoriuz/ArtCNN/releases/download/v1.6.2/"
            "ArtCNN_C4F32_DN.glsl", 0.73, 2.6),
    Variant("artcnn16", "ArtCNN C4F16",
            "Half the cost of C4F32 and within noise of it everywhere "
            "except a pristine source, where C4F32 leads by 0.26 dB. On the "
            "other seven rungs they were 0.05 dB apart or less.",
            "shader", "ArtCNN_C4F16.glsl", 2, PREDICTOR,
            "https://github.com/Artoriuz/ArtCNN/releases/download/v1.6.2/"
            "ArtCNN_C4F16.glsl", 0.21, 1.4),
    Variant("fsrcnnx16", "FSRCNNX 16",
            "The long-standing video default. A shader, not a generator -- "
            "it resamples rather than inventing. Behind both ArtCNN "
            "variants on every rung tested.",
            "shader", "FSRCNNX_x2_16-0-4-1.glsl", 2, PREDICTOR,
            "https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/"
            "FSRCNNX_x2_16-0-4-1.glsl", 0.24, 1.6),
    Variant("fsrcnnx8", "FSRCNNX 8",
            "The cheaper FSRCNNX. Measured 36.13 against the 16's 36.12 on "
            "the test that chose it, so the larger one buys little.",
            "shader", "FSRCNNX_x2_8-0-4-1.glsl", 2, PREDICTOR,
            "https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/"
            "FSRCNNX_x2_8-0-4-1.glsl", 0.07, 1.2),
    Variant("spline36_sssr", "spline36_SSSR",
            "A corrector rather than a doubler: it resamples, then pulls "
            "the result back towards the source's own structure. Won all "
            "three soft rungs, and lost the sharpened ones by more than "
            "anything else -- the most aggressive predictor here.",
            "shader", "SSimSuperRes.glsl", 2, PREDICTOR,
            "https://gist.githubusercontent.com/igv/"
            "2364ffa6e81540f29cb7ab4c9bc05b6b/raw/"
            "15d93440d0a24fc4b8770070be6a9fa2af6f200b/SSimSuperRes.glsl",
            0.01, 1.1),
    Variant("artcnnr8", "ArtCNN R8F64",
            "Eight residual blocks. Won on 360 stills once, and never "
            "since -- on the eight-rung ladder it placed below C4F16 and "
            "below Lanczos, at forty times C4F16's cost.",
            "onnx", "ArtCNN_R8F64.onnx", 2, PREDICTOR,
            "https://github.com/Artoriuz/ArtCNN/releases/download/v1.6.2/"
            "ArtCNN_R8F64.onnx", 3.5, 94.0),

    # ---- generators: invent detail that is not in the source -------------
    Variant("nvvsr_ultra", "NVIDIA VSR ULTRA",
            "NVIDIA's RTX Video upscaler at its highest setting, and the "
            "closest match measured to a degraded source's own texture. "
            "Scales to any factor up to 4x by itself.",
            "nvvsr", "", 0, GENERATOR, "", 490.0, 0.9, 4.0, "ULTRA"),
    Variant("nvvsr_high", "NVIDIA VSR HIGH",
            "A step below ULTRA and slightly sharper than the truth on the "
            "material tested. Scales to any factor up to 4x.",
            "nvvsr", "", 0, GENERATOR, "", 490.0, 0.7, 4.0, "HIGH"),
    Variant("nvvsr_medium", "NVIDIA VSR MEDIUM",
            "Cheaper, and on a clean source it scored *above* HIGH -- the "
            "levels do not rank consistently, because how much invention "
            "helps depends on the source. Scales to any factor up to 4x.",
            "nvvsr", "", 0, GENERATOR, "", 490.0, 0.4, 4.0, "MEDIUM"),
    Variant("nvvsr_low", "NVIDIA VSR LOW",
            "The cheapest thing in this table, faster than plain Lanczos, "
            "and on a clean 360 still the best-scoring of the four levels. "
            "Scales to any factor up to 4x.",
            "nvvsr", "", 0, GENERATOR, "", 490.0, 0.3, 4.0, "LOW"),
    Variant("compactldl", "Compact ldl",
            "Sharper than a shader without a GAN's invention, because JPEG "
            "degradation was in its training -- it reads a compression "
            "artifact as damage where a faithful model sharpens it.",
            "onnx", "2xNomosUni_compact_multijpg_ldl.onnx", 2, GENERATOR,
            "https://huggingface.co/Phips/2xNomosUni_compact_multijpg_ldl/"
            "resolve/main/2xNomosUni_compact_multijpg_ldl.safetensors", 2.4,
            1.26),
    Variant("liveaction", "LiveActionV1 SPAN",
            "Trained for live action rather than anime, and against "
            "denoising rather than with it. The only generator measured to "
            "beat plain Lanczos while inventing 15% -- on one source; it "
            "was last on another.",
            "onnx", "2xLiveActionV1_SPAN.onnx", 2, GENERATOR,
            "https://raw.githubusercontent.com/jcj83429/upscaling/"
            "f73a3a02874360ec6ced18f8bdd8e43b5d7bba57/2xLiveActionV1_SPAN/"
            "2xLiveActionV1_SPAN_490000.pth", 8.9, 1.57),
    Variant("span", "SPAN",
            "The plain NomosUni SPAN. Consistently the weakest of this "
            "family on the sources tested.",
            "onnx", "2xNomosUni_span_multijpg.onnx", 2, GENERATOR,
            "https://huggingface.co/Phips/2xNomosUni_span_multijpg/resolve/"
            "main/2xNomosUni_span_multijpg.safetensors", 4.5, 1.58),
    Variant("spanldl", "SPAN ldl",
            "SPAN with the loss that suppresses what a GAN would add. "
            "Steadier than SPAN and no sharper.",
            "onnx", "2xNomosUni_span_multijpg_ldl.onnx", 2, GENERATOR,
            "https://huggingface.co/Phips/2xNomosUni_span_multijpg_ldl/"
            "resolve/main/2xNomosUni_span_multijpg_ldl.safetensors", 8.9,
            1.58),
    Variant("esrgan2x", "ESRGAN 2x uni",
            "The ESRGAN architecture at 2x. The one model whose fidelity "
            "*improves* when its invention is suppressed, which is a sign "
            "it guesses wrong more often than right on this material.",
            "onnx", "2xNomosUni_esrgan_multijpg.onnx", 2, GENERATOR,
            "https://huggingface.co/Phips/2xNomosUni_esrgan_multijpg/resolve/"
            "main/2xNomosUni_esrgan_multijpg.safetensors", 33.5, 9.44),
    Variant("lsdir", "LSDIR Compact v2",
            "A 4x graph, so at 2x it computes four times the pixels it "
            "hands back -- which on the matched pair tested came out 1.5 dB "
            "*better*, because averaging four samples damps the model's own "
            "errors. Twice the time for it.",
            "onnx", "4xLSDIRCompactv2.onnx", 4, GENERATOR,
            "https://github.com/Phhofm/models/releases/download/"
            "4xLSDIRCompact2/4xLSDIRCompactv2.safetensors", 1.2, 2.46),
    Variant("siax", "Siax",
            "A 4x graph and the slowest thing here by a wide margin. Reads "
            "sharpest on a frozen frame and invents visibly to get there.",
            "onnx", "4x_NMKD-Siax.onnx", 4, GENERATOR,
            "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/"
            "4x_NMKD-Siax_200k.pth", 67.0, 28.49),
)

#: How far each model magnifies a small frame-to-frame change, measured with
#: the seeded 0.86-level nudge described above. Kept here rather than in the
#: table because not every model has one -- LSDIR was judged against a real
#: 8K frame instead. Read on a video frame, which is the material that
#: governs a pre-pass; the figures from `outdoor.jpg` are lower and are the
#: ones the first table records.
#:
#: Retained as measurement rather than as policy. It used to bar three models
#: from video; watching 120 frames of each showed every one of them steady,
#: and the *untouched* 8K flickering 1.20x the Lanczos floor itself -- so the
#: floor was never the target and a model sitting on it is too smooth rather
#: than admirably still.
AMPLIFY: Dict[str, float] = {
    "fsrcnnx8": 1.11, "fsrcnnx16": 1.10, "artcnn16": 1.15, "artcnnr8": 1.09,
    "spline36_sssr": 1.15, "liveaction": 0.85, "span": 1.43, "spanldl": 1.63,
    "compactldl": 1.87, "esrgan2x": 1.82, "siax": 4.24,
}

BY_CODE: Dict[str, Variant] = {v.code: v for v in VARIANTS}

#: Reached for in order, and the first two are not in this table: Topaz if it
#: is installed, then NVIDIA VSR if its wheel is present *and* its licence
#: accepted -- a GPU alone is not enough, or the default would be something
#: the user has not downloaded. See `default_code`.
NVIDIA_DEFAULT = "nvvsr_ultra"
FREE_DEFAULT = "artcnn32"

#: Kept for callers that predate the two-stage choice. Both now name the same
#: predictor: the stills/video split they encoded was the wrong axis, and
#: what actually decides is whether the source is degraded or pristine.
VIDEO_DEFAULT = FREE_DEFAULT
PHOTO_DEFAULT = FREE_DEFAULT


def get(code: Optional[str]) -> Optional[Variant]:
    return BY_CODE.get((code or "").strip().lower())


def present(v: Variant, root: str = ".") -> bool:
    """Whether it can be used here.

    A resampler is always present -- it is a preset in an ffmpeg this project
    already requires. NVIDIA VSR needs its wheel *and* an accepted licence,
    which is asked of `nvvsr` so that this table holds no opinion about
    someone else's terms.
    """
    if v.kind == "resampler":
        return True
    if v.kind == "nvvsr":
        from . import nvvsr

        return nvvsr.installed() and nvvsr.consented()
    return os.path.exists(os.path.join(root, v.path))


def in_category(category: str, root: Optional[str] = None) -> list:
    """The variants of one category, in the table's order.

    With `root`, only those usable here.
    """
    return [v for v in VARIANTS if v.category == category
            and (root is None or present(v, root))]


def residual(target: float, v: Optional[Variant]) -> float:
    """What is left for a resampler once `v` has done its part.

    1.0 means nothing is left and the second stage is not needed. Below 1.0
    means the model overshot and the remainder is a *downscale*, which is a
    different filter set -- libplacebo takes it on `downscaler` and the onnx
    path uses INTER_AREA.

    With no model the resampler does the whole factor.
    """
    if target <= 0:
        raise ValueError(f"a target of {target:g} is not a scale")
    if v is None:
        return float(target)
    if v.any_scale:
        return 1.0
    return float(target) / float(v.scale)


def fastest(category: str, root: Optional[str] = None) -> Optional[str]:
    """The cheapest usable variant of a category, or None if it has none.

    Chosen from `cost`, which is relative and so survives a change of
    hardware; the ordering within a category is set by architecture rather
    than by the machine that measured it.
    """
    pool = in_category(category, root)
    return min(pool, key=lambda v: v.cost).code if pool else None


def default_code(topaz: bool = False, root: str = ".") -> Optional[str]:
    """The model to select when nothing has been chosen.

    Topaz first where it is installed, then NVIDIA VSR where it is ready,
    then the best free predictor. Returns None for Topaz because it is not
    in this table -- the caller owns that half of the choice.
    """
    if topaz:
        return None
    nvidia = BY_CODE.get(NVIDIA_DEFAULT)
    if nvidia is not None and present(nvidia, root):
        return NVIDIA_DEFAULT
    free = BY_CODE.get(FREE_DEFAULT)
    if free is not None and present(free, root):
        return FREE_DEFAULT
    for v in VARIANTS:
        if v.category != RESAMPLER and present(v, root):
            return v.code
    return None


def for_job(is_photo: bool, root: str = ".") -> Optional[Variant]:
    """The best default actually installed here, or None.

    Falls forward rather than failing, so a machine missing the preferred
    model is offered something usable rather than nothing. `is_photo` no
    longer changes the answer -- the flag it used to consult was removed once
    every model in this table was watched over 120 frames and none crawled --
    and it stays in the signature for the callers that pass it.
    """
    code = default_code(root=root)
    return BY_CODE.get(code) if code else None
