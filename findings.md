# stereo360 — reference and findings

Everything the [README](README.md) does not need: the full flag reference, and
the measurements behind each default.

Most of what follows exists because something looked wrong in a headset and the
cause turned out to be measurable. Where a number is quoted it was measured on
8K footage on the machine noted, not taken from general advice — several of the
defaults here contradict the usual guidance, and the measurement is the reason.

## Usage

```bash
python -m stereo360 input.mp4 -o output.mp4
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--face-size N` | auto (W/4) | Cubemap face resolution; auto uses input width ÷ 4, the lossless value |
| `--crf N` | 18 | Encoder quality (lower = better) — see guide below |
| `--preset` | medium | x264/x265 speed preset |
| `--codec` | libx264 | `libx264`/`libx265` (CPU), or NVENC/QSV/AMF — see hardware encoders below |
| `--bitdepth` | 8 | 8 or 10; 10-bit reduces gradient banding in VR |
| `--max-frames N` | all | Process only first N frames (testing) |
| `--no-cubemap` | off | Skip cubemap round-trip |
| `--passthrough` | off | M1 mode: right eye = left eye (no depth model) |
| `--depth-backend` | **auto** | `auto` probes the machine and uses the fastest runtime available, printing which. Or force `depth-anything` (per-frame torch), `video-depth-anything` (temporal, flicker-free), `onnx` (ONNX Runtime: DirectML/CUDA/CoreML, no PyTorch) |
| `--onnx-model PATH` | models/depth_anything_v2_small.onnx | Exported ONNX depth model (`python scripts/export_onnx.py`) |
| `--ort-provider` | auto | onnxruntime provider override, e.g. `DmlExecutionProvider` |
| `--fp16` | off | Half-precision inference for the video backend (GPU) |
| `--start-frame N` | 0 | Skip first N frames (resume into a new file; concat segments with ffmpeg) |
| `--inpaint` | simple | `simple` = OpenCV Telea (fast); `learned` = LaMa neural inpainting (best quality, slow on CPU) |
| `--no-temporal-fill` | (fill is **on**) | Disable filling holes from other frames in the chunk. On by default: real pixels another frame saw beat anything invented. Needs `--chunk-size` > 1 |
| `--depth-tiles N` | 1 | Split each cubemap face into N×N overlapping tiles for depth (feather-blended). Higher = finer depth on thin structures; N² times slower |
| `--face-overlap F` | 0.15 | How far each depth face reaches past its nominal 90° (0.15 = 98° per face), so neighbours share a band rather than only an edge. 0 restores exact faces, which creases the ground at a seam — see [Cube seams in the depth map](#cube-seams-in-the-depth-map) |
| `--depth-model ID` | **Depth-Anything-V2-Base** | HuggingFace model id, or `small`/`large` for the video backend (which ships those two only and defaults to small). Base measured lowest depth noise and 40% less frame-to-frame flicker than Small for +5% render time — see [Which depth model?](#which-depth-model) |
| `--chunk-size N` | 8 | Temporal chunk length for the video backend (1 = off) |
| `--chunk-overlap N` | 2 | Overlap frames ramp-blended at chunk boundaries |
| `--smooth N` | 0 (off) | Edge-aware depth smoothing radius (guided filter). Off by default: it measured 66% of total runtime and `--gradient-limit` already handles depth cliffs |
| `--device` | auto | `auto`, `cuda`, `mps`, or `cpu` |
| `--strength X` | 1.0 | Stereo baseline strength (higher = stronger 3D) |
| `--gradient-limit X` | **1.0** | Clamp the depth gradient to X x the slope at which the warp stops being injective. Prevents disocclusion instead of filling it. 0 disables; raise it if sharp edges look flat |
| `--input-projection` | auto | How to read the input: `auto` believes the file's own metadata, or force `equirectangular`/`cubemap` — see below |
| `--source-subsampling` | off (4:2:0) | Follow the source's chroma subsampling instead of 4:2:0 — see below |
| `--probe-encoders WxH` | off | Print which video encoders can encode WxH here as JSON, and exit |
| `--probe-backends` | off | Print which depth backends can run here as JSON, with a reason for each that cannot, and exit |
| `--probe-json` | off | Print what we know about the input as JSON and exit (for a GUI) |
| `--spatial-audio` | off | Describe the audio as ambiX ambisonics (ACN/SN3D) by writing an `SA3D` box. The source audio must really be ambiX: 4, 9 or 16 channels. Spherical + stereoscopic are always written and need no flag |
| `--split-baseline` | off | Warp both eyes by half the baseline in opposite directions. Same 3D, far less disocclusion per eye, and holes become monocular |
| `--preview-frame N` | off | Render source frame N as one image and stop — see below |
| `--preview-width W` | 2048 | Cap the preview's width; 0 keeps full resolution |
| `--progress-json` | off | NDJSON events on stdout instead of text, for a parent process — see below |

### Quality guide for VR output

VR headsets magnify each source pixel far more than a monitor does, so
compression artifacts (especially banding in skies/fog/walls) are much more
visible than in desktop viewing.

| Use case | Recommended settings |
|---|---|
| Quick tests / iteration | `--crf 18` (default) |
| Final output for VR viewing | `--crf 15 --preset slow` |
| Archival / master quality | `--crf 13 --codec libx265 --bitdepth 10` |

### 10-bit output, and what it actually costs

The pipeline works in 8-bit RGB, so `--bitdepth 10` does not add source
precision. What it does is stop the *output* stage from throwing precision
away, which is a real effect worth understanding.

**Time.** Measured at 8K top-bottom on real footage, encoder time isolated:

| Config | 8-bit | 10-bit | |
|---|---|---|---|
| x264 medium crf18 | 0.336 s/frame | 0.500 | 1.49× |
| x265 medium crf18 | 0.400 | 0.578 | 1.45× |
| x265 slow crf15 | 1.133 | 1.714 | 1.51× |

So ~50% more encode time, consistently. But the encoder overlaps the pipeline
(0.80 s/frame at 8K), so at `medium` **both depths finish inside that window
and 10-bit costs nothing in wall-clock**. Only at `slow` does it reach the
critical path. 10-bit files are also 11–16% *smaller* at the same CRF, because
higher internal precision leaves smaller residuals.

**A source deeper than 8 bits is flattened, and the run says so.** `decode_frames`
asks ffmpeg for `rgb24` and the encoder is fed `rgb24`, so a 10-bit input loses
its precision at decode — before any pipeline stage sees it. Measured on a
10-bit gradient: **161 distinct levels in, 35 out**. `--bitdepth 10` does not
rescue it and is the more misleading case, since the file really is Main10 and
looks like it kept something — the same test gave 41 levels, a minimum step of
one whole 8-bit code. Both cases now emit a warning naming the source depth
(`source_bit_depth` / `precision_preserved` in `--progress-json`). Unlike
chroma, bit depth has no follow-the-source option, because there is nothing to
follow it with.

#### Future work: dithering a 10-bit source

Not implemented. Dithering the 10→8 bit reduction would trade the banding for
noise, which is the standard fix and looks promising on the numbers below — but
it interacts with stereo in a way that needs testing in a headset before it
would be worth shipping. Recorded here so the groundwork is not repeated.

**It does recover the banding.** TPDF dither (±1 LSB, triangular, so the
quantisation error stops correlating with the signal) on a shallow test
gradient:

| | banding (low-passed error) |
|---|---|
| 10-bit truth | 0.3724 |
| truncated to 8-bit | 0.4657 |
| **dithered to 8-bit** | **0.3725** |

Essentially all of the 8-bit quantisation banding, gone.

**Compression takes it back, and the direction is the intuitive one reversed.**
Dither is low-amplitude high-frequency noise — exactly what a quantiser
discards first — so a *high* CRF strips it and a *low* CRF preserves it.
Fraction of the dither surviving the encode:

| CRF | 12 | 15 | 18 | 20 | 23 | 28 |
|---|---|---|---|---|---|---|
| x264 | 100% | 100% | 67% | 39% | 33% | 13% |
| x265 | 100% | 70% | 46% | 23% | 21% | 23% |

Banding returns in step: by CRF 28 the dithered encode measures the same as the
undithered one. x265 strips it sooner than x264, which is consistent with it
being the better denoiser. So dithering is only worth doing at the CRFs this
project already recommends for VR, and is wasted effort above roughly x264 20 /
x265 18.

**Bitrate cost is negligible on real footage: 0–4%**, and encode time is
unchanged. A synthetic flat gradient suggests a ~90× bitrate increase, which is
true and meaningless — a flat gradient compresses to nothing, so noise added to
it is the entire file. Real 8K footage is already full of grass and bark, and
±1 LSB disappears into what the encoder is spending bits on anyway.

**CPU cost is the real obstacle if we do it ourselves.** Measured at 8K:
generating fresh TPDF noise is 0.42 s per channel per frame (0.24 s in
float32), and the add-clip-cast over a full RGB frame dominates at **1.66
s/frame** — against a current budget of about 2 s/frame, so a naive
implementation nearly doubles render time. A precomputed 256×256 tile drops
generation to 0.026 s per channel, but the per-pixel arithmetic remains. The
GPU warp path is already there and would be the obvious place to put it.

**ffmpeg may do it for free, but not the way we currently decode.** The plain
`-pix_fmt rgb24` request does *not* dither (high-frequency noise measured 0.051
levels), while routing the depth reduction through an explicit `format` filter
does (0.524 levels, banding 0.544 → 0.444). The `-sws_dither` and
`zscale=d=...` options made no difference in any position tried — the filter
path dithers by its own default and ignores them. Worth pinning down the exact
chain: doing it inside ffmpeg costs us no CPU and no memory, whereas doing it
in numpy needs an `rgb48le` decode (double the pipe payload, plus a transient
uint16 frame — 177 MB at 8K).

**The open question is stereo, and it is the reason this is not just a flag.**
Where the dither is applied decides whether the two eyes get the *same* noise
or different noise — and independent inter-eye noise is the rivalry mechanism
measured under [Quality guide for VR output](#quality-guide-for-vr-output),
which shows up far below the amplitude at which common-mode noise is noticed.

* Dither **before** the warp and the right eye receives a resampled copy of the
  left eye's noise: correlated, but bilinear interpolation low-passes and
  attenuates it, so one eye ends up grainier than the other.
* Dither **after** the warp, per eye, and the two eyes get fully independent
  noise — the worst case.

There is a temporal axis too: fresh noise every frame will crawl, and a static
tile will not but adds a fixed pattern that head motion drags across the view.
At 21.3 px/deg the noise sits near 10.7 cycles/degree with about 0.2% contrast,
which is close enough to the detection threshold that it has to be judged in a
headset rather than argued from numbers.

**Why it helps an 8-bit source.** Limited range ("tv"), the delivery standard,
maps 0–255 RGB into Y 16–235 — only 220 levels, *fewer than the 8-bit source
has*. That loss happens in the colour conversion, before the encoder runs;
measured, 85% of a gradient's levels survived. 10-bit limited range is Y
64–940, 877 levels, so the compression stops merging them. What 10-bit cannot
do is recover banding already present in the source.

**The catch.** The benefit is only realised by a playback chain that keeps the
extra bits. Decoded to 16-bit, a 10-bit file measured **4.1× more faithful**
than an 8-bit one; decoded back down to 8-bit RGB it measured *worse* (rmse
2.10 vs 1.29), because the 10→8 crush costs more than the 8-bit path ever
pays. Headsets that decode Main10 into a 10-bit surface win; a chain that
flattens to 8-bit does not.

### Colour handling

Frames are decoded to RGB, processed, and re-encoded, so the source's YUV
never passes through untouched. Two things follow, and both were wrong until
they were measured:

* **The output carries the source's colour matrix and range.** Left untagged,
  a 7680-wide file gets read as BT.709 by essentially every player; against a
  `smpte170m` source that guess measured rmse 2.03 with up to 8 levels of
  error — a systematic shift over the whole frame. Tagging brings it to 1.37.
  (Only matrix and range survive: neither x264 nor x265 writes transfer or
  primaries into the VUI whatever they are asked for. Those two are the pair
  that decides how the image decodes.)
* **Scaling uses `+accurate_rnd+full_chroma_int`.** swscale's default rounding
  is the fast one, and it costs real accuracy: 8-bit output measured rmse
  1.510 with the defaults against 1.285 with accurate rounding. It is free —
  0.99× wall clock at 8K and byte-identical files — so it is always on.

### Input projection

360 files can declare their projection, and this tool reads it. Google's
Spherical Video **V2** puts it in an `sv3d` › `proj` box, one of `equi`
(equirectangular), `cbmp` (cubemap: 3×2 layout plus per-face padding) or
`mshp` (an explicit mesh). **V1 cannot express anything but equirectangular** —
the spec says `ProjectionType` "Must be `equirectangular`" — so only V2 helps.

Equi-Angular Cubemap has **no standard box**: it is not mentioned in the V2
spec and `cbmp` defines only `layout = 0`, the plain 3×2 grid. EAC is a
delivery format YouTube produces server-side and signals through its own
pipeline, not something a file can be tagged as.

What we do with it:

* **Untagged** — assumed equirectangular. Most files declare nothing, and it
  is the only projection V1 can express or YouTube accepts on upload.
* **`cbmp`** — read as a 3×2 cubemap. The file's own faces go straight to
  depth estimation instead of being rebuilt, which skips resampling to
  equirect and back for the depth stage; the left eye is reconstructed once at
  4× the face size. Padding declared by the box is cropped away.
* **Anything else** (`mshp`, an unknown layout) — the run **stops** with a
  message. Treating a non-equirect projection as equirect produces output that
  is geometric nonsense from the first frame, and previously did so silently.

`--input-projection` overrides the tag when a file is labelled wrongly.

The 3×2 face mapping was derived by matching ffmpeg's `v360=e:c3x2` output
tile by tile against our own `equirect_to_cubemap` of the same frame, rather
than read off the spec prose: the correct assignment scored 1–2 mean levels
against runner-ups at 30–37. A wrong face order or rotation looks fine in a
thumbnail and is badly wrong in a headset.

Note that ffmpeg's own spherical side data has no mesh concept — a `mshp`
file is reported by ffprobe as `projection: equirectangular`, so the tag alone
cannot always be trusted.

### Hardware encoders

`--codec` accepts NVENC, Quick Sync and AMF as well as x264/x265, and each
family speaks a different dialect for the same two settings — `-cq` for NVENC,
`-global_quality` for QSV, `-rc cqp -qp_i` for AMF, against x264's `-crf`. You
keep using `--crf` and `--preset`; the translation happens in `ffmpeg_io`.

Which families exist at all is a platform question, so the candidate list
spans all of them and anything this ffmpeg was not built with is simply not
offered — a Windows build has no VideoToolbox, a macOS one has no AMF:

| Family | Platform | Notes |
|---|---|---|
| NVENC | Windows, Linux | NVIDIA |
| Quick Sync | Windows, Linux | Intel |
| AMF | Windows | AMD |
| **VAAPI** | **Linux** | AMD and Intel; needs a render node |
| **VideoToolbox** | **macOS** | Apple Silicon and Intel Macs |

VAAPI is the odd one: it encodes from GPU surfaces, so unlike the others it
needs a device opened before the graph and an explicit upload
(`-vaapi_device /dev/dri/renderD128`, `-vf format=nv12,hwupload`). Set
`STEREO360_VAAPI_DEVICE` if your render node is numbered differently. The probe
builds its trial command from the same helpers the real encoder uses, so it
cannot report an encoder usable under flags the render will not use.

**Availability depends on the output resolution**, which is why it is probed
rather than assumed. Measured on one machine (RTX 5070 Ti + Ryzen iGPU):

| Encoder | 3840×3840 | 7680×7680 |
|---|---|---|
| libx264 / libx265 | ✅ | ✅ |
| hevc_nvenc | ✅ | ✅ |
| h264_nvenc | ✅ | ❌ NVENC's H.264 engine stops at 4096² |
| hevc_amf / h264_amf | ✅ | ❌ rejected at that size |
| hevc_qsv / h264_qsv | ❌ no Intel device here | ❌ |

A top-bottom stereo frame is twice the height of its source, so it lands on the
wrong side of those limits far more often than ordinary video does — `hevc_amf`
handles a 4K project and cannot touch an 8K one.

```bash
python -m stereo360 - --probe-encoders 7680x7680
```

Each candidate is asked to encode one real frame at that exact size; the whole
sweep takes about 2.5 s. The desktop UI runs it once the input's dimensions are
known and populates its **Encoder** list from the answer, marking hardware
entries *not recommended* and listing unusable ones disabled with the reason.

VAAPI and VideoToolbox are implemented but **untested on real hardware** —
there is no Linux or macOS machine here. The probe is the safety net: if the
dialect is wrong they report unavailable rather than failing a render.

They are marked that way because they are the speed option, not the quality
one. At 8K, `hevc_nvenc` at cq19 runs 0.28 s/frame against `libx265 slow`
crf13's 1.90 — but those are not the same picture, and for anything being kept
the CPU encoders win on quality per bit.

### Chroma subsampling

Output is **4:2:0** by default. `--source-subsampling` follows the source
instead, for masters and uploads where the consumer is a transcoder rather than
a headset's hardware decoder.

4:2:0 is the default because it is the only layout headset hardware decoders
accept at 8K — a 4:4:4 stream falls back to software decode, which at
7680×7680 is unusable. That is a constraint on *delivery*, not on mastering,
so it is a default rather than a rule.

But measured against a real 7680×7680 frame from this pipeline, following a
4:2:0 source's subsampling changes nothing, and even for a hypothetical 4:4:4
source the gain is small:

| pix_fmt | rmse vs truth |
|---|---|
| yuv420p (8-bit) | 1.9032 |
| **yuv420p10le** | **0.8732** |
| yuv444p (8-bit) | 1.0016 |
| yuv444p10le | 0.8407 |

**10-bit 4:2:0 beats 8-bit 4:4:4.** Bit depth is worth 2.2× here; chroma format
is worth 4% on top of it, for 25% more encode time, and it moves HEVC from the
widely supported Main 10 profile to Rext. The reason the gain is so small is
that the source was already 4:2:0 — its chroma detail was halved before we saw
it. For a genuinely 4:4:4 source the calculus would differ, which is why the
option exists.

Where the layout cannot be produced — an exotic source format, or a hardware
encoder, which stays on 4:2:0 here — the run continues at 4:2:0 with a warning
rather than failing.

Be aware that the slower presets make the **encoder** the bottleneck rather
than the pipeline. Measured on an 8K top-bottom render (pipeline: 0.80 s per
frame):

| Setting | Encoder cost | |
|---|---|---|
| `libx264 medium --crf 18` (default) | 0.32 s/frame | hidden by overlap |
| `libx265 medium --crf 18` | 0.49 s/frame | hidden by overlap |
| `libx265 slow --crf 15` | 1.26 s/frame | **bottleneck** |
| `libx265 slow --crf 13 --bitdepth 10` | 1.90 s/frame | **bottleneck** |

So the archival recipe roughly triples wall-clock time. `hevc_nvenc` costs
0.28 s/frame and disappears back into the overlap, if you will accept the
quality trade against x265 at those settings.

### Previewing a single frame

`--strength`, `--gradient-limit` and `--depth-tiles` all change how the result
looks, and none of them can be judged from the number. `--preview-frame`
renders one source frame as an image so a setting takes seconds to evaluate
instead of a full render:

```bash
python -m stereo360 input.mp4 -o preview.png --preview-frame 45 --strength 1.5
```

The output is the top-bottom stereo pair exactly as the video would contain
it — the same code path as the real conversion, so what you judge is what you
get. It is downscaled to 2048 px wide by default (a full 8K preview is a
7680×7680 image that costs seconds to encode for nothing you can see);
`--preview-width 0` gives full resolution.

The frame index is absolute in the source and ignores `--start-frame`. On the
8K test footage a preview takes about 6 seconds, most of which is loading the
depth model rather than rendering.

One caveat: with `--depth-backend video-depth-anything`, depth comes from a
chunk of consecutive frames, so a single-frame preview has no temporal context
and will differ slightly from the render. stereo360 warns when you do this.

### Tiled depth inference

```bash
python -m stereo360 input.mp4 -o output.mp4 \
  --depth-backend video-depth-anything --depth-tiles 4
```

Depth models run at ~518px internally, so a 1920² cubemap face is
downsampled ~3.7× and thin structures (curtains, railings) fall between the
model's 14-px patches — visible as blocky squares in the right eye. With
`--depth-tiles N`, each face is split into N×N overlapping perspective
sub-crops (valid perspective views — no reprojection needed), each inferred
near native resolution, then feather-blended back. Since each tile gets its
own relative depth scale, a coarse full-face pass is run first and every
tile is scale-aligned to it (least squares) before blending. `--depth-tiles 4`
gives 480² tiles at 8K (no downsample at all); cost scales with N².

### Temporal hole filling

```bash
python -m stereo360 input.mp4 -o output.mp4 \
  --depth-backend video-depth-anything --temporal-fill --inpaint learned
```

Disoccluded background is usually *visible* in neighboring frames. With
`--temporal-fill`, each chunk is warped without inpainting first, and every
hole pixel is filled from the median of the colors other frames produced at
the same pixel (accepted only when the frames agree). Only holes with no
temporal consensus go to Telea/LaMa — this removes the per-frame fill
flicker that purely spatial inpainting cannot avoid.

This matters most for thin near structures, where disocclusion is far larger
than the object. The hole a structure opens behind it is
`baseline * (fg_disparity - bg_disparity)` wide, independent of how wide the
structure itself is: at 8K and `--strength 1.0`, a 3-px railing against a far
background opens a ~26-px hole — nine times its own width, none of which
exists in the left eye. Spatial inpainting must invent all of it, and invents
it differently every frame, which reads as holes crawling around the railing.
Temporal fill replaces the invention with real pixels wherever another frame
in the chunk actually saw that background.

It only helps where something moved enough to reveal the background. With a
static camera and a static object the same strip stays hidden in every frame,
and those pixels still fall through to `--inpaint`. The other lever there is
`--strength`: hole width scales linearly with it.

### Learned inpainting (M5)

```bash
pip install simple-lama-inpainting
python -m stereo360 input.mp4 -o output.mp4 --inpaint learned
```

LaMa synthesizes plausible texture in disocclusion holes instead of diffusing
boundary colors. Holes are filled per connected component (padded crops), so
memory stays bounded at 8K. CPU reference: ~75 s per 1920×960 frame; use a
GPU for practical throughput.

### Which depth backend?

`--depth-backend` defaults to `auto`, which probes what is actually installed
and reports it on startup:

```
GPU accelerated: Depth Anything V2 on CUDA (NVIDIA GeForce RTX 5070 Ti, fp16)
```

or, when nothing better is found:

```
**WARNING** GPU not being utilized. CPU accelerated: Depth Anything V2 on CPU (torch). Please read the README file.
```

That warning is deliberately loud. Depth on the CPU is roughly an order of
magnitude slower, and the failure mode this avoids is a user concluding the
tool is slow when really their accelerator was never picked up — usually a
CPU-only torch wheel, or `onnxruntime` installed instead of
`onnxruntime-gpu` / `-directml`.

If the accelerator genuinely is not there, depth becomes roughly 90% of the
frame, so see [Fast mode for slow machines](#fast-mode-for-slow-machines):
it trades depth detail for a 3-4x cut in exactly that cost.

Probe order is torch GPU (CUDA, then MPS) before ONNX. Torch leads because
where it has a GPU it is at least as fast and keeps `--depth-model` available
for larger variants; ONNX follows because it reaches hardware torch cannot —
AMD and Intel GPUs on Windows have no ROCm torch at all. To pin a choice
rather than probe, name the backend explicitly.

| your hardware | use | why |
|---|---|---|
| NVIDIA GPU, CUDA torch | `depth-anything` (default) | already GPU-accelerated, fp16, and `--depth-model` can pick a larger variant |
| AMD / Intel GPU on Windows | `onnx` | torch has no ROCm on Windows, so DirectML via ONNX is the only GPU path |
| no GPU, or want to avoid a ~2.5 GB CUDA torch install | `onnx` | ONNX Runtime is far smaller and still uses whatever accelerator exists |
| flicker is the priority, torch GPU available | `video-depth-anything` | temporal model, most consistent depth across frames |

`onnx` additionally needs `pip install onnxruntime` (or `-directml` / `-gpu`) —
it is not in `requirements.txt` — plus a one-time `scripts/export_onnx.py`,
since the graph is not shipped. Both are why it cannot be the default without
breaking a fresh clone; the backend now says exactly that if the model is
missing rather than raising a bare `NO_SUCHFILE`.

### Cube seams in the depth map

Depth is estimated on six cubemap faces rather than on the equirect frame,
because a perspective view is what the models were trained on. Six exact 90°
faces tile the sphere with no slack, though, so neighbouring faces share
nothing but a line — and relative-depth models emit an arbitrary scale per
inference, so the same ground arrives at that line with two different scales.

With only a one-pixel strip to fit the correction from, whatever disagreement
survives lands exactly on the seam. Measured at 8K, the depth step across a
seam was **259× the step between ordinary neighbouring pixels** (worst at the
front/down edge). In a headset that reads as the ground creasing, with the
patch above the crease detaching and floating forward.

The fit was never going to close it, because the mismatch is not affine: the
down face spends its whole range on the few metres of ground around your feet
while the front face spends its range on everything out to the horizon. A
border strip is also the worst possible place to sample — it is where a
monocular model is least reliable, it is thin enough for one occluding contour
to dominate, and the values in it sit at the extreme end of each face's own
depth distribution.

So each face is widened past 90° (`--face-overlap`, default 0.15 → 98°).
Neighbours then genuinely share a band: the scale fit gets a 2-D sample
instead of a strip, and the assembly cross-fades across the band, so any
residual is spread over degrees rather than concentrated into a line.

| | median seam step | vs. ordinary neighbours |
| --- | --- | --- |
| exact 90° faces | 0.0276 | 313× |
| **98° faces, cross-faded** | **0.0004** | **3.5×** |
| 103° faces, cross-faded | 0.0002 | 2.3× |

As a fraction of the disparity range, frame 0 of 8K footage, Depth-Anything-V2
Small. The cross-fade does most of that on its own — with the scale fit
disabled entirely it still reaches 5.4× — and the fit takes it the rest of the
way. The Large model has a *worse* seam problem to begin with (652×), because
its per-face depth is more precisely calibrated and its noise floor lower.

#### Why not wider

Widening is bounded on both sides, and the reason 90° was the original choice
is real: it is close to a normal lens, which is what monocular depth models
were trained on. A tangent-plane projection stretches its corners, and the
stretch grows fast — a face corner sits 54.7° off-axis with a 5.2× area
stretch at 90°, 58.4° and 7.0× at 98°, and 60.5° and 8.4× at 103°.

Measured rather than assumed, using geometry as ground truth: for **any**
plane, a ray in direction `d` meets it at distance `-h/(d·n)`, so inverse depth
is exactly *linear in the direction vector*. Fitting predicted depth over real
flat ground against `(dx, dy, dz)` and looking at what is left over therefore
measures how much the model's answer has been bent, with no reference model to
argue about. Centre angular resolution held constant, inliers picked once from
the 90° render so no width gets an easier set of pixels, residuals broken down
by angular radius since distortion is a corner effect:

| ground-plane error vs. 90° | 98° | 103° |
| --- | --- | --- |
| Small | 0.76–1.04× | 0.79–1.05× |
| Base | — | 0.67–0.89× |
| Large | 1.00–1.07× | 1.13–1.20× |

Two clips, three frames each. At 98° nothing measurably degrades on any model.
At 103° the Large model loses 13–20% — and buys nothing for it, since the seam
is already dead at 98° and the 95th-percentile seam step is if anything
slightly better there (0.0037 against 0.0038). Smaller models come out *ahead*
of 90°: the extra context helps them more than the distortion hurts.

98° also gives up less angular resolution — 9% against 15% — since the model
resizes its input whatever field of view it covers. That is the one real cost
of this approach, and it is the reason the overlap is no wider than it needs
to be.

Runtime is unchanged. The blend tables depend only on the output geometry, so
they are built once per run (~2.5 s at 8K, ~450 MB) rather than per frame; a
24-frame 8K render measured 45.5 s against 45.4 s with exact faces.

### Stereo geometry: the baseline follows your gaze

The right eye is offset to the side **of whichever direction is being looked
at** — omnidirectional stereo, the convention 360 rigs record in and headsets
expect. Offsetting along one fixed world axis instead looks reasonable and is
badly wrong away from the equirect centre. For a point at distance `L` in
direction `(lon, lat)`, a fixed +X offset gives a longitude shift of
`-b·cos(lon)/L`:

| longitude | disparity | |
| --- | --- | --- |
| 0° (equirect centre) | −10.08 px | full, correct |
| ±90° (sides) | 0.00 px | **no stereo at all** |
| 180° (behind you) | +10.08 px | full size, **inverted** |

Measured at 3840 px wide; the analytic and empirical figures agreed to 0.1 px.

Inverted disparity is pseudoscopic — near reads as far — while occlusion,
perspective and texture gradient all still say near. The eyes get a depth that
contradicts every other cue, and it is worst at a high-contrast occluding edge,
which is exactly where it is least ignorable. Reported from a Quest 3 as two
regions being painful to look at while the same scene straight ahead was fine;
both sat behind the viewer.

On real 8K footage, parallax separating an object from its own background
(negative = near object sits in front, which is what the front of the scene
does and what occlusion agrees with):

| region | longitude | fixed +X | ODS |
| --- | --- | --- | --- |
| pond, rock, thin tree | +22° | −3.41 px | −5.45 px |
| bench right armrest | +142° | **+16.39 px** | −19.69 px |
| tree trunks against sky | +172° | **+9.26 px** | −9.25 px |

Sideways-of-the-view-direction is `(cos lon, 0, -sin lon)`, and the horizontal
part of a unit direction is `(cos(lat)·sin lon, _, cos(lat)·cos lon)` — so the
offset is just the direction's own horizontal components, swapped and one
negated, times the baseline. No trig, no separate taper, and algebraically it
is a rotation of the horizontal plane by `atan(baseline/distance)`. The
longitude shift comes out uniform over the entire sphere and vertical disparity
stays second order (0.04 px at 8K).

The `cos(latitude)` taper is still there and still needed: equirect meridians
converge, so an untapered offset costs `1/cos(lat)` pixels of longitude — 58×
the equator's shift at lat 89° — and the polar cap does not translate but
*folds*, driving opposite longitudes onto one meridian. Its behaviour is
unchanged by this fix; only the longitude dependence changed.

One consequence worth noting: disocclusion hole filling read its direction from
the same `cos(lon)` term, so it reversed across half the sphere and had no
direction at all near ±90° — meaning that where disocclusions are widest, the
background was being continued in from the foreground's side. It is now one
constant for the frame.

### Temporal stability

Depth models return *relative* depth, so the map is normalised before it
becomes a disparity. Recomputing that range per frame rescales the whole
disparity field every frame, which moves stationary objects sideways — the
image appears to wobble. Measured on 8K footage the 1st–99th percentile span
swung 20% across six consecutive frames.

The chunked path shares one range across a chunk; the single-frame path now
eases the range with an exponential average (~1s at 30fps), so a real scene
change is still followed but per-frame jitter is not. Measured by tracking
fixed points with sub-pixel correlation, frame-to-frame disparity jitter fell
from **0.399px to 0.215px**, which is about what the temporal backend
achieves (0.243px).

### Which depth model?

Measured on frames 45-47 of 8K footage, depth stage only, on an RTX 5070 Ti.
There is no ground-truth depth, so this reports what *can* be measured: noise
where the image is flat (depth should be flat there too), sharpness at real
depth discontinuities, and how much the depth changes frame to frame.

| Model | s/frame | noise | edge sharpness | sharp/noise | depth flicker |
|---|---|---|---|---|---|
| **Small** | 0.195 | 0.108 | **214.9** | 1995 | 0.00701 |
| **Base** (default) | 0.250 | **0.071** | 180.2 | **2546** | 0.00410 |
| **Large** | 0.351 | 0.074 | 169.7 | 2298 | 0.00441 |
| ONNX 266 (fast) | — | 0.089 | 122.0 | 1369 | 0.00320 |

**Base is the best point on this curve, and is the default.** It has the
lowest depth noise and the best sharpness per unit noise, and **40% less
frame-to-frame flicker than Small** — which is the mechanism behind thin
structures shifting shape between frames. Depth is only about 30% of the
pipeline at 8K, so it costs far less than the +28% the depth column suggests:
measured end to end on a 24-frame 8K render, **1.54 s/frame against Small's
1.47, or +5%**. Large came in at 1.64 s/frame.

Pass `--depth-model depth-anything/Depth-Anything-V2-Small-hf` for the faster
run. The temporal backend is unaffected — it ships small and large only, and
still defaults to small. So does `--depth-backend onnx`, which loads whatever
was exported to `models/` rather than a Hub id.

**Large is not worth it here**: twice Base's extra cost, and no better on any
axis measured. **Small is the sharpest but the noisiest**, and that noise is
what moves between frames.

Treat any single column with suspicion. Flicker alone ranks the *blurriest*
model best — ONNX 266 wins that column while losing real detail. An earlier
attempt at a one-number "edge alignment" score ranked Small above Base, the
reverse of what the depth maps plainly show, because it rewarded Small's noise.
The columns only mean something together.

### Fast mode for slow machines

You do not need a different, weaker depth model — the same Depth Anything V2
Small runs at whatever input resolution you export it at. It is a ViT, so its
cost is driven by the number of 14x14 patches, not by its weights: halving the
input quarters the patch count.

```bash
# for CUDA or CPU
python scripts/export_onnx.py --size 266 --out models/depth_fast.onnx
# for DirectML (AMD/Intel on Windows), which needs a static batch axis
python scripts/export_onnx.py --size 266 --static-batch --out models/depth_fast.onnx

python -m stereo360 in.mp4 -o out.mp4 --depth-backend onnx --onnx-model models/depth_fast.onnx
```

In the desktop UI this is under **Depth**: set Backend to `onnx`, then pick the
exported file with the **ONNX model** Browse button. The row only appears once
the ONNX backend is selected, because that is the only configuration that reads
it — with `auto` on a machine that has a torch GPU, the torch backend wins and
the ONNX file is never opened.

`export_onnx.py` verifies the graph it just wrote by running it at batch 1, 2
and 6 — an export that cannot take the six cubemap faces the pipeline sends is
worse than a failed one, because nothing says so until a conversion is under
way.

The backend reads the resolution out of the graph, so nothing else changes.
Measured on an 8K frame, depth stage only:

| export | on CPU | on GPU (DirectML) | depth vs native | fine detail |
|---|---|---|---|---|
| 518 (native, default) | 3.54 s | 1.50 s | — | 100% |
| **266** | **1.28 s (2.8x)** | 0.80 s (1.9x) | corr **0.989** | 61% |
| 154 | 0.93 s (3.8x) | — | — | — |

The speedup is much larger on a CPU-only machine, which is the case it exists
for: on a GPU, inference is only part of the depth stage, so the ceiling is
lower. Note the quality shape — depth correlates at **0.989** with the native
model, so the overall scene structure is essentially intact, and what is lost
is fine detail (61%): thin railings and small objects get coarser depth. Hole
count stayed at zero, because `--gradient-limit` handles the sharper edges
either way.

Faster alternatives like MiDaS-small exist, but adding one means another
architecture, another download and another preprocessing path — for a smaller
gain than simply exporting the model you already have at a lower resolution.

### ONNX Runtime backend (M4)

```bash
pip install onnxruntime        # or onnxruntime-directml / onnxruntime-gpu
python scripts/export_onnx.py  # one-time export (~100 MB + .onnx.data)
python -m stereo360 input.mp4 -o output.mp4 --depth-backend onnx
```

Auto-selects the best execution provider (CUDA → DirectML → CoreML → CPU).
DirectML gives GPU acceleration on AMD/Intel GPUs on Windows without a
PyTorch install. CPU reference: ~3.2 s per 1920×960 frame end-to-end.

### Temporal depth (M3)

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything third_party/Video-Depth-Anything
python -m stereo360 input.mp4 -o output.mp4 --depth-backend video-depth-anything
```

(The upstream repo has no pip package; clone it into `third_party/` or set
`VIDEO_DEPTH_ANYTHING_PATH` to an existing clone. Checkpoint weights download
from the HuggingFace hub on first run.)

The video model enforces depth consistency across frames within each chunk;
chunk boundaries are re-estimated with overlap and ramp-blended to hide seams.
The guided-filter smoothing stays active as a light edge-aware cleanup.

### GPU note

The default `pip install torch` wheel is CPU-only on Windows. For NVIDIA GPUs,
install the CUDA build for a large speedup:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

The output is a top-bottom stereoscopic MP4 with the original audio copied and
Google spherical metadata (`StereoMode=top-bottom`) injected so VR players
auto-detect the format.

### Spatial media metadata

The output carries what Google's Spatial Media Metadata Injector writes, so it
should not need running afterwards:

| Injector checkbox | Written | Where |
|---|---|---|
| My video is spherical (360) | `GSpherical` RDF/XML **and** `sv3d`/`svhd`/`proj`/`prhd`/`equi` | video `trak` / video sample entry |
| My video is stereoscopic 3D (top/bottom) | `StereoMode>top-bottom` **and** `st3d` (mode 1) | same |
| My video has spatial audio (ambiX ACN/SN3D) | `SA3D` | audio sample entry, only with `--spatial-audio` |

The first two are unconditional: this tool only ever produces stereoscopic 360,
so there is nothing to decide. Both Spherical V1 (the legacy XML blob, which
most desktop players still read) and V2 (the boxes newer players and headsets
prefer) are written, because some players honour only one of them.

Ambisonic audio is a property of the *source*, so it has to be declared:

```bash
python -m stereo360 in.mp4 -o out.mp4 --spatial-audio
```

The ambisonic order comes from the audio track's channel count — 4, 9 or 16
channels for first, second or third order. Anything else is refused rather
than guessed, since labelling stereo as ambisonics breaks playback in a way
that is hard to trace back. Audio is copied through with `-c:a copy`, so ambiX
in the source survives, but this flag cannot create ambisonics that were never
there.

Placement is what matters here: the V1 box belongs inside the video `trak` and
the V2 boxes inside the video sample entry. An earlier version wrote V1 at the
top level of `moov`, where a conforming parser does not look — which is exactly
the symptom of a file that still needs the injector run by hand.

### Input resolution

Nothing is tied to 8K. Face size defaults to input width / 4, the critical
gradient behind `--gradient-limit` is `2*pi / (baseline * width)` and so
normalises itself, and thread count, chunk length and the encoder's memory
limits all come from the frame size and the machine's free memory. 4K input
works with no flags:

| input | output | face size | critical gradient | holes |
|---|---|---|---|---|
| 3840×1920 | 3840×3840 | 960 | 0.0545 | 0.000% |
| 7680×3840 | 7680×7680 | 1920 | 0.0273 | 0.000% |

## Desktop interface

```bash
pip install -r requirements-ui.txt
python -m stereo360_ui
```

A settings panel, a one-frame preview, live progress with an ETA, and a Stop
button that leaves you a playable file. The core runs as a child process, so a
depth model running out of VRAM cannot take the window down with it.

The interface is a thin shell over the CLI — it builds a command, runs it, and
reads the `--progress-json` stream. It shows the command it is about to run in
the log, so anything you set up here can be pasted into a terminal.

**Why PySide6/QML.** Measured on a Windows desktop, comparable windows:

| Toolkit | Memory | Ship size |
|---|---|---|
| PySide6 Widgets | 72 MB | ~28 MB |
| **PySide6 QML** | **101 MB** | ~48 MB |
| Slint | 110 MB | 27 MB |
| WebView2 (Tauri) | ~400 MB | ~10 MB |
| Electron | 600 MB+ | ~150 MB |

Tauri's small binary still drives a full browser engine, which is where the
400 MB goes. That matters more here than in most apps: `fit_chunk_size` and
the encoder guard both size themselves from free memory, so a heavy UI costs
chunk frames on a machine that is already short of them.

`python -m stereo360_ui --selftest --shot ui.png` loads the interface, renders
it, saves a screenshot and exits — QML errors are runtime-only, so this is
what turns a typo into a failed command rather than a broken window.

### Driving stereo360 from another program

`--progress-json` replaces the human-readable output and progress bar with
NDJSON on stdout — one JSON object per line, flushed as it happens — so a GUI
or build script never has to scrape prose or parse a progress bar:

```bash
python -m stereo360 input.mp4 -o out.mp4 --progress-json
```

```json
{"message": "GPU accelerated: Depth Anything V2 on CUDA (RTX 5070 Ti, fp16)", "backend": "depth-anything", "device": "cuda", "gpu": true, "type": "info"}
{"total": 92, "width": 7680, "height": 7680, "fps": 29.97, "input": "input.mp4", "output": "out.mp4", "face_size": 1920, "type": "start"}
{"frame": 5, "total": 92, "elapsed": 7.4, "fps": 0.676, "eta": 133.2, "type": "progress"}
{"output": "out.mp4", "frames": 92, "cancelled": false, "elapsed": 136.1, "type": "done"}
```

Event types are `info`, `warning`, `start`, `progress`, `done` and `error`.
Every message carries structured fields beside the text, so the numbers never
have to be recovered from the prose. Progress is throttled to ~10 events per
second, but the first and last frames always report. Failures arrive as an
`error` event and exit 1, instead of a traceback a parent process cannot use.

**Cancelling.** In this mode a line reading `cancel` on stdin stops the run
cleanly. The encoder is closed rather than killed, so ffmpeg finalizes the
file: what you get is a shorter but completely valid video containing the
frames finished so far, with the spherical metadata written. The exit code is
130. Ctrl-C does the same thing in normal terminal use.

EOF on stdin is *not* a cancel, so spawning with stdin closed is safe.

As a library, `pipeline.convert()` takes the same two hooks directly:

```python
from stereo360 import backends, pipeline
from stereo360.events import Reporter

built = backends.build(depth_backend="auto")       # resolves "auto" for you
result = pipeline.convert("input.mp4", "out.mp4",
                          depth_backend=built.backend,
                          chunk_size=built.chunk_size,
                          reporter=MyReporter(),    # subclass of Reporter
                          cancel=stop_event.is_set)
print(result.frames_written, result.cancelled)
```

With no `reporter` it prints nothing at all.

## Development

```bash
python -m pytest tests/ -v
```


### Eliminating disocclusion (rather than filling it)

Disocclusion is not a rendering defect — it is where the warp `x -> x + s(x)`
stops being injective, which happens exactly when `ds/dx <= -1`. The shift is
proportional to depth, so that is a bound on the depth *gradient*, and two
flags act on it directly:

`--gradient-limit X` clamps the depth gradient to X times that critical slope
(a cone erosion, two running minima per row). Holes cannot form below the
limit. It only lowers values, so depth ordering and the full depth *range*
survive — unlike lowering `--strength`, which scales every depth cue down
whether or not it was causing holes. The cost falls entirely on sharp
structures, which get ramped and lose some pop.

`--split-baseline` warps both eyes by half the baseline in opposite
directions. Total disparity — and therefore the depth effect — is unchanged,
but each eye warps half as far, and the two eyes' holes fall in different
places, so binocular fusion suppresses what remains. It is a headset-only
benefit: viewed as flat images, two lightly damaged eyes look worse than one
pristine and one damaged.

Measured on one 8K frame, all at `--strength 1.0` (i.e. no loss of depth):

| configuration | hole area |
|---|---|
| default | 0.0705% |
| `--gradient-limit 1.0` | 0.0033% |
| `--split-baseline` (per eye) | 0.0094% |
| `--split-baseline --gradient-limit 0.5` | **0.0000%** |

The clamp acts on both axes. Rows alone are not enough: a handrail crossing
the frame has its depth cliff in the *vertical* direction, and the warp is not
purely horizontal — the pole taper shifts latitude by up to half the horizontal
shift, peaking around ±45°. That left a thin dashed line hugging such edges
until the vertical direction was clamped too (at twice the horizontal limit,
which is all the geometry requires).

Whatever survives is filled by continuing the background inward from the side
geometry says it lies on, which is deterministic and therefore stable frame to
frame; `--inpaint` only sees what that cannot reach.
