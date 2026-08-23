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
| `--depth-backend` | **depends on the input** | `depth-anything-v3` for a video, `depth-pro` for a still image — see [Choosing a depth model](#choosing-a-depth-model). Or force `auto` (probe the machine and use the fastest runtime available, printing which), `depth-anything` (per-frame V2 via torch), `video-depth-anything` (temporal, flicker-free), `onnx` (ONNX Runtime: DirectML/CUDA/CoreML, no PyTorch) |
| `--onnx-model PATH` | models/depth_anything_v2_small.onnx | Exported ONNX depth model (`python scripts/export_onnx.py`) |
| `--ort-provider` | auto | onnxruntime provider override, e.g. `DmlExecutionProvider` |
| `--fp16` | off | Half-precision inference for the video backend (GPU) |
| `--start-frame N` | 0 | Skip first N frames (resume into a new file; concat segments with ffmpeg) |
| `--inpaint` | simple | `simple` = OpenCV Telea (fast); `learned` = LaMa neural inpainting (best quality, slow on CPU) |
| `--no-temporal-fill` | (fill is **on**) | Disable filling holes from other frames in the chunk. On by default: real pixels another frame saw beat anything invented. Needs `--chunk-size` > 1 |
| `--depth-tiles N` | 1 | Split each cubemap face into N×N overlapping tiles for depth (feather-blended). Higher = finer depth on thin structures; N² times slower |
| `--face-overlap F` | 0.15 | How far each depth face reaches past its nominal 90° (0.15 = 98° per face), so neighbours share a band rather than only an edge. 0 restores exact faces, which creases the ground at a seam — see [Cube seams in the depth map](#cube-seams-in-the-depth-map) |
| `--face-angular-correction F` | 0 (off) | Pull each depth face's edges back onto their true rays, undoing the field of view V3 believes it has rather than the one it was given. 0.55–0.7 measured best, 1.0 overshoots. Costs ~20% of the depth range, so pair it with `--strength 1.2`. Measured on V3 only — see [The model's lens is narrower than the face it is given](#the-models-lens-is-narrower-than-the-face-it-is-given) |
| `--depth-model ID` | **per backend** | `small` for `depth-anything-v3` (`base`/`large` also accepted, and both measured worse — see [Choosing a depth model](#choosing-a-depth-model)); ignored by `depth-pro`, which ships one checkpoint. For `depth-anything`, a HuggingFace model id, default Depth-Anything-V2-Base: lowest depth noise and 40% less frame-to-frame flicker than Small for +5% render time — see [Which depth model?](#which-depth-model). The temporal backend ships `small`/`large` only and defaults to small |
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
| `--output-mode` | 360 | `360` = a full sphere per eye, stacked top over bottom. `vr180` = the middle 180 degrees, eyes side by side. Input must be a full 360 video either way |
| `--yaw DEG` | 0 | Which way the VR180 field points, in degrees of longitude, positive to the right. Free and lossless: it selects a range of columns rather than rotating anything. Only valid with `--output-mode vr180` |
| `--output-width W` | source width | Deliver a smaller frame than the source implies: 360 becomes WxW, vr180 Wx(W/2). Depth and warping still run at full resolution, so the result is supersampled rather than rendered small, and it costs the same time. Exists because 8K 360 output is 7680x7680, which no HEVC or H.264 level decodes. Applies to stills too, where the ceiling is not the decoder but everything downstream of it — an 11904x11904 stereo JPEG is 40 MB and more than many viewers will open |
| `--ambisonic-codec` | auto | How to write the soundfield back when `--yaw` rotates it: `libfdk_aac`, `aac`, `pcm_s24le`, `libopus`. auto takes the first your ffmpeg has, in that order |
| `--spatial-audio` | off | Describe the audio as ambiX ambisonics (ACN/SN3D) by writing an `SA3D` box. The source audio must really be ambiX: 4, 9 or 16 channels. Spherical + stereoscopic are always written and need no flag |
| `--split-baseline` | off | Warp both eyes by half the baseline in opposite directions. Same 3D, far less disocclusion per eye, and holes become monocular |
| `--preview-frame N` | off | Render source frame N as one image and stop — see below |
| `--preview-width W` | 2048 | Cap the preview's width; 0 keeps full resolution |
| `--progress-json` | off | NDJSON events on stdout instead of text, for a parent process — see below |
| `--version` | — | Print the version and exit. Matches the git tag from v1.0.1 on; v1.0.0 reports `0.1.0`, because the constant was not maintained until then |

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

#### It is not monotonic — more tiling can be worse

Measured on frame 0 of a tram-roof shot, judged on a Quest 3, on a scene with
both thin cables and long diagonal struts:

| | result |
|---|---|
| no tiling | cables kink where they cross a depth boundary |
| 2 | cables resolved but thinned; a tile seam lands on a diagonal strut and aliases it |
| **3** | **best — cables and strut both intact** |
| 4 | the strut's edge is almost gone |

The reason 4 is worse is the trade tiling actually makes: it raises detail per
tile by *shrinking the context* each tile sees. A thin cable only needs local
detail and gains. A long diagonal spans several tiles, each of which now
judges its depth from a fragment; the pieces disagree slightly and the
cross-fade softens the edge away. At 4×4 there is not enough of the strut in
any one tile to place it.

So the optimum moves with the scale of the structures in the shot, and the
"quality at any cost" framing above is wrong past a point: beyond it, tiling
costs quality as well as time. A shot with fine detail and no long diagonals
may well prefer 4.

#### It also moves with the model, and now the default is 1

Everything above was measured against Depth Anything V2, and stills defaulted
to 3 on the strength of it. Both models that replaced V2 as defaults are hurt
by tiling instead — same photo, same everything else:

| | chair gap (→ 1.0) | wall wobble (→ 0%) | floor rms (→ 0%) | time |
|---|---|---|---|---|
| Depth Pro, no tiling | 1.49 | **53.1** | 30.1 | **11 s** |
| Depth Pro, tiles 3 | 1.56 | 176.4 | **26.2** | 46 s |
| V3 small, no tiling | **1.42** | **20.1** | **27.6** | **3 s** |
| V3 small, tiles 3 | 1.57 | 23.4 | 34.2 | 13 s |

Same mechanism as 4×4 being worse than 3×3 — a tile cannot see outside
itself — but it bites much harder here, because what these two are good at is
global. V3 fuses six views into one solution, and Depth Pro predicts *metric*
depth, so tiles that disagree about absolute scale have nothing to reconcile
them; 176% wall wobble is that disagreement, three times worse than the bowed
wall the tiling was meant to help. Nor is there detail left to buy: resolving
thin structure inside a whole face is precisely what Depth Pro does better
than tiled V2 managed.

`--depth-tiles 3` is still the right call with `--depth-backend
depth-anything`, which is where it was measured.

Cost is far gentler than N² suggests, because this box is encode-bound at 8K
and much of the extra depth work hides behind the encoder:

| | s/frame | vs default | 8446-frame render |
|---|---|---|---|
| none | 1.51 | 1.00× | 3 h 32 m |
| 2 | 1.94 | 1.28× | 4 h 33 m |
| 3 | 2.70 | 1.79× | 6 h 20 m |

**A caution on measuring this.** The kink was first "measured" at 22 px by
tracing the cable through both eyes and differencing. That number was wrong —
the tracer lost the cable behind an occluder and re-acquired a different one,
and the statistic described the tracer rather than the footage. The real
displacements are a few pixels. Human judgement in the headset got the
ordering right when the metric did not; if a metric disagrees with the
headset here, distrust the metric first.

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

#### `--inpaint` is close to decorative, and here is why

Worth knowing before installing anything for it: on a real frame the inpainter
never runs at all. Instrumented on the reference photo at 7680×3840, strength
1.2, by wrapping the shipped functions rather than reimplementing them:

| | |
| --- | --- |
| disocclusion holes | 17,689 px — 0.060% of the eye |
| filled by `_directional_fill` | 17,689 px — **100%** |
| reaching Telea or LaMa | **0 px** |

`fill_holes` continues each hole from the background side first and hands on
only what that cannot reach, so the inpainter is dead code whenever the
extension succeeds — which here was everywhere. `--inpaint learned` produced a
**byte-identical** file, and LaMa was never even constructed.

Forcing the issue does not help either. With the directional fill disabled so
everything falls through to the inpainter, Telea against LaMa differs by
0.001% of the eye — a few hundred pixels.

The holes are small to begin with, because `--gradient-limit` exists to prevent
them:

| | hole area |
| --- | --- |
| `--gradient-limit 1.0` (default) | 1,584 px — 0.0054% |
| `--gradient-limit 0` | 3,985 px — 0.0135% |

So the streaking beside a near object's trailing edge — the artifact that sends
people looking for a better inpainter — is not hole filling. It is the warp
stretching the object across the depth cliff, which is what the gradient limit
trades a hole for. Changing how the residue is filled cannot touch it.

Measured leverage on the synthesised eye, same frame, one variable at a time:

| lever | share of the eye it changes |
| --- | --- |
| `--strength` 1.2 → 0.8 | **31.85%** |
| `--gradient-limit` 1.0 → 0 | 0.451% |
| directional fill on → off | 0.001% |
| `--inpaint simple` → `learned` | 0.001% |

`--strength` is three orders of magnitude more consequential than the fill
strategy. If disocclusion artifacts are the complaint, that is the knob.

A `--no-directional-fill` flag was built to expose the third row and then
reverted: it is a real switch that measurably does nothing, and the finding is
worth more than the option.

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

The corner stretch is the reason given here for why wider hurts, and it is not
the whole one: V3's own camera estimate stops tracking the truth at about 62°,
so every width in this table is already past what it believes it is looking
through, and the widths degrade in the order that mismatch predicts. See
[The model's lens is narrower than the face it is given](#the-models-lens-is-narrower-than-the-face-it-is-given).

Runtime is unchanged. The blend tables depend only on the output geometry, so
they are built once per run (~2.5 s at 8K, ~450 MB) rather than per frame; a
24-frame 8K render measured 45.5 s against 45.4 s with exact faces.

### The model's lens is narrower than the face it is given

Reported from a Quest 3, and the reason the section above is not the whole
story: patches of ground *near the camera* sitting higher than they should. It
looks like the seam problem and it is not.

Geometry settles it without a reference model, the same way the widening was
measured. A floor one camera height down puts a surface at distance
`1/sin(-lat)` along any ray, exactly, so the estimate can be turned into a
height above the true plane and plotted against distance from the tripod. Frame
0 of the outdoor 8K clip, calibrated on the near apron (0.15–0.5 camera
heights) where the ground is unambiguously flat concrete:

| distance out, in camera heights | 0.3 | 0.5 | 0.7 | 0.85 | **1.0** | 1.2 | 1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| height above the true floor | −0.01 | +0.02 | +0.07 | +0.12 | **+0.18** | +0.19 | +0.18 |

Level under the tripod, then a steady climb to 0.18 camera heights — about
30 cm on a 1.6 m tripod — reached at one camera height out, which is exactly
where the down face's nominal edge crosses the floor. Reproduces on frame 240
within 0.01 throughout.

**It is not the seam.** Three things say so. The lift starts around half a
camera height out, deep inside the down face where no neighbour contributes
anything. Yawing the whole cube 45° moves the seam ring without moving the
bulge. And a two-pass scheme that takes every direction from whichever cube
sees it further from its own seam measured *worse* than one pass (+0.21 against
+0.18) — as did an oracle allowed to pick the better of the two per pixel
(+0.18, i.e. no gain), which is what you get when both passes are wrong in the
same direction by the same amount.

#### The camera head saturates

Depth Anything V3 predicts its own intrinsics, alongside depth and confidence.
Fed the same six views at a sweep of fields of view, the focal length it
reports says it stops believing wide lenses:

| the face really spans | 61.9° | 73.7° | 90° | **98°** | 106.9° | 116° |
| --- | --- | --- | --- | --- | --- | --- |
| the model says | 58.9° | 64.3° | 65.7° | **65.6°** | 67.9° | 69.1° |
| ratio | 0.95× | 0.87× | 0.73× | **0.67×** | 0.64× | 0.60× |

It tracks the truth to about 62° and then flattens out around 65–69°, which is
roughly a 28 mm-equivalent lens — a very common thing to have been trained on,
and nothing like what this pipeline hands it.

So it reconstructs for a much longer lens than it has. A ray genuinely 45° off
the face axis is treated as though it were about 30° off; the distance it must
travel to reach a surface is under-estimated, and the surface is placed too
near. The error is zero on the axis and grows with angle off it, which is the
shape measured above — and the reason it looks like a seam problem is only that
a face edge is where that angle is largest before the face runs out.

It also explains why wider faces measure worse, which the widening study saw as
a mild cost and could not account for:

| face width | 91.1° | 98° | 106.9° |
| --- | --- | --- | --- |
| height error at one camera height | +0.156 | +0.182 | +0.228 |

Same ordering as the mismatch ratio.

#### Why the scale fit cannot see it

`align_overlapping_faces` reconciles the faces *with each other*. All six carry
the same bias, so they agree with each other while all being wrong together —
on every frame tested it chose a shift of exactly **0.000** for all six. A
committee where everyone makes the same mistake votes unanimously.

Nor is it fixable by fitting the faces to the ground instead. That was built
and measured first: for any plane, inverse depth is exactly linear in the
direction vector, so a four-parameter least squares over ground pixels gives
the plane and the offset together, and the offset is the correction. It fits
well — R² 0.962–0.968 across eight frames, offset stable to 2.6% — and it still
fails, structurally. An offset in inverse depth acts on the depth *range*,
while the error is in the *angle*; enough offset to flatten the near floor
exceeds the entire inverse depth of anything far away:

| ground fit looks below | worst floor error | frame clipped to infinity |
| --- | --- | --- |
| 8° | 0.157 | 17.6% |
| 20° | 0.118 | **53.5%** |
| 30° | 0.266 | 65.5% |

There is no setting where the floor comes flat and the horizon survives.

#### The correction

Per-ray, then, matching the shape of the error: divide each face's inverse
depth by `1 + F·(sec θ − 1)`, where `sec θ = √(1 + a² + b²)` is the ray's own
foreshortening at face coords `(a, b)` — 1.00 at the face centre, 1.41 at the
nominal cube edge, 1.91 at the widened corner. A cached per-face table and one
divide per pixel: no second inference, no extra pass, no measurable runtime.

Applied **before** the faces are fitted together, since it moves the values
that fit would otherwise read.

| `F` | worst floor error out to 1.2 camera heights | clipped |
| --- | --- | --- |
| 0.0 (today) | 0.196 | 0.00% |
| 0.4 | 0.079 | 0.00% |
| **0.55** | **0.032** | 0.00% |
| 0.7 | 0.036 | 0.00% |
| 1.0 (the full ray-versus-axis conversion) | 0.127 | 0.00% |

`F = 1` is the complete conversion and overshoots, landing the ground 0.12
camera heights *below* true. The best value per frame across the clip came out
0.60, 0.50, 0.55, 0.60, 0.55 — one constant holds.

On the reference photo's scorecard, through the V3 path at 11904×5952:

| | chair gap (→1.0) | wall wobble (→0%) | floor rms (→0%) | depth span (keep) |
| --- | --- | --- | --- | --- |
| off | 1.42 | 19.6 | 27.7 | 1.30 |
| **F = 0.55** | 1.39 | **8.5** | 24.5 | 1.07 |
| F = 0.7 | 1.39 | **7.6** | 23.5 | 1.07 |
| F = 1.0 | 1.38 | 9.4 | **20.9** | 1.07 |

Wall wobble more than halves, which was the score most at risk — an angular
correction touches every pixel, and a vertical plane crosses a face periphery
the same way a floor does. It improves for the same reason the floor does.

The depth span falls 17%, and score.py is right to flag that, so: by latitude
band the loss is 3–5% on the bands sitting at face centres (nadir, zenith,
horizon) against 13–15% on the bands at 45°, which are the seam latitudes. What
is being removed is the false nearness at the peripheries, not stereo. The
other three scores are ratios and scale-invariant by construction, so
`--strength 1.2` restores the parallax without moving any of them.

Seam agreement improves as a side effect it was not aiming at — the spread
between overlapping faces over the lower hemisphere falls from 6.9% to 4.5% of
local depth — because the faces were disagreeing precisely where each was least
reliable.

#### Why it is off by default

Two scenes, one backend, one face width, and never yet judged in a headset.
`F` is empirical rather than derived: the model's own predicted intrinsics
imply about 0.36, so it is absorbing something the field-of-view story alone
does not account for, and it is tied to V3 at 98° faces. Depth Pro — the stills
default — is untested and would need its own constant, or none.

Two things measured along the way and not used. Tiling narrows the field of
view per inference, which ought to help, and makes it much worse (+0.41 at
`--depth-tiles 2` against +0.20): `estimate_tiled` pins each tile to a coarse
full-face reference with a scale-only fit, re-imposing the shape it was
supposed to escape. And a second pass with the cube *tilted* rather than yawed
does work — 0.196 → 0.150 for two passes, 0.137 for three — because a tilt
changes each ray's angle off the axis where a yaw does not. It is a quarter of
the benefit for double the inference, so it is worth remembering only if the
correction above turns out not to transfer.

### Negative result: a thin trim the synthesised eye loses

Reported from a headset on the reference photo: the plaster trim between the
kitchen and the front door is clear in the left eye and gone in the right, so
it reads as sinking into the wall behind it. This is the metallic-trim artifact
that made Depth Pro the stills default, met again on the V3 path — which
matters because V3 is what video uses, and an indoor video of the same room
would show it.

Nine things were tried and none fixed it. Recorded because most of them look
obviously right beforehand.

#### What is actually happening

The trim is about 7 px wide. At 518² inference the depth boundary lands roughly
**4 px past the picture's own boundary**, on the far side of the trim, so the
whole strip is given the kitchen wall's depth. The near column then sweeps over
it in the warp — correctly, given a depth map that says the column's surface
starts where the trim is. Profiles across the edge in the `-X` face, row 1035:

| face column | 174 | 176 | 178 | 180 | 182 | 184 | 186 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| picture (grey) | 174 | **161** | **125** | **76** | **50** | **47** | 54 |
| depth, 518 | 0.590 | 0.591 | 0.595 | 0.678 | 0.896 | 1.193 | 1.566 |
| depth, 1036 | 0.576 | **0.620** | **1.189** | **1.530** | 1.569 | 1.552 | 1.552 |

The edge itself is *not* soft — the depth transition measures 5 px against the
picture's 3. It is in the wrong place. At 1036 it moves onto the picture's
edge and part of the trim finally reaches the near surface.

#### What was tried

Measured on rendered pixels: how far the trim falls below the walls either
side, averaged over the 20 rows where the source trim is strong. The source
falls 46.6 grey levels; the left eye is the source in every case except
`--split-baseline`.

| | right eye | of source |
| --- | --- | --- |
| **as shipped** | **28.3** | **61%** |
| `--fg-erode 0` | 28.3 | 61% |
| `--depth-tiles 3` | 28.2 | 61% |
| `--smooth 8` | 27.5 | 59% |
| `--split-baseline` | 27.4 | 59% |
| two rotated cubes, best-of | 23.6 | 51% |
| `--gradient-limit 0` | 21.9 | 47% |

Nothing improves on the default, and two things measurably hurt. The guided
filter was also swept properly (`--smooth` is a joint filter guided by the
full-resolution face image): radius 8 → 64 widens the depth transition
monotonically from 5 px to 19 px, because it is a smoother and there was never
a soft edge to snap.

`1036` at two views was scored against the shipped `518` at six, since six
views at 1036 exhausts a 16 GB machine:

| | chair gap | wall wobble | floor rms | depth span |
| --- | --- | --- | --- | --- |
| 518, 6 views | **1.42** | **19.6** | **27.7** | 1.30 |
| 1036, 2 views | 1.48 | 22.3 | 29.7 | 1.47 |

Worse on all three, including `chair_gap`, which is the thin-structure score.
Losing the whole-sphere fusion costs more than the resolution buys. That does
not say resolution is useless — 1036 at six views is the experiment that would
isolate it, and it is not runnable here.

#### The measurement trap, which is the real lesson

Four conclusions were drawn and withdrawn before the numbers above were
trusted, and three of the four came from the same two mistakes.

**Measuring the wrong feature.** The trim is a *dark* line and the brightest
thing within 200 px of it is the kitchen's white wall. A top-hat looking for a
bright line reported the trim as 85–97% intact while the picture plainly showed
it gone, and it ranked `--gradient-limit 0` as the worst option when the sound
measure ranks it merely bad.

**Measuring one row.** At row 2000 `--gradient-limit 0` scores 23 against the
default's 8 and looks like a fix. Over all 20 strong rows it averages 21.9
against 28.3. The trim is lost over part of its height and kept over the rest;
any single row can say whatever you want.

And one that invalidated four separate conclusions on its own: a "ramp width"
helper that returned the span from the first to the last pixel lying between
10% and 90% of a window's min–max. Plateau noise straying inside that band put
the last index 85 px past the edge, so a 4 px transition measured 89 px. Every
theory built on "the depth ramp is twelve times the feature" — the cross-fade
smearing it, the face periphery softening it, two cubes narrowing it — was
built on that number.

The habits that would have caught all three: print the profile before trusting
a summary statistic, and check that the statistic moves when the feature moves.

#### Two sizes of thin, and only one of them is the trim

A second complaint from the same photo turned out to have the same root and a
different size. The chair's back post is a solid piece of wood about 24 px
wide; in the synthesised eye it comes out about 7. Printed across it, at row
2545, the reason is plain — V3 does not describe it as a surface:

| across the post (24 px) | depth span |
| --- | --- |
| Depth Anything V3 | 0.07 → 0.67, **range 0.60** |
| Depth Pro | 0.58 → 0.73, **range 0.15** |

V3 climbs steadily across the wood, so the near edge shifts about twice as far
as the far edge and the post is squeezed. Depth Pro steps at the post's left
edge — the same columns where the picture's brightness goes 124 → 152 → 169 —
and then holds flat. A plateau warps as one piece; a ramp does not.

Both normalised to their own face's 5th–95th percentile, sampled in the +X face
so no assembly or scale fit sits between the model and the number.

So "thin structure" is two problems. A **7 px** trim is below what either model
resolves as its own surface, and gets assigned wholesale to one side of a step.
A **24 px** post is above what Depth Pro needs and below what V3 needs, so the
backend choice decides it. That distinction matters when reading the advice
above: the stills default fixes the post, and is not known to fix the trim.

#### Where it stands

For stills, use the default. Depth Pro resolves this edge and is the stills
default for exactly this reason — see [Choosing a depth
model](#choosing-a-depth-model). For video, where Depth Pro's ~2 s a frame is
prohibitive, it is unsolved.

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

This section picks the best *V2* variant, and Base wins it. V2 is no longer
the default either way — see [Choosing a depth model](#choosing-a-depth-model)
below, which compares across model families. This still decides what
`--depth-backend depth-anything` should be given.

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

### Post-processing cannot fix the depth model

No amount of pipeline work touched the artifacts below. The dead ends are
written up at length because they were expensive, and because each one looked
obviously right beforehand.

The scoring harness lives in `experiments/` -- see [Depth scoring
harness](#depth-scoring-harness). Each score compares the depth map against
something the world guarantees rather than against anyone's judgement.

#### The problem

Five artifacts reported against `indoor.jpg`, all one fault: **the depth map
is smooth where the world is sharp, and lumpy where the world is flat.**

| observed | what the depth map does |
|---|---|
| kitchen wall trim vanishes in the right eye | thin feature overtaken by the near surface sweeping across it |
| pillar trim far too wide | the same feature stretched instead, opposite ramp direction |
| wall by the entrance door is not straight | disparity wanders 62% peak to peak down a flat wall |
| chair is uncomfortable to look at | see-through gaps between the slats filled in at 2.5x too near |
| mop on the vacuum reads wrong | pad and shell merged into one smooth dome |

#### What does not work

Measured against the chair-gap ratio, all giving the same 1.8x:

- more tiles (`--depth-tiles`)
- model input resolution at 518, 1036, 1456 and 1820
- guided filtering (`--smooth` 8 and 24)
- the Large depth model

It is not a resolution problem. The model does not fail to *see* the gaps; it
decides the chair back is solid. That is a learned prior, and no amount of
pixels changes a semantic judgement.

`--strength` cannot help either, and this is provable rather than measured:
all three scores are ratios, so they are invariant under any uniform scaling
of disparity. Turning the stereo down makes the errors less visible without
making them smaller.

#### What does

Two depth estimates disagree where the model is guessing. Damping disparity
*toward far* where they disagree -- one-sided, because these errors are
one-sided: the gaps, the floor ridge and the wall's lower half all read too
near.

| | chair gap (-> 1.0) | wall wobble (-> 0%) | floor rms (-> 0%) |
|---|---|---|---|
| baseline | 1.76 | 103.2 | 40.7 |
| any `--strength` | 1.76 | 103.2 | 40.7 |
| damped, both sources, 0.6 | **1.29** | **49.6** | **35.5** |

Two sources of disagreement, and they catch different things: a second pass
with different tiling is better for the wall and floor, a second pass on the
mirrored image is better for the chair gaps. Mirroring is the cleaner signal
-- same settings, so disagreement is the model being unsure rather than a
change of configuration.

Damping past about 0.8 makes the wall worse again, and at 1.0 the chair gaps
overshoot to 0.94: too far rather than too near.

#### Open

- Not yet judged in a headset. The scores are all scale-invariant ratios with
  a physical law behind them, which is more than could be said for several
  measurements in this investigation -- but the headset decides.
- Costs a second depth pass.
- One scene. Needs the tram footage and an outdoor still before it means
  anything general.

#### Result: the damping idea does not work

Two formulations, both rejected.

**Scale depth by confidence.** Improves the three ratios but takes 38% of the
depth range with it, and the loss lands in the near field -- confidence is
anti-correlated with depth at -0.59, so near surfaces are damped hardest,
which is the opposite of where anyone wants to lose separation. It also turns
every straight line to sawtooth in the right eye, and there is a reason it
must: d(d*f) = f.dd + d.df, so multiplying by a spatially varying factor
*injects* gradient wherever confidence varies. Confidence varies at object
boundaries. It manufactures depth discontinuities at exactly the edges it is
supposed to be protecting.

**Limit the depth gradient where confidence is low.** No effect at all, at
any floor or iteration count. The reason is worth keeping: these artifacts
are not high-frequency depth noise. The chair's gaps are wrong across their
whole extent and the wall is wrong over hundreds of pixels -- regional depth
that is confidently incorrect. No local operator reaches that, which is also
why guided filtering and smoothing did nothing.

#### What did help, slightly

Not damping at all, just using a better depth map. A pass on the mirrored
image scores better than the normal pass on every artifact while keeping the
depth range:

| | chair gap | wall wobble | floor rms | depth span |
|---|---|---|---|---|
| normal pass | 1.76 | 103.2% | 40.7% | 9.26 |
| mirrored pass | 1.71 | 74.0% | 39.7% | 9.13 |
| per-pixel min of three passes | 1.71 | 58.7% | 43.4% | 8.40 |

Ordinary test-time augmentation, in other words. It costs a second depth pass
and gives maybe a quarter off the wall wobble for nothing. One image, so it
could be luck; it would need the tram and an outdoor still before it meant
anything.

#### For anyone picking this up

Three guards exist here because three reviews went out broken and were caught
by eye rather than by score:

- `depth_span`, because the other scores are ratios and could not see the
  stereo collapsing to nothing.
- `cli_defaults()`, because the library's defaults are not the CLI's and a
  render with `gradient_limit` at its library default of 0.0 tears fine
  structure into fragments.
- Validate an edge measurement on the *left* eye first. It is the untouched
  source, so a tracer reporting hundreds of pixels of bend there is measuring
  itself. Two of the measurements in this investigation did exactly that.

#### Depth Anything V3 is a different proposition

V2 Large scored identically to V2 Base on the chair gaps, so the family
looked like a dead end. V3 is not the same kind of model.

Its ONNX signature gives it away:

    input : pixel_values  [batch, num_images, 3, H, W]
    output: predicted_depth, confidence, extrinsics, intrinsics

Multi-view, and it emits its own confidence. Both matter here. A cubemap is
six views sharing a centre, so the faces can go in together and come back as
one consistent geometry instead of six independent guesses that then have to
be reconciled -- which is where the 45-degree seam ridge comes from. And the
confidence map is the thing this experiment spent a day failing to synthesise
from disagreement between passes.

Scored on indoor.jpg, all six faces in a single 1.8 s pass on CPU:

| | chair gap (-> 1.0) | wall wobble (-> 0%) | floor rms (-> 0%) |
|---|---|---|---|
| V2 Base, current default | 1.76 | 103.2 | 40.7 |
| V3 small | **1.42** | **20.2** | **28.1** |

The wall is five times flatter and the floor is the best any configuration
has produced.

Caveats: `depth_span` is not comparable across models, because V3 returns
metric depth and V2 returns relative inverse depth on an arbitrary scale --
so these numbers say nothing about whether the stereo feels right. The 1/d
conversion used here is a guess. And it has not been rendered or looked at in
a headset, which is the check that caught every mistake in this file.

The official `depth-anything/DA3*` repos are not transformers-loadable (no
`model_type`), but `onnx-community/depth-anything-v3-{small,base,large}` are
ready ONNX exports, and this project already has an ONNX backend -- so trying
it properly is a smaller job than it looks. Note the input is 5-D with a
num_images axis, which the current OnnxDepthBackend does not expect.

#### Depth Pro and V3 are complementary, and this scorecard only sees one of them

Scored on indoor.jpg, six faces, everything else identical:

| | chair gap (-> 1.0) | wall wobble (-> 0%) | floor rms (-> 0%) | cost |
|---|---|---|---|---|
| V2 Base, current default | 1.76 | 103.2 | 40.7 | ~8 s GPU |
| V3 small | **1.42** | **20.2** | **28.1** | 1.8 s CPU, 105 MB |
| Depth Pro | 1.49 | 53.1 | 30.1 | 2 s GPU, 3.6 GB |

By those numbers V3 wins outright. Judged in a headset the ranking was Depth
Pro, then V3, then V2 -- and the reason is a gap in the scorecard, not noise.

All three metrics measure large-scale geometry: see-through gaps, a vertical
plane, a floor. None measures thin structure. Thin structure is exactly what
Depth Pro is designed for, and exactly the artifact that started this whole
investigation -- the metallic trim at the kitchen wall edge, which vanishes
from the right eye. Depth Pro keeps it. V2 and V3 both replace it with a hard
boundary.

An attempt at a thin-structure metric failed and is not kept: measuring the
brightness peak's prominence conflates "the trim is there" with "the trim is
gone and the bare kitchen-to-stone edge is sharper", so it ranked the models
backwards. The crop comparison is unambiguous and was believed instead.

So: V3 for geometry, Depth Pro for edges, and the perceptual weight seems to
sit with edges. Both loose ends — whether a larger V3 closes the gap, and
whether feeding V3 more than the 518 px used here to match V2 helps — were
then measured, and neither did. That is the next section.

### Choosing a depth model

The conclusion of the above: the ceiling is the model. Measured on the same
photo, six cube faces, everything else identical.

| | chair gap (-> 1.0) | wall wobble (-> 0%) | floor rms (-> 0%) | thin trim | cost |
|---|---|---|---|---|---|
| V2 Base (was the default) | 1.76 | 103.2 | 40.7 | lost | ~8 s GPU |
| V3 small @518 | 1.42 | **20.2** | 28.1 | lost | 1.8 s CPU, 105 MB |
| V3 small @1036 | **1.28** | 24.2 | 31.3 | lost | slower |
| V3 base @518 | 1.45 | 27.0 | **26.5** | lost | 413 MB |
| V3 large @518 | 1.66 | 26.3 | 28.6 | - | 1383 MB |
| Depth Pro | 1.49 | 53.1 | 30.1 | **kept** | 2 s GPU, 3.6 GB |

Two things decide it.

**Capacity does not help.** V3 large scores *worse* than V3 small on the chair
gaps. Paying 1.4 GB for the large export buys nothing.

**They are complementary, and the scorecard only sees one axis.** All three
scores measure large-scale geometry, where V3's multi-view design wins by a
distance. None measures thin structure, which is Depth Pro's stated design
goal -- and Depth Pro is the only model that keeps the metallic trim at the
kitchen wall edge, the artifact that started the whole investigation. Judged
in a headset the order was Depth Pro, then V3, then V2, which is the opposite
of what the numbers say and is why the numbers are not the last word.

Hence the split: V3 small for video, where 105 MB and CPU-speed inference
matter and no single frame is studied for long; Depth Pro for stills, where
the download is paid once and the picture is looked at closely.

### DirectML, even on an NVIDIA card

The video default is an ONNX graph, so which ONNX Runtime wheel is installed
decides whether it uses the GPU at all. On Windows the answer is
`onnxruntime-directml` regardless of vendor, which is not the obvious one.

Measured on an RTX 5070 Ti, Depth Anything V3 small, six cubemap faces in one
call — the exact shape the pipeline sends:

| | six faces, raw inference | through the pipeline |
|---|---|---|
| CPU provider | 1.91 s | 1.87 s |
| **DirectML** | **0.15 s** (12.7×) | **0.36 s** (5.2×) |

The pipeline column is smaller because resizing six 1920 px faces down to 518
and the results back up is CPU work either way, and it now dominates. That is
where the next speedup is, if one is wanted.

**The output is identical, which had to be checked rather than assumed.** Max
absolute difference against the CPU provider: 0.00000. After the warp's own
1/99-percentile normalisation, 3e-5. All three geometry scores agree to two
decimals. Every number elsewhere in this document was measured on the CPU
provider, so a fast provider that quietly computed something else would have
invalidated the lot.

**`onnxruntime-gpu` is the trap.** It is the obvious wheel for an NVIDIA card,
and the published build carries no `sm_120` kernels — on any RTX 50 series
card its CUDA provider fails with *no kernel image is available for execution
on the device*
([onnxruntime#26245](https://github.com/microsoft/onnxruntime/issues/26245)).
That is the same failure mode as a mismatched torch build, one layer along,
and the reason the installer runs a graph through the GPU rather than
trusting `get_available_providers()`. Writing that check found its own bug
first: `onnx` emits the newest IR version it knows, onnxruntime rejects
anything above its maximum, and the probe failed against a runtime that was
working perfectly.

Torch is unaffected — it has proper `sm_120` kernels, so Depth Pro gets CUDA
on the same machine. The two defaults use two runtimes and each takes its own
route to the GPU.

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

### Depth scoring harness

`experiments/` holds the scoring code behind every depth number quoted above.
It is not part of the tool and nothing imports it.

`score.py` scores a depth map against three artifacts measured on
`indoor.jpg`, each checked against a law the world guarantees rather than
anyone's judgement: a flat floor's inverse depth follows sin(latitude), a flat
wall's follows cos(elevation), and the gaps between chair slats must read as
the wall they show. `damp.py` holds the rejected damping experiments and
`cli_defaults()`, which any script rendering through the library needs.

**`indoor.jpg` is not in the repository** and these scores cannot be
reproduced without it. It is a 360 photo of a private house, which is not
something to publish, and a 15 MB file is not something to put in every
clone. The harness expects it beside the checkout, or wherever
`STEREO360_INDOOR` points, and says so if it is missing.

Substituting a different photo does not work. Every landmark is a pixel
coordinate in that one 11904x5952 frame, so another image scores whatever
happens to sit at those coordinates — meaningless numbers rather than wrong
ones, which is the harder failure to notice. The numbers quoted throughout
this document are therefore a record rather than something a reader can
re-run; what transfers is the method, which is the three laws above.

Three guards exist because three reviews went out broken and were caught by
eye rather than by score:

- `depth_span`, because the other scores are ratios and could not see the
  stereo collapsing to nothing.
- `cli_defaults()`, because the library's defaults are not the CLI's: a render
  with `gradient_limit` at its library default of 0.0 tears fine structure
  into fragments.
- Validate an edge measurement on the *left* eye first. It is the untouched
  source, so a tracer reporting hundreds of pixels of bend there is measuring
  itself. Two measurements here did exactly that.


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

### Depth Anything 3 metric: the scale is real, and the input size is part of it

`DA3METRIC-LARGE` is a plain single-view DinoV2-L with a DPT head — its config
has no camera decoder and no cross-view attention, so the six faces are
independent and none of the multi-view machinery applies. On a CPU-only torch
it runs a face in 3.2 s at its default `process_res` of 504, which is about the
wall time the ONNX V3 small path takes on the GPU for all six.

What it buys is the one thing relative depth cannot give: a scale. Every render
before this had an arbitrary one — the same `--strength 1.2` produced a 49 mm
eye separation outdoors and 35 mm indoors, against a human 65 mm, and there was
no way to know which was which. The metric output can be checked against common
sense with no ground truth at all:

| scene | fitted camera height | check |
|---|---|---|
| road | 1.58 m | straight down measures 1.6 m |
| indoor | 1.41 m | plus 1.01 m to the ceiling = a 2.41 m room |

Both are what a tripod and a house actually are. With that, the baseline stops
being a taste setting. The warp forms `lam = 1 / (dn + _MIN_INV_DEPTH)` and
shifts by `strength * _BASELINE_SCALE` in those units, and normalisation is
affine, so choosing `hi - lo = 1/S` and `lo = _MIN_INV_DEPTH/S` makes one
relative unit mean `S` metres. The eye separation is then
`strength * _BASELINE_SCALE * S` metres, and asking for 65 mm is arithmetic.

**`process_res` is not a quality knob.** Raising it from 504 to 1008 sharpens
the depth edge across a thin upright from 6 px to 2 px, and breaks the metric
scale: the road camera drops from 1.58 m to 0.84 m and the room from 2.41 m to
1.72 m. The input size is part of what the model was calibrated against. Nor is
one constant enough to repair it — the ratio between the two runs has a median
of 1.98 but an interquartile spread of 1.23, so a single rescale leaves about
23% of depth-dependent distortion.

Both can be had at once, because the two errors live at different spatial
frequencies. The scale error is smooth (the model reading the wrong implied
focal length); the detail is local. Dividing the 1008 map by a heavily
low-passed version of its own ratio against the 504 map restores the calibrated
scale and keeps the sharp edge:

| | camera | depth edge |
|---|---|---|
| 504, calibrated | 1.58 m | 6 px |
| 1008, sharp | 0.84 m | 2 px |
| fused | **1.59 m** | **2 px** |

Two smaller things the metric path forces:

**The up face has nothing to measure.** Shown only sky it answers about 10 m
where the side faces put the same sky at 88 m, which paints a ring across the
sky at the latitude the faces meet — a step from 0.9 px of parallax to 8. It is
fixed by rescaling that one face against the ratio the aligner wanted for the
others (0.187 here), leaving the five faces that contain ground untouched.
Indoors, with no sky, nothing needs rescuing.

**The repo's tuned constants assume percentile normalisation.**
`_erode_foreground` weights by `clip((rng - 0.05)/0.05, 0, 1)`, which is
calibrated for a scene spread across most of [0, 1]. A metric encoding squeezes
the midground toward zero: the van at 8.5 m and the road at 14.7 m came out at
0.0676 and 0.0180, a contrast of 0.0496 — just under the threshold, so the
erosion computed a weight of exactly zero and did nothing at all. Any constant
expressed in normalised depth units needs re-reading when the normalisation
changes.

### There is no halo

For most of an investigation into why a van's edge looked thicker in one eye,
the working theory was that the depth map's foreground overhangs its silhouette
by about 7 px — a halo — so a strip of road carries the vehicle's depth and
travels with it. Four fixes were built against that theory and all four failed:

| approach | depth-edge offset | edge width |
|---|---|---|
| none | +7 px | 3 px |
| guided filter (r=8) | +10 px | 18 px |
| joint bilateral (r=20) | +2 px | 26 px |
| weighted mode, per face | +8 px | 3 px |
| weighted mode, post-assembly | +2 px | 0 px |

The guided filter and the joint bilateral are weighted *means*, and a mean
across a depth step returns values between its two sides, so moving the edge to
the right place costs it its sharpness — and a 26 px depth ramp is exactly what
`--gradient-limit` exists to suppress. The weighted mode (split the window by
colour similarity, take the side that wins, never average across) does move the
edge without blurring it, but only after face assembly: the van sits at
longitude -146 degrees, inside a face overlap, and sharpening two faces
separately before cross-fading them puts the ramp straight back.

`fg_erode` "fixed" it and cost more than it saved. At the reach needed to
remove the overhang (8 px) it eats the van's own bodywork — the white strip
between its rear window and its outer edge is only 10 px wide, and came out
8 px in one eye against 13 in the other. There is no good setting: at 2-4 the
strip is perfect and the road still drags, and at 6 and 7 the probe reads +4 and
-13 px of disparity where the truth is -4, worse than either end. On the mesh
path it was worse still, leaving railings and indoor chair posts visibly eaten.

**Then the premise was measured, and it was wrong.** Across 38 edges with a
step in both colour and depth, the two agree to within a pixel — median -1.0 px
where the foreground lies left, +0.0 px where it lies right. Individually:
handrail post +0, sign post -1, kerb -1, bin -1. There is no systematic halo to
erode.

The van is a special case, and the reason is visible in the raw pixels. Across
its rear-right edge the body reads 128-139 and the road beside it 117-127:

| edge | colour contrast across the depth step |
|---|---|
| **van rear-right** | **1 level** |
| bin | 20 levels |
| kerb / steps | 32 levels |
| handrail post | 54 levels |
| sign post | 73 levels |

About one level. The only visible feature is a single dark trim line one pixel
wide. There is no image evidence there for any algorithm to localise the
boundary with, which is precisely why three colour-guided methods failed on it:
they relocate a depth edge by following the picture's edge, and at the van
there is no picture edge to follow.

The original 7 px was also partly a measurement error — it compared the
*midpoint of the depth ramp* (727) against the *dark trim line* (721), which
are different landmarks. Measured like for like, the foreground ends at 724
against a trim line at 721, about 3 px.

The lesson is the ordinary one: measure the premise before building against it.
Four implementations, each sound in itself, were aimed at a defect that the
population statistics say is not there.

### Rendering the warp as a mesh

`right_eye_from_disparity` treats every source pixel as an independent point:
lift it by its inverse depth, translate to the other eye, reproject, scatter it
into a 2x2 footprint, keep the nearest. Nothing in that says two neighbouring
pixels belong to one surface, so a small object's pixels land unevenly and
shear along the warp direction — which is opposite in the two eyes. A bollard
lamp's finial, the nearest object in one scene at 1.6 m, came out as a block
leaning right in the left eye and left in the right eye.

A mesh keeps them joined: vertices from the depth samples, quads between
neighbours, each quad either kept or cut, and the surface between vertices
interpolated rather than left to chance. Depth values are never rewritten, so
`--gradient-limit` is not needed — the cut takes its place, and the object
keeps its true depth. Measured as error against the source after allowing for
parallax, above each renderer's own noise floor:

| scene | mesh | splat |
|---|---|---|
| sign post | **0.99** | 4.59 |
| lamp finial | **2.63** | 6.11 |
| handrail | 5.53 | **5.05** |

Two wins and a tie. The handrail is where a mesh has least to offer: a thin
diagonal bar disoccludes along its whole length, so it produces the most cut
area of anything in the frame.

**Cut on projected stretch, not on depth ratio.** A ratio test is scale-free
and fires just as hard on a tree forty metres away — where a leaf and the gap
behind it differ by a fraction of a pixel — as on a silhouette two metres away.
It left the canopy riddled: 1131 separate holes, 818 of them 4 px or smaller.
Cutting instead when the warp pulls a quad wider than 2.5 output pixels drops
that to 132 holes and takes the canopy and the van edge to zero, because it
only removes geometry that would have been a visible rubber sheet.

The prototype is not production, and its two remaining defects share one cause.
It scatters samples rounded to the nearest pixel and composites them
far-to-near, where a real rasteriser computes coverage per output pixel. That
gives it a noise floor the splat does not have — rendering at zero baseline,
where a perfect renderer must return the source exactly, it leaves a residual
of 1.9 to 5.7 depending on the crop, against the splat's 0.00 — and it produces
four hairlines at longitudes +/-45 and +/-135 degrees, where ties in the
compositing order flip under float32 jitter and the texture coordinate steps a
quarter of a pixel. Breaking ties by proximity to the pixel centre was tried
and made it worse.

Speed is not the obstacle it looks like. Profiling one band: argsort 46%, the
sample blend 30%, rounding and scatter 23% — and the actual geometry,
projecting vertices and deciding cuts, **1.6%**. Nearly all of the cost is the
brute-force stand-in for a rasteriser, and a mesh built at depth resolution
rather than image resolution would start from about 16x fewer quads, since the
depth resolves roughly one independent value per 7 px anyway.

### A metric scale makes the baseline a dial, and indoors it is worth turning

Once the depth is metric the eye separation stops being a taste setting and
becomes a number with units, which makes an old trade newly measurable: how
much depth is being bought with how much damage.

The damage scales with how near the scene is, and a room is much nearer than a
street. Measured over the same two scenes, at a true 65 mm:

| scene | median depth | median disparity | p99 disparity | depth edges |
|---|---|---|---|---|
| road | 22.5 m | 3.5 px | 51.6 px | 5203 |
| indoor | 1.6 m | **49.0 px** | 93.7 px | **32448** |

The typical indoor pixel moves further than the road's 99th percentile, and
there are six times as many depth discontinuities for it to move across — five
times the total disocclusion area. None of that is a defect. At 0.8 m a 65 mm
baseline subtends 4.4 degrees and a real pair of eyes verges 4.65, so the
geometry is right; the scene is simply demanding, which is why VR capture
generally avoids objects that close.

Dropping to 40 mm indoors, changing nothing else:

| | cut geometry, left / right |
|---|---|
| 65 mm | 0.42% / 0.36% |
| 40 mm | **0.18% / 0.14%** |

A 38% smaller baseline removed about 60% of it — steeper than linear, as the
disocclusion numbers above predict. Judged in a headset the room still read
with good depth and visibly less breakup, so indoors this is a trade worth
making. It is not worth making on the road, which has almost no disocclusion to
save at 3.5 px of median disparity.

What it does *not* do is improve the renderer. The same artifacts are present
in the same places; there is simply less warping for them to attach to. That
distinction matters when reading any before-and-after: less work done is not
the same as work done better.

A second thing this comparison ruled out. The suspicion was that indoor scenes
suffer because they are full of low-contrast boundaries — white chair against
cream wall — and that the depth model cannot localise them. The population says
otherwise: the median colour contrast across a depth step is 40 levels indoors
against 39 on the road, and 12% of indoor edges fall below 10 levels against
16% outdoors. Indoor is not lower contrast. It is just closer.

### Consistency between the eyes may matter more than fidelity in either

An argument from the user, recorded because it reframes what the renderer is
for and applies to the splat path as much as to any mesh:

> Because the depth map is low resolution, small details do not get rendered
> correctly. With the whole baseline in one eye, the source eye has the detail
> right and the reconstructed eye has it wrong, and the mismatch is what causes
> fatigue. It would be better if the same detail were rendered *incorrectly in
> both eyes*.

The claim is that binocular agreement is worth more than per-eye accuracy. Two
eyes that agree on a slightly wrong shape fuse into a slightly wrong object,
which is comfortable; one right eye and one wrong eye fuse into nothing, and
the visual system keeps trying.

This is a different phenomenon from the one `--split-baseline` was justified
by, and the two should not be confused:

* A **monocular region** — content one eye can see and the other cannot — is
  normal. Every real occluding edge produces one, and the brain suppresses the
  unmatched side without complaint. That is what the existing note means by
  each eye's holes falling in different places so fusion hides them, and it
  stands.
* A **shape mismatch** on a feature both eyes *can* see is not normal. It
  presents conflicting disparity across the feature, and nothing in ordinary
  vision produces it. This is what the argument above is about, and the
  existing justification says nothing about it.

Today's measurements support the distinction, and are not kind to the current
split. On a bollard lamp's finial the splat rendered the ball leaning right in
the left eye and leaning left in the right — the distortion is *mirrored*,
because the two eyes are warped in opposite directions from the same source.
That is the worst case for this argument, not the best: whole baseline would
give one correct ball and one leaning ball, and split gives two balls leaning
opposite ways.

So there are three arrangements, and the repo has only ever measured the first
two:

| | left eye | right eye | shape error |
|---|---|---|---|
| whole baseline | pristine | distorted | present in one eye |
| split baseline | distorted | distorted | mirrored between eyes |
| chained | distorted | distorted | **shared** |

The chained arrangement is the user's proposal: warp the source to the right
eye, then warp *that* to the left eye rather than going back to the source.
Whatever the depth map got wrong about a detail is then baked into the right
eye and inherited by the left, so both carry the same error and the pair
agrees.

Open questions, none of them settled:

* The left eye becomes a warp of a warp — two resamplings and two hole fills,
  so it is softer than the right and inherits invented content as if it were
  real. The pair agrees, but on something partly fictional.
* Going right-to-left re-invents content the source actually contains. A
  composite that prefers real source pixels where they exist would avoid that,
  at the cost of reintroducing the very inconsistency the scheme is for.
* It is testable without judging by eye. The measure is not each eye's error
  against the source but the *disagreement between the eyes* — for the finial,
  the splat's two eyes scored 5.99 and 6.22 above their floor and the mesh's
  4.64 and 4.42, gaps of about 0.2 in both. Chaining should drive that gap
  toward zero while leaving the absolute error roughly where it was.

### How the baseline is divided between the eyes, measured

The preceding argument says the pair matters more than either half. That is
testable: align each eye to the source by brute-force integer shift, which
divides out where a feature sits and leaves how far its shape had to bend, and
read the *difference between the eyes* rather than either eye's fidelity.

Whole baseline, a reconstructed left eye, a round trip and an even split turn
out not to be four ideas but four points on one axis — what fraction of the
total separation the left eye takes. Measured on three small features that the
depth map is known to get wrong, at a constant 65 mm total:

| left eye's share | lamp finial | sign post | handrail |
|---|---|---|---|
| 0% (pristine) | 12.30 | 8.83 | 10.39 |
| 15% | 1.77 | 2.65 | 5.01 |
| 30% | 0.79 | 2.82 | 2.66 |
| 50% | **0.09** | **1.27** | **0.63** |

Two things fall out, and one of them was not expected.

**A pristine eye is the worst case, by an order of magnitude.** Leaving one eye
untouched guarantees maximum mismatch on exactly the small features the depth
map gets wrong, because the other eye carries all of the error. This is the
opposite of the intuition that an unmodified eye must be the safe one.

**The curve has a knee, not a slope.** The expectation was that error would
scale with displacement, so a 15% share would buy about 15% of the benefit.
It buys 86% of it on the finial and 70% on the sign post. Once both eyes are
being reconstructed at all they are in the same regime, and what remains is
only the difference in how far each was pushed. Where there is a reason to keep
one eye close to the original, 15/85 is therefore a real option; with no such
reason 50/50 is still the minimum and is what the renderer does.

Two arrangements that reconstruct the left eye *without* displacing it were
also measured, and neither reaches an even split:

| left eye | finial | sign post | handrail |
|---|---|---|---|
| reconstructed at zero baseline | 10.59 | 3.94 | 6.97 |
| round trip out to the right eye and back | 8.82 | 1.34 | 5.19 |

Reconstructing at zero baseline cannot work in principle: displacement is zero
for every pixel whatever the depth says, so the eye picks up the rasteriser's
resampling character and none of the depth map's geometry. The round trip does
carry the geometry — the two warps fail to cancel wherever the depth is wrong —
and it nearly closes on the sign post at 11 m, which barely moves. It fails on
the finial at 1.6 m, which moves 27 px, opens large holes, and comes back
carrying invented fill rather than its own distortion.

Chaining the left eye off the right, rather than off the source, was the
proposal this set out to test. It wins once (sign post 1.27 to 0.29), loses
once (handrail 0.63 to 1.41) and ties once, while raising the absolute error
every time through double resampling. Not worth its complexity on this
evidence.

**A caveat that limits all of the above.** The measure is the *magnitude* of
each eye's distortion, not its direction. A 0.09 at 50/50 says the two eyes are
distorted equally, not identically — and the finial is visibly distorted in
*opposite* directions in the two eyes, leaning right in one and left in the
other. If mirroring is itself uncomfortable, an even split scores well on a
metric blind to its main flaw. Settling that needs the eyes aligned to each
other rather than each to the source.

### Two mesh renderer notes

**A fold is not a stretch, and the cut test only sees stretch.** A quad whose
warped corners reverse order has folded the surface back through itself; what
gets drawn is its back face, mirrored. It is *narrow*, not wide, so a test on
projected width passes it. Measured on one indoor scene, 12043 quads fold. In
practice culling them changed 0.0001% of pixels because the folded faces were
already losing the depth test to the surface in front — the cull is correct and
nearly free, and it is not the fix for anything visible.

**The mirrored fill duplicates objects.** `_directional_fill` continues
background into a hole by mirroring across the hole boundary, which is right
for grass or carpet and wrong when the neighbour is recognisable: a tap beside
a stone pillar came back reflected into a symmetric phantom. Telea inpainting
leaves a soft smear instead, differing on 0.07% of the frame, and in stereo the
smear is much the lesser evil. The mirroring exists for temporal stability, so
this is a still-image versus video trade rather than a defect: prefer
inpainting for photos, keep mirroring for footage.

## The mesh renderer was wrong, not slow, and the same change fixes both

The mesh prototype was moved to the GPU to make it fast. Profiling the result
said the port had missed the point: 90% of its time went to `round+mask`,
`zbuffer+scatter` and `blend` -- the brute-force stand-in for a rasteriser --
and 0.7% to the geometry. It was faster hardware running the wrong algorithm,
and 2.9x was the ceiling that bought.

**The wrong algorithm.** Each 1x1 source quad was sampled at 16 fixed points,
each point rounded to the nearest output pixel and scattered. Wasteful where
the warp compresses (16 samples onto one pixel), starved where it stretches
(4 samples across 2.5 px), and wrong everywhere: the texture coordinate came
from where a *sample* fell rather than from where the *output pixel* is, so
the rounding error went into the resampled image.

**The replacement** asks which output columns a segment actually covers -- the
integers in `[u0, u1)` -- and solves for the source position at each. Segment i
ends where segment i+1 begins, so on a connected surface every output column
lies in exactly one segment: measured, 99.88% of candidates win their pixel
uncontested, against 78% before. `max_stretch` bounds coverage to four
columns, so the variable-length expansion a general rasteriser needs -- and
which DirectML has no primitive for -- collapses to four masked passes.

**Two tests with a knowable right answer** decided it, because equality with
the renderer being replaced is the wrong bar.

*Identity.* At baseline 0 the warp is a no-op, so the output must be the
source; anything else is the renderer's own noise floor.

*Constant depth.* `_eye_offset` is a rotation of the horizontal plane, so the
whole sphere shifts by one uniform angle and `map_x` must step by exactly 1 per
column. Departure from that is the hairline artifact, measured rather than
eyeballed. The map is recovered by warping a float32 ramp whose value is the
column index -- it must be float, since a byte-packed ramp wraps every 256
columns and interpolating across that wrap returns nonsense.

| | identity: pixels wrong | mean error | step error | worst column |
|---|---|---|---|---|
| scatter | 37.9% | 1.03 levels | 0.68 px | 4517x median at +134.8 deg |
| scanline | **0.000%** | **0.0000** | **0.00000** | none |

The scatter renderer's worst column landing on +134.8 degrees reproduces the
documented +/-135 hairline from first principles. **The noise floor and the
hairlines were one bug**, and exact coverage removes both -- so the "expect the
mesh render to look slightly softer" caveat no longer applies.

On the real 7680x3840 frame: **99.8 s to 8.7 s, 11.5x**, with *less* left
unfilled (0.094% against 0.138%) because exact coverage wastes nothing.

### Three porting bugs, and what they have in common

None of them produced anything that looked broken.

**The eye-offset sign.** `_eye_offset` is called as `_eye_offset(lam, d,
-baseline)`; inlining it with `+baseline` renders the *opposite eye*. The
output is a clean, plausible stereo frame the whole time. It took reading the
numpy source line by line, not looking at the image, and it was worth 89% of
pixels differing and double the cut area.

**Half a pixel.** The projection ends `+ (w/2 - 1/2)`, not `+ w/2` -- pixel
centres, not corners. Worth only 0.1% on its own, but it changes every
rounding decision.

**A large scalar subtraction.** Building a global scatter index and reducing it
to band-local afterwards -- `flat = cat(flats) - y0 * w` -- is the obvious way
to write it and it silently destroys the result. The offset passes 2**24 at row
2184, and DirectML puts int64-minus-large-scalar through float32, where the
spacing at 20M is 2. The low bit of the target column is lost, every write
lands on an even column, and 21.6% of the frame comes out empty -- *all of it
on odd columns*, and none of it below row 2184. Keeping the index band-local
before the multiply keeps every value under 2**21, which float32 holds exactly
whatever the backend does underneath.

The device primitives were all innocent: `ceil`, `remainder`, `round`,
`.long()`, `index_put_`, `scatter_reduce_`, `expand().reshape()` and int64
arithmetic each tested exact, at scale and at the magnitudes involved. What
found it was dumping the scatter index itself and noticing the k-loop parts
were 50/50 even/odd while their concatenation was 100% even. The lesson is the
project's usual one in a new place: measure the intermediate, not the output.
