# Stereoscopic 360 photos from a single 360 image

Branch `image-input`, off `vr180` at `fba5213`.

## It already works

Before planning anything, the thing was tried:

```bash
python -m stereo360 equirect_8k.jpg -o stereo.jpg --preview-frame 0 --preview-width 0
```

An 8K equirect JPEG in, a **7680x7680 stereo JPEG out, in 5.4 seconds**, with no
code change at all. ffmpeg reads a still as a one-frame video, `--preview-frame`
already writes an image, and depth, warp and stacking never cared where the
frame came from.

So this is not a new pipeline. It is a front door, an encoder setting, and a
metadata writer around one that exists.

## Not breaking video

Same rule as `vr180.md`, and it is cheap to keep here. Nothing in `projection`,
`warp` or `pipeline`'s frame path needs to change. What is new sits at the two
ends:

- **input**: recognising an image and refusing to ask a still for its frame rate
- **output**: a JPEG encoder configured properly, and XMP instead of MP4 boxes

`spherical.py` is untouched. Stills carry **XMP packets in an APP1 segment**,
which is a different mechanism from `st3d`/`sv3d`/`SA3D` entirely, so this is a
new writer alongside it rather than a change to it.

## What the metadata can and cannot say

Researched rather than assumed, and one of the answers is a plain no.

### GPano has no StereoMode

Google's Photo Sphere spec defines these, and **no stereo property of any
kind**:

| Required | Optional |
|---|---|
| `ProjectionType` (only `equirectangular`) | `UsePanoramaViewer`, `PoseHeadingDegrees`, `PosePitchDegrees`, `PoseRollDegrees` |
| `CroppedAreaImageWidthPixels` | `InitialViewHeadingDegrees`, `InitialViewPitchDegrees`, `InitialViewRollDegrees` |
| `CroppedAreaImageHeightPixels` | `InitialHorizontalFOVDegrees`, `InitialVerticalFOVDegrees` |
| `FullPanoWidthPixels` | `CaptureSoftware`, `StitchingSoftware`, `SourcePhotosCount` |
| `FullPanoHeightPixels` | `FirstPhotoDate`, `LastPhotoDate`, `ExposureLockUsed` |
| `CroppedAreaLeftPixels`, `CroppedAreaTopPixels` | `InitialCameraDolly` |

`GSpherical:StereoMode` — the `top-bottom` / `left-right` field this tool
already writes — belongs to the **video** spec and lives in an MP4 box. There is
no photo equivalent. So of the two values proposed for the metadata,
`ProjectionType` exists and `StereoMode` does not.

### Google's own stereo photo format is a different shape

The Cardboard Camera / VR Photo format does not stack the eyes at all:

- the **container JPEG is the left eye**
- the **right eye is a whole JPEG, base64-encoded into `GImage:Data`**, with
  `GImage:Mime = image/jpeg`
- optional ambient audio goes the same way in `GAudio:Data`
- GPano describes the projection as usual

Two consequences worth knowing before committing to it:

1. A base64 JPEG does not fit in one APP1 segment, so this needs **ExtendedXMP**
   — `xmpNote:HasExtendedXMP` plus a hash and multiple segments. Real work, not
   a field.
2. Google Photos is **fussy about the serialization**: of several semantically
   equivalent XMP forms, only the one with properties as *attributes* of a
   single `rdf:Description` renders in stereo. The others show the same image to
   both eyes — failing in the way that looks like success.

### So there are two targets, not one

| | Google Photos / Cardboard | VR photo viewers |
|---|---|---|
| shape | left eye + right embedded | one stacked frame |
| stereo signalled by | `GImage:Data` | filename, or the app's own toggle |
| projection signalled by | GPano | GPano, or filename |
| effort | ExtendedXMP, exact serialization | write the file |

The pipeline naturally produces the right-hand column. The left-hand one is a
second output shape, and worth building only if the device testing says the
stacked form is not read.

## Filenames are load-bearing here

With no stereo field in the metadata, the filename is how most players are told.
The conventions converge across SKYBOX, DeoVR and HereSphere, and are
case-insensitive and separator-agnostic:

| Meaning | Tokens |
|---|---|
| projection | `360`, `180`, or `360x180` / `180x180` |
| top-bottom | `TB`, `OU`, `3DV` |
| side-by-side | `SBS`, `LR`, `3DH` |
| mono | no stereo token at all |

So `garden_360_TB.jpg` and `garden_180x180_3dh.jpg` are self-describing. The tool
should **suggest an output name that follows this**, since it knows both facts
and the user would otherwise have to know the convention.

## Encoding: measured, on a real 8K stereo frame

Current behaviour is `cv2.imencode(".jpg", ...)` with **defaults** — quality 95
and 4:2:0 chroma. On the 7680x7680 render, against the lossless PNG:

| setting | MB | vs default | rms error |
|---|---|---|---|
| q95 4:2:0 *(what it does today)* | 10.8 | 1.00x | 1.188 |
| q95 **4:4:4** | 12.7 | 1.17x | 0.879 |
| q98 4:4:4 | 16.4 | 1.52x | 0.692 |
| q100 4:4:4 | 21.8 | 2.02x | 0.606 |
| **q100 4:4:4 + optimize** | **19.6** | 1.81x | **0.606** |

Two things fall out:

- **Dropping 4:2:0 is the cheapest win by far** — 26% less error for 17% more
  bytes. It is the same argument as chroma subsampling in video, and it bites
  harder in a still that will be magnified and stared at.
- **`IMWRITE_JPEG_OPTIMIZE` is free**: identical pixels, 10% smaller, because it
  only computes better Huffman tables. It should always be on.

Encoding at the top setting costs 0.4 s against a 5.4 s render. There is no
reason not to.

### MozJPEG: not worth pursuing here

Not installed and not reachable — no `cjpeg`, no `mozjpeg`, no bindings, and
OpenCV links plain libjpeg-turbo. It could be added as a dependency, but the
case is weak: MozJPEG's advantage is **smaller files at equal quality**, largely
through trellis quantisation and better progressive scans. At q100 4:4:4 for an
archival still, size is not the binding constraint, and it would not make the
image *better*. `IMWRITE_JPEG_OPTIMIZE` already takes the free part.

Worth revisiting only if file size becomes a complaint.

## Depth: the proposal is contradicted by the measurement

"Use the largest depth model since speed is not an issue" assumes Large is
better and merely slower. It is not — from `findings.md`, measured on 8K frames:

| Model | noise | edge sharpness | sharp/noise |
|---|---|---|---|
| **Base** (default) | **0.071** | **180.2** | **2546** |
| Large | 0.074 | 169.7 | 2298 |

**Large is worse on every axis measured, not just slower.** Free time does not
make it the right choice. Base stays the default for stills too.

Free time *does* buy two other things, and these are the levers to reach for:

- **`--depth-tiles N`** — N×N tiles per cube face, finer depth on thin
  structures like railings and cables, N² times slower. Prohibitive for video;
  for one still, minutes. This is the real "quality at any cost" setting.
- **`--inpaint learned`** — LaMa, much better texture in disocclusions, slow on
  CPU. One frame makes that irrelevant.

`--smooth` stays off even so: it measured 66% of runtime for no visible gain,
and free time does not improve a setting that does nothing.

## Interface

The video controls are mostly meaningless here, and showing them would imply
they do something. What survives, and what does not:

| Keep | Drop |
|---|---|
| output format (360 / VR180) and direction | encoding preset, CRF, codec, bit depth |
| output resolution | frame range, preview frame |
| strength, gradient limit, split baseline | spatial audio, chunk size/overlap, temporal fill |
| depth model, tiles, inpaint, device | the temporal depth backend entirely |

**The preview pane stops being a preview.** For video it exists so you can judge
one frame before committing to an hour; for a still, that frame *is* the
deliverable. So the pane becomes: the **input** when a file is opened, replaced
by the **output** once converted. One panel, two states, no "render preview"
button.

Whether this is a mode inside the same window or a separate view is a UI
question, not an architectural one — `options.build_argv` already keys off a
plain dict, so an image mode is a different set of controls feeding the same
builder.

## Answered on a Quest 3

Nine files, one variable at a time, neutral filenames wherever the name was not
the thing under test.

| file | MP | XMP | filename | stereo? |
|---|---|---|---|---|
| `a_size_big` | 59.0 | — | — | no |
| `b_size_small` | 33.2 | — | — | no |
| `e_stereo_gpano_oneeye` | 59.0 | GPano 2:1 | — | **yes** |
| `f_stereo_gpano_fullframe` | 59.0 | GPano 1:1 | — | **yes** |
| `g_stereo_360_TB` | 59.0 | GPano 2:1 | `_360_TB` | **yes** |
| `h_stereo_bare_360_TB` | 59.0 | — | `_360_TB` | **yes** |
| `i_vr180_180x180_3dh` | 29.5 | GPano 180 | `_180x180_3dh` | **yes** |

### 59 megapixels is fine for an image

Four of the five that worked are 7680x7680. **The 35.6 Mpx cap does not apply
to stills** — as expected, because it constrains the video decoder and a JPEG
is a texture. The size that makes 8K 360 *video* unplayable is a non-issue for
8K 360 *photos*, so the full-resolution output needs no compromise here and
`--output-width` is not needed for images.

### GPano alone is enough, and so is the filename alone

`e` and `f` carry no filename tokens and worked. `h` carries no metadata at all
and worked. `a` has neither and did not. So either signal is sufficient and
neither is required — which means writing both is belt and braces, not a
choice to agonise over.

### The GPano dimensions do not matter

`e` and `f` disagree about what the panorama is — one says 7680x3840 on a
7680x7680 file, the other says 7680x7680 — and **both worked**. So the reader
is not doing arithmetic on those fields to infer the layout. The presence of
GPano marks the file as a panorama; the 1:1 aspect then reads as top-bottom.

That also removes the question the plan could not answer from the spec. Either
form is acceptable, so write the honest one: GPano describing **one eye**, which
is the panorama the projection fields are actually about.

### The Cardboard / VR Photo format is not needed

This is the significant one. Step 7 was the only genuinely large piece of work
here — ExtendedXMP, base64 of a whole second JPEG, and an exact `rdf`
serialization or Google Photos silently shows both eyes the same. **A plain
stacked frame is read as stereo**, so none of it is required.

It stays worth knowing about for Google Photos specifically, which was never
tested here and is a different reader from the headset gallery. But it is no
longer on the path.

### VR180 stills work too

`i` is the other output mode, side by side, and it worked with GPano plus
`_180x180_3dh`. Both output modes are viable for photos.

## Still open

- **Google Photos** was not tested, and is the one reader likely to want the
  Cardboard format. Only matters if sharing through it is wanted.
- **Whether `c` wrapped and `d` stayed flat** — the mono pair, testing whether
  GPano drives projection on its own. Informational only: every output this
  tool produces is stereo, and stereo detection is already settled.

## Superseded: needing the headset

- **Does 7680x7680 display as an image?** The 35.6 Mpx cap is a property of the
  *video decoder*; a JPEG is decoded and uploaded as a texture, where the limit
  is `GL_MAX_TEXTURE_SIZE` — typically 16384 on Adreno, so 7680 should be
  comfortable. Expected to work, not verified. Three files are on the Desktop to
  settle it:

  | file | tests |
  |---|---|
  | `imgtest_360_mono_30mp.jpg` | control: one eye, 2:1, GPano tagged |
  | `imgtest_360_stereo_59mp.jpg` | the question, GPano tagged |
  | `imgtest_plain_59mp.jpg` | same pixels, no metadata — separates "too big" from "tagged wrong" |

- **Does a stacked top-bottom JPEG read as stereo anywhere**, with GPano and/or
  a filename token? This decides whether the Cardboard format is needed at all.
- **Does the Quest gallery read GPano** for the projection, or does it need the
  filename?

## Suggested order

1. **Input**: accept image extensions, and a real entry point rather than
   `--preview-frame 0 --preview-width 0`.
2. **Encoding**: q100, 4:4:4, optimize. Small, measured, immediately better.
3. ~~**Device testing**~~ — **done**, nine files, results above.
4. **GPano XMP writer**, projection only — it is all the spec has, and it is
   enough. Describe one eye; the dimensions are not read for layout.
5. **Filename suggestion** following the player conventions.
6. **UI**: the reduced control set and the input/output panel.
7. ~~**Cardboard/VR Photo output**~~ — **not needed.** The stacked frame is
   read as stereo, so the largest piece of work in this plan is cancelled.
   Revisit only for Google Photos, which is a different reader and untested.

Sources: [GPano spec](https://developers.google.com/streetview/spherical-metadata),
[Cardboard Camera VR Photo format](https://developers.google.com/vr/reference/cardboard-camera-vr-photo-format),
[Google Photos VR180 XMP behaviour](https://github.com/imrivera/google-photos-vr180-test),
[SKYBOX filename rules](https://forum.skybox.xyz/d/157-filename-rules-for-vr-format),
[DeoVR naming options](https://forum.deovr.com/d/40-play-options-as-file-naming-options)
