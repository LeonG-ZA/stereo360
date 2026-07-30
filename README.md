# stereo360

Convert monoscopic 360° equirectangular video into stereoscopic top-bottom 360°
video for VR headsets.

## What it does

Ordinary 360 footage is flat: you can look around, but nothing has depth. This
takes a single-lens 360 video and synthesises the second eye, so the result has
real stereo in every direction.

It works by estimating depth on six cubemap faces — a perspective view is what
depth models were trained on, an equirect frame is not — reconciling those six
into one depth map, then re-rendering the scene from a second viewpoint.

- **Depth in every direction.** The virtual eye separation follows your gaze
  rather than one fixed axis, so the stereo is correct behind you as well as in
  front.
- **No seams.** Faces overlap and cross-fade, so the ground does not crease
  where two of them meet.
- **Steady between frames.** Depth ranges are smoothed over time, and a
  temporal model is available for footage where flicker matters most.
- **Ready to upload.** Google spherical metadata is written, the original audio
  is copied through, and ambisonic audio can be tagged as such.
- **8K capable**, with hardware encoder support on NVIDIA, Intel, AMD, Linux
  and macOS.

Output is a top-bottom MP4 that VR players and YouTube detect automatically.

## Requirements

- Python 3.10+
- FFmpeg and ffprobe on `PATH`
- A GPU is strongly recommended. On the CPU, depth estimation is roughly an
  order of magnitude slower and becomes ~90% of the runtime.

## Installing

```bash
git clone https://github.com/LeonG-ZA/stereo360.git
cd stereo360
pip install -r requirements.txt
```

Then add the right accelerator for your machine. **This step matters more than
any other setting** — the wrong one silently runs everything on the CPU.

### NVIDIA

The default `pip install torch` wheel is CPU-only on Windows, so install the
CUDA build explicitly:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Nothing else to configure. The tool detects CUDA and uses it.

### AMD or Intel

PyTorch has no ROCm build on Windows, so the GPU path is DirectML through ONNX
Runtime instead:

```bash
pip install -r requirements-onnx.txt
pip install onnxruntime-directml
python scripts/export_onnx.py --static-batch    # one-time, ~100 MB
python -m stereo360 input.mp4 -o output.mp4 --depth-backend onnx
```

`--static-batch` is required: DirectML rejects the graph once the batch axis is
dynamic, even at batch 1.

On **Linux with AMD**, ROCm PyTorch does exist — install that instead and use
the default backend, as for NVIDIA.

### Apple Silicon

```bash
pip install -r requirements-onnx.txt
```

Torch uses Metal (MPS) automatically; ONNX Runtime picks CoreML if you prefer
that path.

### No GPU

It will still run. Use [fast
mode](findings.md#fast-mode-for-slow-machines), which trades depth detail for
a 3–4× cut in the part that dominates.

### Checking it worked

```bash
python -m stereo360 --probe-backends
```

Reports which backends can actually run here, and what is missing for each that
cannot. Every run also prints what it selected on startup — if it says **GPU
not being utilized**, the accelerator above did not take.

## Running the desktop UI

```bash
pip install -r requirements-ui.txt
python -m stereo360_ui
```

A settings panel, a one-frame preview so you can judge settings without sitting
through a render, live progress with an ETA, and a Stop button that leaves you
a playable file.

The interface is a thin shell over the command line: it builds a command, runs
it as a child process, and shows that command in the log — so anything you set
up here can be pasted into a terminal. Running the core separately also means a
depth model exhausting VRAM cannot take the window down with it.

## Running from the command line

```bash
python -m stereo360 input.mp4 -o output.mp4
```

Useful starting points:

```bash
# see one frame before committing to a full render
python -m stereo360 input.mp4 -o preview.png --preview-frame 0

# final quality for VR viewing
python -m stereo360 input.mp4 -o output.mp4 --codec libx265 --crf 16 --preset slow

# steadiest depth, if flicker is what bothers you
python -m stereo360 input.mp4 -o output.mp4 --depth-backend video-depth-anything
```

Every flag, and the measurements behind each default, are in
[findings.md](findings.md).

## Documentation

- **[findings.md](findings.md)** — full flag reference, and the measurements
  behind the defaults: which depth model and why, what CRF is actually visually
  lossless in a headset, cube seam removal, stereo geometry, hardware encoders,
  colour handling.
- **[plans/360-stereo-converter-design.md](plans/360-stereo-converter-design.md)**
  — design and milestone roadmap.

## Licence

MIT — see [LICENSE](LICENSE).

The models this tool drives are not included and carry their own terms:

| Component | Where it comes from | Licence |
|---|---|---|
| Depth Anything V2 (Small / Base / Large) | downloaded from the HuggingFace Hub on first use | Apache-2.0 |
| Video Depth Anything | cloned into `third_party/` by you — see [`third_party/README.md`](third_party/README.md) | see that repository |
| FFmpeg | must be on `PATH` | LGPL/GPL, depending on the build |

Nothing in this repository redistributes them. `models/` and `third_party/`
hold generated and cloned content and are deliberately not committed — a fresh
clone is under 1 MB, and each directory's README says how to populate it.
