# experiments

Working code behind the numbers in `findings.md`. **None of it is part of the
tool and nothing in `stereo360/` imports it.** It is kept so the measurements
can be reproduced and built on, not because it is finished.

Every script here expects `STEREO360_REPO` to point at the checkout, and most
read `7680p.jpg` or `indoor.jpg` from it. `indoor.jpg` is not in the
repository — see the note in `findings.md`.

## `score.py`, `damp.py`

The depth scorer and the rejected damping experiments. See `findings.md`.

## `metric_depth/`

Running Depth Anything 3's metric variant, which is the only model here that
produces a scale rather than relative depth.

| file | what it does |
|---|---|
| `da3_shim.py` | imports the package without its export-only dependencies, and replaces a `forward` that assumes CUDA |
| `da3_metric.py` | runs the model over six widened faces and reports whether they agree |
| `da3_fuse.py` | combines a sharp high-resolution run with a calibrated low-resolution one |
| `da3_render.py` | metric normalisation, the sky-face rescue, and a baseline in millimetres |

The model is `depth-anything/DA3METRIC-LARGE`, 1.34 GB, and the package is
`depth-anything-3` — install it with `--no-deps`, or its solver will try to
compile xformers against this torch.

## `stereo_geometry/`

How the separation is divided between the eyes, and a renderer that keeps the
depth map's surface connected instead of splatting independent points.

| file | what it does |
|---|---|
| `mesh_warp.py` | the mesh renderer: quads from depth samples, cut on projected stretch, composited far to near |
| `mesh_render.py` | drives it for a whole frame and writes a stereo pair |
| `splat_share.py` | the same, through the existing splat path, at an arbitrary share |
| `three_way.py` | whole against split against chained, by how much the eyes disagree |
| `left_variants.py` | a left eye reconstructed without moving it, and a round trip |
| `asym_split.py` | the share sweep that found the knee at about 15% |

The measure throughout is not each eye's fidelity but the difference between
them, on the argument recorded in `findings.md` that a pair which agrees on a
slightly wrong shape is easier to look at than one right eye and one wrong one.

`mesh_warp.py` is a prototype rasteriser and has two known defects, both from
scattering rounded samples rather than computing coverage per output pixel: a
noise floor the splat does not have, and four hairlines at longitudes +/-45
and +/-135 degrees. It also leaves some of its cuts unfilled. See `plans/todo.md`.
