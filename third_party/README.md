# third_party/

Upstream checkouts live here. They are **not** in the repository: they are
other people's code, under their own licences, and cloning them is a one-line
command rather than something to redistribute.

Only needed for `--depth-backend video-depth-anything`. Every other backend
works without anything in this directory.

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything third_party/Video-Depth-Anything
pip install -r requirements-vda.txt
```

`stereo360/backends.py` looks for `third_party/Video-Depth-Anything` by
default, or wherever `VIDEO_DEPTH_ANYTHING_PATH` points if you already have a
clone elsewhere. Checkpoint weights download on first use.

`--probe-backends` reports which backends can actually run on this machine and
says what is missing for the ones that cannot.
