# models/

Exported ONNX depth models live here. They are **not** in the repository —
together they run to about 194 MB, nothing in them is authored here, and
`scripts/export_onnx.py` regenerates them on demand.

Only needed for `--depth-backend onnx`. The default backend downloads its
weights from the HuggingFace Hub and needs nothing in this directory.

```bash
pip install -r requirements-onnx.txt

# The default the CLI looks for (~100 MB, plus an .onnx.data sidecar)
python scripts/export_onnx.py

# Fast mode for slow machines: much quicker, much coarser depth
python scripts/export_onnx.py --size 266 --out models/depth_fast.onnx

# DirectML rejects the graph once the batch axis is dynamic, even at batch 1
python scripts/export_onnx.py --size 266 --static-batch --out models/depth_fast.onnx
```

Point `--onnx-model` at any of them. See "ONNX Runtime backend" and "Fast mode
for slow machines" in the top-level README for what each size costs and buys.
