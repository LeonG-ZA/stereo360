"""A/B: dump depth maps under different alignment/reassembly variants."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from stereo360 import projection
from stereo360.depth.onnx_backend import OnnxDepthBackend

frame = cv2.imread("frame.png")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
h, w = frame.shape[:2]
face_size = min(512, h)

backend = OnnxDepthBackend("models/depth_anything_v2_small.onnx")
faces_rgb = projection.equirect_to_cubemap(frame, face_size)
raw = {f: backend.estimate(faces_rgb[f]) for f in projection.FACES}


def dump(faces, name, nearest):
    disp = projection.cubemap_to_equirect(faces, w, h, nearest=nearest)[..., 0]
    dn = (disp - disp.min()) / (disp.max() - disp.min() + 1e-9)
    cv2.imwrite(f"align_{name}.png", (dn * 255).astype(np.uint8))
    return disp


# 1. no alignment, nearest
dump({f: d.copy() for f, d in raw.items()}, "none_nearest", True)
# 2. full current alignment (affine w/ fallback), bilinear padded
faces2 = {f: d.copy() for f, d in raw.items()}
projection.align_face_scales(faces2)
dump(faces2, "affine_bilinear", False)
dump({f: d for f, d in faces2.items()}, "affine_nearest", True)

# 3. scale-only alignment (monkeypatch affine solver off), bilinear padded
import stereo360.projection as P
orig = P._solve_face_affine
P._solve_face_affine = lambda *a, **k: None
faces3 = {f: d.copy() for f, d in raw.items()}
projection.align_face_scales(faces3)
P._solve_face_affine = orig
dump(faces3, "scaleonly_bilinear", False)

# Report affine params per face
pairs_params = projection._solve_face_affine
print("done")
