#!/usr/bin/env python3
"""Reproducible export of on-device models for the ML Playground Android app.

Produces (in app/src/main/assets/):
  efficientdet_lite0.tflite      MediaPipe object detector (float32, downloaded)
  yoloe_pf_640.tflite / .onnx    YOLOE-11s prompt-free, 640x640 input, compact output
  yoloe_pf_320.tflite / .onnx    same model, 320x320 input
  yoloe_pf_labels.txt            4585 class names, one per line, index order

Compact output (both runtimes): float32 [1, N, 6] = (cx, cy, w, h, score, class_id)
  * cx, cy, w, h normalized to 0..1 relative to the model input (letterboxed) image
  * score = max over classes of sigmoid class scores (already sigmoided)
  * class_id = argmax class index stored as float
  * N = 8400 for 640 input, 2100 for 320 input. No NMS applied (do NMS in app).
The segmentation mask coefficients / prototypes are dropped.

Usage:
  uv venv --python 3.12 tools/.venv
  (install deps, see README.md)
  tools/.venv/bin/python tools/export_models.py [--sizes 640 320]
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent / "app" / "src" / "main" / "assets"
WORK = ROOT / "work"

EFFICIENTDET_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"
)
YOLOE_WEIGHTS = "yoloe-11s-seg-pf.pt"


def download_efficientdet() -> None:
    dst = ASSETS / "efficientdet_lite0.tflite"
    if not dst.exists():
        print(f"Downloading {EFFICIENTDET_URL}")
        urllib.request.urlretrieve(EFFICIENTDET_URL, dst)
    print(f"{dst.name}: {dst.stat().st_size} bytes")


def patch_compact_head() -> None:
    """Make YOLOE segment head return a compact [B, N, 6] tensor in export mode."""
    from ultralytics.nn.modules.head import Detect, YOLOEDetect, YOLOESegment

    def forward(self, x):
        if not self.export:
            return _orig_forward(self, x)
        y = YOLOEDetect.forward(self, x)  # [B, 4 + nc + nm, N]; Segment._inference appends mask coefs
        if isinstance(y, tuple):
            y = y[0]
        nc = self.nc
        box = y[:, :4]  # cx, cy, w, h in input pixels
        scores = y[:, 4 : 4 + nc]  # already sigmoid
        conf, idx = scores.max(1, keepdim=True)  # [B, 1, N]
        s0 = int(self.stride[0])
        h, w = int(self.shape[2]) * s0, int(self.shape[3]) * s0  # input size from P3 grid
        norm = torch.tensor([w, h, w, h], dtype=box.dtype, device=box.device).view(1, 4, 1)
        out = torch.cat([box / norm, conf, idx.to(box.dtype)], 1)  # [B, 6, N]
        return out.transpose(1, 2).contiguous()  # [B, N, 6]

    _orig_forward = YOLOESegment.forward
    YOLOESegment.forward = forward
    assert issubclass(YOLOESegment, Detect)


def export_yoloe(sizes: list[int], skip_tflite: bool = False) -> None:
    from ultralytics import YOLOE

    WORK.mkdir(exist_ok=True)
    patch_compact_head()

    names = None
    for imgsz in sizes:
        wdir = WORK / f"yoloe_{imgsz}"
        wdir.mkdir(exist_ok=True)
        os.chdir(wdir)  # ultralytics downloads weights / writes exports relative to cwd
        pt = YOLOE_WEIGHTS
        model = YOLOE(pt)
        model.model.task = "detect"  # single output tensor (no proto output)
        names = model.names

        onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=17, dynamic=False)
        shutil.copy(onnx_path, ASSETS / f"yoloe_pf_{imgsz}.onnx")

        if not skip_tflite:
            model = YOLOE(pt)
            model.model.task = "detect"
            # 'saved_model' runs the onnx2tf path (NHWC input) and writes *_float32/_float16.tflite
            sm_dir = Path(model.export(format="saved_model", imgsz=imgsz))
            tfl = next(sm_dir.glob("*_float16.tflite"))
            shutil.copy(tfl.resolve(), ASSETS / f"yoloe_pf_{imgsz}.tflite")

    labels = ASSETS / "yoloe_pf_labels.txt"
    labels.write_text("\n".join(names[i] for i in range(len(names))) + "\n")
    print(f"labels: {len(names)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 320])
    ap.add_argument("--skip-tflite", action="store_true")
    args = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    download_efficientdet()
    export_yoloe(args.sizes, args.skip_tflite)
    for f in sorted(ASSETS.iterdir()):
        print(f"{f.name:32s} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
