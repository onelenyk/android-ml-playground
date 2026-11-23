#!/usr/bin/env python3
"""Run the exported YOLOE prompt-free models with onnxruntime and LiteRT and compare detections.

Mirrors what the Kotlin postprocessor must do:
  1. letterbox image to SxS (keep aspect, pad with 114 gray, centered), RGB, float32 /255 -> 0..1
  2. ONNX input  'images'  float32 [1, 3, S, S] (NCHW)
     TFLite input           float32 [1, S, S, 3] (NHWC)
  3. output float32 [1, N, 6] = (cx, cy, w, h, score, class_id); boxes normalized 0..1 of letterboxed input
  4. score threshold -> class-agnostic or per-class NMS -> undo letterbox

Usage: tools/.venv/bin/python tools/verify_models.py [image] [--sizes 640 320]
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent / "app" / "src" / "main" / "assets"
BUS_URL = "https://ultralytics.com/images/bus.jpg"


def letterbox(img: Image.Image, s: int):
    w, h = img.size
    r = min(s / w, s / h)
    nw, nh = round(w * r), round(h * r)
    px, py = (s - nw) // 2, (s - nh) // 2
    canvas = Image.new("RGB", (s, s), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    x = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC RGB 0..1
    return x, r, px, py


def nms(boxes, scores, iou_thr=0.5):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return keep


def decode(out, s, r, px, py, conf=0.25, iou=0.5):
    out = out.reshape(-1, 6)
    out = out[out[:, 4] >= conf]
    cx, cy, w, h = (out[:, :4] * s).T  # -> letterboxed pixels
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    keep = nms(boxes, out[:, 4], iou)  # class-agnostic NMS
    boxes = boxes[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / r  # undo letterbox -> original image pixels
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / r
    return [(int(c), float(sc), b) for c, sc, b in zip(out[keep, 5], out[keep, 4], boxes)]


def run_onnx(path, x):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    i, o = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"  ONNX   in  {i.name} {i.shape} {i.type} | out {o.name} {o.shape} {o.type}")
    return sess.run(None, {i.name: x.transpose(2, 0, 1)[None]})[0]


def run_tflite(path, x):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    print(f"  TFLite in  {i['name']} {list(i['shape'])} {i['dtype'].__name__} | out {o['name']} {list(o['shape'])} {o['dtype'].__name__}")
    it.set_tensor(i["index"], x[None])
    it.invoke()
    return it.get_tensor(o["index"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=str(ROOT / "work" / "bus.jpg"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 320])
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        img_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BUS_URL, img_path)
    labels = (ASSETS / "yoloe_pf_labels.txt").read_text().splitlines()
    print(f"labels: {len(labels)}  image: {img_path}")
    img = Image.open(img_path).convert("RGB")

    ok = True
    for s in args.sizes:
        print(f"\n=== {s}x{s} ===")
        x, r, px, py = letterbox(img, s)
        res = {}
        for name, fn, ext in (("onnx", run_onnx, "onnx"), ("tflite", run_tflite, "tflite")):
            p = ASSETS / f"yoloe_pf_{s}.{ext}"
            if not p.exists():
                print(f"  missing {p.name}")
                continue
            raw = fn(p, x)
            res[name] = raw
            dets = decode(raw, s, r, px, py, args.conf)
            print(f"  [{name}] {len(dets)} detections (conf>={args.conf}, agnostic NMS 0.5):")
            for c, sc, b in sorted(dets, key=lambda d: -d[1])[:10]:
                print(f"     {labels[c]:24s} {sc:.3f}  [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")
        if len(res) == 2:
            a, b = res["onnx"].reshape(-1, 6), res["tflite"].reshape(-1, 6)
            box_err = np.abs(a[:, :4] - b[:, :4]).max()
            score_err = np.abs(a[:, 4] - b[:, 4]).max()
            top = a[:, 4] > args.conf
            cls_agree = (a[top, 5] == b[top, 5]).mean() if top.any() else 1.0
            print(f"  diff: max|box|={box_err:.4f} (normalized) max|score|={score_err:.4f} "
                  f"class agreement on conf>{args.conf} anchors={cls_agree:.3f}")
            ok &= box_err < 0.02 and score_err < 0.05 and cls_agree > 0.9
    print("\nRESULT:", "PASS" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
