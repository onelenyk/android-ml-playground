# ML Models Export & Verification Tools

## Setup Python environment
System Python 3.14 may lack wheels for PyTorch/Ultralytics, so use Python 3.12 via `uv`:

```bash
uv venv --python 3.12 tools/.venv
uv pip install --python tools/.venv/bin/python ultralytics onnx onnxruntime onnxsim onnx2tf ai-edge-litert pillow
```

## Export Detection Models
Exports `efficientdet_lite0.tflite` (MediaPipe COCO 80), plus `yoloe-11s-seg-pf.pt` (YOLOE prompt-free, 4,585 classes) to both `.tflite` (FP16, NHWC) and `.onnx` (opset 17, NCHW) at 640×640 and 320×320 with compact head `[1, N, 6] = (cx, cy, w, h, score, class_id)`:

```bash
tools/.venv/bin/python tools/export_models.py --sizes 640 320
```

## Verify Exported Models
Runs both ONNX Runtime and LiteRT on a test image (`bus.jpg` by default) and compares top detections:

```bash
tools/.venv/bin/python tools/verify_models.py
```

## Gemma 4 E2B (LiteRT-LM) Setup for "Опис AI"
Download `gemma-4-E2B-it.litertlm` (~2.59 GB, Apache 2.0) and push to the Pixel 7a app external storage:

```bash
mkdir -p tools/work
curl -L -C - "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm" -o tools/work/gemma-4-E2B-it.litertlm
adb shell mkdir -p /sdcard/Android/data/dev.onelenyk.mlplayground/files/models
adb push tools/work/gemma-4-E2B-it.litertlm /sdcard/Android/data/dev.onelenyk.mlplayground/files/models/gemma-4-E2B-it.litertlm
```
