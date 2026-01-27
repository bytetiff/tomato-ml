# Tomato ViT

Vision Transformer pipeline for PlantVillage tomato leaf disease classification, ONNX export, and INT8 quantization aimed at Android (Poco X3 NFC).

## Quickstart
- Place PlantVillage dataset under `data/raw/plantvillage/` (tomato classes already present).
- Create and activate a Python 3.10+ env, then install deps: `pip install -r requirements.txt`.
- Prepare CSVs: `python -m src.data.prepare_plantvillage` (creates `data/processed/splits/{all,train,val,test}.csv`).
- Train teacher (example ViT): `python -m src.training.train_teacher --accelerator xpu --model-name vit_tiny_patch16_224`.
- Train teacher (example ConvNeXt): `python -m src.training.train_teacher --accelerator xpu --model-name convnext_tiny.fb_in22k_ft_in1k`.
- Train student: `python -m src.training.train_student`.
- Distill: `python -m src.training.distill`.
- Export ONNX FP32: `python -m src.export.export_onnx`.
- Quantize ONNX INT8: `python -m src.quant.quantize_onnx`.

## Interpretability / Debug
- ViT/DeiT attention rollout: `python -m src.analysis.visualize_attention --ckpt <...> --csv data/processed/splits/val.csv --class-map data/processed/class_to_idx.json --per-class --limit 10 --out outputs/attn`
- CNN/ConvNeXt Grad-CAM: `python -m src.analysis.grad_cam --ckpt <...> --csv data/processed/splits/val.csv --class-map data/processed/class_to_idx.json --per-class --limit 10 --out outputs/gradcam`
- Gradient attributions (saliency/IG): `python -m src.analysis.saliency_attribution --ckpt <...> --csv data/processed/splits/val.csv --class-map data/processed/class_to_idx.json --per-class --limit 10 --out outputs/saliency --method ig`
- Occlusion tests (corners/border/center): `python -m src.analysis.occlusion_sensitivity --ckpt <...> --csv data/processed/splits/val.csv --class-map data/processed/class_to_idx.json --per-class --limit 10 --threshold 0.15`

## Intel Arc (torch.xpu) check
Run:
```bash
python - <<'PY'
import torch
print("torch xpu available:", torch.xpu.is_available() if hasattr(torch, "xpu") else False)
if hasattr(torch, "xpu") and torch.xpu.is_available():
    print("xpu count:", torch.xpu.device_count())
    print("current device:", torch.xpu.current_device())
    print("device name:", torch.xpu.get_device_name(torch.xpu.current_device()))
PY
```
Then verify Intel Extension for PyTorch loads:
```bash
python - <<'PY'
try:
    import intel_extension_for_pytorch as ipex
    print("IPEX imported:", ipex.__version__)
except Exception as e:
    print("IPEX import failed:", e)
PY
```

## Android integration notes
- Input: RGB 224x224, ImageNet mean/std normalization (same as training).
- ONNX Runtime Mobile or convert ONNX→TFLite. Run on CPU/GPU if available.
- Outputs logits; apply softmax/argmax to get class id. Map ids to tomato disease names (see `src/data/prepare_plantvillage.py` label map).

## Layout
See `configs/` for hydra configs, `src/` for data/model/training/eval/export/quant modules, `experiments/` for checkpoints and reports, and `mobile/onnx` for deployment artifacts.
