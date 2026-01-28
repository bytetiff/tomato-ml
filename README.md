# Tomato Leaf Disease Classification (Continuous Learning, Field Conditions)

This project focuses on tomato leaf disease recognition under field conditions and cross‑dataset transfer. The baseline is PlantVillage, followed by adding a domain‑shifted dataset (PlantDoc), leakage/duplicate cleaning, and fine‑tuning. The practical goal is robust classification across domains; the scientific goal is a reproducible continuous learning loop (data audit → adaptation → re‑evaluation).

## Scientific relevance
In agri‑tech, accuracy often drops when moving from lab images to real field photos (background, illumination, artifacts). This project demonstrates a repeatable continuous learning workflow: regular dataset leakage checks, class expansion from new sources, model adaptation, and re‑validation on a combined test set.

## Current model
Primary model: ConvNeXt‑Tiny `convnext_tiny.fb_in22k_ft_in1k` (timm), trained on XPU (Intel Arc B580).

Best checkpoint (mixed, lr=5e-5):
`experiments/teacher/convnext_tiny_mix_clean_lr5e5/convnext_tiny.fb_in22k_ft_in1k-xpu-best.pth`

Datasets:
- PlantVillage (base, 10 classes) — `data/raw/plantvillage`
- PlantDoc (8 classes, domain adaptation) — `data/raw/plantdoc`
- Additional sources: `data/raw/kaggle_TLDS`, `data/raw/mendeley_TLID` (audited for leakage/duplicates)

## Results (latest, 5 decimals)
PlantVillage (clean, test split):
Acc=0.99793, Balanced Acc=0.99801, Macro F1=0.99791, PR-AUC=0.99973, ROC-AUC=0.99997  
Artifacts: `outputs/metrics_clean`

PlantDoc only (drop-missing, 8 classes):
Acc=0.75758, Balanced Acc=0.74722, Macro F1=0.74868, PR-AUC=0.85427, ROC-AUC=0.96994  
Artifacts: `outputs/metrics_plantdoc_only`, confusion matrix: `outputs/confmat_plantdoc_only/confusion_matrix.*`

Combined test (PlantVillage test + PlantDoc test, 10 classes):
Acc=0.98385, Balanced Acc=0.97791, Macro F1=0.98038, PR-AUC=0.99784, ROC-AUC=0.99979  
Artifacts: `outputs/metrics_combined`, confusion matrix: `outputs/confmat_combined/confusion_matrix.*`

## Interpretability and visualization
Grad‑CAM (ConvNeXt):
```
python -m src.analysis.grad_cam --ckpt <ckpt> --csv <csv> --class-map data/processed/class_to_idx.json --per-class --limit 10 --out outputs/gradcam
```
Examples (PlantVillage baseline): `outputs/gradcam_clean/`

Attention rollout (ViT):
```
python -m src.analysis.visualize_attention --ckpt <ckpt> --csv data/processed/splits/val.csv --class-map data/processed/class_to_idx.json --per-class --limit 10 --out outputs/attn
```

## Evaluation
Classifier metrics:
```
python -m src.analysis.eval_classifier --ckpt <ckpt> --csv <csv> --class-map data/processed/class_to_idx.json --out <out> --device auto --batch-size 64
```
Confusion matrix:
```
python -m src.analysis.eval_confusion_matrix --ckpt <ckpt> --data <csv> --class-map data/processed/class_to_idx.json --output-dir <out> --img-size 224 --batch-size 64 --device auto
```

## Leakage and duplicate checks
SHA1 + pHash:
```
python -m src.analysis.check_leakage --splits train=<csv> val=<csv> test=<csv> --root . --phash --phash-threshold 4
```
pHash sweep (threshold selection):
```
python -m src.analysis.check_leakage --splits train=<csv> test=<csv> --root . --phash-sweep 0-16 --phash-sweep-chunk 512
```

## Quickstart
1) Install dependencies:
```
pip install -r requirements.txt
```
2) Prepare PlantVillage:
```
python -m src.data.prepare_plantvillage
```
3) Train ConvNeXt:
```
python -m src.training.train_teacher --accelerator xpu --precision 32-true --devices 1 --model-name convnext_tiny.fb_in22k_ft_in1k
```

## Project structure
`src/` — data, training, analysis  
`outputs/` — metrics, matrices, plots, visualizations  
`experiments/` — checkpoints  
`data/` — raw and processed datasets  
`docs/` — logs and plan

## Note
The workflow is built for continuous learning: new datasets are converted to CSV, audited for leakage, and then used for fine‑tuning or mixed training.
