"""
Оценка чекпойнта (timm-модель) по CSV: accuracy, balanced accuracy, macro F1, macro PR-AUC, macro ROC-AUC.
Сохраняет графики ROC/PR по классам.

Пример:
python -m src.analysis.eval_classifier ^
  --ckpt experiments\\teacher\\convnext_tiny\\convnext_tiny.fb_in22k_ft_in1k-xpu-best.pth ^
  --csv data\\processed\\splits\\test.csv ^
  --class-map data\\processed\\class_to_idx.json ^
  --out outputs\\metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.data.plantvillage_csv import PlantVillageCsvDataset
from src.training.transforms import build_transforms, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Оценка модели на CSV split.")
    p.add_argument("--ckpt", type=Path, required=True, help="Путь к .pth чекпойнту.")
    p.add_argument("--csv", type=Path, required=True, help="CSV (train/val/test) с path,label,class_name.")
    p.add_argument("--class-map", type=Path, required=True, help="data/processed/class_to_idx.json.")
    p.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Корень проекта (для относительных путей в CSV).")
    p.add_argument("--model-name", type=str, default=None, help="Имя модели timm (если не сохранено в ckpt).")
    p.add_argument("--img-size", type=int, default=224, help="Размер входа.")
    p.add_argument("--batch-size", type=int, default=64, help="batch size для инференса.")
    p.add_argument("--num-workers", type=int, default=4, help="num_workers для DataLoader.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "xpu"], help="Устройство.")
    p.add_argument("--device-index", type=int, default=0, help="Индекс XPU при device=auto/xpu.")
    p.add_argument("--out", type=Path, required=True, help="Каталог для графиков.")
    p.add_argument(
        "--drop-missing",
        action="store_true",
        help="Игнорировать классы, отсутствующие в y_true и y_pred (полезно, если тест покрывает не все классы карты).",
    )
    return p.parse_args()


def extract_state_dict(state) -> Dict[str, torch.Tensor]:
    if isinstance(state, dict) and "model_state" in state:
        sd = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    if isinstance(sd, dict) and any(k.startswith("model.") for k in sd.keys()):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    return sd


def infer_model_name(args: argparse.Namespace, ckpt_state: dict) -> str:
    if args.model_name:
        return args.model_name
    if isinstance(ckpt_state, dict):
        ckpt_args = ckpt_state.get("args")
        if isinstance(ckpt_args, dict):
            name = ckpt_args.get("model_name")
            if isinstance(name, str) and name:
                return name
    return "vit_tiny_patch16_224"


def load_class_map(path: Path) -> Dict[int, str]:
    with path.open(encoding="utf-8") as f:
        class_to_idx = json.load(f)
    return {int(v): str(k) for k, v in class_to_idx.items()}


def plot_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = y_prob.shape[1]

    # PR
    plt.figure(figsize=(7, 6))
    for c in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true == c, y_prob[:, c])
        plt.plot(recall, precision, label=class_names[c])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close()

    # ROC
    plt.figure(figsize=(7, 6))
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true == c, y_prob[:, c])
        plt.plot(fpr, tpr, label=class_names[c])
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    idx_to_class = load_class_map(args.class_map)
    num_classes = len(idx_to_class)

    # device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU недоступен.")
        device = torch.device(f"xpu:{args.device_index}")
        torch.xpu.set_device(args.device_index)
    else:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device(f"xpu:{args.device_index}")
            torch.xpu.set_device(args.device_index)
        else:
            device = torch.device("cpu")

    ckpt_state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_name = infer_model_name(args, ckpt_state)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(extract_state_dict(ckpt_state), strict=False)
    model.to(device).eval()

    tfm = build_transforms(args.img_size)["eval"]
    ds = PlantVillageCsvDataset(args.csv, root=args.root, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob_list: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(probs.argmax(dim=1).cpu().tolist())
            y_prob_list.append(probs.cpu().numpy())

    y_true_np = np.array(y_true, dtype=np.int64)
    y_pred_np = np.array(y_pred, dtype=np.int64)
    y_prob_np = np.concatenate(y_prob_list, axis=0)

    # Опционально урезаем классы до реально встречающихся
    if args.drop_missing:
        labels_keep = sorted(set(y_true_np) | set(y_pred_np))
        map_old_new = {old: i for i, old in enumerate(labels_keep)}
        y_true_eval = np.array([map_old_new[t] for t in y_true_np if t in map_old_new], dtype=np.int64)
        y_pred_eval = np.array([map_old_new[p] for p in y_pred_np if p in map_old_new], dtype=np.int64)
        class_names_eval = [idx_to_class[i] for i in labels_keep]
        y_prob_eval = y_prob_np[:, labels_keep] if y_prob_np.shape[1] >= len(labels_keep) else y_prob_np
        num_classes_eval = len(labels_keep)
    else:
        y_true_eval, y_pred_eval = y_true_np, y_pred_np
        class_names_eval = [idx_to_class[i] for i in range(num_classes)]
        y_prob_eval = y_prob_np
        num_classes_eval = num_classes

    acc = accuracy_score(y_true_eval, y_pred_eval)
    bal_acc = balanced_accuracy_score(y_true_eval, y_pred_eval)
    f1_macro = f1_score(y_true_eval, y_pred_eval, average="macro")

    # приведение к единому числу классов
    if y_prob_eval.shape[1] != num_classes_eval:
        if y_prob_eval.shape[1] < num_classes_eval:
            pad = np.zeros((y_prob_eval.shape[0], num_classes_eval - y_prob_eval.shape[1]), dtype=y_prob_eval.dtype)
            y_prob_eval = np.concatenate([y_prob_eval, pad], axis=1)
        else:
            y_prob_eval = y_prob_eval[:, :num_classes_eval]
    # one-hot для y_true
    y_true_oh = np.zeros((y_true_eval.shape[0], num_classes_eval), dtype=np.int32)
    for i, t in enumerate(y_true_eval):
        if 0 <= t < num_classes_eval:
            y_true_oh[i, t] = 1
    pr_auc_macro = average_precision_score(y_true_oh, y_prob_eval, average="macro")
    roc_auc_macro = roc_auc_score(y_true_oh, y_prob_eval, multi_class="ovr", average="macro")

    print(f"Overall Acc       : {acc:.5f}")
    print(f"Balanced Acc      : {bal_acc:.5f}")
    print(f"Macro F1          : {f1_macro:.5f}")
    print(f"Macro PR-AUC      : {pr_auc_macro:.5f}")
    print(f"Macro ROC-AUC     : {roc_auc_macro:.5f}")

    plot_curves(y_true_eval, y_prob_eval, class_names_eval, args.out)
    print(f"Сохранены графики PR/ROC в {args.out}")


if __name__ == "__main__":
    main()
