"""
Оценка ConvNeXt-Tiny: строит и сохраняет матрицу ошибок (сырой и нормализованной),
heatmap и печатает метрики (accuracy, classification report, топ-5 ошибок).

Поддерживает CSV (path,label) или ImageFolder. Порядок классов:
- если указан --class-map (JSON вида {class_name: idx}), используется он;
- для ImageFolder — dataset.classes;
- для CSV без class_map — сортировка уникальных int-меток и сохранение classes.json.

Пример:
python -m src.analysis.eval_confusion_matrix ^
  --ckpt experiments\\teacher\\convnext_tiny\\convnext_tiny.fb_in22k_ft_in1k-xpu-best.pth ^
  --data data\\processed\\splits\\test.csv ^
  --class-map data\\processed\\class_to_idx.json ^
  --output-dir outputs\\confmat ^
  --img-size 224 ^
  --batch-size 64
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Матрица ошибок для ConvNeXt-Tiny.")
    p.add_argument("--ckpt", type=Path, required=True, help="Путь к чекпойнту (.pth).")
    p.add_argument("--data", type=Path, required=True, help="Путь к CSV (path,label) или к директории ImageFolder.")
    p.add_argument("--class-map", type=Path, default=None, help="JSON {class_name: idx}. Опционально.")
    p.add_argument("--output-dir", type=Path, required=True, help="Куда сохранять CSV/PNG.")
    p.add_argument("--img-size", type=int, default=224, help="Размер входа модели.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    p.add_argument("--num-workers", type=int, default=4, help="num_workers для DataLoader.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "xpu"], help="Устройство.")
    p.add_argument("--device-index", type=int, default=0, help="Индекс для cuda/xpu.")
    p.add_argument("--model-name", type=str, default=None, help="Имя модели timm (по умолчанию берётся из ckpt или convnext_tiny.fb_in22k_ft_in1k).")
    p.add_argument(
        "--drop-missing",
        action="store_true",
        help="Удалить классы, которые отсутствуют в y_true и y_pred (полезно, если тест содержит не все классы карты).",
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
    return "convnext_tiny.fb_in22k_ft_in1k"


class CSVDataset(Dataset):
    def __init__(self, csv_path: Path, root: Path, transform, class_to_idx: Dict[str, int]):
        self.samples: List[Tuple[Path, int]] = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("CSV должен содержать столбцы 'path' и 'label'")
            for row in reader:
                p = Path(row["path"])
                if not p.is_absolute():
                    p = (root / p).resolve()
                label = int(row["label"])
                self.samples.append((p, label))
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def load_class_map(class_map_path: Optional[Path], labels: List[int]) -> Tuple[Dict[int, str], Dict[str, int]]:
    if class_map_path and class_map_path.exists():
        with class_map_path.open(encoding="utf-8") as f:
            class_to_idx = json.load(f)
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        return idx_to_class, class_to_idx
    # иначе — по меткам, сортировка уникальных
    uniq = sorted(set(labels))
    idx_to_class = {int(i): str(i) for i in uniq}
    class_to_idx = {str(i): int(i) for i in uniq}
    return idx_to_class, class_to_idx


def choose_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        return torch.device(f"cuda:{args.device_index}") if torch.cuda.is_available() else torch.device("cpu")
    if args.device == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.set_device(args.device_index)
            return torch.device(f"xpu:{args.device_index}")
        return torch.device("cpu")
    # auto
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(args.device_index)
        return torch.device(f"xpu:{args.device_index}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.device_index}")
    return torch.device("cpu")


def build_transform(img_size: int):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Resize(img_size + 32),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    m = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            m[t, p] += 1
    return m


def top_k_errors(mat: np.ndarray, idx_to_class: Dict[int, str], k: int = 5) -> List[Tuple[str, str, int]]:
    errs = []
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(mat[i, j])
            if c > 0:
                errs.append((idx_to_class.get(i, str(i)), idx_to_class.get(j, str(j)), c))
    errs.sort(key=lambda x: x[2], reverse=True)
    return errs[:k]


def save_matrix_csv(mat: np.ndarray, class_names: List[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for name, row in zip(class_names, mat):
            writer.writerow([name] + list(row))


def plot_heatmap_single(mat_norm: np.ndarray, class_names: List[str], out_png: Path):
    # Одна нормализованная матрица, крупнее и с подписями значений.
    fig_w = max(8, len(class_names) * 0.9)
    fig_h = max(8, len(class_names) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat_norm, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    ax.set_xlabel("Pred label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_title("Confusion Matrix (row-normalized)")

    # подписи ячеек (до 2 знаков)
    thresh = mat_norm.max() / 2.0 if mat_norm.size > 0 else 0
    for i in range(mat_norm.shape[0]):
        for j in range(mat_norm.shape[1]):
            val = mat_norm[i, j]
            text = f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="white" if val > thresh else "black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = choose_device(args)

    # загрузка чекпойнта
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_name = infer_model_name(args, ckpt)

    # определяем трансформ
    tfm = build_transform(args.img_size)

    # загружаем датасет
    if args.data.is_dir():
        ds = datasets.ImageFolder(args.data, transform=tfm)
        idx_to_class = {i: c for i, c in enumerate(ds.classes)}
        class_to_idx = {c: i for i, c in idx_to_class.items()}
    else:
        # CSV: нужна карта классов
        # сначала читаем все labels, чтобы построить или загрузить карту
        labels_tmp: List[int] = []
        with args.data.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "label" not in reader.fieldnames or "path" not in reader.fieldnames:
                raise ValueError("CSV должен содержать столбцы 'path' и 'label'")
            for row in reader:
                labels_tmp.append(int(row["label"]))
        idx_to_class, class_to_idx = load_class_map(args.class_map, labels_tmp)
        ds = CSVDataset(args.data, PROJECT_ROOT, tfm, class_to_idx)
        # если не было class_map — сохраним порядок
        if args.class_map is None:
            out_classes = args.output_dir / "classes.json"
            out_classes.parent.mkdir(parents=True, exist_ok=True)
            with out_classes.open("w", encoding="utf-8") as f:
                json.dump({v: k for k, v in idx_to_class.items()}, f, ensure_ascii=False, indent=2)

    num_classes = len(idx_to_class)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # модель
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(extract_state_dict(ckpt), strict=False)
    model.to(device).eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())

    # при необходимости урежем классы до реально встречающихся (true или pred)
    if args.drop_missing:
        labels_keep = sorted(set(y_true) | set(y_pred))
        map_old_new = {old: i for i, old in enumerate(labels_keep)}
        y_true_mapped = [map_old_new[t] for t in y_true if t in map_old_new]
        y_pred_mapped = [map_old_new[p] for p in y_pred if p in map_old_new]
        class_names = [idx_to_class[l] for l in labels_keep]
        num_classes_eff = len(labels_keep)
    else:
        y_true_mapped = y_true
        y_pred_mapped = y_pred
        class_names = [idx_to_class[i] for i in range(num_classes)]
        num_classes_eff = num_classes

    # матрица ошибок
    mat = confusion_matrix(y_true_mapped, y_pred_mapped, num_classes_eff)
    row_sums = mat.sum(axis=1, keepdims=True).astype(np.float64)
    mat_norm = np.divide(mat, row_sums, out=np.zeros_like(mat, dtype=np.float64), where=row_sums != 0)

    # сохранение
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_matrix_csv(mat, class_names, args.output_dir / "confusion_matrix_raw.csv")
    save_matrix_csv(mat_norm, class_names, args.output_dir / "confusion_matrix_normalized.csv")
    plot_heatmap_single(mat_norm, class_names, args.output_dir / "confusion_matrix.png")

    # метрики
    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    labels_all = list(range(num_classes_eff))
    report = classification_report(
        y_true_mapped,
        y_pred_mapped,
        labels=labels_all,
        target_names=class_names,
        digits=5,
        zero_division=0,
    )
    idx_to_class_eff = dict(enumerate(class_names))
    top_errs = top_k_errors(mat, idx_to_class_eff)

    print(f"Accuracy: {acc:.5f}")
    print("\nClassification report:\n", report)
    print("Top-5 ошибок (true -> pred : count):")
    for t, p, c in top_errs:
        print(f"  {t} -> {p}: {c}")


if __name__ == "__main__":
    main()
