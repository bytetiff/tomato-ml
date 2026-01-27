"""Проверка чувствительности модели к фону/углам через окклюзию (occlusion test).

Идея: берём одно изображение, считаем предсказание, затем закрываем части кадра
(углы/рамку/центр) и смотрим, насколько меняются вероятности и top-k.

Запуск из корня:
python -m src.analysis.occlusion_sensitivity --ckpt experiments/teacher/vit-tiny-xpu-best.pth --csv data/processed/splits/val.csv --index 0 --class-map data/processed/class_to_idx.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import timm
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms as T


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Prediction:
    topk: List[Tuple[int, float]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Occlusion sensitivity для ViT (проверка зависимости от фона/углов).")
    p.add_argument("--ckpt", type=Path, required=True, help="Путь к чекпойнту (xpu/manual или Lightning).")
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Имя модели timm (если не задано, берётся из чекпойнта при наличии).",
    )
    p.add_argument("--class-map", type=Path, required=True, help="Путь к data/processed/class_to_idx.json.")
    p.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Корень проекта (для относительных путей в CSV).")
    p.add_argument("--csv", type=Path, help="CSV с path,label,class_name (если не задан --image).")
    p.add_argument("--index", type=int, default=0, help="Индекс строки в CSV (0-based).")
    p.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Список индексов CSV через запятую (например: 0,10,25). Игнорируется при --per-class.",
    )
    p.add_argument(
        "--per-class",
        action="store_true",
        help="Взять по одному примеру на класс из CSV (в порядке строк).",
    )
    p.add_argument("--limit", type=int, default=10, help="Лимит примеров при --per-class.")
    p.add_argument(
        "--filter-class",
        type=str,
        default=None,
        help="Ограничить выборку одним class_name (например Tomato_Bacterial_spot).",
    )
    p.add_argument("--image", type=Path, help="Путь к изображению (альтернатива CSV).")
    p.add_argument("--img-size", type=int, default=224, help="Размер входа для модели.")
    p.add_argument("--topk", type=int, default=5, help="Сколько top-k выводить.")
    p.add_argument("--occ-size", type=float, default=0.2, help="Размер окклюзии (доля стороны, 0..1).")
    p.add_argument("--border", type=float, default=0.12, help="Толщина рамки (доля стороны, 0..1).")
    p.add_argument("--fill", type=str, default="gray", choices=["gray", "black", "mean"], help="Чем заполнять окклюзию.")
    p.add_argument("--save", type=Path, default=None, help="Сохранить коллаж (оригинал + окклюзии) в файл.")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Порог падения уверенности, считаем долю случаев > threshold.",
    )
    p.add_argument(
        "--deshadow",
        action="store_true",
        help="Добавить вариант 'deshadow': поднять тени (HSV, поднимаем канал V до порога).",
    )
    p.add_argument(
        "--deshadow-bg-only",
        action="store_true",
        help="Поднимать тени только на фоне (тени на листе не трогаем; лист выделяем HSV/RGB эвристикой).",
    )
    p.add_argument(
        "--deshadow-leaf-only",
        action="store_true",
        help="Поднимать тени только на листе (лист выделяем HSV/RGB эвристикой).",
    )
    p.add_argument(
        "--shadow-vmin",
        type=float,
        default=0.25,
        help="Минимальный порог V в [0..1]. Итоговый порог = max(shadow_vmin, percentile(V)).",
    )
    p.add_argument(
        "--shadow-percentile",
        type=float,
        default=15.0,
        help="Перцентиль по V (0..100) для вычисления порога теней.",
    )
    p.add_argument(
        "--shadowfill",
        action="store_true",
        help="Добавить вариант 'shadowfill': заменить тёмные пиксели (V ниже порога) на нейтральный цвет.",
    )
    p.add_argument(
        "--shadowfill-bg-only",
        action="store_true",
        help="Для 'shadowfill' стараться закрашивать только тени фона (пиксели листа исключаем простой HSV/RGB эвристикой).",
    )
    p.add_argument(
        "--shadowfill-leaf-only",
        action="store_true",
        help="Для 'shadowfill' закрашивать только тени на самом листе (лист выделяем простой HSV/RGB эвристикой).",
    )
    p.add_argument(
        "--leaf-s-min",
        type=float,
        default=0.25,
        help="Порог насыщенности S (0..1) для эвристики маски листа (только для --shadowfill-bg-only).",
    )
    p.add_argument(
        "--leaf-h-min",
        type=float,
        default=35.0,
        help="Минимальный hue в градусах (0..360) для 'зелёной' маски листа (только для --shadowfill-bg-only).",
    )
    p.add_argument(
        "--leaf-h-max",
        type=float,
        default=160.0,
        help="Максимальный hue в градусах (0..360) для 'зелёной' маски листа (только для --shadowfill-bg-only).",
    )
    p.add_argument(
        "--leaf-refine",
        type=int,
        default=5,
        help="Размер фильтра (нечётное >=3) для сглаживания маски листа (только для --shadowfill-bg-only).",
    )
    p.add_argument(
        "--per-class-random",
        action="store_true",
        help="Для --per-class брать случайный пример каждого класса (иначе берётся первое вхождение в CSV).",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed для --per-class-random.")
    return p.parse_args()


def load_class_map(path: Path) -> Dict[int, str]:
    with path.open(encoding="utf-8") as f:
        class_to_idx = json.load(f)
    return {idx: cls for cls, idx in class_to_idx.items()}


def load_checkpoint(model, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = extract_state_dict(state)
    model.load_state_dict(sd, strict=False)


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


def get_transforms(img_size: int):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    overlay_tfm = T.Compose([T.Resize(img_size + 32), T.CenterCrop(img_size)])
    model_tfm = T.Compose([overlay_tfm, T.ToTensor(), T.Normalize(mean, std)])
    return overlay_tfm, model_tfm


def read_csv_row(csv_path: Path, root: Path, index: int) -> Tuple[Path, Optional[int], Optional[str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == index:
                p = Path(row["path"])
                if not p.is_absolute():
                    p = (root / p).resolve()
                label = int(row["label"]) if "label" in row and row["label"] != "" else None
                cls = row.get("class_name")
                return p, label, cls
    raise IndexError(f"index={index} вне диапазона CSV {csv_path}")


def parse_indices(text: Optional[str]) -> List[int]:
    if text is None or text.strip() == "":
        return []
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    return out


def read_per_class(csv_path: Path, root: Path, limit: int) -> List[Tuple[Path, Optional[int], Optional[str]]]:
    samples: List[Tuple[Path, Optional[int], Optional[str]]] = []
    seen: Dict[str, bool] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row.get("class_name") or row.get("label") or "unknown"
            if cls in seen:
                continue
            seen[cls] = True
            p = Path(row["path"])
            if not p.is_absolute():
                p = (root / p).resolve()
            label = int(row["label"]) if "label" in row and row["label"] != "" else None
            samples.append((p, label, cls))
            if len(samples) >= limit:
                break
    return samples


def read_per_class_random(
    csv_path: Path, root: Path, limit: int, seed: int
) -> List[Tuple[Path, Optional[int], Optional[str]]]:
    import random

    rng = random.Random(seed)
    by_cls: Dict[str, List[Tuple[Path, Optional[int], Optional[str]]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row.get("class_name") or row.get("label") or "unknown"
            p = Path(row["path"])
            if not p.is_absolute():
                p = (root / p).resolve()
            label = int(row["label"]) if "label" in row and row["label"] != "" else None
            by_cls.setdefault(cls, []).append((p, label, cls))

    classes = sorted(by_cls.keys())
    picked: List[Tuple[Path, Optional[int], Optional[str]]] = []
    for cls in classes:
        items = by_cls.get(cls) or []
        if not items:
            continue
        picked.append(rng.choice(items))
        if len(picked) >= limit:
            break
    return picked


def read_by_class(
    csv_path: Path, root: Path, class_name: str, limit: int
) -> List[Tuple[Path, Optional[int], Optional[str]]]:
    samples: List[Tuple[Path, Optional[int], Optional[str]]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row.get("class_name")
            if cls != class_name:
                continue
            p = Path(row["path"])
            if not p.is_absolute():
                p = (root / p).resolve()
            label = int(row["label"]) if "label" in row and row["label"] != "" else None
            samples.append((p, label, cls))
            if len(samples) >= limit:
                break
    return samples


def to_fill_color(img: Image.Image, mode: str) -> Tuple[int, int, int]:
    if mode == "black":
        return (0, 0, 0)
    if mode == "gray":
        return (128, 128, 128)
    if mode == "mean":
        arr = torch.from_numpy(__import__("numpy").array(img)).float()
        mean = arr.view(-1, 3).mean(dim=0).clamp(0, 255).to(torch.int).tolist()
        return (int(mean[0]), int(mean[1]), int(mean[2]))
    raise ValueError(mode)


def occlude_corners(img: Image.Image, frac: float, fill: Tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    s = int(min(w, h) * frac)
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, s, s], fill=fill)
    draw.rectangle([w - s, 0, w, s], fill=fill)
    draw.rectangle([0, h - s, s, h], fill=fill)
    draw.rectangle([w - s, h - s, w, h], fill=fill)
    return out


def occlude_border(img: Image.Image, frac: float, fill: Tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    b = int(min(w, h) * frac)
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, w, b], fill=fill)  # top
    draw.rectangle([0, h - b, w, h], fill=fill)  # bottom
    draw.rectangle([0, 0, b, h], fill=fill)  # left
    draw.rectangle([w - b, 0, w, h], fill=fill)  # right
    return out


def occlude_center(img: Image.Image, frac: float, fill: Tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    s = int(min(w, h) * frac)
    cx, cy = w // 2, h // 2
    x0, y0 = cx - s // 2, cy - s // 2
    x1, y1 = x0 + s, y0 + s
    out = img.copy()
    ImageDraw.Draw(out).rectangle([x0, y0, x1, y1], fill=fill)
    return out


def deshadow_hsv(img: Image.Image, vmin: float, percentile: float) -> Image.Image:
    """
    Простое «поднятие теней»:
    - переводим RGB->HSV
    - считаем порог по каналу V: max(vmin, percentile(V))
    - все пиксели, у которых V ниже порога, поднимаем до порога (H,S сохраняем)

    Это не «идеальное удаление теней», но даёт объективный тест: насколько модель зависит от тёмных участков.
    """
    if not (0.0 <= vmin <= 1.0):
        raise ValueError(f"shadow-vmin должен быть в [0..1], получено: {vmin}")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"shadow-percentile должен быть в [0..100], получено: {percentile}")

    import numpy as np

    hsv = np.array(img.convert("HSV"), dtype=np.uint8)  # [H,W,3] в 0..255
    v = hsv[..., 2].astype(np.float32)
    thr = float(max(vmin * 255.0, np.percentile(v, percentile)))
    hsv[..., 2] = np.maximum(v, thr).clip(0, 255).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def estimate_leaf_mask_np(img: Image.Image, s_min: float, h_min_deg: float, h_max_deg: float, refine: int) -> "object":
    """
    Грубая эвристика маски листа (только для диагностики shadowfill-bg-only):
    - HSV: S >= s_min и H в [h_min_deg, h_max_deg] (примерно «зелёные» оттенки)
    - RGB: G доминирует над R и B
    - затем MaxFilter/MinFilter для закрытия дыр (refine = нечётное >= 3)

    Возвращает numpy bool [H,W].
    """
    if not (0.0 <= s_min <= 1.0):
        raise ValueError(f"--leaf-s-min должен быть в [0..1], получено: {s_min}")
    if not (0.0 <= h_min_deg <= 360.0) or not (0.0 <= h_max_deg <= 360.0):
        raise ValueError(f"--leaf-h-min/max должны быть в [0..360], получено: {h_min_deg}, {h_max_deg}")
    if h_min_deg > h_max_deg:
        raise ValueError("--leaf-h-min должен быть <= --leaf-h-max (wrap-around не поддержан)")

    import numpy as np

    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)

    hsv = np.array(img.convert("HSV"), dtype=np.uint8)
    h = hsv[..., 0].astype(np.float32)  # 0..255 ~ 0..360
    s = hsv[..., 1].astype(np.float32)  # 0..255

    h_min = h_min_deg / 360.0 * 255.0
    h_max = h_max_deg / 360.0 * 255.0
    greenish = (h >= h_min) & (h <= h_max)
    saturated = s >= (s_min * 255.0)
    g_dom = (g >= r + 5) & (g >= b + 5)

    mask = (greenish & saturated & g_dom).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L")
    if refine and refine >= 3 and refine % 2 == 1:
        mask_img = mask_img.filter(ImageFilter.MaxFilter(refine)).filter(ImageFilter.MinFilter(refine))
    return np.array(mask_img, dtype=np.uint8) > 127


def shadow_fill_hsv(
    img: Image.Image, vmin: float, percentile: float, fill_rgb: Tuple[int, int, int]
) -> Tuple[Image.Image, float]:
    """
    «Удаление теней» для теста чувствительности:
    - переводим RGB->HSV
    - считаем порог по V: max(vmin, percentile(V))
    - пиксели с V < порога заменяем на fill_rgb (в RGB)

    Это специально более «жёсткая» операция, чем deshadow.
    """
    if not (0.0 <= vmin <= 1.0):
        raise ValueError(f"shadow-vmin должен быть в [0..1], получено: {vmin}")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"shadow-percentile должен быть в [0..100], получено: {percentile}")

    import numpy as np

    hsv = np.array(img.convert("HSV"), dtype=np.uint8)
    v = hsv[..., 2].astype(np.float32)
    thr = float(max(vmin * 255.0, np.percentile(v, percentile)))
    mask = v < thr

    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    rgb[mask] = np.array(fill_rgb, dtype=np.uint8)
    mask_frac = float(mask.mean())
    return Image.fromarray(rgb, mode="RGB"), mask_frac


def predict(
    model,
    x: torch.Tensor,
    topk: int,
) -> Prediction:
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    vals, idxs = torch.topk(probs, k=min(topk, probs.numel()))
    return Prediction(topk=[(int(i.item()), float(v.item())) for i, v in zip(idxs, vals)])


def format_topk(pred: Prediction, idx_to_class: Dict[int, str]) -> str:
    parts = []
    for idx, p in pred.topk:
        parts.append(f"{idx_to_class.get(idx, str(idx))}={p:.3f}")
    return ", ".join(parts)


def save_collage(out_path: Path, images: List[Tuple[str, Image.Image]]) -> None:
    # все уже одного размера
    w, h = images[0][1].size
    pad = 8
    title_h = 18
    cols = 1
    rows = len(images)
    canvas = Image.new("RGB", (w + 2 * pad, rows * (h + title_h + pad) + pad), (255, 255, 255))
    y = pad
    for title, img in images:
        canvas.paste(img, (pad, y + title_h))
        y += h + title_h + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    idx_to_class = load_class_map(args.class_map)
    num_classes = len(idx_to_class)

    if args.image is None and args.csv is None:
        raise ValueError("Нужно указать --image или --csv.")

    if getattr(args, "shadowfill_bg_only", False) and getattr(args, "shadowfill_leaf_only", False):
        raise ValueError("Нельзя одновременно использовать --shadowfill-bg-only и --shadowfill-leaf-only.")

    if getattr(args, "shadowfill_bg_only", False) and getattr(args, "shadowfill_leaf_only", False):
        raise ValueError("Нельзя одновременно использовать --shadowfill-bg-only и --shadowfill-leaf-only.")
    if getattr(args, "deshadow_bg_only", False) and getattr(args, "deshadow_leaf_only", False):
        raise ValueError("Нельзя одновременно использовать --deshadow-bg-only и --deshadow-leaf-only.")

    overlay_tfm, model_tfm = get_transforms(args.img_size)

    device = torch.device("xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    if device.type == "xpu":
        torch.xpu.set_device(0)

    ckpt_state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_name = infer_model_name(args, ckpt_state)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    sd = extract_state_dict(ckpt_state)
    try:
        model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise RuntimeError(
            f"Не удалось загрузить чекпойнт в модель '{model_name}'. "
            f"Проверьте, что --model-name соответствует обучению. Исходная ошибка: {e}"
        ) from e
    model.to(device).eval()

    print(f"Model: {model_name} | Device: {device}")

    # формируем список изображений для прогона
    samples: List[Tuple[Path, Optional[int], Optional[str]]] = []
    if args.image is not None:
        img_path = args.image if args.image.is_absolute() else (args.root / args.image).resolve()
        samples = [(img_path, None, None)]
    else:
        assert args.csv is not None
        if args.filter_class:
            samples = read_by_class(args.csv, args.root, args.filter_class, args.limit)
        elif args.per_class:
            if args.per_class_random:
                samples = read_per_class_random(args.csv, args.root, args.limit, args.seed)
            else:
                samples = read_per_class(args.csv, args.root, args.limit)
        else:
            idxs = parse_indices(args.indices) or [args.index]
            for idx in idxs:
                samples.append(read_csv_row(args.csv, args.root, idx))

    drops_corners: List[float] = []
    drops_border: List[float] = []
    drops_center: List[float] = []
    drops_deshadow: List[float] = []
    drops_shadowfill: List[float] = []
    deshadow_mask_fracs: List[float] = []
    deshadow_leaf_fracs: List[float] = []
    shadowfill_mask_fracs: List[float] = []
    shadowfill_leaf_fracs: List[float] = []

    for sample_idx, (img_path, label, cls_name) in enumerate(samples):
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
        img_in = overlay_tfm(img_rgb)
        fill = to_fill_color(img_in, args.fill)
        variants: List[Tuple[str, Image.Image]] = [
            ("orig", img_in),
            ("corners", occlude_corners(img_in, args.occ_size, fill)),
            ("border", occlude_border(img_in, args.border, fill)),
            ("center", occlude_center(img_in, args.occ_size, fill)),
        ]
        deshadow_enabled = bool(args.deshadow) or bool(getattr(args, "deshadow_bg_only", False)) or bool(
            getattr(args, "deshadow_leaf_only", False)
        )
        deshadow_mask_frac: Optional[float] = None
        deshadow_leaf_frac: Optional[float] = None
        if deshadow_enabled:
            if getattr(args, "deshadow_bg_only", False) or getattr(args, "deshadow_leaf_only", False):
                import numpy as np

                leaf_mask = estimate_leaf_mask_np(
                    img_in, args.leaf_s_min, args.leaf_h_min, args.leaf_h_max, args.leaf_refine
                )  # np.bool_
                hsv = np.array(img_in.convert("HSV"), dtype=np.uint8)
                v = hsv[..., 2].astype(np.float32)
                thr = float(max(args.shadow_vmin * 255.0, np.percentile(v, args.shadow_percentile)))
                shadow = v < thr
                if getattr(args, "deshadow_bg_only", False):
                    apply_mask = shadow & (~leaf_mask)
                else:
                    apply_mask = shadow & leaf_mask

                hsv[..., 2] = np.where(apply_mask, thr, v).clip(0, 255).astype(np.uint8)
                deshadow_img = Image.fromarray(hsv, mode="HSV").convert("RGB")
                deshadow_mask_frac = float(apply_mask.mean())
                deshadow_leaf_frac = float(leaf_mask.mean())
            else:
                deshadow_img = deshadow_hsv(img_in, args.shadow_vmin, args.shadow_percentile)
            variants.append(("deshadow", deshadow_img))
        if args.shadowfill:
            mask_frac = 0.0
            leaf_frac: Optional[float] = None
            if getattr(args, "shadowfill_bg_only", False):
                import numpy as np

                leaf_mask = estimate_leaf_mask_np(
                    img_in, args.leaf_s_min, args.leaf_h_min, args.leaf_h_max, args.leaf_refine
                )  # np.bool_
                hsv = np.array(img_in.convert("HSV"), dtype=np.uint8)
                v = hsv[..., 2].astype(np.float32)
                thr = float(max(args.shadow_vmin * 255.0, np.percentile(v, args.shadow_percentile)))
                shadow = v < thr
                mask = shadow & (~leaf_mask)

                rgb = np.array(img_in.convert("RGB"), dtype=np.uint8)
                rgb[mask] = np.array(fill, dtype=np.uint8)
                shadow_img = Image.fromarray(rgb, mode="RGB")

                mask_frac = float(mask.mean())
                leaf_frac = float(leaf_mask.mean())
            elif getattr(args, "shadowfill_leaf_only", False):
                import numpy as np

                leaf_mask = estimate_leaf_mask_np(
                    img_in, args.leaf_s_min, args.leaf_h_min, args.leaf_h_max, args.leaf_refine
                )  # np.bool_
                hsv = np.array(img_in.convert("HSV"), dtype=np.uint8)
                v = hsv[..., 2].astype(np.float32)
                thr = float(max(args.shadow_vmin * 255.0, np.percentile(v, args.shadow_percentile)))
                shadow = v < thr
                mask = shadow & leaf_mask

                rgb = np.array(img_in.convert("RGB"), dtype=np.uint8)
                rgb[mask] = np.array(fill, dtype=np.uint8)
                shadow_img = Image.fromarray(rgb, mode="RGB")

                mask_frac = float(mask.mean())
                leaf_frac = float(leaf_mask.mean())
            else:
                shadow_img, mask_frac = shadow_fill_hsv(img_in, args.shadow_vmin, args.shadow_percentile, fill)
            variants.append(("shadowfill", shadow_img))

        preds: Dict[str, Prediction] = {}
        probs_by_name: Dict[str, torch.Tensor] = {}
        for name, img_var in variants:
            x = model_tfm(img_var).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu()
            probs_by_name[name] = probs
            vals, idxs = torch.topk(probs, k=min(args.topk, probs.numel()))
            preds[name] = Prediction(topk=[(int(i.item()), float(v.item())) for i, v in zip(idxs, vals)])

        base_top1 = preds["orig"].topk[0][0]
        base_p_top1 = float(probs_by_name["orig"][base_top1].item())
        corners_p = float(probs_by_name["corners"][base_top1].item())
        border_p = float(probs_by_name["border"][base_top1].item())
        center_p = float(probs_by_name["center"][base_top1].item())
        drop_c = base_p_top1 - corners_p
        drop_b = base_p_top1 - border_p
        drop_ctr = base_p_top1 - center_p
        drops_corners.append(drop_c)
        drops_border.append(drop_b)
        drops_center.append(drop_ctr)

        drop_ds: Optional[float] = None
        if deshadow_enabled:
            ds_p = float(probs_by_name["deshadow"][base_top1].item())
            drop_ds = base_p_top1 - ds_p
            drops_deshadow.append(drop_ds)
            if deshadow_mask_frac is not None:
                deshadow_mask_fracs.append(deshadow_mask_frac)
            if deshadow_leaf_frac is not None:
                deshadow_leaf_fracs.append(deshadow_leaf_frac)

        drop_sf: Optional[float] = None
        if args.shadowfill:
            sf_p = float(probs_by_name["shadowfill"][base_top1].item())
            drop_sf = base_p_top1 - sf_p
            drops_shadowfill.append(drop_sf)
            shadowfill_mask_fracs.append(mask_frac)
            if (getattr(args, "shadowfill_bg_only", False) or getattr(args, "shadowfill_leaf_only", False)) and leaf_frac is not None:
                shadowfill_leaf_fracs.append(leaf_frac)

        header = f"[{sample_idx}] {img_path.name}"
        if cls_name is not None:
            header += f" | true={cls_name}"
        header += f" | base_top1={idx_to_class.get(base_top1, str(base_top1))} ({base_p_top1:.3f})"
        print(header)
        order = ["orig", "corners", "border", "center"]
        if deshadow_enabled:
            order.append("deshadow")
        if args.shadowfill:
            order.append("shadowfill")
        for name in order:
            print(f"  {name:7s}: {format_topk(preds[name], idx_to_class)}")
        drop_line = f"  drop(top1) corners={drop_c:.3f}, border={drop_b:.3f}, center={drop_ctr:.3f}"
        if drop_ds is not None:
            if deshadow_mask_frac is not None:
                extra = f"mask={deshadow_mask_frac:.1%}"
                if deshadow_leaf_frac is not None:
                    extra += f", leaf={deshadow_leaf_frac:.1%}"
                drop_line += f", deshadow={drop_ds:.3f} ({extra})"
            else:
                drop_line += f", deshadow={drop_ds:.3f}"
        if drop_sf is not None:
            extra = f"mask={mask_frac:.1%}"
            if (getattr(args, "shadowfill_bg_only", False) or getattr(args, "shadowfill_leaf_only", False)) and leaf_frac is not None:
                extra += f", leaf={leaf_frac:.1%}"
            drop_line += f", shadowfill={drop_sf:.3f} ({extra})"
        drop_line += f" (threshold={args.threshold})"
        print(drop_line)

        if args.save is not None and len(samples) == 1:
            out_path = args.save if args.save.is_absolute() else (args.root / args.save).resolve()
            save_collage(out_path, variants)
            print(f"Saved collage: {out_path}")

    if len(samples) > 1:
        import numpy as np

        dc = np.array(drops_corners, dtype=float)
        db = np.array(drops_border, dtype=float)
        dctr = np.array(drops_center, dtype=float)
        print("\n=== Summary (drop of base top1 prob) ===")
        print(f"samples: {len(samples)} | threshold: {args.threshold}")
        print(f"corners mean={dc.mean():.3f}, median={np.median(dc):.3f}, >thr={(dc > args.threshold).mean():.2%}")
        print(f"border  mean={db.mean():.3f}, median={np.median(db):.3f}, >thr={(db > args.threshold).mean():.2%}")
        print(f"center  mean={dctr.mean():.3f}, median={np.median(dctr):.3f}, >thr={(dctr > args.threshold).mean():.2%}")
        if deshadow_enabled:
            dds = np.array(drops_deshadow, dtype=float)
            print(f"deshadow mean={dds.mean():.3f}, median={np.median(dds):.3f}, >thr={(dds > args.threshold).mean():.2%}")
            if len(deshadow_mask_fracs) == len(drops_deshadow):
                dm = np.array(deshadow_mask_fracs, dtype=float)
                print(f"deshadow_mask mean={dm.mean():.1%}, median={np.median(dm):.1%}")
            if len(deshadow_leaf_fracs) == len(drops_deshadow):
                dl = np.array(deshadow_leaf_fracs, dtype=float)
                print(f"leaf_mask (deshadow) mean={dl.mean():.1%}, median={np.median(dl):.1%}")
        if args.shadowfill:
            dsf = np.array(drops_shadowfill, dtype=float)
            print(f"shadowfill mean={dsf.mean():.3f}, median={np.median(dsf):.3f}, >thr={(dsf > args.threshold).mean():.2%}")
            sm = np.array(shadowfill_mask_fracs, dtype=float)
            print(f"shadowfill_mask mean={sm.mean():.1%}, median={np.median(sm):.1%}")
            if (getattr(args, "shadowfill_bg_only", False) or getattr(args, "shadowfill_leaf_only", False)) and len(shadowfill_leaf_fracs) == len(shadowfill_mask_fracs):
                sl = np.array(shadowfill_leaf_fracs, dtype=float)
                print(f"leaf_mask mean={sl.mean():.1%}, median={np.median(sl):.1%}")


if __name__ == "__main__":
    main()
