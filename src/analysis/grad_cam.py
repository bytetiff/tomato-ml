"""
Grad-CAM для моделей timm с 2D feature map (например, ConvNeXt).

Скрипт сохраняет изображение, которое видит модель (Resize+CenterCrop 224),
оверлей Grad-CAM и саму карту (grayscale).

Примеры:
  python -m src.analysis.grad_cam ^
    --ckpt experiments\\teacher\\convnext_tiny\\convnext_tiny.fb_in22k_ft_in1k-xpu-best.pth ^
    --csv data\\processed\\splits\\val.csv ^
    --class-map data\\processed\\class_to_idx.json ^
    --index 0 ^
    --out outputs\\gradcam

  python -m src.analysis.grad_cam --ckpt ... --csv ... --class-map ... --per-class --limit 10 --out outputs\\gradcam
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grad-CAM для timm моделей (ConvNeXt/CNN).")
    p.add_argument("--ckpt", type=Path, required=True, help="Путь к .pth (xpu/manual или Lightning).")
    p.add_argument("--csv", type=Path, required=True, help="CSV с колонками path,label,class_name.")
    p.add_argument("--class-map", type=Path, required=True, help="Путь к data/processed/class_to_idx.json.")
    p.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Корень проекта (для относительных путей в CSV).")
    p.add_argument("--model-name", type=str, default=None, help="Имя модели timm (если не хранится в чекпойнте).")
    p.add_argument("--img-size", type=int, default=224, help="Размер входа модели.")
    p.add_argument("--index", type=int, default=0, help="Индекс строки в CSV (0-based).")
    p.add_argument("--indices", type=str, default=None, help="Список индексов через запятую, например: 0,10,25.")
    p.add_argument("--per-class", action="store_true", help="Взять по одному примеру на класс.")
    p.add_argument("--per-class-random", action="store_true", help="Как --per-class, но случайный пример на класс.")
    p.add_argument("--seed", type=int, default=42, help="Seed для --per-class-random.")
    p.add_argument("--filter-class", type=str, default=None, help="Ограничить выборку одним классом (class_name из CSV).")
    p.add_argument("--limit", type=int, default=10, help="Лимит примеров для --per-class/--filter-class.")

    p.add_argument(
        "--target",
        type=str,
        default="pred",
        choices=["pred", "true"],
        help="Для какого класса считать Grad-CAM: предсказанного или истинного.",
    )
    p.add_argument(
        "--target-layer",
        type=str,
        default=None,
        help=(
            "Слой для Grad-CAM (путь как в named_modules, например: stages.3). "
            "Если не указан, попробуем выбрать автоматически."
        ),
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "xpu"], help="Где считать Grad-CAM.")
    p.add_argument("--device-index", type=int, default=0, help="Индекс XPU (если --device xpu/auto).")

    p.add_argument("--alpha", type=float, default=0.55, help="Максимальная интенсивность наложения (0..1).")
    p.add_argument("--gamma", type=float, default=1.0, help="Гамма для карты (cam^gamma). >1 делает карту более контрастной.")
    p.add_argument("--cmap", type=str, default="jet", help="Matplotlib colormap (например jet, turbo, magma).")
    p.add_argument("--out", type=Path, required=True, help="Каталог для сохранения PNG.")
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


def load_class_map(path: Path) -> Dict[int, str]:
    with path.open(encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
    return idx_to_class


def get_transforms(img_size: int):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    overlay_tfm = T.Compose([T.Resize(img_size + 32), T.CenterCrop(img_size)])
    model_tfm = T.Compose([overlay_tfm, T.ToTensor(), T.Normalize(mean, std)])
    return overlay_tfm, model_tfm


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
    raise IndexError(f"index={index} не найден в CSV {csv_path}")


def read_per_class(
    csv_path: Path, root: Path, limit: int
) -> List[Tuple[Path, Optional[int], Optional[str]]]:
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


def pick_samples(args: argparse.Namespace) -> List[Tuple[Path, Optional[int], Optional[str]]]:
    indices = parse_indices(args.indices)
    if indices:
        return [read_csv_row(args.csv, args.root, i) for i in indices]
    if args.filter_class:
        return read_by_class(args.csv, args.root, args.filter_class, args.limit)
    if args.per_class:
        if args.per_class_random:
            return read_per_class_random(args.csv, args.root, args.limit, args.seed)
        return read_per_class(args.csv, args.root, args.limit)
    return [read_csv_row(args.csv, args.root, args.index)]


def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if name in modules:
        return modules[name]
    raise KeyError(f"Не найден модуль '{name}'. Доступные примеры: {list(modules.keys())[:10]} ...")


def auto_target_layer(model: torch.nn.Module) -> Tuple[torch.nn.Module, str]:
    stages = getattr(model, "stages", None)
    if stages is not None and hasattr(stages, "__len__") and len(stages) > 0:
        return stages[-1], f"stages.{len(stages) - 1}"

    layer4 = getattr(model, "layer4", None)
    if layer4 is not None and hasattr(layer4, "__len__") and len(layer4) > 0:
        return layer4[-1], "layer4.-1"

    last_name: Optional[str] = None
    last_mod: Optional[torch.nn.Module] = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            last_name, last_mod = name, mod
    if last_mod is not None and last_name is not None:
        return last_mod, last_name

    raise RuntimeError("Не удалось автоматически выбрать слой для Grad-CAM. Укажите --target-layer.")


def compute_grad_cam(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    x: torch.Tensor,
    target_idx: int,
    out_size: int,
    gamma: float,
) -> torch.Tensor:
    activations: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None

    def fwd_hook(_m, _inp, out):
        nonlocal activations, gradients
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("Выбранный слой возвращает не Tensor, Grad-CAM невозможен.")
        activations = out

        def _grad_hook(g):
            nonlocal gradients
            gradients = g

        out.register_hook(_grad_hook)

    handle = layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        score = logits[:, target_idx].sum()
        score.backward()
    finally:
        handle.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Не удалось получить активации/градиенты. Проверьте слой (--target-layer).")
    if activations.ndim != 4 or gradients.ndim != 4:
        raise RuntimeError(
            f"Grad-CAM ожидает 4D tensor (B,C,H,W), но слой дал {tuple(activations.shape)}."
        )

    # Grad-CAM: weights = GAP(dL/dA), cam = ReLU(sum_c w_c * A_c)
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    # normalизация 0..1
    cam = cam - cam.amin(dim=(2, 3), keepdim=True)
    cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-6)

    if gamma != 1.0:
        cam = cam.clamp(0, 1).pow(gamma)

    cam = F.interpolate(cam, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return cam[0, 0].detach()


def save_triptych(
    img: Image.Image,
    cam: torch.Tensor,
    out_path: Path,
    title: str,
    cmap: str,
    alpha: float,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    cam_np = cam.cpu().numpy().astype(np.float32)
    cam_np = np.clip(cam_np, 0.0, 1.0)
    img_np = np.asarray(img).astype(np.float32) / 255.0

    color_map = mpl.colormaps.get_cmap(cmap)
    heat = color_map(cam_np)[..., :3]  # 0..1 RGB

    # пиксельная альфа: слабые значения карты почти не окрашивают изображение
    a = np.clip(alpha * cam_np, 0.0, 1.0)[..., None]
    overlay = img_np * (1.0 - a) + heat * a
    overlay = np.clip(overlay, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title("Оригинал (crop)")

    axes[1].imshow(overlay)
    axes[1].axis("off")
    axes[1].set_title(title)

    axes[2].imshow(cam_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].axis("off")
    axes[2].set_title("Grad-CAM")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    idx_to_class = load_class_map(args.class_map)
    num_classes = len(idx_to_class)
    overlay_tfm, model_tfm = get_transforms(args.img_size)

    # устройство
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU недоступен в текущем окружении.")
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

    if args.target_layer:
        layer = get_module_by_name(model, args.target_layer)
        layer_name = args.target_layer
    else:
        layer, layer_name = auto_target_layer(model)

    print(f"Model: {model_name} | Device: {device} | Target layer: {layer_name}")

    samples = pick_samples(args)
    for path, label, cls_name in samples:
        with Image.open(path) as im:
            img_rgb = im.convert("RGB")

        img_vis = overlay_tfm(img_rgb)
        x = model_tfm(img_rgb).unsqueeze(0).to(device)

        # pred/true
        with torch.no_grad():
            logits = model(x)
            pred_idx = int(logits.argmax(dim=1).item())
            probs = torch.softmax(logits, dim=1)[0]
            topk = torch.topk(probs, k=min(5, num_classes))
            topk_str = ", ".join([f"{idx_to_class.get(int(i), int(i))}={float(p):.3f}" for i, p in zip(topk.indices, topk.values)])

        if args.target == "true" and label is not None:
            target_idx = int(label)
        else:
            target_idx = pred_idx

        cam = compute_grad_cam(
            model=model,
            layer=layer,
            x=x,
            target_idx=target_idx,
            out_size=args.img_size,
            gamma=args.gamma,
        )

        pred_name = idx_to_class.get(pred_idx, str(pred_idx))
        true_name = cls_name or (idx_to_class.get(int(label), str(label)) if label is not None else "unknown")
        target_name = idx_to_class.get(target_idx, str(target_idx))
        title = f"pred: {pred_name} | true: {true_name} | cam: {target_name}"

        out_path = args.out / f"{path.stem}_gradcam.png"
        save_triptych(img_vis, cam, out_path, title, cmap=args.cmap, alpha=args.alpha)

        print(f"Image: {path}")
        print(f"Top5: {topk_str}")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
