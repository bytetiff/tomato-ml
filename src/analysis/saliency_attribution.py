"""
Градиентная интерпретация (saliency / grad*input / SmoothGrad / integrated gradients) для timm ViT.

Почему это нужно:
attention-rollout часто выглядит "шумно" и не является причинным объяснением.
Градиентные карты ближе к вопросу "на какие пиксели влияет предсказание".

Примеры:
  python -m src.analysis.saliency_attribution ^
    --ckpt experiments/teacher/vit-tiny-xpu-best.pth ^
    --csv data/processed/splits/val.csv ^
    --class-map data/processed/class_to_idx.json ^
    --index 0 ^
    --out outputs/saliency

  python -m src.analysis.saliency_attribution --ckpt ... --csv ... --class-map ... --per-class --limit 10 --method ig
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import timm
import torch
from PIL import Image
from torchvision import transforms as T


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Градиентная интерпретация: saliency / integrated gradients.")
    p.add_argument("--ckpt", type=Path, required=True, help="Чекпойнт .pth (из нашего обучения).")
    p.add_argument("--model-name", type=str, default=None, help="Имя модели timm (если не задано, берём из чекпойнта).")
    p.add_argument("--class-map", type=Path, required=True, help="Путь к data/processed/class_to_idx.json.")
    p.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Корень проекта (для относительных путей в CSV).")
    p.add_argument("--csv", type=Path, required=True, help="CSV с колонками path,label,class_name.")
    p.add_argument("--index", type=int, default=0, help="Индекс строки в CSV (0-based).")
    p.add_argument("--per-class", action="store_true", help="Взять по одному примеру на класс из CSV.")
    p.add_argument("--limit", type=int, default=10, help="Лимит примеров при --per-class.")
    p.add_argument("--per-class-random", action="store_true", help="Для --per-class брать случайный пример каждого класса.")
    p.add_argument("--seed", type=int, default=42, help="Seed для --per-class-random.")
    p.add_argument("--img-size", type=int, default=224, help="Размер входа модели.")
    p.add_argument(
        "--method",
        type=str,
        default="saliency",
        choices=["saliency", "gradxinput", "smoothgrad", "smoothgrad_gradxinput", "ig"],
        help="Метод атрибуции.",
    )
    p.add_argument("--ig-steps", type=int, default=16, help="Число шагов для Integrated Gradients.")
    p.add_argument("--sg-samples", type=int, default=24, help="Число сэмплов для SmoothGrad.")
    p.add_argument(
        "--sg-noise-std",
        type=float,
        default=0.12,
        help="Std гауссова шума для SmoothGrad (в шкале НОРМАЛИЗОВАННОГО тензора).",
    )
    p.add_argument(
        "--target",
        type=str,
        default="pred",
        choices=["pred", "true"],
        help="По какому классу строить карту: по предсказанному или по истинному (если есть label).",
    )
    p.add_argument("--out", type=Path, required=True, help="Папка для сохранения PNG.")
    p.add_argument("--alpha", type=float, default=0.45, help="Максимальная прозрачность наложения тепловой карты.")
    p.add_argument(
        "--alpha-gamma",
        type=float,
        default=1.0,
        help="Гамма для карты прозрачности (alpha * sal^gamma). >1 делает подсветку более точечной.",
    )
    p.add_argument(
        "--vis-percentile",
        type=float,
        default=99.0,
        help="Клиппинг для визуализации: значения выше перцентиля приравниваются к максимуму (0..100).",
    )
    return p.parse_args()


def load_class_map(path: Path) -> Dict[int, str]:
    with path.open(encoding="utf-8") as f:
        class_to_idx = json.load(f)
    return {idx: cls for cls, idx in class_to_idx.items()}


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

    picked: List[Tuple[Path, Optional[int], Optional[str]]] = []
    for cls in sorted(by_cls.keys()):
        items = by_cls.get(cls) or []
        if not items:
            continue
        picked.append(rng.choice(items))
        if len(picked) >= limit:
            break
    return picked


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


def tensor_saliency(grad: torch.Tensor) -> torch.Tensor:
    # grad: [1,3,H,W] -> [H,W]
    s = grad.detach().abs().mean(dim=1)[0]
    s = s - s.min()
    s = s / (s.max().clamp_min(1e-8))
    return s


def tensor_gradxinput(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    # x, grad: [1,3,H,W] -> [H,W]
    gxi = (x.detach() * grad.detach()).abs().mean(dim=1)[0]
    gxi = gxi - gxi.min()
    gxi = gxi / (gxi.max().clamp_min(1e-8))
    return gxi


def normalize_for_vis(m: torch.Tensor, percentile: float) -> torch.Tensor:
    """Робастная нормализация [0..1] для визуализации (клиппинг по перцентилю)."""
    p = float(percentile)
    if not (0.0 < p <= 100.0):
        raise ValueError(f"--vis-percentile должен быть в (0..100], получено: {percentile}")
    v = m.detach().float()
    v = v - v.min()
    vmax = v.max().clamp_min(1e-8)
    v = v / vmax
    if p < 100.0:
        thr = torch.quantile(v.flatten(), p / 100.0).clamp_min(1e-8)
        v = torch.clamp(v / thr, 0.0, 1.0)
    return v


def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_idx: int,
    steps: int,
) -> torch.Tensor:
    # x: [1,3,H,W] normalized tensor
    steps = int(max(2, steps))
    baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)
    for i in range(1, steps + 1):
        a = float(i) / float(steps)
        xi = baseline + a * (x - baseline)
        xi.requires_grad_(True)
        logits = model(xi)
        score = logits[0, target_idx]
        model.zero_grad(set_to_none=True)
        if xi.grad is not None:
            xi.grad.zero_()
        score.backward()
        total_grad += xi.grad.detach()
    avg_grad = total_grad / float(steps)
    ig = (x - baseline) * avg_grad
    return tensor_saliency(ig)


def smoothgrad(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_idx: int,
    samples: int,
    noise_std: float,
    use_gradxinput: bool,
) -> torch.Tensor:
    samples = int(max(4, samples))
    noise_std = float(max(0.0, noise_std))
    acc = torch.zeros_like(x)
    for _ in range(samples):
        xn = (x + torch.randn_like(x) * noise_std).detach()
        xn.requires_grad_(True)
        logits = model(xn)
        score = logits[0, target_idx]
        model.zero_grad(set_to_none=True)
        if xn.grad is not None:
            xn.grad.zero_()
        score.backward()
        if use_gradxinput:
            # накапливаем уже grad*input в тензоре формы [1,3,H,W]
            acc += (xn.detach() * xn.grad.detach())
        else:
            acc += xn.grad.detach()
    acc = acc / float(samples)
    return tensor_gradxinput(x, acc) if use_gradxinput else tensor_saliency(acc)


def overlay_heatmap(img_rgb: Image.Image, sal: torch.Tensor, alpha: float) -> Image.Image:
    import numpy as np

    base = np.array(img_rgb.convert("RGB"), dtype=np.float32) / 255.0
    m = sal.detach().cpu().numpy().astype(np.float32)
    m = np.clip(m, 0.0, 1.0)
    # Важно: не красим "нулевые" области (иначе вся картинка уходит в синий/зеленый).
    # Делаем альфу пиксельно: alpha_map = alpha * m^gamma, и красим в "hot" (черный->красный->желтый).
    a = float(max(0.0, min(1.0, alpha)))
    alpha_map = (a * m).clip(0.0, 1.0)
    # "hot": r=m, g=m^2, b=0
    r = m
    g = np.clip(m * m, 0.0, 1.0)
    b = np.zeros_like(m)
    heat = np.stack([r, g, b], axis=-1)
    out = (1.0 - alpha_map[..., None]) * base + alpha_map[..., None] * heat
    out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def save_triptych(out_path: Path, left: Image.Image, mid: Image.Image, right: Image.Image) -> None:
    pad = 10
    w, h = left.size
    canvas = Image.new("RGB", (w * 3 + pad * 4, h + pad * 2), (255, 255, 255))
    canvas.paste(left, (pad, pad))
    canvas.paste(mid, (pad * 2 + w, pad))
    canvas.paste(right, (pad * 3 + w * 2, pad))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    idx_to_class = load_class_map(args.class_map)
    num_classes = len(idx_to_class)

    overlay_tfm, model_tfm = get_transforms(args.img_size)
    device = torch.device("xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    if device.type == "xpu":
        torch.xpu.set_device(0)

    ckpt_state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_name = infer_model_name(args, ckpt_state)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(extract_state_dict(ckpt_state), strict=False)
    model.to(device).eval()

    if args.per_class:
        if args.per_class_random:
            samples = read_per_class_random(args.csv, args.root, args.limit, args.seed)
        else:
            samples = read_per_class(args.csv, args.root, args.limit)
    else:
        samples = [read_csv_row(args.csv, args.root, args.index)]

    out_dir = args.out if args.out.is_absolute() else (args.root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for (img_path, label, cls_name) in samples:
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB")
        im_in = overlay_tfm(im_rgb)
        x = model_tfm(im_rgb).unsqueeze(0).to(device)
        x.requires_grad_(True)

        with torch.no_grad():
            logits0 = model(x)
            probs0 = torch.softmax(logits0, dim=1)[0]
            pred_idx = int(torch.argmax(probs0).item())

        if args.target == "true" and label is not None:
            target_idx = int(label)
        else:
            target_idx = pred_idx

        # Считаем атрибуцию
        if args.method == "saliency":
            logits = model(x)
            score = logits[0, target_idx]
            model.zero_grad(set_to_none=True)
            score.backward()
            sal = tensor_saliency(x.grad)
        elif args.method == "gradxinput":
            logits = model(x)
            score = logits[0, target_idx]
            model.zero_grad(set_to_none=True)
            score.backward()
            sal = tensor_gradxinput(x, x.grad)
        elif args.method == "smoothgrad":
            sal = smoothgrad(
                model, x.detach(), target_idx=target_idx, samples=args.sg_samples, noise_std=args.sg_noise_std, use_gradxinput=False
            )
        elif args.method == "smoothgrad_gradxinput":
            sal = smoothgrad(
                model, x.detach(), target_idx=target_idx, samples=args.sg_samples, noise_std=args.sg_noise_std, use_gradxinput=True
            )
        else:
            sal = integrated_gradients(model, x.detach(), target_idx=target_idx, steps=args.ig_steps)

        sal_vis = normalize_for_vis(sal, percentile=args.vis_percentile)
        # Применяем gamma к прозрачности, чтобы не подсвечивать низкие значения
        if args.alpha_gamma != 1.0:
            sal_overlay = torch.clamp(sal_vis, 0.0, 1.0) ** float(args.alpha_gamma)
        else:
            sal_overlay = sal_vis
        overlay = overlay_heatmap(im_in, sal_overlay, alpha=args.alpha)
        # Отдельно визуализируем саму карту в градациях серого
        sal_img = (sal_vis.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        sal_pil = Image.fromarray(sal_img, mode="L").convert("RGB")

        pred_name = idx_to_class.get(pred_idx, str(pred_idx))
        true_name = cls_name or (idx_to_class.get(label, str(label)) if label is not None else "unknown")
        stem = img_path.stem.replace(" ", "_")
        out_path = out_dir / f"{stem}__pred={pred_name}__true={true_name}__{args.method}.png"
        save_triptych(out_path, im_in, overlay, sal_pil)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
