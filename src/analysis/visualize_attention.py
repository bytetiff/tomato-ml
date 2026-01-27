"""Генерация attention rollout (ViT) по изображениям в CSV-сплитах.

Запуск из корня:
python -m src.analysis.visualize_attention --ckpt experiments/teacher/vit-tiny-xpu-best.pth --csv data/processed/splits/val.csv --class-map data/processed/class_to_idx.json --limit 16 --out outputs/attn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import timm
from PIL import Image
from torchvision import transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Визуализация attention rollout для ViT.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Путь к чекпойнту (xpu/manual или Lightning).")
    parser.add_argument("--csv", type=Path, required=True, help="CSV с колонками path,label,class_name.")
    parser.add_argument("--class-map", type=Path, required=True, help="class_to_idx.json для декодирования меток.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Корень проекта (для относительных путей в CSV).")
    parser.add_argument("--limit", type=int, default=16, help="Сколько примеров визуализировать.")
    parser.add_argument("--per-class", action="store_true", help="Взять по одному примеру на класс (не более limit).")
    parser.add_argument("--out", type=Path, default=Path("outputs/attn"), help="Каталог для сохранения карт.")
    parser.add_argument("--img-size", type=int, default=224, help="Размер входа для ViT.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_tiny_patch16_224",
        help="Имя модели timm, должно совпадать с обучением.",
    )
    parser.add_argument(
        "--head-reduce",
        type=str,
        default="mean",
        choices=["mean", "max", "min"],
        help="Как агрегировать attention по головам: mean/max/min.",
    )
    return parser.parse_args()


def load_class_map(path: Path) -> Dict[int, str]:
    with path.open(encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return idx_to_class


def load_checkpoint(model, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state" in state:
        sd = state["model_state"]
    elif "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    # Если ключи с префиксом "model.", убираем
    has_prefix = any(k.startswith("model.") for k in sd.keys())
    if has_prefix:
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Предупреждение: отсутствуют ключи при загрузке: {missing}")
    if unexpected:
        print(f"Предупреждение: лишние ключи при загрузке: {unexpected}")


def get_eval_transform(img_size: int):
    """Возвращает два трансформа: для оверлея (PIL) и для модели (тензор)."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    overlay_tfm = T.Compose(
        [
            T.Resize(img_size + 32),
            T.CenterCrop(img_size),
        ]
    )
    model_tfm = T.Compose(
        [
            overlay_tfm,
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    return overlay_tfm, model_tfm


def read_samples(csv_path: Path, root: Path, limit: int, per_class: bool) -> List[Tuple[Path, int, str]]:
    import csv

    samples: List[Tuple[Path, int, str]] = []
    seen_classes: Dict[str, bool] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = Path(row["path"])
            if not path.is_absolute():
                path = (root / path).resolve()
            label = int(row["label"])
            cls = row.get("class_name", str(label))
            if per_class:
                if cls in seen_classes:
                    continue
                seen_classes[cls] = True
            samples.append((path, label, cls))
            if (per_class and len(samples) >= limit) or (not per_class and len(samples) >= limit):
                break
    return samples


def register_attn_hooks(model) -> List:
    raise NotImplementedError("Hooks не используются, см. manual_attn_forward.")


def attention_rollout(attn_list: List[torch.Tensor], head_reduce: str = "mean") -> torch.Tensor:
    if len(attn_list) == 0:
        raise RuntimeError("Не удалось собрать карты внимания (attn_list пуст). Проверьте хуки и модель.")
    # attn_list: L элементов, каждый [B, heads, tokens, tokens]
    attn = torch.stack(attn_list)  # [L, B, H, T, T]
    if head_reduce == "mean":
        attn = attn.mean(dim=2)
    elif head_reduce == "max":
        attn = attn.max(dim=2).values
    elif head_reduce == "min":
        attn = attn.min(dim=2).values
    else:
        raise ValueError(f"Неизвестный head_reduce: {head_reduce}")
    # добавляем identity и нормализуем
    eye = torch.eye(attn.size(-1), device=attn.device).unsqueeze(0).unsqueeze(0)
    attn = attn + eye
    attn = attn / attn.sum(dim=-1, keepdim=True)
    # rollout
    rollout = attn[0]
    for i in range(1, attn.size(0)):
        rollout = attn[i] @ rollout
    # берем связи от CLS токена к остальным патчам
    cls_attn = rollout[:, 0, 1:]  # [B, tokens-1]
    return cls_attn


def to_heatmap(mask: torch.Tensor, img_size: int) -> torch.Tensor:
    # mask: [B, tokens-1], reshaped to grid
    b, n = mask.shape
    side = int(n**0.5)
    mask = mask.reshape(b, 1, side, side)
    mask = torch.nn.functional.interpolate(mask, size=(img_size, img_size), mode="bilinear", align_corners=False)
    # нормализация 0..1
    mask_min = mask.amin(dim=(1, 2, 3), keepdim=True)
    mask_max = mask.amax(dim=(1, 2, 3), keepdim=True)
    mask = (mask - mask_min) / (mask_max - mask_min + 1e-6)
    return mask.squeeze(1)  # [B, H, W]


def overlay_and_save(img: Image.Image, mask: torch.Tensor, out_path: Path, title: str) -> None:
    """Сохраняет коллаж: слева оригинал, справа оверлей внимания."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Оригинал")

    axes[1].imshow(img)
    axes[1].imshow(mask.cpu().numpy(), alpha=0.5, cmap="jet")
    axes[1].axis("off")
    axes[1].set_title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    idx_to_class = load_class_map(args.class_map)
    overlay_tfm, model_tfm = get_eval_transform(args.img_size)

    # поднимаем модель и внимание
    num_classes = len(idx_to_class)
    model = timm.create_model(args.model_name, pretrained=False, num_classes=num_classes)
    load_checkpoint(model, args.ckpt)
    model.eval()
    required_attrs = ["patch_embed", "cls_token", "pos_embed", "pos_drop", "blocks", "norm", "head"]
    if not all(hasattr(model, a) for a in required_attrs):
        raise RuntimeError(
            "visualize_attention работает только для ViT/DeiT из timm. "
            "Для ConvNeXt/CNN используйте saliency_attribution или occlusion_sensitivity."
        )

    samples = read_samples(args.csv, args.root, args.limit, args.per_class)

    def manual_attn_forward(model, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attn_list: List[torch.Tensor] = []
        B = x.shape[0]
        # patch embedding
        x = model.patch_embed(x)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        dist_token = getattr(model, "dist_token", None)
        if dist_token is not None:
            dist_token = dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + model.pos_embed
        x = model.pos_drop(x)

        for blk in model.blocks:
            # нормировка
            x_norm = blk.norm1(x)
            B, N, C = x_norm.shape
            qkv = blk.attn.qkv(x_norm).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, tokens, dim
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
            attn = attn.softmax(dim=-1)
            attn_list.append(attn.detach())
            attn_out = attn @ v  # B, heads, tokens, dim
            attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
            attn_out = blk.attn.proj(attn_out)
            attn_out = blk.attn.proj_drop(attn_out)
            dp_attn = getattr(blk, "drop_path", None) or getattr(blk, "drop_path1", None)
            dp_mlp = getattr(blk, "drop_path", None) or getattr(blk, "drop_path2", None)
            if dp_attn is None or dp_mlp is None:
                raise RuntimeError("В блоке ViT нет drop_path/drop_path1/drop_path2.")
            x = x + dp_attn(attn_out)
            mlp_out = blk.mlp(blk.norm2(x))
            x = x + dp_mlp(mlp_out)

        x = model.norm(x)
        cls_out = x[:, 0]
        logits = model.head(cls_out)
        return logits, attn_list

    for path, label, cls_name in samples:
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
        overlay_img = overlay_tfm(img_rgb)
        x = model_tfm(overlay_img).unsqueeze(0)
        with torch.no_grad():
            logits, attn_list = manual_attn_forward(model, x)
            pred_idx = int(logits.argmax(dim=1).item())
        # собираем карты внимания
        cls_mask = attention_rollout(attn_list, head_reduce=args.head_reduce)  # [1, tokens-1]
        heatmap = to_heatmap(cls_mask, args.img_size)[0]
        title = f"pred: {idx_to_class.get(pred_idx, pred_idx)} | true: {cls_name}"
        out_path = args.out / f"{path.stem}_attn.png"
        overlay_and_save(overlay_img, heatmap, out_path, title)
        print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
