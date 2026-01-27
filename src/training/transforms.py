"""Базовые аугментации для обучения."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms.functional as F
from torchvision import transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RandomBorderOcclusion:
    """Случайно закрывает рамку изображения, чтобы модель не опиралась на края/фон.

    Заполняет рамку значением `fill` (0..1), работает на тензоре (C,H,W) после ToTensor.
    """

    def __init__(
        self,
        p: float = 0.3,
        frac_range: Tuple[float, float] = (0.05, 0.15),
        fill: float = 0.5,
        min_sides: int = 1,
        max_sides: int = 4,
    ) -> None:
        self.p = p
        self.frac_range = frac_range
        self.fill = fill
        self.min_sides = min_sides
        self.max_sides = max_sides

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return x
        if x.ndim != 3:
            return x
        _, h, w = x.shape
        frac = float(torch.empty(1).uniform_(self.frac_range[0], self.frac_range[1]).item())
        bw = max(1, int(min(h, w) * frac))
        sides = ["top", "bottom", "left", "right"]
        k = int(torch.randint(self.min_sides, self.max_sides + 1, (1,)).item())
        perm = torch.randperm(len(sides))[:k]
        out = x.clone()
        fill = float(self.fill)
        for idx in perm.tolist():
            side = sides[idx]
            if side == "top":
                out[:, :bw, :] = fill
            elif side == "bottom":
                out[:, h - bw :, :] = fill
            elif side == "left":
                out[:, :, :bw] = fill
            elif side == "right":
                out[:, :, w - bw :] = fill
        return out


class RandomShadowStripe:
    """Имитация тени/засвета: вытянутая полоса вдоль края кадра с затемнением."""

    def __init__(
        self,
        p: float = 0.35,
        width_frac: Tuple[float, float] = (0.08, 0.25),
        darken: Tuple[float, float] = (0.35, 0.65),
    ) -> None:
        self.p = p
        self.width_frac = width_frac
        self.darken = darken

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p or x.ndim != 3:
            return x
        _, h, w = x.shape
        frac = float(torch.empty(1).uniform_(self.width_frac[0], self.width_frac[1]).item())
        stripe_w = max(1, int(min(h, w) * frac))
        side = torch.randint(0, 4, (1,)).item()  # 0: left, 1: right, 2: top, 3: bottom
        factor = float(torch.empty(1).uniform_(self.darken[0], self.darken[1]).item())
        out = x.clone()
        if side == 0:
            out[:, :, :stripe_w] *= factor
        elif side == 1:
            out[:, :, w - stripe_w :] *= factor
        elif side == 2:
            out[:, :stripe_w, :] *= factor
        else:
            out[:, h - stripe_w :, :] *= factor
        return out.clamp(0.0, 1.0)


def build_transforms(img_size: int) -> dict:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train = T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.1),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
            T.RandomApply([T.RandomAutocontrast(), T.RandomEqualize()], p=0.2),
            T.RandomApply([T.RandomSolarize(threshold=0.5), T.RandomPosterize(bits=4)], p=0.15),
            T.RandomGrayscale(p=0.08),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            T.ToTensor(),
            RandomShadowStripe(p=0.35, width_frac=(0.08, 0.22), darken=(0.3, 0.6)),
            RandomBorderOcclusion(p=0.35, frac_range=(0.05, 0.15), fill=0.5),
            T.Normalize(mean, std),
            # базовый erasing + вытянутые пятна для имитации полос/теней
            T.RandomErasing(p=0.35, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
            T.RandomErasing(p=0.20, scale=(0.02, 0.10), ratio=(0.1, 6.0)),
        ]
    )
    eval_tfm = T.Compose(
        [
            T.Resize(img_size + 32),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    return {"train": train, "eval": eval_tfm}
