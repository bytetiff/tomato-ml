"""
Демонстрация аугментации тенями/полосами (RandomShadowStripe).

Пример:
python -m src.analysis.demo_shadow_aug ^
  --image data/raw/plantvillage/Tomato__Target_Spot/994414de-c6a6-4d55-8a61-eb0cd8d930c0___Com.G_TgS_FL\ 8367.JPG ^
  --out outputs/shadow_aug_demo ^
  --samples 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms as T

from src.training.transforms import RandomShadowStripe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Сохраняет примеры тенеовой аугментации.")
    p.add_argument("--image", type=Path, required=True, help="Путь к исходному изображению.")
    p.add_argument("--out", type=Path, required=True, help="Каталог для сохранения результатов.")
    p.add_argument("--samples", type=int, default=4, help="Сколько сгенерировать вариантов.")
    p.add_argument("--seed", type=int, default=None, help="Фиксировать seed для воспроизводимости.")
    p.add_argument("--width-min", type=float, default=0.08, help="Мин. доля ширины полосы.")
    p.add_argument("--width-max", type=float, default=0.22, help="Макс. доля ширины полосы.")
    p.add_argument("--darken-min", type=float, default=0.3, help="Мин. коэффициент затемнения.")
    p.add_argument("--darken-max", type=float, default=0.6, help="Макс. коэффициент затемнения.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    with Image.open(args.image) as img:
        img = img.convert("RGB")

    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()
    shadow = RandomShadowStripe(
        p=1.0,
        width_frac=(args.width_min, args.width_max),
        darken=(args.darken_min, args.darken_max),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    img.save(args.out / "original.png")

    for i in range(args.samples):
        t = to_tensor(img)
        aug = shadow(t)
        to_pil(aug).save(args.out / f"shadow_aug_{i}.png")

    print(f"Сохранено: original.png и {args.samples} вариантов в {args.out}")


if __name__ == "__main__":
    main()

