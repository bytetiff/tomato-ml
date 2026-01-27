"""
Конвертация PlantDoc (папки с классами) в CSV с классами, совпадающими с class_to_idx.json.

Мэппинг папок PlantDoc -> наши классы задан в словаре name_map ниже.
Если какая-то папка не найдена в словаре или в class_to_idx.json — она пропускается.

Пример:
python -m src.data.convert_plantdoc_to_csv ^
  --root data/raw/plantdoc ^
  --class-map data/processed/class_to_idx.json ^
  --out outputs/plantdoc_test.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Конвертирует PlantDoc в CSV под наши классы.")
    p.add_argument("--root", type=Path, required=True, help="Корень PlantDoc с папками классов.")
    p.add_argument("--class-map", type=Path, required=True, help="JSON {class_name: idx} (наш порядок классов).")
    p.add_argument("--out", type=Path, required=True, help="Путь для сохранения CSV.")
    return p.parse_args()


def get_name_map() -> Dict[str, str]:
    # PlantDoc папки -> наши классы
    return {
        "Tomato Early blight leaf": "Tomato_Early_blight",
        "Tomato leaf": "Tomato_healthy",
        "Tomato leaf bacterial spot": "Tomato_Bacterial_spot",
        "Tomato leaf late blight": "Tomato_Late_blight",
        "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus",
        "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato mold leaf": "Tomato_Leaf_Mold",
        "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
        "Tomato two spotted spider mites leaf": "Tomato_Spider_mites_Two_spotted_spider_mite",
    }


def main() -> None:
    args = parse_args()
    name_map = get_name_map()
    class_map = json.load(args.class_map.open(encoding="utf-8"))

    rows = []
    for cls_dir in args.root.iterdir():
        if not cls_dir.is_dir():
            continue
        src_name = cls_dir.name
        tgt_name = name_map.get(src_name)
        if tgt_name is None:
            print(f"Пропускаю {src_name} (нет в name_map)")
            continue
        if tgt_name not in class_map:
            print(f"Пропускаю {src_name} -> {tgt_name} (нет в class_to_idx)")
            continue
        label = class_map[tgt_name]
        for img in cls_dir.rglob("*"):
            if img.is_file():
                rows.append({"path": str(img), "label": label, "class_name": tgt_name})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "class_name"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Сохранено {len(rows)} записей в {args.out}")


if __name__ == "__main__":
    main()
