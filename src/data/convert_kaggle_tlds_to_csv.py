from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Конвертация Kaggle TLDS в CSV под PlantVillage-классы.")
    p.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kaggle_TLDS",
        help="Корень датасета Kaggle TLDS (с папками Train/Test).",
    )
    p.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["Train", "Test"],
        help="Какие сплиты конвертировать (Train/Test).",
    )
    p.add_argument(
        "--class-map",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "class_to_idx.json",
        help="Базовая карта классов (PlantVillage).",
    )
    p.add_argument(
        "--class-map-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "class_to_idx_extended.json",
        help="Куда сохранить расширенную карту классов.",
    )
    p.add_argument(
        "--map-json",
        type=Path,
        default=None,
        help="JSON-словарь: имя папки -> имя класса. Если не задан, используется дефолтный маппинг.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Папка для CSV.",
    )
    return p.parse_args()


def normalize_new_class(name: str) -> str:
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", name.strip()) if p]
    if not parts:
        raise ValueError(f"Пустое имя класса после нормализации: {name!r}")
    parts = [parts[0].capitalize()] + [p.lower() for p in parts[1:]]
    return "Tomato_" + "_".join(parts)


def default_mapping() -> Dict[str, str]:
    return {
        "Bacterial spot": "Tomato_Bacterial_spot",
        "Late blight": "Tomato_Late_blight",
        "health": "Tomato_healthy",
    }


def load_class_map(path: Path) -> Dict[str, int]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_class_map(path: Path, class_to_idx: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=True, indent=2)


def iter_images(split_dir: Path) -> Iterable[Tuple[Path, str]]:
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTS and img_path.is_file():
                yield img_path, class_dir.name


def map_class_name(raw_name: str, mapping: Dict[str, str]) -> str:
    if raw_name in mapping:
        return mapping[raw_name]
    return normalize_new_class(raw_name)


def write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.map_json:
        with args.map_json.open(encoding="utf-8") as f:
            mapping = json.load(f)
    else:
        mapping = default_mapping()

    class_to_idx = load_class_map(args.class_map)

    for split in args.splits:
        split_dir = args.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Не найден сплит: {split_dir}")

        rows: List[Dict[str, str]] = []
        added_new = []

        for img_path, raw_class in iter_images(split_dir):
            class_name = map_class_name(raw_class, mapping)
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)
                added_new.append(class_name)

            rel_path = img_path.resolve()
            try:
                rel_path = rel_path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel_path = img_path.resolve()

            rows.append(
                {
                    "path": str(rel_path).replace("\\", "/"),
                    "label": str(class_to_idx[class_name]),
                    "class_name": class_name,
                    "split": split.lower(),
                }
            )

        out_path = args.out_dir / f"kaggle_tlds_{split.lower()}.csv"
        write_csv(rows, out_path)
        if added_new:
            print(f"[INFO] {split}: добавлены новые классы: {', '.join(added_new)}")
        print(f"[OK] {split}: сохранено {len(rows)} строк в {out_path}")

    save_class_map(args.class_map_out, class_to_idx)
    print(f"[OK] Расширенная карта классов: {args.class_map_out}")


if __name__ == "__main__":
    main()
