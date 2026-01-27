"""Готовит сплиты PlantVillage по томатам (train/val/test) без копирования изображений.

Сканирует `data/raw/plantvillage`, делает стратифицированный сплит 70/15/15
с фиксированным сидом и сохраняет CSV в `data/processed/splits/`.
Запускать из корня репозитория:

    python -m src.data.prepare_plantvillage
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "plantvillage"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = OUTPUT_DIR / "splits"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str
    split: str


def discover_classes(raw_root: Path) -> List[Path]:
    """Директории классов, отсортированные по имени."""
    return sorted([p for p in raw_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def collect_images(class_dir: Path) -> List[Path]:
    """Файлы изображений в папке класса (без рекурсии)."""
    files = [
        p
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    files.sort(key=lambda p: p.name.lower())
    return files


def split_files(
    files: Sequence[Path], rng: random.Random
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Разбивает список на train/val/test по заданным долям."""
    shuffled = list(files)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_n = int(n * TRAIN_RATIO)
    val_n = int(n * VAL_RATIO)
    test_n = n - train_n - val_n
    return (
        shuffled[:train_n],
        shuffled[train_n : train_n + val_n],
        shuffled[train_n + val_n :],
    )


def make_samples(
    class_dirs: Sequence[Path], rng: random.Random
) -> Tuple[List[Sample], Dict[str, int]]:
    """Формирует Sample для всех сплитов и карту класс->индекс."""
    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
    samples: List[Sample] = []

    for class_dir in class_dirs:
        files = collect_images(class_dir)
        train_files, val_files, test_files = split_files(files, rng)
        for split_name, files_in_split in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ]:
            for path in files_in_split:
                samples.append(
                    Sample(
                        path=path,
                        label=class_to_idx[class_dir.name],
                        class_name=class_dir.name,
                        split=split_name,
                    )
                )
    return samples, class_to_idx


def write_csv(path: Path, samples: Iterable[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "label", "class_name", "split"]
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "path": sample.path.relative_to(PROJECT_ROOT).as_posix(),
                    "label": sample.label,
                    "class_name": sample.class_name,
                    "split": sample.split,
                }
            )


def save_class_map(path: Path, class_to_idx: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)


def summarize(samples: Sequence[Sample]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for sample in samples:
        summary.setdefault(sample.class_name, {"train": 0, "val": 0, "test": 0})
        summary[sample.class_name][sample.split] += 1
    return summary


def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_ROOT}")

    rng = random.Random(RANDOM_SEED)
    class_dirs = discover_classes(RAW_ROOT)
    samples, class_to_idx = make_samples(class_dirs, rng)

    write_csv(SPLITS_DIR / "all.csv", samples)
    write_csv(SPLITS_DIR / "train.csv", [s for s in samples if s.split == "train"])
    write_csv(SPLITS_DIR / "val.csv", [s for s in samples if s.split == "val"])
    write_csv(SPLITS_DIR / "test.csv", [s for s in samples if s.split == "test"])
    save_class_map(OUTPUT_DIR / "class_to_idx.json", class_to_idx)

    summary = summarize(samples)
    total = len(samples)
    print(f"Total samples: {total}")
    for class_name in sorted(summary):
        counts = summary[class_name]
        class_total = sum(counts.values())
        print(
            f"{class_name}: "
            f"{counts['train']} train, {counts['val']} val, {counts['test']} test "
            f"(total {class_total})"
        )
    print(f"CSV files written to: {SPLITS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Class map saved to: {(OUTPUT_DIR / 'class_to_idx.json').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
