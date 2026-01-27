"""Датасет PlantVillage по CSV-сплитам."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class PlantVillageCsvDataset(Dataset):
    """Читает строки CSV (path, label) и загружает изображения."""

    def __init__(
        self,
        csv_path: Path,
        root: Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = self._read_csv(self.csv_path)

    def _read_csv(self, csv_path: Path) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = Path(row["path"])
                if not img_path.is_absolute():
                    img_path = (self.root / img_path).resolve()
                label = int(row["label"])
                samples.append((img_path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
