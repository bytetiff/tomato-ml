"""LightningDataModule для PlantVillage CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.plantvillage_csv import PlantVillageCsvDataset
from src.training.transforms import build_transforms, PROJECT_ROOT


class PlantVillageDataModule(pl.LightningDataModule):
    """Создает даталоадеры для train/val/test по CSV."""

    def __init__(
        self,
        train_csv: Path,
        val_csv: Path,
        test_csv: Optional[Path],
        root: Path = PROJECT_ROOT,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.test_csv = Path(test_csv) if test_csv else None
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.transforms = build_transforms(img_size)

        self.train_ds: Optional[PlantVillageCsvDataset] = None
        self.val_ds: Optional[PlantVillageCsvDataset] = None
        self.test_ds: Optional[PlantVillageCsvDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_ds = PlantVillageCsvDataset(
                self.train_csv, root=self.root, transform=self.transforms["train"]
            )
            self.val_ds = PlantVillageCsvDataset(
                self.val_csv, root=self.root, transform=self.transforms["eval"]
            )
        if stage in (None, "test") and self.test_csv:
            self.test_ds = PlantVillageCsvDataset(
                self.test_csv, root=self.root, transform=self.transforms["eval"]
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("test_csv не задан, тест-даталоадер недоступен")
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
