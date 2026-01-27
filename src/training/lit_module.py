"""LightningModule для ViT-tiny классификатора."""

from __future__ import annotations

import torch
from torch import nn
import pytorch_lightning as pl
import timm
from torchmetrics.classification import MulticlassAccuracy


class LitViTTiny(pl.LightningModule):
    """ViT-tiny с кросс-энтропией и AdamW."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "vit_tiny_patch16_224",
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        max_epochs: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        if stage == "train":
            acc = self.train_acc(logits, y)
        elif stage == "val":
            acc = self.val_acc(logits, y)
        else:
            acc = self.test_acc(logits, y)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
