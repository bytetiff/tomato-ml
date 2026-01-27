"""Запуск обучения модели из timm (ViT/DeiT/ConvNeXt и т.п.) на PlantVillage (CSV-сплиты).

Поддерживает:
- accelerator=cpu/cuda/mps/...: обучение через Lightning.
- accelerator=xpu: ручной тренинг на torch.xpu (Lightning не умеет xpu).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import timm
from torchmetrics.classification import MulticlassAccuracy
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

from src.training.transforms import build_transforms, PROJECT_ROOT
from src.data.plantvillage_csv import PlantVillageCsvDataset


def _safe_ckpt_stem(name: str) -> str:
    """Делаем имя безопасным для файловой системы (Windows)."""
    import re

    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(" .-_")
    return stem or "model"


def parse_args() -> argparse.Namespace:
    default_train = PROJECT_ROOT / "data" / "processed" / "splits" / "train.csv"
    default_val = PROJECT_ROOT / "data" / "processed" / "splits" / "val.csv"
    default_test = PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv"
    default_class_map = PROJECT_ROOT / "data" / "processed" / "class_to_idx.json"
    parser = argparse.ArgumentParser(description="Train a timm image classifier on PlantVillage CSV.")
    parser.add_argument("--train-csv", type=Path, default=default_train, help="Путь к train.csv")
    parser.add_argument("--val-csv", type=Path, default=default_val, help="Путь к val.csv")
    parser.add_argument("--test-csv", type=Path, default=default_test, help="Путь к test.csv")
    parser.add_argument("--class-map", type=Path, default=default_class_map, help="Путь к class_to_idx.json")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Корень проекта (для относительных путей в CSV)")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча")
    parser.add_argument("--num-workers", type=int, default=4, help="num_workers для DataLoader")
    parser.add_argument("--img-size", type=int, default=224, help="Размер входа для ViT")
    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_tiny_patch16_224",
        help="Имя модели timm (например vit_tiny_r_s16_p8_224)",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Базовый learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay")
    parser.add_argument("--max-epochs", type=int, default=30, help="Число эпох")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator: auto/cpu/cuda/mps/xpu")
    parser.add_argument("--devices", type=int, default=1, help="Сколько девайсов использовать")
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Индекс XPU устройства (используется только при --accelerator xpu).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="precision: 32-true/16-mixed/bf16-mixed (для xpu: 32 или bf16)",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "teacher", help="Каталог для чекпойнтов")
    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup alpha (0 чтобы выключить)")
    parser.add_argument("--cutmix", type=float, default=0.0, help="CutMix alpha (0 чтобы выключить)")
    parser.add_argument("--mixup-prob", type=float, default=1.0, help="Вероятность применения mixup/cutmix")
    parser.add_argument("--mixup-switch-prob", type=float, default=0.0, help="Вероятность переключения между mixup/cutmix")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Путь к чекпойнту для возобновления (загрузятся только веса модели).",
    )
    return parser.parse_args()


def load_num_classes(class_map_path: Path) -> int:
    with class_map_path.open(encoding="utf-8") as f:
        class_to_idx = json.load(f)
    return len(class_to_idx)


def train_xpu(args: argparse.Namespace, num_classes: int) -> None:
    """Ручной тренинг на torch.xpu (Lightning не умеет xpu)."""
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("torch.xpu недоступен, проверьте установку IPEX/XPU PyTorch")

    device_idx = int(getattr(args, "device_index", 0))
    device = torch.device(f"xpu:{device_idx}")
    torch.xpu.set_device(device_idx)

    transforms = build_transforms(args.img_size)
    train_ds = PlantVillageCsvDataset(args.train_csv, root=args.root, transform=transforms["train"])
    val_ds = PlantVillageCsvDataset(args.val_csv, root=args.root, transform=transforms["eval"])
    test_ds = PlantVillageCsvDataset(args.test_csv, root=args.root, transform=transforms["eval"]) if args.test_csv else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = None
    if test_ds:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)
    # resume (только веса модели)
    best_acc = 0.0
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = state.get("model_state", state)
            if isinstance(sd, dict) and any(k.startswith("model.") for k in sd.keys()):
                sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"Resume: пропущено {len(missing)} ключей, лишних {len(unexpected)}")
            best_acc = float(state.get("val_acc", best_acc))
            print(f"Возобновление с чекпойнта {ckpt_path}, стартовый best_acc={best_acc:.4f}")
        else:
            print(f"Внимание: resume_from={ckpt_path} не найден, стартуем с нуля.")
    model.to(device)
    use_mix = (args.mixup > 0 or args.cutmix > 0) and args.mixup_prob > 0
    if use_mix and (args.batch_size % 2 != 0):
        raise ValueError("При mixup/cutmix batch-size должен быть чётным.")
    mixup_fn = None
    if use_mix:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode="batch",
            num_classes=num_classes,
        )
    criterion = SoftTargetCrossEntropy() if use_mix else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    metric = MulticlassAccuracy(num_classes=num_classes).to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / f"{_safe_ckpt_stem(args.model_name)}-xpu-best.pth"

    use_bf16 = args.precision.lower().startswith("bf16")
    autocast_dtype = torch.bfloat16 if use_bf16 else None

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_true = y
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
            optimizer.zero_grad()
            if autocast_dtype:
                with torch.autocast(device_type="xpu", dtype=autocast_dtype):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Эпоха {epoch} | шаг {batch_idx+1}/{len(train_loader)} | loss {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))

        # Валидация
        model.eval()
        metric.reset()
        val_loss = 0.0
        with torch.no_grad():
            ce_loss = nn.CrossEntropyLoss()
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if autocast_dtype:
                    with torch.autocast(device_type="xpu", dtype=autocast_dtype):
                        logits = model(x)
                        loss = ce_loss(logits, y)
                else:
                    logits = model(x)
                    loss = ce_loss(logits, y)
                val_loss += loss.item()
                metric.update(logits, y)
        val_acc = metric.compute().item()
        val_loss /= max(1, len(val_loader))
        print(f"[Эпоха {epoch}] train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            args_to_save = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "args": args_to_save,
                },
                best_path,
            )
            print(f"Новый лучший чекпойнт сохранен: {best_path} (val_acc={val_acc:.4f})")

    # Тест на лучшем чекпойнте
    if test_loader and best_path.exists():
        print("Запуск теста на лучшем чекпойнте...")
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        model.eval()
        metric.reset()
        test_loss = 0.0
        with torch.no_grad():
            ce_loss = nn.CrossEntropyLoss()
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if autocast_dtype:
                    with torch.autocast(device_type="xpu", dtype=autocast_dtype):
                        logits = model(x)
                        loss = ce_loss(logits, y)
                else:
                    logits = model(x)
                    loss = ce_loss(logits, y)
                test_loss += loss.item()
                metric.update(logits, y)
        test_acc = metric.compute().item()
        test_loss /= max(1, len(test_loader))
        print(f"[Тест] loss={test_loss:.4f} acc={test_acc:.4f}")
    else:
        print("Тест не выполнялся (нет test_loader или чекпойнта).")


def main() -> None:
    args = parse_args()
    num_classes = load_num_classes(args.class_map)

    if args.accelerator == "xpu":
        train_xpu(args, num_classes)
        return

    # Импортируем Lightning только для non-XPU пути
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from src.training.datamodule import PlantVillageDataModule
    from src.training.lit_module import LitViTTiny

    dm = PlantVillageDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    model = LitViTTiny(
        num_classes=num_classes,
        model_name=args.model_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"{_safe_ckpt_stem(args.model_name)}-{{epoch:02d}}-{{val_acc:.4f}}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        default_root_dir=args.output_dir,
        callbacks=[ckpt_cb],
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=dm)

    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        print(f"Лучший чекпойнт: {best_ckpt}")
        trainer.test(ckpt_path=best_ckpt, datamodule=dm)
    else:
        print("Чекпойнты не сохранены, тест не запущен.")


if __name__ == "__main__":
    main()
