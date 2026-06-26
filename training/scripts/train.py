"""Train an EfficientNet-B0 binary classifier (real vs. fake face crops).

Two-phase fine-tune: freeze backbone + train head, then unfreeze last N blocks at lower LR.

Usage:
    python train.py --config ../configs/train_config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(model_name: str) -> nn.Module:
    model = timm.create_model(model_name, pretrained=True, num_classes=1)
    return model


def build_dataloaders(cfg: dict):
    train_tfms = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = Path(cfg["data_dir"])
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tfms)

    # ImageFolder sorts classes alphabetically -> {"fake": 0, "real": 1}; confirm and use as label map.
    assert train_ds.classes == ["fake", "real"], f"unexpected classes: {train_ds.classes}"

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)
    return train_loader, val_loader


def set_backbone_frozen(model: nn.Module, frozen: bool, unfreeze_blocks: int):
    for param in model.parameters():
        param.requires_grad = not frozen
    if frozen:
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    elif unfreeze_blocks:
        # re-freeze everything except classifier + last N blocks of the feature extractor
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True
        blocks = list(model.blocks)
        for block in blocks[-unfreeze_blocks:]:
            for param in block.parameters():
                param.requires_grad = True


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    all_labels, all_probs = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        with torch.set_grad_enabled(train):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    acc = np.mean((np.array(all_probs) > 0.5) == np.array(all_labels))
    return avg_loss, auc, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg["model_name"]).to(device)
    criterion = nn.BCEWithLogitsLoss()

    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    epochs_without_improvement = 0
    history = []

    # Phase 1: freeze backbone, train head only
    set_backbone_frozen(model, frozen=True, unfreeze_blocks=0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["freeze_lr"])
    epoch = 0
    for _ in range(cfg["freeze_epochs"]):
        epoch += 1
        train_loss, train_auc, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_auc, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"[head] epoch {epoch}: train_loss={train_loss:.4f} train_auc={train_auc:.4f} "
              f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}")
        history.append({"epoch": epoch, "phase": "head", "train_loss": train_loss, "train_auc": train_auc,
                         "val_loss": val_loss, "val_auc": val_auc, "val_acc": val_acc})
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    # Phase 2: unfreeze last blocks, fine-tune at lower lr
    set_backbone_frozen(model, frozen=False, unfreeze_blocks=cfg["unfreeze_blocks"])
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["finetune_lr"],
                                  weight_decay=cfg["weight_decay"])
    for _ in range(cfg["finetune_epochs"]):
        epoch += 1
        train_loss, train_auc, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_auc, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"[finetune] epoch {epoch}: train_loss={train_loss:.4f} train_auc={train_auc:.4f} "
              f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}")
        history.append({"epoch": epoch, "phase": "finetune", "train_loss": train_loss, "train_auc": train_auc,
                         "val_loss": val_loss, "val_auc": val_auc, "val_acc": val_acc})

        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch} (best val_auc={best_auc:.4f})")
                break

    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"history": history, "best_val_auc": best_auc}, f, indent=2)

    print(f"done. best val_auc={best_auc:.4f}. best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
