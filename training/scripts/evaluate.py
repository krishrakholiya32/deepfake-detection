"""Evaluate a trained checkpoint against the held-out test split.

Usage:
    python evaluate.py --config ../configs/train_config.yaml --checkpoint /kaggle/working/runs/effnetb0_v1/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfms = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(Path(cfg["data_dir"]) / args.split, transform=tfms)
    assert ds.classes == ["fake", "real"], f"unexpected classes: {ds.classes}"
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = timm.create_model(cfg["model_name"], pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds = (np.array(all_probs) > 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    cm = confusion_matrix(all_labels, all_preds)

    print(f"split={args.split} n={len(all_labels)}")
    print(f"AUC={auc:.4f} accuracy={acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    print(f"confusion matrix [[tn fp] [fn tp]] (0=fake, 1=real):\n{cm}")


if __name__ == "__main__":
    main()
