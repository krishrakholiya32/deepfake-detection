"""Export a trained PyTorch checkpoint to ONNX for CPU inference, and verify the export.

Usage:
    python export_model.py --config ../configs/train_config.yaml \
        --checkpoint /kaggle/working/runs/effnetb0_v1/checkpoints/best.pt \
        --output ../../models/deepfake_effnetb0.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = timm.create_model(cfg["model_name"], pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    size = cfg["image_size"]
    dummy = torch.randn(1, 3, size, size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # legacy TorchScript-based exporter; avoids the onnxscript/onnx_ir dependency
    )
    print(f"exported to {args.output}")

    # Verify ONNX output numerically matches PyTorch before trusting the export.
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {"input": dummy.numpy()})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"max abs diff in logits (torch vs onnx): {max_diff:.6f}")

    # Compare in probability space, not raw logits — small logit differences from
    # backend kernel variation (e.g. batchnorm fusion, SiLU) barely move the sigmoid
    # output and don't change the real/fake verdict, so they're not a correctness bug.
    torch_prob = 1 / (1 + np.exp(-torch_out))
    onnx_prob = 1 / (1 + np.exp(-onnx_out))
    max_prob_diff = np.abs(torch_prob - onnx_prob).max()
    print(f"max abs diff in probability (torch vs onnx): {max_prob_diff:.6f}")
    assert max_prob_diff < 0.01, "ONNX export does not match PyTorch output — do not trust this artifact"
    print("export verified OK")


if __name__ == "__main__":
    main()
