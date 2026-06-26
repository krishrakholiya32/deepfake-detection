"""Face detection/cropping shared by training preprocessing and inference.

Single source of truth so train-time and inference-time face crops never drift apart.
"""
from __future__ import annotations

import numpy as np
import torch
from facenet_pytorch import MTCNN

FACE_MARGIN = 0.30  # expand bbox by this fraction on each side to capture blending seams
FACE_SIZE = 224

_mtcnn: MTCNN | None = None


def get_detector() -> MTCNN:
    global _mtcnn
    if _mtcnn is None:
        # Auto-detects GPU on Kaggle (training) and falls back to CPU on the
        # Arm VM (inference), which has no GPU — no separate config needed.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
    return _mtcnn


def _expand_box(box: np.ndarray, width: int, height: int, margin: float = FACE_MARGIN) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 -= w * margin
    y1 -= h * margin
    x2 += w * margin
    y2 += h * margin
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(width, int(x2)),
        min(height, int(y2)),
    )


def extract_face(image: np.ndarray) -> np.ndarray | None:
    """Detect the most prominent face in an RGB uint8 HxWx3 image and return a resized crop.

    Returns None if no face is detected.
    """
    height, width = image.shape[:2]
    try:
        boxes, probs = get_detector().detect(image)
    except RuntimeError:
        # facenet-pytorch's torch.cat raises RuntimeError on images with no face candidates
        return None
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = int(np.argmax(probs))
    x1, y1, x2, y2 = _expand_box(boxes[best_idx], width, height)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    return _resize(crop, FACE_SIZE)


def _resize(crop: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
