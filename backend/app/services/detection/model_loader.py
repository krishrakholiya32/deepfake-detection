"""Loads the exported ONNX deepfake classifier once and reuses it (LRU-cached single instance)."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.core.config import settings

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@lru_cache(maxsize=1)
def get_session() -> ort.InferenceSession:
    model_path = Path(settings.MODELS_PATH) / settings.MODEL_FILENAME
    # Thread pinning matches the surveillance project's convention for predictable CPU usage on the Arm VM.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def preprocess(face_crop: np.ndarray) -> np.ndarray:
    """face_crop: HxWx3 RGB uint8, already resized to model input size. Returns NCHW float32 batch of 1."""
    img = face_crop.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return img[np.newaxis, ...].astype(np.float32)


def predict_fake_probability(face_crop: np.ndarray) -> float:
    """Returns P(fake). Training labels are ImageFolder-sorted {"fake": 0, "real": 1},
    so the model's raw sigmoid output is P(real) — invert it here."""
    session = get_session()
    inputs = preprocess(face_crop)
    logits = session.run(["logits"], {"input": inputs})[0]
    prob_real = 1.0 / (1.0 + np.exp(-logits[0, 0]))
    return float(1.0 - prob_real)
