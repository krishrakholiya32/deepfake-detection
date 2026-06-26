import numpy as np

from app.core.config import settings
from app.services.detection.face_extractor import extract_face
from app.services.detection.model_loader import predict_fake_probability


class NoFaceDetectedError(Exception):
    pass


def run_image_detection(image_rgb: np.ndarray) -> dict:
    crop = extract_face(image_rgb)
    if crop is None:
        raise NoFaceDetectedError("No face detected in image")
    fake_probability = predict_fake_probability(crop)
    verdict = "fake" if fake_probability >= settings.DETECTION_THRESHOLD else "real"
    return {"verdict": verdict, "fake_probability": fake_probability}
