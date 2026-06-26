import cv2
import numpy as np

from app.core.config import settings
from app.services.detection.face_extractor import extract_face
from app.services.detection.model_loader import predict_fake_probability


def sample_video_frames(video_path: str, max_frames: int, sample_fps: float):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_secs = total_frames / fps if fps else 0

    interval = max(1, int(fps / sample_fps)) if sample_fps > 0 else int(fps)
    candidate_indices = list(range(0, total_frames, interval))

    if len(candidate_indices) > max_frames:
        # uniformly sample across the full duration rather than only the opening seconds
        positions = np.linspace(0, len(candidate_indices) - 1, max_frames).astype(int)
        candidate_indices = [candidate_indices[p] for p in positions]

    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.release()
    return duration_secs


def run_video_detection(video_path: str) -> dict:
    frame_scores: list[dict] = []

    for frame_idx, frame_rgb in sample_video_frames(
        video_path, settings.VIDEO_MAX_SAMPLED_FRAMES, settings.VIDEO_SAMPLE_FPS
    ):
        crop = extract_face(frame_rgb)
        if crop is None:
            continue
        fake_probability = predict_fake_probability(crop)
        frame_scores.append({"frame_index": frame_idx, "fake_probability": fake_probability})

    if not frame_scores:
        raise ValueError("No face detected in any sampled frame")

    probs = [f["fake_probability"] for f in frame_scores]
    mean_prob = float(np.mean(probs))
    max_prob = float(np.max(probs))
    pct_fake_frames = float(np.mean([p >= settings.DETECTION_THRESHOLD for p in probs]))

    verdict = "fake" if mean_prob >= settings.DETECTION_THRESHOLD else "real"

    return {
        "verdict": verdict,
        "fake_probability": mean_prob,
        "frame_results": {
            "frames": frame_scores,
            "max_fake_probability": max_prob,
            "pct_fake_frames": pct_fake_frames,
            "sampled_frame_count": len(frame_scores),
        },
    }
