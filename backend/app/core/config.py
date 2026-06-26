from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "Deepfake Detection"

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:80",
    ]

    DATABASE_URL: str = "postgresql+asyncpg://deepfake:deepfake@db:5432/deepfake"

    UPLOADS_PATH: str = "/app/uploads"
    MODELS_PATH: str = "/app/models"
    MODEL_FILENAME: str = "deepfake_effnetb0.onnx"

    DETECTION_THRESHOLD: float = 0.5
    VIDEO_MAX_SAMPLED_FRAMES: int = 30
    VIDEO_SAMPLE_FPS: float = 1.0
    MAX_VIDEO_DURATION_SECS: int = 300

    class Config:
        env_file = ".env"


settings = Settings()
