from datetime import datetime

from sqlalchemy import String, DateTime, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.database import Base


class DetectionJob(Base):
    __tablename__ = "detection_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    media_type: Mapped[str] = mapped_column(String(16))          # image | video
    original_filename: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|processing|done|error
    verdict: Mapped[str | None] = mapped_column(String(8), nullable=True)         # real | fake
    fake_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
