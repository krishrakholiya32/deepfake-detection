from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DetectionResult(BaseModel):
    job_id: int
    status: str
    verdict: Optional[str] = None
    fake_probability: Optional[float] = None
    frame_results: Optional[dict] = None
    error_message: Optional[str] = None


class DetectionJobOut(BaseModel):
    id: int
    media_type: str
    original_filename: str
    status: str
    verdict: Optional[str] = None
    fake_probability: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DetectionJobDetailOut(DetectionJobOut):
    frame_results: Optional[dict] = None
