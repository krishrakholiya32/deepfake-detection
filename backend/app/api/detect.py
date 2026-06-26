import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.detection_job import DetectionJob
from app.schemas.detection import DetectionResult
from app.services.detection.image_inference import NoFaceDetectedError, run_image_detection
from app.services.job_service import run_video_job

router = APIRouter(prefix="/api/detect", tags=["detect"])

IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_CONTENT_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}


@router.post("/image", response_model=DetectionResult)
async def detect_image(file: UploadFile, db: AsyncSession = Depends(get_db)):
    if file.content_type not in IMAGE_CONTENT_TYPES:
        raise HTTPException(400, f"unsupported image content type: {file.content_type}")

    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "could not decode image")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    job = DetectionJob(media_type="image", original_filename=file.filename, status="processing")
    db.add(job)
    await db.commit()
    await db.refresh(job)

    try:
        result = run_image_detection(image_rgb)
    except NoFaceDetectedError as e:
        job.status = "error"
        job.error_message = str(e)
        await db.commit()
        raise HTTPException(422, str(e))

    job.status = "done"
    job.verdict = result["verdict"]
    job.fake_probability = result["fake_probability"]
    await db.commit()

    return DetectionResult(
        job_id=job.id, status="done", verdict=job.verdict, fake_probability=job.fake_probability
    )


@router.post("/video", response_model=DetectionResult)
async def detect_video(
    file: UploadFile, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
):
    if file.content_type not in VIDEO_CONTENT_TYPES:
        raise HTTPException(400, f"unsupported video content type: {file.content_type}")

    uploads_dir = Path(settings.UPLOADS_PATH)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    video_path = uploads_dir / f"{uuid.uuid4()}_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    job = DetectionJob(media_type="video", original_filename=file.filename, status="pending")
    db.add(job)
    await db.commit()
    await db.refresh(job)

    background_tasks.add_task(run_video_job, job.id, str(video_path))

    return DetectionResult(job_id=job.id, status="pending")


@router.get("/job/{job_id}", response_model=DetectionResult)
async def get_job_status(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(DetectionJob, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return DetectionResult(
        job_id=job.id,
        status=job.status,
        verdict=job.verdict,
        fake_probability=job.fake_probability,
        frame_results=job.frame_results,
        error_message=job.error_message,
    )
