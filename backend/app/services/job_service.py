import logging
from pathlib import Path

from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.detection_job import DetectionJob
from app.services.detection.video_inference import run_video_detection

log = logging.getLogger(__name__)


async def run_video_job(job_id: int, video_path: str):
    async with AsyncSessionLocal() as db:
        job = (await db.execute(select(DetectionJob).where(DetectionJob.id == job_id))).scalar_one()
        job.status = "processing"
        await db.commit()

    try:
        result = run_video_detection(video_path)
        async with AsyncSessionLocal() as db:
            job = (await db.execute(select(DetectionJob).where(DetectionJob.id == job_id))).scalar_one()
            job.status = "done"
            job.verdict = result["verdict"]
            job.fake_probability = result["fake_probability"]
            job.frame_results = result["frame_results"]
            await db.commit()
    except Exception as e:
        log.exception(f"video detection job {job_id} failed")
        async with AsyncSessionLocal() as db:
            job = (await db.execute(select(DetectionJob).where(DetectionJob.id == job_id))).scalar_one()
            job.status = "error"
            job.error_message = str(e)
            await db.commit()
    finally:
        Path(video_path).unlink(missing_ok=True)
