from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.detection_job import DetectionJob
from app.schemas.detection import DetectionJobDetailOut, DetectionJobOut

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=List[DetectionJobOut])
async def list_history(
    media_type: Optional[str] = None,
    verdict: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    q = select(DetectionJob).order_by(desc(DetectionJob.created_at)).offset(offset).limit(limit)
    if media_type:
        q = q.where(DetectionJob.media_type == media_type)
    if verdict:
        q = q.where(DetectionJob.verdict == verdict)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{job_id}", response_model=DetectionJobDetailOut)
async def get_history_detail(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(DetectionJob, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job
