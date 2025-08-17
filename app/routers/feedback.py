from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.session import get_db
from app.models.pydantic_models import FeedbackCreate, FeedbackOut
from app.models.sqlalchemy_models import Feedback
from app.services import email_service

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def submit_feedback(feedback: FeedbackCreate, db: AsyncSession = Depends(get_db)):
    feedback_obj = Feedback(feedback_message=feedback.feedback_message)
    db.add(feedback_obj)
    await db.commit()
    await db.refresh(feedback_obj)

    await email_service.send_feedback_confirmation(message=feedback.feedback_message)
    return {"message": "Feedback submitted successfully."}


@router.get("", response_model=list[FeedbackOut])
async def get_all_feedback(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Feedback))
    feedback_list = result.scalars().all()
    return feedback_list


@router.patch("/{feedback_id}/status")
async def update_feedback_status(
    feedback_id: int, new_status: str, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Feedback).where(Feedback.feedback_id == feedback_id)
    )
    feedback_obj = result.scalar_one_or_none()

    if not feedback_obj:
        raise HTTPException(status_code=404, detail="Feedback not found")

    feedback_obj.feedback_status = new_status
    await db.commit()
    await db.refresh(feedback_obj)

    return {"message": f"Feedback status updated to '{new_status}'."}
