import hashlib
import os
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.dependencies import get_db
from app.models import pydantic_models as pyd_models
from app.models import sqlalchemy_models as sql_models
from app.services.risk_profile_service import calculate_risk_score, get_risk_category, get_risk_level, generate_recommendations
from app.utils.avatar import select_random_avatar

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("", response_model=List[pyd_models.UserResponse])
async def get_users_list(db: AsyncSession = Depends(get_db)):
   query = select(sql_models.UserModel).order_by(sql_models.UserModel.id)
   result = await db.execute(query)
   users = result.scalars().all()
   return users


@router.get("/{user_id}/balance", response_model=pyd_models.UserBalanceResponse)
async def get_user_balance(user_id: int, db: AsyncSession = Depends(get_db)):
   result = await db.execute(
       select(sql_models.UserBalanceModel).filter(
           sql_models.UserBalanceModel.user_id == user_id
       )
   )
   user_balance = result.scalars().first()

   if not user_balance:
       raise HTTPException(status_code=404, detail="User balance not found")
   return user_balance


@router.get("/{user_id}/risk-profile", response_model=pyd_models.RiskProfileResponse)
async def get_risk_profile(user_id: int, db: AsyncSession = Depends(get_db)):
   result = await db.execute(
       select(sql_models.RiskProfile)
       .options(selectinload(sql_models.RiskProfile.recommendations))
       .filter(sql_models.RiskProfile.user_id == user_id)
   )
   risk_profile = result.scalars().first()
   
   if not risk_profile:
       raise HTTPException(status_code=404, detail="Risk profile not found")
   return risk_profile


@router.post("/{user_id}/risk-profile/assessment", response_model=pyd_models.RiskAssessmentResponse)
async def create_risk_assessment(
   user_id: int,
   assessment_request: pyd_models.RiskAssessmentRequest,
   db: AsyncSession = Depends(get_db)
):
   if assessment_request.user_id != user_id:
       raise HTTPException(status_code=400, detail="User ID mismatch")
   
   assessment = sql_models.RiskAssessment(
       user_id=user_id,
       age=assessment_request.age,
       investment_experience=assessment_request.investment_experience,
       risk_tolerance=assessment_request.risk_tolerance,
       investment_timeline=assessment_request.investment_timeline,
       financial_goals=assessment_request.financial_goals,
       income_stability=assessment_request.income_stability
   )
   
   db.add(assessment)
   await db.flush()
   
   existing_profile = await db.execute(
       select(sql_models.RiskProfile).filter(sql_models.RiskProfile.user_id == user_id)
   )
   existing = existing_profile.scalars().first()
   
   if existing:
       await db.delete(existing)
   
   risk_score = calculate_risk_score(assessment_request)
   portfolio_alignment = {"score": 75.0, "status": "aligned", "message": "Portfolio aligns well with risk profile"}
   breakdown = {
       "questionnaire_score": risk_score,
       "portfolio_score": 70.0,
       "age_factor": max(0, 100 - assessment_request.age),
       "experience_factor": {"beginner": 30, "intermediate": 60, "advanced": 90}[assessment_request.investment_experience],
       "financial_situation": 65.0,
       "time_horizon": min(100, assessment_request.investment_timeline * 10)
   }
   
   risk_profile = sql_models.RiskProfile(
       user_id=user_id,
       risk_score=risk_score,
       risk_category=get_risk_category(risk_score),
       risk_level=get_risk_level(risk_score),
       last_assessment_date=assessment.created_at,
       next_assessment_due="2026-02-15",
       has_completed_assessment=True,
       portfolio_alignment=portfolio_alignment,
       breakdown=breakdown
   )
   
   db.add(risk_profile)
   await db.flush()
   
   recommendations = generate_recommendations(risk_score)
   for rec in recommendations:
       db_rec = sql_models.RiskRecommendation(
           risk_profile_id=risk_profile.id,
           type=rec["type"],
           title=rec["title"],
           description=rec["description"],
           priority=rec["priority"],
           impact=rec["impact"]
       )
       db.add(db_rec)
   
   await db.commit()
   
   result = await db.execute(
       select(sql_models.RiskProfile)
       .options(selectinload(sql_models.RiskProfile.recommendations))
       .filter(sql_models.RiskProfile.id == risk_profile.id)
   )
   final_profile = result.scalars().first()
   
   return pyd_models.RiskAssessmentResponse(
       user_id=user_id,
       assessment_id=assessment.id,
       risk_profile=final_profile,
       created_at=assessment.created_at
   )


@router.post("/{user_id}/profile-image")
async def upload_profile_image(
   user_id: int, file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
   result = await db.execute(select(sql_models.UserModel).filter_by(id=user_id))
   user = result.scalar_one_or_none()
   if not user:
       raise HTTPException(status_code=404, detail="User not found")

   content = await file.read()
   file_hash = hashlib.sha256(content).hexdigest()
   ext = file.filename.split(".")[-1]
   filename = f"{user_id}{file_hash}.{ext}"
   filepath = os.path.join(settings.MEDIA_ROOT, filename)
   new_url = settings.MEDIA_URL + filename

   if user.image_url and user.image_url != new_url:
       old_filename = user.image_url.replace(settings.MEDIA_URL, "")
       old_filepath = os.path.join(settings.MEDIA_ROOT, old_filename)

       if os.path.exists(old_filepath):
           os.remove(old_filepath)

   if not os.path.exists(filepath):
       with open(filepath, "wb") as f:
           f.write(content)

   user.image_url = new_url
   await db.commit()

   return {"message": "Profile image uploaded", "url": new_url}


@router.delete("/{user_id}/profile-image")
async def delete_profile_image(user_id: int, db: AsyncSession = Depends(get_db)):
   result = await db.execute(select(sql_models.UserModel).filter_by(id=user_id))
   user = result.scalar_one_or_none()
   if not user:
       raise HTTPException(status_code=404, detail="User not found")

   if user.image_url and user.image_url.startswith(settings.MEDIA_URL):
       local_path = user.image_url.replace(
           settings.MEDIA_URL, settings.MEDIA_ROOT + "/"
       )
       try:
           os.remove(local_path)
       except FileNotFoundError:
           pass

   user.image_url = select_random_avatar()
   await db.commit()

   return {"message": "Profile image removed", "url": user.image_url}