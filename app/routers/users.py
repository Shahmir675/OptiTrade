import hashlib
import os
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.core.dependencies import get_db
from app.models import pydantic_models as pyd_models
from app.models import sqlalchemy_models as sql_models
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
