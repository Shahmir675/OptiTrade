from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List
from app.models import sqlalchemy_models as sql_models
from app.models import pydantic_models as pyd_models
from app.core.dependencies import get_db

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("", response_model=List[pyd_models.UserResponse])
async def get_users_list(page: int = 1, page_size: int = 10, db: AsyncSession = Depends(get_db)):
    offset = (page - 1) * page_size
    query = select(sql_models.UserModel).offset(offset).limit(page_size)
    result = await db.execute(query)
    users = result.scalars().all()
    return users

@router.get("/{user_id}/balance", response_model=pyd_models.UserBalanceResponse)
async def get_user_balance(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(sql_models.UserBalanceModel)
        .filter(sql_models.UserBalanceModel.user_id == user_id)
    )
    user_balance = result.scalars().first()

    if not user_balance:
        raise HTTPException(status_code=404, detail="User balance not found")
    return user_balance