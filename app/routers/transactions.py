from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.dependencies import get_db
from app.models import pydantic_models as pyd_models
from app.models import sqlalchemy_models as sql_models

router = APIRouter(prefix="/transactions", tags=["Transactions"])


@router.get("/{user_id}", response_model=List[pyd_models.TransactionResponse])
async def get_transactions_by_user(
    user_id: int, session: AsyncSession = Depends(get_db)
):
    result = await session.execute(
        select(sql_models.Transaction).where(sql_models.Transaction.user_id == user_id)
    )
    transactions = result.scalars().all()

    return transactions
