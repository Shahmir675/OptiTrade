from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List
from app.models import sqlalchemy_models as sql_models
from app.models import pydantic_models as pyd_models
from app.core.dependencies import get_db

router = APIRouter(prefix="/transactions", tags=["Transactions"])

@router.get("", response_model=List[pyd_models.TransactionResponse])
async def get_transactions(session: AsyncSession = Depends(get_db)):
    result = await session.execute(select(sql_models.Transaction))
    transactions = result.scalars().all()
    return transactions