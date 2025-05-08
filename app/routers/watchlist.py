from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List
from app.models import sqlalchemy_models as sql_models
from app.models import pydantic_models as pyd_models
from app.core.dependencies import get_db
from app.services.data_service import get_stock_info_from_json


router = APIRouter(prefix="/watchlist", tags=["Watchlist"])

@router.get("/{user_id}", response_model=List[pyd_models.WatchlistResponse])
async def get_watchlist(user_id: int, db: AsyncSession = Depends(get_db)):
    query = select(sql_models.Watchlist).where(sql_models.Watchlist.user_id == user_id)
    result = await db.execute(query)
    watchlist_items = result.scalars().all()
    return watchlist_items


@router.post("/{user_id}/{stock_symbol}", response_model=pyd_models.WatchlistResponse)
async def add_to_watchlist(
    user_id: int,
    stock_symbol: str,
    db: AsyncSession = Depends(get_db)
):
    query = select(sql_models.Watchlist).where(
        sql_models.Watchlist.user_id == user_id,
        sql_models.Watchlist.stock_symbol == stock_symbol
    )
    result = await db.execute(query)
    existing_entry = result.scalars().first()

    if existing_entry:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Stock already in watchlist")

    if not get_stock_info_from_json(stock_symbol):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stock symbol not found")

    new_watchlist_item = sql_models.Watchlist(user_id=user_id, stock_symbol=stock_symbol)
    db.add(new_watchlist_item)
    await db.commit()
    await db.refresh(new_watchlist_item)
    return new_watchlist_item

@router.delete("/{user_id}/{stock_symbol}")
async def remove_from_watchlist(
    user_id: int,
    stock_symbol: str,
    db: AsyncSession = Depends(get_db)
):
    query = select(sql_models.Watchlist).where(
        sql_models.Watchlist.user_id == user_id,
        sql_models.Watchlist.stock_symbol == stock_symbol
    )
    result = await db.execute(query)
    watchlist_item = result.scalars().first()

    if not watchlist_item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stock not in watchlist")

    await db.delete(watchlist_item)
    await db.commit()
    return {"detail": "Stock removed from watchlist"}