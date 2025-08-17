import os
import sys
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.dependencies import get_db
from app.models import pydantic_models as pyd_models
from app.models import sqlalchemy_models as sql_models

script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts"))
sys.path.insert(0, script_path)

from portfolio_management import buy_stock, sell_stock

router = APIRouter(prefix="/portfolio", tags=["Portfolio Actions"])


@router.post("/buy")
async def buy_stock_endpoint(
    request: pyd_models.BuyStockRequest, db: AsyncSession = Depends(get_db)
):
    try:
        result = await buy_stock(
            user_id=request.user_id,
            symbol=request.symbol,
            quantity=request.quantity,
            db=db,
            order_type=request.order_type,
            limit_price=request.limit_price,
        )
        return {"message": "Stock purchase processed.", "details": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/sell")
async def sell_stock_endpoint(
    request: pyd_models.SellStockRequest, db: AsyncSession = Depends(get_db)
):
    try:
        result = await sell_stock(
            user_id=request.user_id,
            symbol=request.symbol,
            quantity=request.quantity,
            db=db,
            order_type=request.order_type,
            limit_price=request.limit_price,
        )
        return {"message": "Stock purchase processed.", "details": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/{user_id}", response_model=pyd_models.PortfolioResponse)
async def get_portfolio(
    user_id: int,
    page: int = 1,
    page_size: int = 10,
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size
    query = (
        select(sql_models.Portfolio)
        .where(sql_models.Portfolio.user_id == user_id)
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(query)
    portfolios_db = result.scalars().all()

    portfolio_items = []
    for p in portfolios_db:
        portfolio_items.append(
            pyd_models.PortfolioResponseItem(
                user_id=p.user_id,
                symbol=p.symbol,
                quantity=p.quantity,
                average_price=p.average_price,
                current_value=p.current_value,
                total_invested=p.total_invested,
            )
        )
    return pyd_models.PortfolioResponse(portfolio=portfolio_items)


@router.get(
    "/history/{user_id}",
    response_model=List[pyd_models.PortfolioHistoryItem],
    tags=["Portfolio History"],
)
async def get_portfolio_history(
    user_id: int, symbol: Optional[str] = None, session: AsyncSession = Depends(get_db)
):
    query = select(sql_models.PortfolioHistory).where(
        sql_models.PortfolioHistory.user_id == user_id
    )
    if symbol:
        query = query.where(sql_models.PortfolioHistory.symbol == symbol.upper())

    query = query.order_by(sql_models.PortfolioHistory.snapshot_date.desc())

    result = await session.execute(query)
    history_items = result.scalars().all()

    if not history_items:
        return []

    return history_items
