from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models import sqlalchemy_models as sql_models
from app.models import pydantic_models as pyd_models
from app.core.dependencies import get_db
import os
import sys

script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
sys.path.insert(0, script_path)

from portfolio_management import buy_stock, sell_stock 

router = APIRouter(prefix="/portfolio", tags=["Portfolio Actions"])

@router.post("/buy")
def buy_stock_endpoint(request: pyd_models.BuyStockRequest):
    try:
        result = buy_stock(
            request.user_id,
            request.symbol,
            request.quantity,
            request.order_type,
            request.limit_price
        )
        return {"message": "Stock purchase processed.", "details": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/sell")
def sell_stock_endpoint(request: pyd_models.SellStockRequest):
    try:
        result = sell_stock(
            request.user_id,
            request.symbol,
            request.quantity,
            request.order_type,
            request.limit_price
        )
        return {"message": "Stock sale processed.", "details": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
        portfolio_items.append(pyd_models.PortfolioResponseItem(
            user_id=p.user_id,
            symbol=p.symbol,
            quantity=p.quantity,
            average_price=p.average_price,
            current_value=p.current_value,
            total_invested=p.total_invested
        ))
    return pyd_models.PortfolioResponse(portfolio=portfolio_items)