import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.services import data_service, price_feed_service

router = APIRouter(prefix="/stocks", tags=["Stocks"])


@router.get("")
async def get_all_stocks_list():
    data = data_service.get_all_stocks_from_json()
    return JSONResponse(content=data)


@router.get("/prices")
async def get_prices_endpoint():
    return price_feed_service.get_current_price_data()


@router.get("/{stock_symbol}")
async def get_stock_historical_data(
    stock_symbol: str, start_date: str = None, end_date: str = None
):
    data = await data_service.get_yfinance_stock_data_async(
        stock_symbol, start_date, end_date
    )

    if data is None or data.empty:
        stock_info = data_service.get_stock_info_from_json(stock_symbol)
        if stock_info:
            return JSONResponse(content=stock_info)
        raise HTTPException(
            status_code=404, detail="Stock data not found from yfinance or local JSON."
        )

    if not data.empty and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    cols_to_keep = ["Open", "High", "Low", "Close", "Volume"]
    existing_cols = [col for col in cols_to_keep if col in data.columns]

    if not existing_cols:
        raise HTTPException(
            status_code=404, detail="Stock data found but not in expected format."
        )

    data_filtered = data[existing_cols].round(2)

    response = {
        str(index.date()): row.to_dict() for index, row in data_filtered.iterrows()
    }
    return JSONResponse(content=response)
