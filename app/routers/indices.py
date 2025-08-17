import asyncio
import json

import httpx
import yfinance as yf
from bs4 import BeautifulSoup
from fastapi import APIRouter

router = APIRouter(prefix="", tags=["Indices"])


async def fetch_current_price():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://www.tradingview.com/symbols/NASDAQ-IXIC/")
        soup = BeautifulSoup(response.text, "html.parser")
        script_tags = soup.find_all(
            "script", {"type": "application/prs.init-data+json"}
        )
        if not script_tags or len(script_tags) < 4:
            raise ValueError("Unexpected HTML structure from TradingView")
        raw_json = script_tags[3].string or script_tags[3].text
        data = json.loads(raw_json)
        outer_key = next(iter(data))
        return round(float(data[outer_key]["data"]["symbol"]["trade"]["price"]), 2)


@router.get("/NASDAQ-summary")
async def nasdaq_summary():
    try:
        current = await fetch_current_price()
        ticker = yf.Ticker("^IXIC")
        previous_close = await asyncio.to_thread(
            lambda: round(ticker.info.get("previousClose", 0), 2)
        )
        daily_change = round(current - previous_close, 2)
        pct_change = round((daily_change / previous_close) * 100, 2)

        return {
            "current": current,
            "previous_close": previous_close,
            "daily_change": daily_change,
            "percent_change": pct_change,
        }

    except Exception as e:
        return {"error": str(e)}


@router.get("/NASDAQ-intraday")
async def nasdaq_intraday():
    try:
        ticker = yf.Ticker("^IXIC")
        intraday_df = await asyncio.to_thread(
            lambda: ticker.history(period="1d", interval="1h")
        )
        if intraday_df.empty:
            return {"intraday": []}

        intraday_df.reset_index(inplace=True)
        intraday = [
            {
                "time": row["Datetime"].isoformat(),
                "open": round(row["Open"], 2),
                "close": round(row["Close"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "volume": int(row["Volume"]),
            }
            for _, row in intraday_df.iterrows()
        ]

        return {"intraday": intraday}

    except Exception as e:
        return {"error": str(e)}
