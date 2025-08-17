import asyncio
import os
import sys

from app.services.data_service import get_all_tickers

script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts"))
sys.path.insert(0, script_path)

from prices import fetch_prices_infinite as external_fetch_prices
from prices import price_data as external_price_data


async def start_price_fetching_task():
    tickers = get_all_tickers()
    if tickers:
        asyncio.create_task(external_fetch_prices(tickers))


def get_current_price_data():
    return external_price_data
