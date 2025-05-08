import json
import pandas as pd
import yfinance as yf
import curl_cffi.requests
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import ChunkedEncodingError
from app.core.config import settings
import asyncio

session = curl_cffi.requests.Session(impersonate=settings.CURL_CFFI_IMPERSONATE)

proxy_config = {}
if settings.YFINANCE_PROXY_HTTP:
    proxy_config["http"] = settings.YFINANCE_PROXY_HTTP
if settings.YFINANCE_PROXY_HTTPS:
    proxy_config["https"] = settings.YFINANCE_PROXY_HTTPS

if proxy_config:
    yf.set_config(proxy=proxy_config)

executor = ThreadPoolExecutor(max_workers=4)

_stocks_df = None
_tickers_list = []
_news_data_list = []

def load_initial_data():
    global _stocks_df, _tickers_list, _news_data_list
    try:
        _stocks_df = pd.read_json(settings.STOCKS_JSON_PATH, lines=True)
        _tickers_list = list(_stocks_df['symbol'])
    except Exception as e:
        print(f"Error loading stocks.json: {e}")
        _stocks_df = pd.DataFrame() 
        _tickers_list = []

    try:
        with open(settings.NEWS_JSON_PATH, 'r') as file:
            _news_data_list = [json.loads(line) for line in file]
    except Exception as e:
        print(f"Error loading news.json: {e}")
        _news_data_list = []
        
load_initial_data()


def get_all_tickers() -> list:
    return _tickers_list

def get_stock_info_from_json(symbol: str) -> dict | None:
    if _stocks_df is not None and not _stocks_df.empty:
        stock = _stocks_df[_stocks_df['symbol'] == symbol]
        if not stock.empty:
            return stock.to_dict(orient='records')[0]
    return None

def get_all_stocks_from_json() -> list:
    if _stocks_df is not None and not _stocks_df.empty:
        df_copy = _stocks_df.copy()
        df_for_json = df_copy.astype(object).where(pd.notnull(df_copy), None)
        return df_for_json.to_dict(orient='records')
    return []

def download_stock_data_sync(stock_symbol: str, start_date: str = None, end_date: str = None, period: str = 'max'):
    try:
        ticker = yf.Ticker(stock_symbol, session=session)
        data = ticker.history(start=start_date, end=end_date, period=period)
        return data
    except ChunkedEncodingError:
        return None
    except Exception:
        return None


async def get_yfinance_stock_data_async(stock_symbol: str, start_date: str = None, end_date: str = None):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, download_stock_data_sync, stock_symbol, start_date, end_date)

def get_news_data(page: int = 1, page_size: int = 10) -> list:
    start = (page - 1) * page_size
    end = start + page_size
    return _news_data_list[start:end]