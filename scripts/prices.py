import aiohttp
import asyncio
from aiohttp_socks import ProxyConnector
from stem import Signal
import random
from datetime import datetime, UTC
from pytz import timezone
import re
import json
from fastapi import FastAPI

app = FastAPI()

PAKISTAN_TIMEZONE = timezone('Asia/Karachi')
missed_stocks = set()
pause_for_403 = False
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.64',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
]

price_data = []
stocks = []

path = '/home/shahmir/Backend/OptiTrade/app/static/stocks.json'
with open(path) as f:
    for line in f:
        stock = json.loads(line)
        stocks.append(stock['symbol'])
        utc_time = datetime.now(UTC)
        pst_time = utc_time.astimezone(PAKISTAN_TIMEZONE)
        formatted_time = pst_time.strftime("%Y-%m-%d %H:%M:%S")
        price_data.append({
            "symbol": stock['symbol'],
            "price": stock.get('close', 0.0),
            "time_fetched": formatted_time
        })

async def fetch_last_price_async(ticker, session, retries=3):
    global pause_for_403
    url = f"https://www.barchart.com/stocks/quotes/{ticker}/price-history/historical"
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }
    connector = ProxyConnector.from_url('socks5://127.0.0.1:9050')

    for attempt in range(retries):
        if pause_for_403:
            print(f"Pausing for 1 minute due to 403 error...")
            await asyncio.sleep(60)
            pause_for_403 = False

        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data_init = await response.text()
                    last_price = re.search(r'"lastPrice":"(\d+\.\d+)"', data_init)
                    if last_price:
                        utc_time = datetime.now(UTC)
                        pst_time = utc_time.astimezone(PAKISTAN_TIMEZONE)
                        formatted_time = pst_time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        existing_entry = next((item for item in price_data if item['symbol'] == ticker), None)

                        if existing_entry:
                            existing_entry["price"] = round(float(last_price.group(1)), 2)
                            existing_entry["time_fetched"] = formatted_time
                        else:
                            price_data.append({
                                "symbol": ticker,
                                "price": round(float(last_price.group(1)), 2),
                                "time_fetched": formatted_time,
                            })
                            
                        missed_stocks.discard(ticker)
                        
                        print(f"Fetched successfully for {ticker}")
                        
                        return
                elif response.status == 403:
                    print(f"{ticker}: HTTP 403 Forbidden - Pausing for 1 minutes...")
                    pause_for_403 = True
                    break
            break
        except Exception as e:
            print(f"Error fetching {ticker} on attempt {attempt + 1}: {e}")
            await asyncio.sleep(2 ** attempt)

    missed_stocks.add(ticker)

async def fetch_prices_infinite(tickers, batch_size=150):
    connector = ProxyConnector.from_url('socks5://127.0.0.1:9050')
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            if missed_stocks:
                tasks = [fetch_last_price_async(ticker, session) for ticker in missed_stocks]
                await asyncio.gather(*tasks)
                missed_stocks.clear()
                await asyncio.sleep(random.uniform(1, 5))

            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                tasks = [fetch_last_price_async(ticker, session) for ticker in batch if ticker not in missed_stocks]
                await asyncio.gather(*tasks)
                await asyncio.sleep(random.uniform(1, 5))
