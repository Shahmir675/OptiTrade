import pandas as pd
import requests
import yfinance as yf
import logging
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import curl_cffi

pd.options.mode.chained_assignment = None

warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO)

yf.set_config(proxy={"http": "socks5h://127.0.0.1:9050", "https": "socks5h://127.0.0.1:9050"})
session = curl_cffi.requests.Session(impersonate="chrome")

def fetch_nasdaq_data():
    """Fetch Nasdaq Tickers Data"""
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&exchange=nasdaq&download=true"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch data from NASDAQ API: {e}")
        raise Exception(f"Failed to fetch data from NASDAQ API: {e}")

def filter_stocks(df):
    df = df[df['name'].str.contains('common share|common stock|common|stock|share', case=False, na=False)]
    df['name'] = df['name'].str.lower()
    
    excluded_patterns = ['preferred', 'depositary', 'preference', 'unit ', ' right', 'units ']
    df = df[~df['name'].str.contains('|'.join(excluded_patterns), case=False, na=False)]
    
    df = df[df['country'] == 'United States']
    
    filter_words = ['ordinary', 'common']
    exclude_df = df[~df['name'].str.contains('|'.join(filter_words), case=False, na=False)]
    
    merged = df.merge(exclude_df, how='outer', indicator=True)
    return merged[merged['_merge'] == 'left_only'].drop(columns='_merge')

def generate_logo_urls(ticker):
    base_url = "https://companiesmarketcap.com/img/company-logos/"
    return {
        'light': f"{base_url}64/{ticker}.png",
        'dark': f"{base_url}64/{ticker}.D.png",
        'high_light': f"{base_url}128/{ticker}.png",
        'high_dark': f"{base_url}128/{ticker}.D.png",
    }

def get_clearbit_logo(ticker):
    try:
        stock = yf.Ticker(ticker, session=session)
        website = stock.info.get("website", "")
        if website:
            domain = website.replace("http://", "").replace("https://", "").split('/')[0]
            return f"https://logo.clearbit.com/{domain}"
        return None
    except Exception as e:
        logging.error(f"Error fetching Clearbit logo for {ticker}: {e}")
        return None

def get_sector_and_industry(ticker):
    try:
        stock = yf.Ticker(ticker, session=session)
        return stock.info.get("sector"), stock.info.get("industry")
    except Exception as e:
        logging.error(f"Error fetching sector and industry for {ticker}: {e}")
        return None, None

def fetch_ohlcv_data(ticker):
    url = f'https://eodhd.com/financial-summary/{ticker}.US'
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        ohlcv = {}

        relevant_props = {
            'Prev. Close': 'Close',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Volume': 'Volume',
        }

        props = soup.find_all('div', class_='ticker__props__item')
        for prop in props:
            name = prop.find('span', class_='ticker__props__name').text.strip()
            value = prop.find('span', class_='ticker__props__value').text.strip()

            if name in relevant_props:
                value = value.replace(' K', '000').replace(' M', '000000')
                if name == 'Volume':
                    ohlcv[relevant_props[name]] = int(value.replace(' ', ''))
                else:
                    ohlcv[relevant_props[name]] = float(value)

        if not ohlcv:
            raise ValueError('No data found on EODHD')

        return ohlcv

    except (requests.HTTPError, ValueError):
        stock = yf.Ticker(ticker, session=session)
        hist = stock.history(period="1d")
        if hist.empty:
            raise ValueError('No data found on both EODHD and yfinance')

        ohlcv = {
            'Open': hist['Open'].iloc[0],
            'High': hist['High'].iloc[0],
            'Low': hist['Low'].iloc[0],
            'Close': hist['Close'].iloc[0],
            'Volume': int(hist['Volume'].iloc[0]),
        }

        return ohlcv
    
def get_dividend_data(ticker):
    stock = yf.Ticker(ticker, session=session)
    dividends = stock.dividends
    last_dividend_date = dividends.index[-1].strftime('%Y-%m-%d') if not dividends.empty else None
    last_dividend_amount = dividends.iloc[-1] if not dividends.empty else None
    dividend_yield = stock.info.get('dividendYield', None)
    
    if not dividends.empty:
        dividend_dates = dividends.index.to_series().diff().dt.days.dropna()
        avg_days_between = dividend_dates.mean()
        
        if avg_days_between <= 100:
            frequency = 'Quarterly'
        elif avg_days_between <= 200:
            frequency = 'Semi-Annually'
        else:
            frequency = 'Annually'
    else:
        frequency = None
        
    return {
        'last_dividend_date': last_dividend_date,
        'last_dividend_amount': last_dividend_amount,
        'dividend_yield': f"{dividend_yield}%" if dividend_yield else None,
        'payment_frequency': frequency
    }

def process_ticker(ticker, volume):
    """Process a single ticker: Fetch logos, sector/industry, OHLCV data, and stock name."""
    try:
        urls = generate_logo_urls(ticker)
        
        sector, industry = get_sector_and_industry(ticker)
        
        stock = yf.Ticker(ticker, session=session)
        stock_name = stock.info.get('longName', stock.info.get('shortName', ticker))
        
        ohlcv_data = fetch_ohlcv_data(ticker)
        if ohlcv_data is None:
            ohlcv_data = {'Open': None, 'High': None, 'Low': None, 'Close': None, 'Volume': None}
        
        dividend_data = get_dividend_data(ticker)
        
        logging.info(f"Processed ticker: {ticker}")
        
        return {
            'ticker': ticker,
            'logo_light': urls['light'],
            'logo_dark': urls['dark'],
            'logo_high_light': urls['high_light'],
            'logo_high_dark': urls['high_dark'],
            'clearbit_logo': get_clearbit_logo(ticker),
            'sector': sector,
            'industry': industry,
            'name': stock_name,
            'open': ohlcv_data.get('Open'),
            'high': ohlcv_data.get('High'),
            'low': ohlcv_data.get('Low'),
            'close': ohlcv_data.get('Close'),
            'volume': volume,
            'last_dividend_date': dividend_data['last_dividend_date'],
            'last_dividend_amount': dividend_data['last_dividend_amount'],
            'dividend_yield': dividend_data['dividend_yield'],
            'payment_frequency': dividend_data['payment_frequency']
        }
    except Exception as e:
        logging.error(f"Error processing ticker {ticker}: {e}")
        return None

def add_logos_and_info(df, volume_data):
    tickers = df['symbol'].tolist()
    results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_ticker, ticker, volume_data.get(ticker, 0)): ticker for ticker in tickers}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    processed_df = pd.DataFrame(results)
    
    if processed_df.empty:
        return df

    merged_df = df.merge(
        processed_df,
        left_on='symbol',
        right_on='ticker',
        how='left',
        suffixes=('_OLD', '')
    )

    merged_df = merged_df.drop(columns=['ticker'] + 
                        [col for col in merged_df.columns if col.endswith('_OLD')],
                        errors='ignore')

    return merged_df

if __name__ == '__main__':
    try:
        logging.info("Script Started")
        nasdaq_data = fetch_nasdaq_data()
        df = pd.DataFrame(nasdaq_data['data']['rows'])
        df['url'] = 'https://nasdaq.com' + df['url']
        volume_data = {row['symbol']: row['volume'] for row in nasdaq_data['data']['rows']}
        df = filter_stocks(df)
        df = df.reset_index(drop=True)
        print("Done")
        df_with_logos_and_info = add_logos_and_info(df, volume_data)

        columns_to_int = ['volume', 'marketCap']
        columns_to_float = ['open', 'high', 'low', 'close', 'lastsale', 'pctchange', 'netchange']

        df_with_logos_and_info['lastsale'] = pd.to_numeric(df_with_logos_and_info['lastsale'].str[1:]).round(2)
        df_with_logos_and_info['pctchange'] = df_with_logos_and_info['pctchange'].str[:-1].round(2)
        df_with_logos_and_info[columns_to_int] = df_with_logos_and_info[columns_to_int].apply(pd.to_numeric, errors='coerce', downcast='integer')
        df_with_logos_and_info[columns_to_float] = df_with_logos_and_info[columns_to_float].apply(pd.to_numeric, errors='coerce').astype(float)
        df_with_logos_and_info[columns_to_float] = df_with_logos_and_info[columns_to_float].round(2)
        df_with_logos_and_info
        df_with_logos_and_info = df_with_logos_and_info.drop(['ipoyear', 'country'], axis=1)
        df_with_logos_and_info = df_with_logos_and_info[['symbol', 'name', 'open', 'high', 
                                                            'low', 'close', 'volume', 'lastsale', 
                                                            'netchange', 'pctchange', 'marketCap', 'last_dividend_date',
                                                            'last_dividend_amount', 'dividend_yield', 'payment_frequency', 
                                                            'industry', 'sector', 'url', 'logo_light', 
                                                            'logo_dark', 'logo_high_light', 'logo_high_dark', 
                                                            'clearbit_logo']]
        df_with_logos_and_info = df_with_logos_and_info.sort_values(by='symbol').reset_index(drop=True)
        path = 'app/static/stocks.json'
        df_with_logos_and_info.to_json(path, index=False, orient='records', lines=True)
        
        logging.info(f"Data saved to {path}")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
