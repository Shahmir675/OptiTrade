import pandas as pd
import requests
import json
import yfinance as yf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

def fetch_nasdaq_data():
    """
    Fetch NASDAQ stock data from the API.
    
    Returns:
        dict: JSON response containing stock data.
        
    Raises:
        Exception: If the API request fails or returns an error.
    """
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&exchange=nasdaq&download=true"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch data from NASDAQ API: {e}")

def filter_stocks(df):
    """
    Filter stocks for common shares and U.S. stocks.
    
    Args:
        df (pd.DataFrame): DataFrame containing NASDAQ stock data.
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing relevant common U.S. stocks.
    """
    to_include = 'common share|common stock|common|stock|share'
    df = df[df['name'].str.lower().str.contains(to_include, case=False, na=False)]
    df.loc[:, 'name'] = df.loc[:, 'name'].str.lower()

    to_exclude = ['preferred', 'depositary', 'preference', 'unit ', ' right', 'units ']
    pattern = '|'.join(to_exclude)
    df = df[~df['name'].str.contains(pattern, case=False, na=False)].copy()

    df = df[df['country'] == 'United States']

    filter_words = ['ordinary', 'common']
    pattern = '|'.join(filter_words)
    exclude_df = df[~df['name'].str.contains(pattern, case=False, na=False)].copy()

    merged = df.merge(exclude_df, how='outer', indicator=True)
    return merged[merged['_merge'] == 'left_only'].drop(columns='_merge')

def get_company_domain(ticker):
    """
    Fetch the company's website domain using Yahoo Finance data.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        str: The company's domain (e.g., 'apple.com') or None if not found.
    """
    try:
        stock = yf.Ticker(ticker)
        website = stock.info.get("website", "")
        if website:
            return website.replace("http://", "").replace("https://", "").split('/')[0]
        return None
    except Exception as e:
        print(f"Error fetching domain for {ticker}: {e}")
        return None

def fetch_logo(ticker):
    """
    Fetch the company's logo using the ticker symbol and Clearbit Logo API.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        str: URL of the company's logo or None if not found.
    """
    domain = get_company_domain(ticker)
    if domain:
        return f"https://logo.clearbit.com/{domain}"
    return None

def add_logos_by_ticker(df):
    """
    Add a 'logo_url' column to the DataFrame with logo URLs for each ticker symbol.

    Args:
        df (pd.DataFrame): DataFrame containing stock data (must have 'ticker' column).

    Returns:
        pd.DataFrame: Updated DataFrame with logo URLs.
    """
    tickers = df['symbol'].tolist()

    logo_urls = []

    def fetch_with_delay(ticker):
        logo = fetch_logo(ticker)
        time.sleep(0.5)
        return logo

    with ThreadPoolExecutor() as executor:
        logo_urls = list(tqdm(executor.map(fetch_with_delay, tickers), total=len(tickers), desc="Fetching logos"))

    df['logo_url'] = logo_urls
    return df

if __name__ == "__main__":
    try:
        nasdaq_data = fetch_nasdaq_data()
        df = pd.DataFrame(nasdaq_data['data']['rows'])
        filtered_df = filter_stocks(df)
        df_with_logos = add_logos_by_ticker(filtered_df)
        path = 'filtered_stocks_with_logos.csv'
        df_with_logos.to_csv(path, index=False)

        print(f"Filtered data with logos saved to {csv_file_path}")
        print(json.dumps(df_with_logos.to_dict(orient='records')))
    except Exception as e:
        print(f"An error occurred: {e}")
