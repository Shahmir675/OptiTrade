import pandas as pd
import requests

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
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:85.0) Gecko/20100101 Firefox/85.0'
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

if __name__ == "__main__":
    nasdaq_data = fetch_nasdaq_data()
    df = pd.DataFrame(nasdaq_data['data']['rows'])
    filtered_df = filter_stocks(df)
    print(filtered_df)
