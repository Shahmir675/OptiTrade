import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from finvizfinance.quote import finvizfinance


def fetch_headlines(ticker, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        try:
            news_dfs[ticker] = get_news(ticker)
            break
        except Exception as e:
            print(f"Error fetching headlines for {ticker}: {e}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying {ticker}... (Attempt {attempt}/{retries})")
                time.sleep(delay)
            else:
                print(
                    f"Failed to fetch headlines for {ticker} after {retries} attempts. Skipping..."
                )


def get_news(ticker):
    print(f"Fetching headlines for {ticker}...")
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()
    return news_df


def main():
    file_path = "app/static/stocks.json"

    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]

    print(f"Loaded {len(data)} stocks.")

    tickers = [ticker_info["symbol"] for ticker_info in data]
    print(f"Tickers: {tickers[:10]}")

    global news_dfs
    news_dfs = {}

    with ThreadPoolExecutor(max_workers=1) as executor:
        with tqdm(total=len(tickers)) as pbar:
            futures = [executor.submit(fetch_headlines, ticker) for ticker in tickers]
            for future in futures:
                future.result()
                pbar.update(1)

    combined_df = pd.concat(
        [df.assign(Ticker=ticker) for ticker, df in news_dfs.items()], ignore_index=True
    )

    combined_df = combined_df[["Date", "Ticker", "Title", "Link"]]

    combined_df = combined_df.sort_values(by="Date", ascending=False).reset_index(
        drop=True
    )

    missing_tickers = list(set(tickers) - set(combined_df["Ticker"].unique()))
    print(f"Missing tickers (no news fetched): {missing_tickers}")

    output_file_path = "app/static/news.json"
    combined_df.to_json(output_file_path, orient="records", lines=True)

    print(f"News data saved to {output_file_path}")


if __name__ == "__main__":
    main()
