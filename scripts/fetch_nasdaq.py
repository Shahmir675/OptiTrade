import logging
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import curl_cffi
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# --- Configuration ---
MAX_WORKERS = 32  # Increased workers as the task is I/O bound
ROLLING_WINDOW = 30
RETRY_LIMIT = 3
FAILURE_THRESHOLD = 20  # Increased threshold for larger runs
SAVE_PATH = "../app/static/stocks.json"

# --- Setup Logging and Session ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None

# --- Setup yfinance and Session for Network Requests ---
# Using a SOCKS5 proxy via Tor, as configured in the original script
yf.set_config(
    proxy={"http": "socks5h://127.0.0.1:9050", "https": "socks5h://127.0.0.1:9050"}
)
session = curl_cffi.requests.Session(impersonate="chrome")

# --- Robustness: Script Restart on Consecutive Failures ---
failure_lock = threading.Lock()
consecutive_failures = {"count": 0}


def restart_script():
    """Restarts the script if a critical number of failures occur."""
    logging.critical(
        f"Restarting script after {FAILURE_THRESHOLD} consecutive failures"
    )
    python = sys.executable
    os.execv(python, [python] + sys.argv)


def fetch_nasdaq_list():
    """Fetches the list of all Nasdaq-listed stocks from the Nasdaq API."""
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&exchange=nasdaq&download=true"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        logging.info("Fetching list of Nasdaq stocks...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        logging.info("Successfully fetched Nasdaq stock list.")
        return response.json()
    except requests.RequestException as e:
        logging.error(f"FATAL: Failed to fetch initial data from NASDAQ API: {e}")
        raise


def filter_stocks(df):
    """Applies filters to the DataFrame to select common stocks from the US."""
    logging.info(f"Filtering stocks. Initial count: {len(df)}")
    df = df[
        df["name"].str.contains(
            "common share|common stock|common|stock|share", case=False, na=False
        )
    ]
    df["name"] = df["name"].str.lower()

    excluded_patterns = [
        "preferred",
        "depositary",
        "preference",
        "unit ",
        " right",
        "units ",
    ]
    df = df[~df["name"].str.contains("|".join(excluded_patterns), case=False, na=False)]

    df = df[df["country"] == "United States"]

    filter_words = ["ordinary", "common"]
    exclude_df = df[
        ~df["name"].str.contains("|".join(filter_words), case=False, na=False)
    ]

    merged = df.merge(exclude_df, how="outer", indicator=True)
    df = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    logging.info(f"Finished filtering. Final count: {len(df)}")
    return df


def get_dividend_frequency(dividends):
    """Estimates dividend frequency based on payment dates."""
    if dividends.empty or len(dividends) < 2:
        return None

    dividend_dates = dividends.index.to_series().diff().dt.days.dropna()
    avg_days_between = dividend_dates.mean()

    if 80 <= avg_days_between <= 100:
        return "Quarterly"
    elif 170 <= avg_days_between <= 200:
        return "Semi-Annually"
    elif 350 <= avg_days_between <= 380:
        return "Annually"
    else:
        return "Irregular"


def process_ticker_data(ticker, volume_from_api):
    """
    Consolidated function to fetch ALL data for a single ticker.
    This is the core of the optimization, reducing multiple network calls to just a few.
    """
    attempt = 0
    last_error = None

    while attempt < RETRY_LIMIT:
        try:
            # 1. Create a single Ticker object
            stock = yf.Ticker(ticker, session=session)

            # 2. Fetch history and info (two main network calls)
            hist = stock.history(period="60d", interval="1d")
            info = stock.info

            # --- Data Validation ---
            if (
                hist.empty
                or "Close" not in hist
                or hist["Close"].dropna().shape[0] < 40
            ):
                raise ValueError(
                    "Insufficient historical data for volatility calculation."
                )
            if not info or info.get("quoteType") != "EQUITY":
                raise ValueError("Not an equity or no info available.")

            # --- A. Volatility Calculation ---
            prices = hist["Close"].dropna()
            log_returns = np.log(prices / prices.shift(1)).dropna()
            vol_series = log_returns.rolling(window=ROLLING_WINDOW).std().dropna()
            if vol_series.empty:
                raise ValueError("Volatility series is empty after calculation.")
            annualized_vol = vol_series.iloc[-1] * np.sqrt(252)

            # --- B. OHLCV Data ---
            latest_ohlcv = hist.iloc[-1]

            # --- C. Dividend Data ---
            dividends = stock.dividends
            last_dividend_date = (
                dividends.index[-1].strftime("%Y-%m-%d")
                if not dividends.empty
                else None
            )
            last_dividend_amount = dividends.iloc[-1] if not dividends.empty else None

            # --- D. Other Information from .info ---
            website = info.get("website", "")
            domain = (
                website.replace("http://", "").replace("https://", "").split("/")[0]
                if website
                else None
            )
            clearbit_logo_url = (
                f"https://logo.clearbit.com/{domain}" if domain else None
            )

            # --- Assemble the complete result dictionary ---
            result = {
                "symbol": ticker,
                "name": info.get("longName", info.get("shortName", ticker)),
                "open": round(latest_ohlcv["Open"], 2),
                "high": round(latest_ohlcv["High"], 2),
                "low": round(latest_ohlcv["Low"], 2),
                "close": round(latest_ohlcv["Close"], 2),
                "volume": int(latest_ohlcv["Volume"]),
                "marketCap": info.get("marketCap"),
                "30d_realized_volatility_annualized": round(annualized_vol, 6),
                "last_dividend_date": last_dividend_date,
                "last_dividend_amount": last_dividend_amount,
                "dividend_yield": (
                    f"{info.get('dividendYield', 0) * 100:.2f}%"
                    if info.get("dividendYield")
                    else None
                ),
                "payment_frequency": get_dividend_frequency(dividends),
                "industry": info.get("industry"),
                "sector": info.get("sector"),
                "logo_light": f"https://companiesmarketcap.com/img/company-logos/64/{ticker}.png",
                "logo_dark": f"https://companiesmarketcap.com/img/company-logos/64/{ticker}.D.png",
                "logo_high_light": f"https://companiesmarketcap.com/img/company-logos/128/{ticker}.png",
                "logo_high_dark": f"https://companiesmarketcap.com/img/company-logos/128/{ticker}.D.png",
                "clearbit_logo": clearbit_logo_url,
            }

            logging.info(f"[{ticker}] Successfully processed.")
            with failure_lock:
                consecutive_failures["count"] = 0  # Reset on success
            return result

        except Exception as e:
            last_error = e
            attempt += 1
            logging.warning(f"[{ticker}] Attempt {attempt}/{RETRY_LIMIT} failed: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(0.5)

    logging.error(f"[{ticker}] All {RETRY_LIMIT} attempts failed. Error: {last_error}")
    with failure_lock:
        consecutive_failures["count"] += 1
        if consecutive_failures["count"] >= FAILURE_THRESHOLD:
            restart_script()
    return None


if __name__ == "__main__":
    try:
        logging.info("--- Script Started ---")

        # 1. Fetch initial list from Nasdaq
        nasdaq_json = fetch_nasdaq_list()
        df_initial = pd.DataFrame(nasdaq_json["data"]["rows"])
        volume_data = {
            row["symbol"]: row["volume"] for row in nasdaq_json["data"]["rows"]
        }  # Keep original volume if needed

        # 2. Filter stocks to get the target list
        df_filtered = filter_stocks(df_initial)
        tickers_to_process = df_filtered["symbol"].tolist()
        logging.info(
            f"Starting to process {len(tickers_to_process)} tickers with {MAX_WORKERS} workers."
        )

        # 3. Process all tickers in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a future for each ticker
            future_to_ticker = {
                executor.submit(
                    process_ticker_data, ticker, volume_data.get(ticker)
                ): ticker
                for ticker in tickers_to_process
            }

            # As futures complete, collect results
            for future in as_completed(future_to_ticker):
                result = future.result()
                if result:
                    all_results.append(result)

        if not all_results:
            logging.error("No data could be processed. Exiting.")
            sys.exit(1)

        # 4. Create final DataFrame from results
        df_final = pd.DataFrame(all_results)

        # Merge with original data to get missing columns like 'lastsale', 'pctchange', etc.
        df_final = pd.merge(
            df_final,
            df_initial[["symbol", "lastsale", "netchange", "pctchange", "url"]],
            on="symbol",
            how="left",
        )
        df_final["url"] = "https://nasdaq.com" + df_final["url"]

        # 5. Final data cleaning and formatting
        df_final["lastsale"] = pd.to_numeric(
            df_final["lastsale"].str.strip().str.replace("$", ""), errors="coerce"
        ).round(2)
        df_final["pctchange"] = pd.to_numeric(
            df_final["pctchange"].str.strip().str.replace("%", ""), errors="coerce"
        ).round(2)
        df_final["netchange"] = pd.to_numeric(
            df_final["netchange"], errors="coerce"
        ).round(2)

        # 6. Define final column order and save
        final_columns = [
            "symbol",
            "name",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "lastsale",
            "netchange",
            "pctchange",
            "30d_realized_volatility_annualized",
            "marketCap",
            "last_dividend_date",
            "last_dividend_amount",
            "dividend_yield",
            "payment_frequency",
            "industry",
            "sector",
            "url",
            "logo_light",
            "logo_dark",
            "logo_high_light",
            "logo_high_dark",
            "clearbit_logo",
        ]
        # Ensure all columns exist, fill missing with None
        for col in final_columns:
            if col not in df_final.columns:
                df_final[col] = None

        df_final = df_final[final_columns]
        df_final = df_final.sort_values(by="symbol").reset_index(drop=True)

        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        df_final.to_json(SAVE_PATH, index=False, orient="records", lines=True)

        logging.info(f"--- Script Finished Successfully. Data saved to {SAVE_PATH} ---")

    except Exception as e:
        logging.critical(
            f"A critical error occurred in the main script block: {e}", exc_info=True
        )
        sys.exit(1)
