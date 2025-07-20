# src/data_processing/data_ingestion.py

import yfinance as yf
import os
import pandas as pd
from datetime import datetime
from src.utils.config import RAW_DATA_DIR # Import our config

def download_stock_data(ticker, start_date, end_date, interval='1d'):
    """
    Downloads historical stock data for a given ticker from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Data interval (e.g., '1d', '1h', '1wk').
                        Note: Hourly data is limited to past 7 days, daily for longer.

    Returns:
        pd.DataFrame: DataFrame containing historical data, or None if download fails.
    """
    print(f"Downloading {ticker} data from {start_date} to {end_date} with interval {interval}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            print(f"No data found for {ticker} in the specified range.")
            return None
        return data
    except Exception as e:
        print(f"Error downloading {ticker} data: {e}")
        return None

def save_data(df, ticker, data_type='market_data', file_format='csv'):
    """
    Saves the DataFrame to the appropriate raw data directory.

    Args:
        df (pd.DataFrame): DataFrame to save.
        ticker (str): Stock ticker symbol.
        data_type (str): Subdirectory under raw (e.g., 'market_data', 'fundamental_data').
        file_format (str): 'csv' or 'parquet'.
    """
    output_dir = os.path.join(RAW_DATA_DIR, data_type)
    os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists

    filename = f"{ticker}_raw.{file_format}"
    filepath = os.path.join(output_dir, filename)

    if file_format == 'csv':
        df.to_csv(filepath, index=True) # index=True to save the DateTime index
    elif file_format == 'parquet':
        df.to_parquet(filepath, index=True)
    else:
        print(f"Unsupported file format: {file_format}")
        return

    print(f"Saved {ticker} raw data to {filepath}")

if __name__ == "__main__":
    # Example usage:
    # Let's download data for a few tech stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d') # Today's date

    for ticker in tickers:
        df = download_stock_data(ticker, start_date, end_date, interval='1d')
        if df is not None:
            save_data(df, ticker, data_type='market_data', file_format='csv')

    print("\nRaw data ingestion complete.")