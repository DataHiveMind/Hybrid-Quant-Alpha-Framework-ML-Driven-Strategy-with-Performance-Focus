# src/data_processing/feature_engineering.py

import pandas as pd
import numpy as np
import talib as ta
import os
from src.utils.config import RAW_DATA_DIR, FEATURES_DIR, LABELS_DIR

def calculate_technical_indicators(df):
    """
    Calculates various technical indicators and adds them to the DataFrame.
    Assumes input DataFrame has 'Open', 'High', 'Low', 'Close', 'Volume' columns.
    """
    print("Calculating technical indicators...")
    # Moving Averages
    df['SMA_10'] = ta.SMA(df['Close'], timeperiod=10)
    df['EMA_10'] = ta.EMA(df['Close'], timeperiod=10)
    df['SMA_50'] = ta.SMA(df['Close'], timeperiod=50)
    df['EMA_50'] = ta.EMA(df['Close'], timeperiod=50)

    # RSI
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)

    # MACD
    macd, macdsignal, macdhist = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist

    # Bollinger Bands
    upper, middle, lower = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    # Average True Range
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Chaikin A/D Line (requires High, Low, Close, Volume)
    df['AD_Line'] = ta.AD(df['High'], df['Low'], df['Close'], df['Volume'])

    return df

def calculate_statistical_features(df):
    """
    Calculates various statistical features.
    """
    print("Calculating statistical features...")
    # Log Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Volatility (rolling standard deviation of log returns)
    df['Volatility_10d'] = df['Log_Return'].rolling(window=10).std() * np.sqrt(252) # Annulized
    df['Volatility_20d'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)

    # Skewness and Kurtosis (rolling)
    df['Skewness_10d'] = df['Log_Return'].rolling(window=10).skew()
    df['Kurtosis_10d'] = df['Log_Return'].rolling(window=10).kurt()

    return df

def create_lagged_features(df, lags=[1, 2, 3, 5]):
    """
    Creates lagged features for selected columns.
    """
    print("Creating lagged features...")
    for lag in lags:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Log_Return'].shift(lag) # Lagged returns

    return df

def generate_features(ticker, raw_data_path):
    """
    Loads raw data, calculates all features, and returns the DataFrame.
    """
    print(f"Generating features for {ticker} from {raw_data_path}...")
    try:
        df = pd.read_csv(raw_data_path, index_col='Date', parse_dates=True)
        # Rename columns to standard 'Open', 'High', 'Low', 'Close', 'Volume' for TA-Lib
        df.columns = [col.replace(' ', '_') for col in df.columns] # Handle spaces in yfinance columns
        df.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low',
            'Close': 'Close', 'Adj_Close': 'Adj Close', 'Volume': 'Volume'
        }, inplace=True)

        df = calculate_technical_indicators(df)
        df = calculate_statistical_features(df)
        df = create_lagged_features(df)

        # Drop rows with NaN values introduced by rolling windows or lags
        df.dropna(inplace=True)

        print(f"Features generated for {ticker}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error generating features for {ticker}: {e}")
        return None

def save_features(df, ticker):
    """
    Saves the feature DataFrame to the features directory.
    """
    filepath = os.path.join(FEATURES_DIR, f'{ticker}_features.csv')
    df.to_csv(filepath, index=True)
    print(f"Saved features for {ticker} to {filepath}")

def generate_target_variable(df, forward_days=1, threshold=0.001):
    """
    Generates a classification target variable:
    1: Price increases by at least 'threshold' in 'forward_days'
    0: Price changes within +/- 'threshold'
    -1: Price decreases by at least 'threshold' in 'forward_days'
    """
    print(f"Generating target variable (forward {forward_days} days, threshold {threshold})...")
    future_close = df['Close'].shift(-forward_days)
    return_future = (future_close / df['Close']) - 1

    conditions = [
        return_future > threshold,
        (return_future >= -threshold) & (return_future <= threshold),
        return_future < -threshold
    ]
    choices = [1, 0, -1]
    df['Target'] = np.select(conditions, choices, default=np.nan)

    # Drop rows where target is NaN (at the end of the series due to shifting)
    df.dropna(subset=['Target'], inplace=True)
    df['Target'] = df['Target'].astype(int) # Ensure integer type

    print(f"Target variable generated. Remaining data points: {df.shape[0]}")
    return df

def save_labels(df, ticker):
    """
    Saves the target variable (labels) to a separate CSV.
    """
    labels_df = df[['Target']]
    filepath = os.path.join(LABELS_DIR, f'{ticker}_labels.csv')
    labels_df.to_csv(filepath, index=True)
    print(f"Saved labels for {ticker} to {filepath}")

if __name__ == "__main__":
    # Process the raw data downloaded in Phase 1
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

    for ticker in tickers:
        raw_data_path = os.path.join(RAW_DATA_DIR, 'market_data', f'{ticker}_raw.csv')
        
        # 1. Generate features
        features_df = generate_features(ticker, raw_data_path)
        
        if features_df is not None:
            # 2. Generate target variable on the feature-engineered data
            # Use a slightly smaller lookback for target to avoid dropping too many rows
            # Ensure the target is based on the 'Close' that aligns with the features
            final_df = generate_target_variable(features_df.copy(), forward_days=5, threshold=0.005) # Predict 5-day return direction

            # 3. Save features (without the target for pure feature set)
            # The 'Target' column is included in final_df, but we specifically save
            # features and labels separately for clarity and ML pipeline.
            # When we use this data for training, we'll merge them.
            save_features(final_df.drop(columns=['Target']), ticker) # Save features without target
            save_labels(final_df, ticker) # Save labels with index
            
            print(f"--- Finished processing {ticker} ---\n")