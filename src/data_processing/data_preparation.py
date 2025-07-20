# src/data_processing/data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from src.utils.config import FEATURES_DIR, LABELS_DIR, BACKTEST_DATA_DIR

def load_processed_data(ticker):
    """
    Loads features and labels for a given ticker.
    """
    features_path = os.path.join(FEATURES_DIR, f'{ticker}_features.csv')
    labels_path = os.path.join(LABELS_DIR, f'{ticker}_labels.csv')

    print(f"Loading features from: {features_path}")
    print(f"Loading labels from: {labels_path}")

    try:
        features_df = pd.read_csv(features_path, index_col='Date', parse_dates=True)
        labels_df = pd.read_csv(labels_path, index_col='Date', parse_dates=True)

        # Align indices to ensure features and labels correspond correctly
        aligned_df = pd.merge(features_df, labels_df, left_index=True, right_index=True, how='inner')
        print(f"Aligned data shape for {ticker}: {aligned_df.shape}")
        
        X = aligned_df.drop(columns=['Target'])
        y = aligned_df['Target']
        
        return X, y
    except FileNotFoundError:
        print(f"Error: Data files for {ticker} not found. Please run feature_engineering.py first.")
        return None, None
    except Exception as e:
        print(f"Error loading or aligning data for {ticker}: {e}")
        return None, None

def temporal_train_test_split(X, y, test_size=0.2):
    """
    Performs a temporal train-test split to avoid look-ahead bias.
    The training set will be chronologically before the test set.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target Series.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Performing temporal train-test split with test_size={test_size}...")
    split_index = int(len(X) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Train set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler, fitting only on the training data.
    """
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, scaler

def prepare_data_for_ml(ticker, test_size=0.2):
    """
    Main function to load, split, and scale data for a given ticker.
    """
    X, y = load_processed_data(ticker)
    if X is None:
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=test_size)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save test data for backtesting purposes (aligned with original index)
    # We'll need the original (unscaled) prices for backtesting
    backtest_data_df = X_test.copy()
    backtest_data_df['Target'] = y_test # Add target back for backtest data
    # Ensure original 'Close' price is available for backtesting (from X, not X_scaled)
    # If 'Close' was a feature, it's already there. If not, we might need to retrieve it.
    # For now, assuming 'Close' is one of the features.

    backtest_filepath = os.path.join(BACKTEST_DATA_DIR, f'{ticker}_backtest_data.csv')
    backtest_data_df.to_csv(backtest_filepath, index=True)
    print(f"Saved backtest data for {ticker} to {backtest_filepath}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    # Example usage:
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    for ticker in tickers:
        print(f"\n--- Preparing data for {ticker} ---")
        X_train, X_test, y_train, y_test, scaler = prepare_data_for_ml(ticker)
        
        if X_train is not None:
            print(f"Data preparation for {ticker} complete. Shapes:")
            print(f"  X_train_scaled: {X_train.shape}")
            print(f"  X_test_scaled: {X_test.shape}")
            print(f"  y_train: {y_train.shape}")
            print(f"  y_test: {y_test.shape}")
            # You can add code here to save the scaler object if needed for deployment