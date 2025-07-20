# src/alpha_strategies/strategy_logic.py

import pandas as pd
import numpy as np
import yaml
import os
from src.utils.config import CONFIG_DIR
from src.models.model_zoo import get_model
from src.data_processing.data_preparation import prepare_data_for_ml # We'll need the scaler from here


def load_strategy_config(config_name="default_strategy.yaml"):
    """Loads strategy parameters from a YAML file."""
    config_path = os.path.join(CONFIG_DIR, 'strategy_configs', config_name)
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Loaded strategy configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Strategy config file not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error loading strategy config: {e}")
        return None

def generate_signals(ticker, strategy_config, production_model=False):
    """
    Generates trading signals for a given ticker based on ML model predictions.

    Args:
        ticker (str): The stock ticker symbol.
        strategy_config (dict): Dictionary of strategy parameters.
        production_model (bool): Whether to load production or archived model.

    Returns:
        pd.DataFrame: DataFrame with 'Close' prices, ML predictions, and 'Signal' column.
                      Signal: 1 for Buy, -1 for Sell, 0 for Hold/Cash.
    """
    print(f"\nGenerating signals for {ticker} using {strategy_config['model_to_use']} model...")

    # Load prepared test data and the scaler used for training
    # We load the full processed data and then scale X_test for prediction
    X, y = prepare_data_for_ml(ticker, test_size=0.2)[0:2] # Get X, y from preparation
    
    # We need the scaler used during data preparation.
    # For simplicity, we'll re-run data_preparation.py to get the scaler object directly.
    # In a real pipeline, the scaler would be saved and loaded.
    _, X_test_scaled, _, _, scaler = prepare_data_for_ml(ticker, test_size=0.2)
    
    if X_test_scaled is None:
        print(f"Could not load prepared data for {ticker}. Skipping signal generation.")
        return None

    # Load the trained model
    model_type = strategy_config['model_to_use']
    model = get_model(ticker, model_type, production=production_model)

    if model is None:
        print(f"No {model_type} model found for {ticker}. Skipping signal generation.")
        return None
    
    # Ensure X_test_scaled columns match training columns
    # This is critical if the order of columns changed or some were dropped during feature engineering
    # For now, we assume `prepare_data_for_ml` returns consistent columns.
    
    # Get ML predictions on the scaled test data
    ml_predictions = model.predict(X_test_scaled)

    # Map XGBoost predictions back if necessary
    if model_type == 'xgb':
        # Assuming XGBoost predicted 0, 1, 2 for original -1, 0, 1
        prediction_mapping_back = {0: -1, 1: 0, 2: 1}
        ml_predictions = np.array([prediction_mapping_back[p] for p in ml_predictions])

    # Create a DataFrame to hold predictions and signals, indexed by Date
    signals_df = X_test_scaled.copy() # Use the test set index
    signals_df['Close'] = X.loc[signals_df.index, 'Close'] # Get original Close prices
    signals_df['ML_Prediction'] = ml_predictions

    # Convert ML predictions to trading signals based on strategy config
    signals_df['Signal'] = 0 # Default to hold/cash

    # Map predictions to actions
    # This loop needs to be careful with the order of conditions if any overlap
    # We're directly mapping ML_Prediction to Signal (1, 0, -1) which aligns with buy/hold/sell
    
    # Example: If ML_Prediction is 1, set Signal to 1 (BUY)
    signals_df.loc[signals_df['ML_Prediction'] == 1, 'Signal'] = 1
    # Example: If ML_Prediction is -1, set Signal to -1 (SELL/SHORT)
    signals_df.loc[signals_df['ML_Prediction'] == -1, 'Signal'] = -1
    # If ML_Prediction is 0, Signal remains 0 (HOLD)

    print(f"Signals generated for {ticker}. First few signals:\n{signals_df['Signal'].head()}")
    return signals_df[['Close', 'Signal', 'ML_Prediction']] # Return relevant columns for backtesting

if __name__ == "__main__":
    ticker = 'AAPL'
    strategy_config = load_strategy_config()
    if strategy_config:
        signals_df = generate_signals(ticker, strategy_config, production_model=False)
        if signals_df is not None:
            print(f"\nGenerated signals for {ticker}:\n")
            print(signals_df.head())
            print(signals_df['Signal'].value_counts())
            # Check for any NaNs that might have crept in
            print(f"NaNs in signals_df: {signals_df.isnull().sum().sum()}")