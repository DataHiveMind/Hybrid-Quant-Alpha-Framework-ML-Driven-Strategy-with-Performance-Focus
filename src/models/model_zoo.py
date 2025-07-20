# src/models/model_zoo.py

import os
from src.utils.config import ARCHIVED_MODELS_DIR, PRODUCTION_MODELS_DIR
import joblib # Using joblib for simplicity, can be extended for other formats

# Import individual model loaders
from src.models.traditional_ml.rf_model import load_model as load_rf_model
from src.models.traditional_ml.xgb_model import load_model as load_xgb_model

def get_model(ticker, model_type, production=False):
    """
    Central function to load a trained model by ticker and type.

    Args:
        ticker (str): Stock ticker symbol.
        model_type (str): Type of model ('rf', 'xgb', etc.).
        production (bool): If True, loads from production directory, else from archived.

    Returns:
        Trained model object, or None if not found/supported.
    """
    if model_type == 'rf':
        return load_rf_model(ticker, production=production)
    elif model_type == 'xgb':
        return load_xgb_model(ticker, production=production)
    # Add more model types here as you implement them
    # elif model_type == 'lstm':
    #     from src.models.deep_learning.lstm_model import load_model as load_lstm_model
    #     return load_lstm_model(ticker, production=production)
    else:
        print(f"Error: Model type '{model_type}' not supported in model_zoo.")
        return None

if __name__ == "__main__":
    # Example usage
    ticker = 'AAPL'
    
    # Try loading an RF model
    rf_model = get_model(ticker, 'rf')
    if rf_model:
        print(f"Successfully loaded RandomForest model for {ticker}.")

    # Try loading an XGB model
    xgb_model = get_model(ticker, 'xgb')
    if xgb_model:
        print(f"Successfully loaded XGBoost model for {ticker}.")

    # Example of a non-existent model
    non_existent_model = get_model(ticker, 'non_existent_model')