# src/models/traditional_ml/xgb_model.py

import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib # For saving/loading models
import os
from src.utils.config import ARCHIVED_MODELS_DIR, PRODUCTION_MODELS_DIR

def train_xgboost_model(X_train, y_train, **kwargs):
    """
    Trains an XGBoostClassifier model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        kwargs: Hyperparameters for XGBClassifier.

    Returns:
        xgboost.XGBClassifier: Trained model.
    """
    print("Training XGBoostClassifier model...")
    # Default parameters, can be overridden by kwargs
    # Use 'multi:softprob' for multi-class classification if targets are 0,1,2 etc.
    # Our targets are -1, 0, 1. XGBoost expects 0, 1, 2 for multi-class.
    # We need to map y_train and y_test to 0, 1, 2 for XGBoost and then map back.
    
    # Map target values: -1 -> 0, 0 -> 1, 1 -> 2
    y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
    
    params = {
        'objective': 'multi:softmax', # For multi-class classification
        'num_class': 3, # Because we have -1, 0, 1 (mapped to 0, 1, 2)
        'eval_metric': 'mlogloss',
        'n_estimators': 150,
        'learning_rate': 0.1,
        'max_depth': 5,
        'use_label_encoder': False, # Suppress warning, not actually using it for direct integer labels
        'random_state': 42
    }
    params.update(kwargs)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train_mapped)
    print("XGBoostClassifier training complete.")
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates the trained XGBoost model and prints a classification report.
    Handles the target mapping for evaluation.

    Args:
        model: Trained XGBoost model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target (original -1, 0, 1).
        model_name (str): Name of the model for reporting.
    """
    print(f"\nEvaluating {model_name}...")
    
    # Map y_test for evaluation to match model's expected labels
    y_test_mapped = y_test.map({-1: 0, 0: 1, 1: 2})
    y_pred_mapped = model.predict(X_test)
    
    # Map predictions back to original labels (-1, 0, 1) for report clarity
    # Ensure all mapped classes (0,1,2) are present in model.classes_
    class_mapping_back = {0: -1, 1: 0, 2: 1}
    y_pred = pd.Series(y_pred_mapped).map(class_mapping_back).values # Convert to values for report

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    # Ensure all original labels are accounted for in target_names
    target_names = [str(x) for x in sorted(y_test.unique().tolist())] # Sort existing unique labels
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])) # Explicitly define labels order

def save_model(model, ticker, model_type="xgb", production=False):
    """
    Saves the trained model.

    Args:
        model: Trained model object.
        ticker (str): Stock ticker symbol.
        model_type (str): Type of model (e.g., 'rf', 'xgb').
        production (bool): If True, saves to production directory, else to archived.
    """
    model_dir = PRODUCTION_MODELS_DIR if production else ARCHIVED_MODELS_DIR
    os.makedirs(model_dir, exist_ok=True)
    
    filename = f"{ticker}_{model_type}_model.joblib"
    filepath = os.path.join(model_dir, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")
    return filepath

def load_model(ticker, model_type="xgb", production=False):
    """
    Loads a trained model.

    Args:
        ticker (str): Stock ticker symbol.
        model_type (str): Type of model.
        production (bool): If True, loads from production directory, else from archived.

    Returns:
        Trained model object.
    """
    model_dir = PRODUCTION_MODELS_DIR if production else ARCHIVED_MODELS_DIR
    filename = f"{ticker}_{model_type}_model.joblib"
    filepath = os.path.join(model_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Error: Model not found at {filepath}")
        return None
        
    print(f"Loading model from: {filepath}")
    model = joblib.load(filepath)
    return model

if __name__ == "__main__":
    from src.data_processing.data_preparation import prepare_data_for_ml

    # Example usage:
    ticker = 'MSFT' # Or loop through tickers

    X_train, X_test, y_train, y_test, scaler = prepare_data_for_ml(ticker)

    if X_train is not None:
        # Train the model
        xgb_model = train_xgboost_model(X_train, y_train, n_estimators=200, learning_rate=0.05)
        
        # Evaluate the model
        evaluate_model(xgb_model, X_test, y_test, model_name=f"XGBoost for {ticker}")
        
        # Save the model
        save_model(xgb_model, ticker, model_type="xgb", production=False)
        
        # Load the model back (example of loading)
        loaded_xgb_model = load_model(ticker, model_type="xgb", production=False)
        if loaded_xgb_model:
            print(f"Model loaded successfully. Type: {type(loaded_xgb_model)}")