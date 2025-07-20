# src/models/traditional_ml/rf_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib # For saving/loading models
import os
from src.utils.config import ARCHIVED_MODELS_DIR, PRODUCTION_MODELS_DIR

def train_random_forest_model(X_train, y_train, **kwargs):
    """
    Trains a RandomForestClassifier model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        kwargs: Hyperparameters for RandomForestClassifier.

    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained model.
    """
    print("Training RandomForestClassifier model...")
    # Default parameters, can be overridden by kwargs
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'class_weight': 'balanced' # Important for imbalanced financial data
    }
    params.update(kwargs) # Update with any provided kwargs

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    print("RandomForestClassifier training complete.")
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates the trained model and prints a classification report.

    Args:
        model: Trained scikit-learn model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        model_name (str): Name of the model for reporting.
    """
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    # Ensure all possible labels are included in target_names
    # This is crucial if a class (e.g., -1 or 0) doesn't appear in y_test
    # For a 3-class target (-1, 0, 1)
    target_names = [str(x) for x in sorted(y_test.unique().tolist() + list(model.classes_))]
    target_names = sorted(list(set(target_names))) # Remove duplicates and sort
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=sorted(model.classes_)))

def save_model(model, ticker, model_type="rf", production=False):
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

def load_model(ticker, model_type="rf", production=False):
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
    ticker = 'AAPL' # Or loop through tickers

    X_train, X_test, y_train, y_test, scaler = prepare_data_for_ml(ticker)

    if X_train is not None:
        # Train the model
        rf_model = train_random_forest_model(X_train, y_train, n_estimators=150, max_depth=15)
        
        # Evaluate the model
        evaluate_model(rf_model, X_test, y_test, model_name=f"RandomForest for {ticker}")
        
        # Save the model
        save_model(rf_model, ticker, model_type="rf", production=False)
        
        # Load the model back (example of loading)
        loaded_rf_model = load_model(ticker, model_type="rf", production=False)
        if loaded_rf_model:
            print(f"Model loaded successfully. Type: {type(loaded_rf_model)}")