# src/utils/config.py

import os

# Base project directory (set dynamically)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data paths
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, 'features')
LABELS_DIR = os.path.join(PROCESSED_DATA_DIR, 'labels')
BACKTEST_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'backtest_data')

# Model paths
MODELS_TRAINED_DIR = os.path.join(PROJECT_ROOT, 'models_trained')
PRODUCTION_MODELS_DIR = os.path.join(MODELS_TRAINED_DIR, 'production')
ARCHIVED_MODELS_DIR = os.path.join(MODELS_TRAINED_DIR, 'archived')

# Reports paths
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
BACKTEST_RESULTS_DIR = os.path.join(REPORTS_DIR, 'backtest_results')

# Other configurations
# Example: API keys (store securely, e.g., using environment variables in production)
# ALPHAVANTAGE_API_KEY = os.environ.get('ALPHAVANTAGE_API_KEY', 'YOUR_DEFAULT_OR_TEST_KEY')

# Create directories if they don't exist
for path in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, LABELS_DIR, BACKTEST_DATA_DIR,
    MODELS_TRAINED_DIR, PRODUCTION_MODELS_DIR, ARCHIVED_MODELS_DIR,
    REPORTS_DIR, BACKTEST_RESULTS_DIR
]:
    os.makedirs(path, exist_ok=True)