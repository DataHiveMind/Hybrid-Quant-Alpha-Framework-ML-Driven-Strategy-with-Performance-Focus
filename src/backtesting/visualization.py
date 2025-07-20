# src/backtesting/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils.config import BACKTEST_RESULTS_DIR

def plot_equity_curve(equity_curve, ticker, save_path=None):
    """
    Plots the equity curve of the backtest.

    Args:
        equity_curve (pd.Series): Time series of portfolio values.
        ticker (str): Stock ticker symbol.
        save_path (str, optional): Directory to save the plot. Defaults to BACKTEST_RESULTS_DIR.
    """
    plt.figure(figsize=(12, 6))
    equity_curve.plot(title=f'{ticker} Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        filename = f'{ticker}_equity_curve.png'
        filepath = os.path.join(save_path, ticker, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True) # Ensure directory exists
        plt.savefig(filepath)
        print(f"Equity curve plot saved to {filepath}")
    plt.show()

def plot_drawdown(equity_curve, ticker, save_path=None):
    """
    Plots the drawdown from peak equity.

    Args:
        equity_curve (pd.Series): Time series of portfolio values.
        ticker (str): Stock ticker symbol.
        save_path (str, optional): Directory to save the plot. Defaults to BACKTEST_RESULTS_DIR.
    """
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    
    plt.figure(figsize=(12, 6))
    drawdown.plot(title=f'{ticker} Drawdown', color='red')
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        filename = f'{ticker}_drawdown.png'
        filepath = os.path.join(save_path, ticker, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True) # Ensure directory exists
        plt.savefig(filepath)
        print(f"Drawdown plot saved to {filepath}")
    plt.show()

if __name__ == "__main__":
    # Example usage (dummy equity curve)
    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    dummy_equity = pd.Series(np.random.randn(252).cumsum() + 1000, index=dates)
    dummy_equity.loc['2020-03-01':'2020-04-01'] -= 50 # Simulate a large drawdown
    
    plot_equity_curve(dummy_equity, "DUMMY")
    plot_drawdown(dummy_equity, "DUMMY")