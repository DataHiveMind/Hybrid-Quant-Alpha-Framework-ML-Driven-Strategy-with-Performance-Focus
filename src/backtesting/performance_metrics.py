# src/backtesting/performance_metrics.py

import pandas as pd
import numpy as np

def calculate_performance_metrics(equity_curve, risk_free_rate=0.02):
    """
    Calculates various performance metrics for an equity curve.

    Args:
        equity_curve (pd.Series): A time series of portfolio values (indexed by Date).
        risk_free_rate (float): Annual risk-free rate (e.g., 0.02 for 2%).

    Returns:
        dict: A dictionary of performance metrics.
    """
    if equity_curve.empty:
        return {
            'CAGR': 0.0, 'Sharpe Ratio': 0.0, 'Sortino Ratio': 0.0,
            'Max Drawdown': 0.0, 'Max Drawdown Duration': 0, 'Total Return': 0.0,
            'Volatility': 0.0, 'Number of Trades': 0 # We'll need trade count from backtester
        }

    # Calculate daily returns
    returns = equity_curve.pct_change().dropna()

    if returns.empty:
        return {
            'CAGR': 0.0, 'Sharpe Ratio': 0.0, 'Sortino Ratio': 0.0,
            'Max Drawdown': 0.0, 'Max Drawdown Duration': 0, 'Total Return': 0.0,
            'Volatility': 0.0, 'Number of Trades': 0
        }

    # Total Return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # CAGR (Compound Annual Growth Rate)
    # Assumes daily data, 252 trading days per year
    num_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0])**(1/num_years) - 1 if num_years > 0 else 0

    # Volatility (Annualized Standard Deviation of Returns)
    volatility = returns.std() * np.sqrt(252)

    # Sharpe Ratio (Annualized)
    # Assumes risk_free_rate is annual, converting to daily
    daily_risk_free_rate = (1 + risk_free_rate)**(1/252) - 1
    excess_returns = returns - daily_risk_free_rate
    sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252)

    # Sortino Ratio (Annualized)
    # Downside deviation
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() if not downside_returns.empty else 0
    sortino_ratio = (excess_returns.mean() / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0

    # Max Drawdown
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()

    # Max Drawdown Duration
    # Calculate duration of each drawdown period
    # This is a bit more complex. A simpler approximation is:
    # Find the longest period where drawdown is below 0.
    is_in_drawdown = (drawdown < 0)
    if is_in_drawdown.any():
        # Find consecutive 'True' values (in drawdown)
        groups = (is_in_drawdown != is_in_drawdown.shift()).cumsum()
        drawdown_durations = is_in_drawdown.groupby(groups).sum()
        max_drawdown_duration = drawdown_durations.max()
    else:
        max_drawdown_duration = 0 # No drawdown periods

    metrics = {
        'Total Return': total_return,
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Max Drawdown Duration (Days)': max_drawdown_duration
    }
    
    return metrics

if __name__ == "__main__":
    # Example usage (dummy equity curve)
    dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
    dummy_equity = pd.Series(np.random.randn(252).cumsum() + 1000, index=dates)
    
    # Simulate a drawdown
    dummy_equity.loc['2020-03-01':'2020-04-01'] -= 20
    
    metrics = calculate_performance_metrics(dummy_equity)
    print("Dummy Equity Curve Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")