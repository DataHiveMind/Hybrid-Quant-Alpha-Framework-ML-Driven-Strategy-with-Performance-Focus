# src/alpha_strategies/risk_management.py

def calculate_position_size(current_capital, trade_value_usd, current_price):
    """
    Calculates the number of shares to trade based on desired USD trade value.

    Args:
        current_capital (float): Current available capital.
        trade_value_usd (float): Desired USD value for the trade.
        current_price (float): Current price per share of the asset.

    Returns:
        int: Number of shares to trade (rounded down).
    """
    if current_price <= 0:
        return 0
    
    # Ensure we don't try to trade more than available capital
    # Simplified: For now, we assume current_capital is large enough.
    # In a real system, you'd check if trade_value_usd > current_capital and adjust.
    
    shares = trade_value_usd / current_price
    return int(shares) # Trade whole shares

# This module can be expanded for more complex risk rules later:
# - Stop loss/Take profit levels
# - Volatility-based position sizing
# - Portfolio-level risk management