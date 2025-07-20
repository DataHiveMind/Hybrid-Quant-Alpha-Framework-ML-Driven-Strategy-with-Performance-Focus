# src/backtesting/backtester.py

import pandas as pd
import numpy as np
import os
from src.utils.config import BACKTEST_DATA_DIR, BACKTEST_RESULTS_DIR
from src.alpha_strategies.strategy_logic import generate_signals, load_strategy_config
from src.alpha_strategies.risk_management import calculate_position_size
from src.backtesting.performance_metrics import calculate_performance_metrics
from src.backtesting.visualization import plot_equity_curve, plot_drawdown

class VectorizedBacktester:
    """
    A vectorized backtesting engine for trading strategies.
    """
    def __init__(self, ticker, strategy_config, production_model=False):
        self.ticker = ticker
        self.strategy_config = strategy_config
        self.production_model = production_model
        self.initial_capital = strategy_config.get('initial_capital', 100000.0)
        self.position_size_usd = strategy_config.get('position_size_per_trade_usd', 10000)
        self.commission_per_share = strategy_config.get('commission_per_share', 0.005)
        self.slippage_bps = strategy_config.get('slippage_bps', 0.01) / 10000.0 # Convert bps to decimal

        self.data = None
        self.equity_curve = None
        self.trades = None

        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """
        Generates signals and prepares data for backtesting.
        """
        signals_df = generate_signals(self.ticker, self.strategy_config, self.production_model)
        if signals_df is None:
            raise ValueError(f"Could not generate signals for {self.ticker}. Backtest cannot proceed.")
        
        self.data = signals_df.copy()
        # Ensure 'Close' is present for price
        if 'Close' not in self.data.columns:
            raise ValueError("Dataframe must contain 'Close' price column.")
        
        # Initialize columns for backtesting
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Position'] = 0 # 1 for long, -1 for short, 0 for cash
        self.data['Trades'] = 0 # 1 for buy, -1 for sell
        self.data['Shares'] = 0 # Number of shares held
        self.data['Trade_Shares'] = 0 # Number of shares traded at each step
        self.data['Cash'] = 0.0
        self.data['Portfolio_Value'] = 0.0
        self.data['Commission_Cost'] = 0.0
        self.data['Slippage_Cost'] = 0.0

        self.data['Cash'].iloc[0] = self.initial_capital
        self.data['Portfolio_Value'].iloc[0] = self.initial_capital

    def run_backtest(self):
        """
        Executes the vectorized backtest.
        """
        print(f"\nRunning backtest for {self.ticker}...")
        
        for i in range(1, len(self.data)):
            prev_row = self.data.iloc[i-1]
            current_row = self.data.iloc[i]

            current_cash = prev_row['Cash']
            current_shares = prev_row['Shares']
            current_position = prev_row['Position']
            
            signal = current_row['Signal'] # This is the ML-driven signal: 1, 0, -1
            current_price = current_row['Close']
            
            trade_shares = 0
            commission_cost = 0
            slippage_cost = 0
            
            # Simplified: Assume trade execution at current_price
            # In real backtester, you'd typically execute at next day's open or some other realistic fill price
            
            # Handle closing existing positions first if signal changes direction
            if current_position == 1 and signal == -1: # Was long, now signal to sell/short
                # Sell all long shares
                trade_shares = current_shares # Sell existing shares
                trade_value = trade_shares * current_price
                commission_cost += trade_shares * self.commission_per_share
                slippage_cost += trade_value * self.slippage_bps
                current_cash += trade_value - commission_cost - slippage_cost
                current_shares = 0
                current_position = 0 # Now cash/flat
                self.data.loc[self.data.index[i], 'Trades'] = -1 # Record as a sell trade for closing
                self.data.loc[self.data.index[i], 'Trade_Shares'] = -trade_shares
                
            elif current_position == -1 and signal == 1: # Was short, now signal to buy/long
                # Buy to cover all short shares
                # This backtester does not explicitly handle shorting mechanics (e.g. margin, borrow fees)
                # It simply treats -1 position as if shares were sold and need to be bought back.
                # To simplify for vectorized, we assume it's like buying to cover.
                trade_shares = abs(current_shares) # Buy back existing short shares
                trade_value = trade_shares * current_price
                commission_cost += trade_shares * self.commission_per_share
                slippage_cost += trade_value * self.slippage_bps
                current_cash -= trade_value + commission_cost + slippage_cost # Buying costs cash
                current_shares = 0
                current_position = 0 # Now cash/flat
                self.data.loc[self.data.index[i], 'Trades'] = 1 # Record as a buy trade for closing
                self.data.loc[self.data.index[i], 'Trade_Shares'] = trade_shares

            # Execute new signal
            if signal == 1 and current_position == 0: # Buy signal and currently flat
                shares_to_buy = calculate_position_size(current_cash, self.position_size_usd, current_price)
                cost_of_trade = shares_to_buy * current_price
                if current_cash >= cost_of_trade: # Check if enough cash
                    trade_shares += shares_to_buy
                    commission_cost += shares_to_buy * self.commission_per_share
                    slippage_cost += cost_of_trade * self.slippage_bps
                    current_cash -= (cost_of_trade + commission_cost + slippage_cost)
                    current_shares += shares_to_buy
                    current_position = 1
                    self.data.loc[self.data.index[i], 'Trades'] = 1 # Record as a buy trade
                    self.data.loc[self.data.index[i], 'Trade_Shares'] = trade_shares
                else:
                    self.data.loc[self.data.index[i], 'Signal'] = 0 # Not enough cash, revert signal to hold

            elif signal == -1 and current_position == 0: # Sell/Short signal and currently flat
                # This backtester does not implement actual shorting.
                # For simplicity, if we get a SELL signal and are flat, we will go CASH (do nothing).
                # To implement shorting, you'd need to add logic for borrowing, margin, etc.
                # For now, treat -1 as 'sell if long', otherwise 'hold'.
                # To simulate a 'short' position, one might set shares to a negative number and
                # adjust P&L accordingly. For this project's scope, we'll keep it simple:
                # no new short positions from flat.
                self.data.loc[self.data.index[i], 'Signal'] = 0 # Change signal to hold if trying to short from flat
                pass # Do nothing (effectively a hold)

            # Update portfolio state
            self.data.loc[self.data.index[i], 'Cash'] = current_cash
            self.data.loc[self.data.index[i], 'Shares'] = current_shares
            self.data.loc[self.data.index[i], 'Position'] = current_position
            self.data.loc[self.data.index[i], 'Commission_Cost'] = commission_cost
            self.data.loc[self.data.index[i], 'Slippage_Cost'] = slippage_cost
            
            # Calculate portfolio value (Cash + Value of Held Shares)
            self.data.loc[self.data.index[i], 'Portfolio_Value'] = current_cash + (current_shares * current_price)

        self.equity_curve = self.data['Portfolio_Value']
        self._record_trades()
        print(f"Backtest for {self.ticker} complete.")
        return self.equity_curve

    def _record_trades(self):
        """Records executed trades."""
        trades_df = self.data[self.data['Trades'] != 0].copy()
        if not trades_df.empty:
            trades_df['Trade_Type'] = trades_df['Trades'].apply(lambda x: 'BUY' if x == 1 else 'SELL')
            trades_df['Trade_Value'] = abs(trades_df['Trade_Shares'] * trades_df['Close'])
            trades_df['Total_Cost'] = trades_df['Commission_Cost'] + trades_df['Slippage_Cost']
            self.trades = trades_df[['Close', 'Trade_Shares', 'Trade_Type', 'Trade_Value', 'Total_Cost', 'ML_Prediction', 'Signal']].copy()
        else:
            self.trades = pd.DataFrame(columns=['Close', 'Trade_Shares', 'Trade_Type', 'Trade_Value', 'Total_Cost', 'ML_Prediction', 'Signal'])

    def analyze_results(self):
        """
        Analyzes backtest results, calculates performance metrics, and generates visualizations.
        """
        if self.equity_curve is None:
            print("Please run the backtest first using run_backtest().")
            return None, None

        print("\nAnalyzing backtest results...")
        metrics = calculate_performance_metrics(self.equity_curve)
        
        # Save results
        os.makedirs(os.path.join(BACKTEST_RESULTS_DIR, self.ticker), exist_ok=True)
        equity_filepath = os.path.join(BACKTEST_RESULTS_DIR, self.ticker, f'{self.ticker}_equity_curve.csv')
        metrics_filepath = os.path.join(BACKTEST_RESULTS_DIR, self.ticker, f'{self.ticker}_metrics.txt')
        trades_filepath = os.path.join(BACKTEST_RESULTS_DIR, self.ticker, f'{self.ticker}_trades.csv')

        self.equity_curve.to_csv(equity_filepath, header=True, index=True)
        print(f"Equity curve saved to {equity_filepath}")

        with open(metrics_filepath, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        print(f"Performance metrics saved to {metrics_filepath}")

        if not self.trades.empty:
            self.trades.to_csv(trades_filepath, index=True)
            print(f"Trade log saved to {trades_filepath}")
        else:
            print("No trades executed during backtest.")

        # Generate visualizations
        plot_equity_curve(self.equity_curve, self.ticker)
        plot_drawdown(self.equity_curve, self.ticker)
        
        return metrics, self.trades

if __name__ == "__main__":
    ticker_to_backtest = 'AAPL'
    
    # Load strategy configuration
    strategy_config = load_strategy_config("default_strategy.yaml")
    
    if strategy_config:
        backtester = VectorizedBacktester(ticker_to_backtest, strategy_config, production_model=False)
        equity_curve = backtester.run_backtest()
        
        if equity_curve is not None and not equity_curve.empty:
            metrics, trades = backtester.analyze_results()
            print("\n--- Backtest Summary ---")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            if trades is not None:
                print(f"\nTotal Trades: {len(trades)}")
                if not trades.empty:
                    print(f"Winning Trades: {len(trades[trades['Trade_Type']=='BUY'])}") # This is simplistic, needs proper P&L per trade
                    print(f"Losing Trades: {len(trades[trades['Trade_Type']=='SELL'])}") # This is simplistic