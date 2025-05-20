#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairs Trading Strategy Implementation.

This strategy identifies cointegrated pairs of stocks and trades based on their spread.
When the spread diverges beyond a threshold, it takes opposite positions in the two stocks,
expecting the spread to revert to its mean.
"""

import backtrader as bt
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS


class PairsTradingStrategy(bt.Strategy):
    """
    A pairs trading strategy that identifies cointegrated pairs of stocks.
    
    The strategy calculates the spread between two assets, normalizes it, and trades
    when the spread exceeds a certain threshold, taking opposing positions in each asset.
    
    Parameters:
        lookback_period (int): Period for calculating spread statistics
        entry_threshold (float): Z-score threshold to trigger trade entry
        exit_threshold (float): Z-score threshold to trigger trade exit
        position_size (int): Base number of shares to trade
        rebalance_freq (int): Days between hedge ratio recalculations
        stop_loss (float): Stop loss percentage
    """
    
    params = (
        ('lookback_period', 60),    # Days to calculate spread statistics
        ('entry_threshold', 2.0),   # Z-score threshold to enter
        ('exit_threshold', 0.5),    # Z-score threshold to exit
        ('position_size', 100),     # Base position size
        ('rebalance_freq', 20),     # Days between hedge ratio recalculations
        ('stop_loss', 0.05),        # Stop loss percentage
    )

    def __init__(self):
        """Initialize the strategy with indicators and variables."""
        # Convert parameters to appropriate types, handling both single values and lists
        # This handles the case where parameters are passed as lists from parameter_grid during optimization
        self.p.lookback_period = int(self.p.lookback_period[0] if isinstance(self.p.lookback_period, list) else self.p.lookback_period)
        self.p.position_size = int(self.p.position_size[0] if isinstance(self.p.position_size, list) else self.p.position_size)
        self.p.rebalance_freq = int(self.p.rebalance_freq[0] if isinstance(self.p.rebalance_freq, list) else self.p.rebalance_freq)
        
        # Handle float parameters that could be passed as lists
        self.p.entry_threshold = float(self.p.entry_threshold[0] if isinstance(self.p.entry_threshold, list) else self.p.entry_threshold)
        self.p.exit_threshold = float(self.p.exit_threshold[0] if isinstance(self.p.exit_threshold, list) else self.p.exit_threshold)
        self.p.stop_loss = float(self.p.stop_loss[0] if isinstance(self.p.stop_loss, list) else self.p.stop_loss)
        
        # Verify we have exactly two data feeds for pairs trading
        if len(self.datas) != 2:
            raise ValueError("PairsTradingStrategy requires exactly two data feeds")
            
        # Set up asset references for clarity
        self.asset1 = self.datas[0]
        self.asset2 = self.datas[1]
        self.asset1_name = self.asset1._name
        self.asset2_name = self.asset2._name
        
        # Initialize hedge ratio and beta
        self.hedge_ratio = 1.0
        self.last_recalc_day = 0
        
        # Track trade metrics
        self.active_trades = {}
        self.trades_history = []
        self.equity_curve = []
        self.cumulative_pnl = 0.0
        
        # Arrays to store price data for statistical calculations
        self.asset1_prices = []
        self.asset2_prices = []
        
        # Track spread values for z-score calculation
        self.spread_values = []
        self.spread_mean = 0
        self.spread_std = 0.0001  # Initialize to small value to avoid division by zero
        
        # Log initialization
        self.log(f"Strategy initialized with parameters: "
                 f"lookback_period={self.p.lookback_period}, "
                 f"entry_threshold={self.p.entry_threshold}, "
                 f"exit_threshold={self.p.exit_threshold}, "
                 f"position_size={self.p.position_size}, "
                 f"rebalance_freq={self.p.rebalance_freq}, "
                 f"stop_loss={self.p.stop_loss}")

    def calculate_hedge_ratio(self):
        """Calculate the hedge ratio between the two assets using OLS regression."""
        if len(self.asset1_prices) < self.p.lookback_period:
            self.log("Not enough data to calculate hedge ratio")
            return
            
        # Extract the most recent lookback_period prices
        y = self.asset1_prices[-self.p.lookback_period:]
        X = self.asset2_prices[-self.p.lookback_period:]
        
        # Add constant to X for regression
        X = np.vstack([np.ones(len(X)), X]).T
        
        # Perform OLS regression to get the hedge ratio
        try:
            model = OLS(y, X)
            results = model.fit()
            beta = results.params[1]
            
            # Update hedge ratio
            old_ratio = self.hedge_ratio
            self.hedge_ratio = abs(beta)
            
            self.log(f"Recalculated hedge ratio: {old_ratio:.4f} -> {self.hedge_ratio:.4f}")
            self.last_recalc_day = len(self)
            
            # Recalculate spread statistics with new hedge ratio
            self.update_spread_statistics()
            
        except Exception as e:
            self.log(f"Error calculating hedge ratio: {str(e)}")

    def update_spread_statistics(self):
        """Update the mean and standard deviation of the spread."""
        # Calculate spreads using current hedge ratio
        spreads = []
        lookback = min(self.p.lookback_period, len(self.asset1_prices))
        
        for i in range(lookback):
            idx = -lookback + i
            spread = self.asset1_prices[idx] - (self.hedge_ratio * self.asset2_prices[idx])
            spreads.append(spread)
        
        # Calculate statistics
        if len(spreads) > 0:
            self.spread_mean = np.mean(spreads)
            self.spread_std = max(np.std(spreads), 0.0001)  # Ensure non-zero std
            self.log(f"Updated spread statistics: mean={self.spread_mean:.4f}, std={self.spread_std:.4f}")

    def calculate_z_score(self, spread):
        """Calculate z-score for the current spread."""
        return (spread - self.spread_mean) / self.spread_std

    def next(self):
        """Execute trading logic on each bar."""
        # Log portfolio value
        portfolio_value = self.broker.getvalue()
        dt = self.datas[0].datetime.date(0)
        
        # Track equity curve
        self.equity_curve.append({
            'Date': dt,
            'Value': portfolio_value,
            'Cash': self.broker.get_cash(),
            'PnL': self.cumulative_pnl
        })
        
        # Store prices for statistical calculations
        self.asset1_prices.append(self.asset1.close[0])
        self.asset2_prices.append(self.asset2.close[0])
        
        # Wait until we have enough data
        if len(self.asset1_prices) < self.p.lookback_period:
            return
            
        # Recalculate hedge ratio periodically
        if (len(self) - self.last_recalc_day) >= self.p.rebalance_freq:
            self.calculate_hedge_ratio()
        
        # Calculate current spread and z-score
        current_spread = self.asset1.close[0] - (self.hedge_ratio * self.asset2.close[0])
        self.spread_values.append(current_spread)
        z_score = self.calculate_z_score(current_spread)
        
        # Log status periodically
        if len(self) % 10 == 0:
            self.log(f"Current z-score: {z_score:.2f}, Spread: {current_spread:.2f}, "
                     f"Hedge ratio: {self.hedge_ratio:.4f}, Portfolio: ${portfolio_value:.2f}")
        
        # Get current positions
        pos1 = self.getposition(self.asset1).size
        pos2 = self.getposition(self.asset2).size
        
        # Check for open pairs position
        in_market = (pos1 != 0 or pos2 != 0)
        
        # ENTRY logic: spread has diverged beyond threshold
        if not in_market and abs(z_score) > self.p.entry_threshold:
            # Determine trade direction based on z-score
            if z_score > 0:  # Spread is positive and above threshold
                # Sell asset1, buy asset2 (expecting spread to decrease)
                direction = -1
                self.log(f"ENTRY SIGNAL: Sell {self.asset1_name}, Buy {self.asset2_name}")
            else:  # Spread is negative and below threshold
                # Buy asset1, sell asset2 (expecting spread to increase)
                direction = 1
                self.log(f"ENTRY SIGNAL: Buy {self.asset1_name}, Sell {self.asset2_name}")
            
            # Calculate position sizes based on current prices and hedge ratio
            price1 = self.asset1.close[0]
            price2 = self.asset2.close[0]
            
            # Base position size for asset1
            size1 = self.p.position_size * direction
            
            # Calculate hedged position size for asset2
            # Adjust asset2 position to maintain dollar-neutral portfolio
            size2 = int(-(size1 * price1) / price2 * (1/self.hedge_ratio)) * direction
            
            # Check if we have enough cash
            cash_needed = (abs(size1) * price1 + abs(size2) * price2) * 1.05  # 5% buffer
            available_cash = self.broker.get_cash()
            
            if available_cash < cash_needed:
                # Adjust position sizes proportionally
                size_ratio = available_cash / cash_needed * 0.95  # 5% safety margin
                size1 = int(size1 * size_ratio)
                size2 = int(size2 * size_ratio)
                self.log(f"Adjusted position sizes due to cash constraints: {size1}, {size2}")
            
            # Execute trades if sizes are non-zero
            if size1 != 0 and size2 != 0:
                self.buy(data=self.asset1, size=size1) if size1 > 0 else self.sell(data=self.asset1, size=abs(size1))
                self.buy(data=self.asset2, size=size2) if size2 > 0 else self.sell(data=self.asset2, size=abs(size2))
                
                # Record entry details
                self.active_trades = {
                    'entry_date': dt,
                    'entry_z': z_score,
                    'entry_spread': current_spread,
                    'asset1': {
                        'name': self.asset1_name,
                        'price': price1,
                        'size': size1,
                        'direction': 'BUY' if size1 > 0 else 'SELL'
                    },
                    'asset2': {
                        'name': self.asset2_name,
                        'price': price2,
                        'size': size2,
                        'direction': 'BUY' if size2 > 0 else 'SELL'
                    }
                }
                
        # EXIT logic: spread has reverted to mean or stop loss hit
        elif in_market:
            # Calculate unrealized PnL for the pair
            if self.active_trades:
                price1 = self.asset1.close[0]
                price2 = self.asset2.close[0]
                entry1 = self.active_trades['asset1']['price']
                entry2 = self.active_trades['asset2']['price']
                size1 = self.active_trades['asset1']['size']
                size2 = self.active_trades['asset2']['size']
                
                pnl1 = (price1 - entry1) * size1
                pnl2 = (price2 - entry2) * size2
                unrealized_pnl = pnl1 + pnl2
                
                # Calculate percentage loss for stop loss check
                entry_value = abs(entry1 * size1) + abs(entry2 * size2)
                pct_loss = -unrealized_pnl / entry_value if entry_value > 0 else 0
                
                # Exit on profit target, mean reversion, or stop loss
                if (abs(z_score) < self.p.exit_threshold) or (pct_loss > self.p.stop_loss):
                    exit_reason = "Mean reversion" if abs(z_score) < self.p.exit_threshold else "Stop loss"
                    self.log(f"EXIT SIGNAL: {exit_reason} - Close pair position")
                    
                    # Close all positions
                    self.close(self.asset1)
                    self.close(self.asset2)
                    
                    # Record trade details
                    if self.active_trades:
                        self.active_trades.update({
                            'exit_date': dt,
                            'exit_z': z_score,
                            'exit_spread': current_spread,
                            'pnl': unrealized_pnl,
                            'exit_reason': exit_reason
                        })
                        
                        self.trades_history.append(self.active_trades.copy())
                        self.cumulative_pnl += unrealized_pnl
                        self.active_trades = {}

    def log(self, txt):
        """Log messages with the current date."""
        dt = self.datas[0].datetime.date(0)
        print(f"{dt}: {txt}")

    def notify_order(self, order):
        """Handle order execution notifications."""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED for {order.data._name}: Price={order.executed.price:.2f}, "
                         f"Size={order.executed.size}, Value=${order.executed.value:.2f}, Comm=${order.executed.comm:.2f}")
            else:
                self.log(f"SELL EXECUTED for {order.data._name}: Price={order.executed.price:.2f}, "
                         f"Size={abs(order.executed.size)}, Value=${order.executed.value:.2f}, Comm=${order.executed.comm:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order for {order.data._name} {order.getstatusname()}: Size={order.size}")

    def notify_trade(self, trade):
        """Handle trade closure notifications."""
        if trade.isclosed:
            self.log(f"Trade closed for {trade.data._name}: P&L Gross=${trade.pnl:.2f}, Net=${trade.pnlcomm:.2f}")

    def stop(self):
        """Save the equity curve and performance metrics when the strategy ends."""
        print(f"\nPairs Trading Strategy Performance Summary:")
        print(f"Initial Portfolio Value: ${self.broker.startingcash:.2f}")
        print(f"Final Portfolio Value: ${self.broker.getvalue():.2f}")
        print(f"Total Return: {(self.broker.getvalue() / self.broker.startingcash - 1):.2%}")
        
        # Get trade statistics
        total_trades = len(self.trades_history)
        
        if total_trades > 0:
            winning_trades = sum(1 for t in self.trades_history if t.get('pnl', 0) > 0)
            losing_trades = total_trades - winning_trades
            
            win_rate = winning_trades / total_trades
            print(f"Number of Trades: {total_trades}")
            print(f"Winning Trades: {winning_trades} ({win_rate:.1%})")
            print(f"Losing Trades: {losing_trades} ({1-win_rate:.1%})")
            
            # Calculate P&L metrics
            total_pnl = sum(t.get('pnl', 0) for t in self.trades_history)
            
            if winning_trades > 0:
                gross_profit = sum(t.get('pnl', 0) for t in self.trades_history if t.get('pnl', 0) > 0)
                avg_win = gross_profit / winning_trades
                print(f"Average Winning Trade: ${avg_win:.2f}")
            
            if losing_trades > 0:
                gross_loss = sum(t.get('pnl', 0) for t in self.trades_history if t.get('pnl', 0) <= 0)
                avg_loss = gross_loss / losing_trades
                print(f"Average Losing Trade: ${avg_loss:.2f}")
                
                if gross_loss != 0:
                    profit_factor = abs(gross_profit / gross_loss) if winning_trades > 0 else 0
                    print(f"Profit Factor: {profit_factor:.2f}")
        
        # Save enhanced equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Calculate drawdown on the equity curve
        if not equity_df.empty and 'Value' in equity_df.columns:
            equity_df['peak'] = equity_df['Value'].cummax()
            equity_df['drawdown'] = (equity_df['peak'] - equity_df['Value']) / equity_df['peak']
            max_dd = equity_df['drawdown'].max()
            print(f"Maximum Drawdown: {max_dd:.2%}")
        
        print(f"Equity curve data points: {len(equity_df)}")