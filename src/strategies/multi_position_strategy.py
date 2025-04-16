#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Position Strategy Implementation.

This strategy trades multiple positions in a portfolio based on SMA signals.
"""

import backtrader as bt
import pandas as pd

class MultiPositionStrategy(bt.Strategy):
    """
    A strategy that can manage multiple positions across different assets.
    
    This strategy uses SMAs to generate buy/sell signals for each ticker
    and manages position sizes accordingly.
    
    Parameters:
        sma_period (int): Period for the Simple Moving Average
        position_size (int): Default number of shares to trade
        max_positions (int): Maximum number of positions to hold at once
    """
    
    params = (
        ('sma_period', 20),      # Period for the Simple Moving Average
        ('position_size', 100),  # Default position size
        ('max_positions', 3),    # Maximum number of positions to hold at once
    )

    def __init__(self):
        """Initialize the strategy with indicators for each data feed."""
        # Create dictionaries to access close prices and indicators by ticker name
        self.data_close = {data._name: data.close for data in self.datas}
        self.sma = {data._name: bt.indicators.SimpleMovingAverage(data.close, period=self.params.sma_period) 
                   for data in self.datas}
        
        # Track active positions with buy prices and entry dates
        self.active_positions = {}
        
        # Add a list to track equity curve
        self.equity_curve = []
        
        # Track trades for performance analysis
        self.trades = []
        
        # Track cumulative PnL
        self.cumulative_pnl = 0.0
        
        # Debug parameters
        self.log(f"Strategy initialized with parameters: sma_period={self.params.sma_period}, "
                 f"position_size={self.params.position_size}, max_positions={self.params.max_positions}")

    def next(self):
        """Execute trading logic on each bar."""
        # Log portfolio value at each step
        portfolio_value = self.broker.getvalue()
        dt = self.datas[0].datetime.date(0)
        self.equity_curve.append({
            'Date': dt, 
            'Value': portfolio_value, 
            'Cash': self.broker.get_cash(),
            'PnL': self.cumulative_pnl
        })
        
        # Count current active positions
        current_positions = sum(1 for data in self.datas if self.getposition(data).size > 0)
        
        # Log status periodically
        if len(self.equity_curve) % 20 == 0:
            self.log(f"Portfolio Value: {portfolio_value:.2f}, Active Positions: {current_positions}/{self.params.max_positions}")

        # Iterate over each data feed (i.e., each ticker)
        for data in self.datas:
            ticker = data._name
            current_close = self.data_close[ticker][0]
            sma_value = self.sma[ticker][0]
            position_size = self.getposition(data).size

            # Buy logic: if close > SMA, no position exists, and we're under max positions
            if current_close > sma_value and position_size == 0 and current_positions < self.params.max_positions:
                size = self.params.position_size
                # Make sure we're not buying more than we can afford
                cash = self.broker.get_cash()
                if cash < current_close * size:
                    # Adjust position size based on available cash
                    affordable_size = int(cash / current_close) - 1  # Subtract 1 for safety
                    if affordable_size <= 0:
                        self.log(f"Not enough cash to buy {ticker}. Available: ${cash:.2f}")
                        continue
                    size = affordable_size
                    self.log(f"Adjusted position size to {size} based on available cash")
                
                # Execute the buy order
                self.buy(data=data, size=size)
                self.log(f"BUY SIGNAL: {ticker} at {current_close:.2f} (SMA: {sma_value:.2f})")
                
                # Update position count
                current_positions += 1
                
                # Store position details in active_positions
                if ticker not in self.active_positions:
                    self.active_positions[ticker] = []
                self.active_positions[ticker].append({
                    'entry_date': dt,
                    'entry_price': current_close,
                    'size': size
                })
                
                # Add trade to our custom log
                self.trades.append({
                    'type': 'open',
                    'ticker': ticker,
                    'date': dt,
                    'price': current_close,
                    'size': size
                })

            # Sell logic: if close < SMA and a position exists
            elif current_close < sma_value and position_size > 0:
                # Execute the sell order
                self.sell(data=data, size=position_size)
                self.log(f"SELL SIGNAL: {ticker} at {current_close:.2f} (SMA: {sma_value:.2f})")
                
                # Update position count
                current_positions -= 1
                
                # Calculate P&L using active_positions tracking
                pnl = 0.0
                if ticker in self.active_positions and self.active_positions[ticker]:
                    # Get FIFO positions to close
                    positions_to_close = []
                    remaining_size = position_size
                    
                    while remaining_size > 0 and self.active_positions[ticker]:
                        position = self.active_positions[ticker][0]  # Get oldest position (FIFO)
                        
                        if position['size'] <= remaining_size:
                            # Close entire position
                            positions_to_close.append(position)
                            remaining_size -= position['size']
                            self.active_positions[ticker].pop(0)
                        else:
                            # Partially close position
                            partial_position = position.copy()
                            partial_position['size'] = remaining_size
                            positions_to_close.append(partial_position)
                            
                            # Update remaining position
                            position['size'] -= remaining_size
                            remaining_size = 0
                    
                    # Calculate PnL for each closed position
                    for position in positions_to_close:
                        pos_pnl = (current_close - position['entry_price']) * position['size']
                        commission = current_close * position['size'] * 0.001  # 0.1% commission
                        pos_pnl -= commission
                        pnl += pos_pnl
                    
                    # Clean up if dictionary is now empty
                    if not self.active_positions[ticker]:
                        del self.active_positions[ticker]
                
                else:
                    # Fallback to old method if position tracking failed
                    buy_price = next((t['price'] for t in reversed(self.trades) 
                                     if t['ticker'] == ticker and t['type'] == 'open'), None)
                    if buy_price:
                        pnl = (current_close - buy_price) * position_size
                        commission = current_close * position_size * 0.001  # 0.1% commission
                        pnl -= commission
                
                # Update cumulative PnL
                self.cumulative_pnl += pnl
                
                # Add trade to our custom log
                self.trades.append({
                    'type': 'close',
                    'ticker': ticker,
                    'date': dt,
                    'price': current_close,
                    'size': position_size,
                    'pnl': pnl,
                    'cumulative_pnl': self.cumulative_pnl
                })

    def log(self, txt):
        """Log messages with the current date."""
        dt = self.datas[0].datetime.date(0)
        print(f"{dt}: {txt}")

    def notify_order(self, order):
        """Handle order execution notifications."""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED for {order.data._name}, Price: {order.executed.price}, "
                         f"Cost: {order.executed.value}, Comm: {order.executed.comm}")
            elif order.issell():
                self.log(f"SELL EXECUTED for {order.data._name}, Price: {order.executed.price}, "
                         f"Cost: {order.executed.value}, Comm: {order.executed.comm}")

    def notify_trade(self, trade):
        """Handle trade closure notifications."""
        if trade.isclosed:
            self.log(f"Trade closed for {trade.data._name}: PnL Gross {trade.pnl}, Net {trade.pnlcomm}")

    def stop(self):
        """Save the equity curve when the strategy ends."""
        # Create summary of strategy performance
        print(f"\nStrategy Performance Summary:")
        print(f"Final Portfolio Value: {self.broker.getvalue():.2f}")
        
        # Get trade statistics
        total_trades = len([t for t in self.trades if t['type'] == 'close'])
        winning_trades = len([t for t in self.trades if t['type'] == 'close' and t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trades if t['type'] == 'close' and t.get('pnl', 0) <= 0])
        
        print(f"Number of Trades: {total_trades}")
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            print(f"Winning Trades: {winning_trades} ({win_rate:.1%})")
            print(f"Losing Trades: {losing_trades} ({1-win_rate:.1%})")
            
            # Calculate P&L metrics
            total_pnl = sum([t.get('pnl', 0) for t in self.trades if t['type'] == 'close'])
            gross_profit = sum([t.get('pnl', 0) for t in self.trades if t['type'] == 'close' and t.get('pnl', 0) > 0])
            gross_loss = sum([t.get('pnl', 0) for t in self.trades if t['type'] == 'close' and t.get('pnl', 0) <= 0])
            
            print(f"Total P&L: ${total_pnl:.2f}")
            print(f"Gross Profit: ${gross_profit:.2f}")
            print(f"Gross Loss: ${gross_loss:.2f}")
            
            if gross_loss != 0:
                profit_factor = abs(gross_profit / gross_loss)
                print(f"Profit Factor: {profit_factor:.2f}")
            
            if winning_trades > 0:
                avg_win = gross_profit / winning_trades
                print(f"Average Win: ${avg_win:.2f}")
            
            if losing_trades > 0:
                avg_loss = gross_loss / losing_trades
                print(f"Average Loss: ${avg_loss:.2f}")
        
        # Save enhanced equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Calculate drawdown on the enhanced equity curve
        if not equity_df.empty and 'Value' in equity_df.columns:
            equity_df['peak'] = equity_df['Value'].cummax()
            equity_df['drawdown'] = (equity_df['peak'] - equity_df['Value']) / equity_df['peak']
            max_dd = equity_df['drawdown'].max()
            print(f"Maximum Drawdown: {max_dd:.2%}")
        
        print(f"Equity curve data points: {len(equity_df)}")