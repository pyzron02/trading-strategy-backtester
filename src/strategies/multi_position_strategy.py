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
    """
    
    params = (
        ('sma_period', 20),   # Period for the Simple Moving Average
        ('position_size', 100), # Default position size
    )

    def __init__(self):
        """Initialize the strategy with indicators for each data feed."""
        # Create dictionaries to access close prices and indicators by ticker name
        self.data_close = {data._name: data.close for data in self.datas}
        self.sma = {data._name: bt.indicators.SimpleMovingAverage(data.close, period=self.params.sma_period) 
                   for data in self.datas}
        
        # Add a list to track equity curve
        self.equity_curve = []
        
        # Track trades for performance analysis
        self.trades = []

    def next(self):
        """Execute trading logic on each bar."""
        # Log portfolio value at each step
        portfolio_value = self.broker.getvalue()
        dt = self.datas[0].datetime.date(0)
        self.equity_curve.append({'Date': dt, 'Value': portfolio_value})
        self.log(f"Portfolio Value: {portfolio_value:.2f}")

        # Iterate over each data feed (i.e., each ticker)
        for data in self.datas:
            ticker = data._name
            current_close = self.data_close[ticker][0]
            sma_value = self.sma[ticker][0]
            position_size = self.getposition(data).size

            # Buy logic: if close > SMA and no position exists
            if current_close > sma_value and position_size == 0:
                size = self.params.position_size
                self.buy(data=data, size=size)
                self.log(f"BUY SIGNAL: {ticker} at {current_close:.2f} (SMA: {sma_value:.2f})")
                
                # Track trade
                self.trades.append({
                    'type': 'open',
                    'ticker': ticker,
                    'date': dt,
                    'price': current_close,
                    'size': size
                })

            # Sell logic: if close < SMA and a position exists
            elif current_close < sma_value and position_size > 0:
                self.sell(data=data, size=position_size)
                self.log(f"SELL SIGNAL: {ticker} at {current_close:.2f} (SMA: {sma_value:.2f})")
                
                # Calculate P&L
                buy_price = next((t['price'] for t in reversed(self.trades) 
                                 if t['ticker'] == ticker and t['type'] == 'open'), None)
                if buy_price:
                    pnl = (current_close - buy_price) * position_size
                else:
                    pnl = 0.0
                
                # Track trade
                self.trades.append({
                    'type': 'close',
                    'ticker': ticker,
                    'date': dt,
                    'price': current_close,
                    'size': position_size,
                    'pnl': pnl
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
        print(f"Number of Trades: {len([t for t in self.trades if t['type'] == 'close'])}")
        
        # Calculate total P&L
        total_pnl = sum([t.get('pnl', 0) for t in self.trades if t['type'] == 'close'])
        print(f"Total P&L: {total_pnl:.2f}")
        
        # Save equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        print(f"Equity curve data points: {len(equity_df)}")