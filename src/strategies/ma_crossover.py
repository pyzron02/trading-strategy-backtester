#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moving Average Crossover Strategy.

This strategy goes long when fast MA crosses above slow MA and exits when fast MA crosses below slow MA.
"""

import backtrader as bt
import numpy as np

class MACrossover(bt.Strategy):
    """
    A Moving Average Crossover Strategy.
    
    This strategy uses two moving averages:
    - When the fast MA crosses above the slow MA, it enters a long position
    - When the fast MA crosses below the slow MA, it exits the position
    """
    
    params = (
        ('fast_period', 5),      # Fast moving average period
        ('slow_period', 20),     # Slow moving average period
        ('position_size', 100),  # Size of the position to take
        ('entry_threshold', 0.0), # Optional threshold for entries (crossover strength)
        ('exit_threshold', 0.0),  # Optional threshold for exits (crossover strength)
    )

    def __init__(self):
        """Initialize the strategy."""
        # Calculate warmup period
        self.warmup_period = max(int(self.params.slow_period) * 2, 50)
        
        # Create dictionaries of indicators for each data feed
        self.fast_ma = {}
        self.slow_ma = {}
        self.crossovers = {}
        
        for data in self.datas:
            # Use standard SMA indicators
            self.fast_ma[data] = bt.indicators.SMA(
                data.close, period=int(self.params.fast_period)
            )
            self.slow_ma[data] = bt.indicators.SMA(
                data.close, period=int(self.params.slow_period)
            )
            
            # Use standard CrossOver indicator
            self.crossovers[data] = bt.indicators.CrossOver(
                self.fast_ma[data], 
                self.slow_ma[data]
            )
        
        # Keep track of the equity curve for analysis
        self.equity_curve = []
        
        # Keep track of trades for analysis
        self.trades = []
        
        # Variables to track if we have enough data for trading
        self.min_bars_required = self.warmup_period
        self.bars_processed = 0

    def next(self):
        """Define the trading logic executed on each bar."""
        # Increment bars processed counter
        self.bars_processed += 1
        
        # Log portfolio value for the equity curve
        date = self.data.datetime.date(0).isoformat()
        value = self.broker.getvalue()
        self.equity_curve.append({'Date': date, 'Value': value})
        
        # Skip trading until we have enough bars for reliable indicator values
        if self.bars_processed < self.min_bars_required:
            return
        
        # Iterate over each data feed (ticker)
        for data in self.datas:
            # Skip if not enough data
            if len(data) < self.warmup_period:
                continue
                
            # Get current position size for this ticker
            position = self.getposition(data).size
            
            # Check crossover value
            crossover = self.crossovers[data][0]
            
            # Calculate crossover strength (percentage difference between MAs)
            fast_ma = self.fast_ma[data][0]
            slow_ma = self.slow_ma[data][0]
            
            # Skip if values are not yet available
            if fast_ma == 0 or slow_ma == 0:
                continue
                
            # Calculate strength as percentage difference
            crossover_strength = abs(fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0
            
            # Buy signal: crossover is positive and strength exceeds threshold
            if crossover > 0 and crossover_strength >= self.params.entry_threshold and position == 0:
                self.buy(data=data, size=self.params.position_size)
                
                # Track trade
                self.trades.append({
                    'type': 'open',
                    'ticker': data._name,
                    'date': date,
                    'price': data.close[0],
                    'size': self.params.position_size,
                    'signal': 'crossover_buy',
                    'strength': crossover_strength
                })
            
            # Sell signal: crossover is negative and strength exceeds threshold
            elif crossover < 0 and crossover_strength >= self.params.exit_threshold and position > 0:
                self.sell(data=data, size=position)
                
                # Calculate P&L
                buy_price = next((t['price'] for t in reversed(self.trades) 
                                 if t['ticker'] == data._name and t['type'] == 'open'), None)
                if buy_price:
                    pnl = (data.close[0] - buy_price) * position
                else:
                    pnl = 0.0
                
                # Track trade
                self.trades.append({
                    'type': 'close',
                    'ticker': data._name,
                    'date': date,
                    'price': data.close[0],
                    'size': position,
                    'pnl': pnl,
                    'signal': 'crossover_sell',
                    'strength': crossover_strength
                }) 