#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moving Average Crossover Strategy.

This strategy goes long when fast MA crosses above slow MA and exits when fast MA crosses below slow MA.
"""

import backtrader as bt

class MACrossover(bt.Strategy):
    """
    A Moving Average Crossover Strategy.
    
    This strategy uses two moving averages:
    - When the fast MA crosses above the slow MA, it enters a long position
    - When the fast MA crosses below the slow MA, it exits the position
    """
    
    params = (
        ('fast_period', 10),     # Fast moving average period
        ('slow_period', 30),     # Slow moving average period
        ('position_size', 100),  # Size of the position to take
    )

    def __init__(self):
        """Initialize the strategy."""
        # Create dictionaries of indicators for each data feed
        self.fast_ma = {}
        self.slow_ma = {}
        
        for data in self.datas:
            self.fast_ma[data] = bt.indicators.SimpleMovingAverage(
                data.close, period=self.params.fast_period
            )
            self.slow_ma[data] = bt.indicators.SimpleMovingAverage(
                data.close, period=self.params.slow_period
            )
            
            # Create the crossover indicator
            self.crossover = bt.indicators.CrossOver(self.fast_ma[data], self.slow_ma[data])
        
        # Keep track of the equity curve for analysis
        self.equity_curve = []

    def next(self):
        """Define the trading logic executed on each bar."""
        # Log portfolio value for the equity curve
        date = self.data.datetime.date(0).isoformat()
        value = self.broker.getvalue()
        self.equity_curve.append({'Date': date, 'Value': value})
        
        # Iterate over each data feed (ticker)
        for data in self.datas:
            # Get current position size for this ticker
            position = self.getposition(data).size
            
            # Log values for this ticker
            print(f"{date} - {data._name}: Close: {data.close[0]:.2f}, " + 
                  f"Fast MA: {self.fast_ma[data][0]:.2f}, " + 
                  f"Slow MA: {self.slow_ma[data][0]:.2f}, " + 
                  f"Position: {position}")
            
            # If not in a position and fast MA crosses above slow MA, buy
            if not position and self.crossover > 0:
                self.buy(data=data, size=self.params.position_size)
                print(f"{date} - BUY signal for {data._name}: Fast MA crossed above Slow MA")
            
            # If in a position and fast MA crosses below slow MA, sell
            elif position and self.crossover < 0:
                self.sell(data=data, size=position)
                print(f"{date} - SELL signal for {data._name}: Fast MA crossed below Slow MA") 