#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Stock Strategy Implementation.

This strategy buys when price is above SMA and sells when price is below SMA.
"""

import backtrader as bt
import numpy as np

class SafeSMA(bt.Indicator):
    """
    A safer implementation of Simple Moving Average that's more resilient 
    to edge cases and initialization issues.
    """
    lines = ('sma',)
    params = (('period', 20),)
    
    def __init__(self):
        # Set the minimum period needed for the indicator
        self.addminperiod(self.params.period)
        # Use Backtrader's own SMA for calculation
        self.sma = bt.indicators.SMA(self.data, period=self.params.period)
    
    def next(self):
        # Simply copy the SMA value - the safety is in the minimum period setting
        self.lines.sma[0] = self.sma[0]

class SimpleStockStrategy(bt.Strategy):
    """
    A simple strategy that buys when price is above SMA and sells when price is below SMA.
    
    Parameters:
        sma_period (int): The period for the Simple Moving Average
        position_size (int): Number of shares to buy/sell
    """
    
    params = (
        ('sma_period', 20),
        ('position_size', 10),
    )

    def __init__(self):
        """Initialize the strategy with the SMA indicator."""
        # Calculate warmup period
        self.warmup_period = self.params.sma_period + 5
        
        # Create a dictionary of SMA indicators for each data feed
        self.smas = {}
        for data in self.datas:
            # Use standard Backtrader SMA for stability
            self.smas[data] = bt.indicators.SMA(data.close, period=self.params.sma_period)
        
        # Keep track of the equity curve for analysis
        self.equity_curve = []
        
        # Keep track of trades for analysis
        self.trades = []
        
        # Variables to track if we have enough data for trading
        self.min_bars_required = max(self.params.sma_period + 10, 30)  # Ensure sufficient warmup
        self.bars_processed = 0

    def next(self):
        """Define the trading logic executed on each bar."""
        # Increment bars processed counter
        self.bars_processed += 1
        
        # Log portfolio value for the equity curve
        date = self.data.datetime.date(0).isoformat()
        value = self.broker.getvalue()
        self.equity_curve.append({'Date': date, 'Value': value})

        # Skip trading until we have enough bars for indicators to be reliable
        if self.bars_processed < self.min_bars_required:
            return

        # Iterate over each data feed (ticker)
        for data in self.datas:
            # Skip if not enough data
            if len(data) < self.warmup_period:
                continue
                
            # Get current values
            close = data.close[0]
            sma = self.smas[data][0]
            position = self.getposition(data).size

            # Trading logic applied to each ticker
            if close > sma and position == 0:
                # Buy signal: price above SMA and no position
                self.buy(data=data, size=self.params.position_size)
                
                # Track trade
                self.trades.append({
                    'type': 'open',
                    'ticker': data._name,
                    'date': date,
                    'price': close,
                    'size': self.params.position_size
                })
                
            elif close < sma and position > 0:
                # Sell signal: price below SMA and has position
                self.sell(data=data, size=position)
                
                # Calculate P&L
                buy_price = next((t['price'] for t in reversed(self.trades) 
                                 if t['ticker'] == data._name and t['type'] == 'open'), None)
                if buy_price:
                    pnl = (close - buy_price) * position
                else:
                    pnl = 0.0
                
                # Track trade
                self.trades.append({
                    'type': 'close',
                    'ticker': data._name,
                    'date': date,
                    'price': close,
                    'size': position,
                    'pnl': pnl
                })