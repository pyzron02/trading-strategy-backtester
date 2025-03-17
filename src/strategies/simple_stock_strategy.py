#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Stock Strategy Implementation.

This strategy buys when price is above SMA and sells when price is below SMA.
"""

import backtrader as bt

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
        # Create a dictionary of SMA indicators for each data feed
        self.sma = {data: bt.ind.SimpleMovingAverage(data.close, period=self.params.sma_period) for data in self.datas}
        
        # Keep track of the equity curve for analysis
        self.equity_curve = []
        
        # Keep track of trades for analysis
        self.trades = []

    def next(self):
        """Define the trading logic executed on each bar."""
        # Log portfolio value for the equity curve
        date = self.data.datetime.date(0).isoformat()
        value = self.broker.getvalue()
        self.equity_curve.append({'Date': date, 'Value': value})

        # Iterate over each data feed (ticker)
        for data in self.datas:
            close = data.close[0]
            sma = self.sma[data][0]
            position = self.getposition(data).size

            # Log values for debugging
            print(f"{date} - {data._name}: Close: {close:.2f}, SMA: {sma:.2f}, Position: {position}")

            # Trading logic applied to each ticker
            if close > sma and position == 0:
                # Buy signal: price above SMA and no position
                self.buy(data=data, size=self.params.position_size)
                print(f"{date} - BUY signal for {data._name}: Price {close:.2f} > SMA {sma:.2f}")
                
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
                print(f"{date} - SELL signal for {data._name}: Price {close:.2f} < SMA {sma:.2f}")
                
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