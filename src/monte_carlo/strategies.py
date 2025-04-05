#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy implementations for the Monte Carlo backtesting framework.
This module contains simplified versions of trading strategies
used for Monte Carlo simulations.
"""

import backtrader as bt

# Strategy Classes
class SimpleStock(bt.Strategy):
    """
    A simple stock trading strategy based on a SMA.
    """
    params = (
        ('sma_period', 20),
        ('position_size', 100),
    )

    def __init__(self):
        # Define the SMA indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.sma_period
        )
        
        # Track portfolio values every day
        self.val_start = self.broker.getvalue()
        self.daily_values = []
        
        # Keep track of all active positions and cash
        self.positions_info = {}  # ticker -> quantity
        self.cash = self.val_start
    
    def notify_order(self, order):
        """Track order executions to update positions accurately"""
        if order.status in [order.Completed]:
            # Get the data name
            data_name = order.data._name
            
            # Update position info for this ticker
            if order.isbuy():
                # Add to position
                current_qty = self.positions_info.get(data_name, 0)
                self.positions_info[data_name] = current_qty + order.size
                
                # Update cash (subtract cost + commission)
                self.cash -= order.executed.price * order.size + order.executed.comm
            else:
                # Reduce position
                current_qty = self.positions_info.get(data_name, 0)
                self.positions_info[data_name] = current_qty - order.size
                
                # Update cash (add proceeds - commission)
                self.cash += order.executed.price * order.size - order.executed.comm
    
    def log_values(self):
        """Log portfolio values for each day"""
        # Get current date
        dt = self.data.datetime.date(0)
        
        # Use broker's getvalue() method which accounts for cash + position values
        value = self.broker.getvalue()
        
        # Store the date and value
        self.daily_values.append({
            'date': dt,
            'value': value,
            'cash': self.broker.get_cash(),
            'positions': self.get_position_values()
        })
    
    def get_position_values(self):
        """Get the current value of all positions"""
        position_values = {}
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size != 0:
                ticker = data._name
                position_values[ticker] = {
                    'size': pos.size,
                    'price': pos.price,
                    'value': pos.size * data.close[0]
                }
        return position_values
        
    def next(self):
        # Log portfolio value at each bar
        self.log_values()
            
        # Trading logic
        position = self.getposition().size
        
        if self.data.close[0] > self.sma[0] and position == 0:
            # Buy signal
            self.buy(size=self.params.position_size)
        elif self.data.close[0] < self.sma[0] and position > 0:
            # Sell signal
            self.sell(size=position)


class MACrossover(bt.Strategy):
    """
    A Moving Average Crossover Strategy.
    """
    params = (
        ('fast_period', 5),      # Fast moving average period
        ('slow_period', 20),     # Slow moving average period
        ('position_size', 100),  # Size of the position to take
    )

    def __init__(self):
        # Calculate warmup period - ensure it's long enough for safe use
        self.warmup_period = max(self.params.slow_period, self.params.fast_period) * 3
        
        # Set minimum periods to ensure indicators have enough data
        self.addminperiod(self.warmup_period)
        
        # Use dictionary-based indicators for safety
        self.fast_ma = {}
        self.slow_ma = {}
        self.crossover = {}
        
        # Initialize indicators for data feeds
        for data in self.datas:
            # Use standard SMA indicators
            self.fast_ma[data] = bt.indicators.SMA(data.close, period=self.params.fast_period)
            self.slow_ma[data] = bt.indicators.SMA(data.close, period=self.params.slow_period)
            
            # Use standard CrossOver indicator
            self.crossover[data] = bt.indicators.CrossOver(self.fast_ma[data], self.slow_ma[data])
        
        # Track portfolio values every day
        self.daily_values = []
        self.val_start = self.broker.getvalue()
        
        # Keep track of positions
        self.positions_info = {}  # ticker -> quantity
        
        # Variables to track if we have enough data for trading
        self.min_bars_required = self.warmup_period
        self.bars_processed = 0

    def notify_order(self, order):
        """Track order executions to update positions accurately"""
        if order.status in [order.Completed]:
            # Get the data name
            data_name = order.data._name
            
            # Update position info for this ticker
            if order.isbuy():
                # Add to position
                current_qty = self.positions_info.get(data_name, 0)
                self.positions_info[data_name] = current_qty + order.size
            else:
                # Reduce position
                current_qty = self.positions_info.get(data_name, 0)
                self.positions_info[data_name] = current_qty - order.size
    
    def log_values(self):
        """Log portfolio values for each day"""
        # Get current date
        dt = self.data.datetime.date(0)
        
        # Use broker's getvalue() method which accounts for cash + position values
        value = self.broker.getvalue()
        
        # Store the date and value
        self.daily_values.append({
            'date': dt,
            'value': value,
            'cash': self.broker.get_cash(),
            'positions': self.get_position_values()
        })
    
    def get_position_values(self):
        """Get the current value of all positions"""
        position_values = {}
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size != 0:
                ticker = data._name
                position_values[ticker] = {
                    'size': pos.size,
                    'price': pos.price,
                    'value': pos.size * data.close[0]
                }
        return position_values

    def next(self):
        # Increment bars processed counter
        self.bars_processed += 1
        
        # Log portfolio value
        self.log_values()
        
        # Skip trading until we have enough bars for reliable indicator values
        if self.bars_processed < self.min_bars_required:
            return
            
        # Process each data feed
        for data in self.datas:
            try:
                # Skip if not enough data
                if len(data) < self.warmup_period:
                    continue
                
                # Safety check to ensure all indicators have valid values
                if (not self.fast_ma[data] or not self.fast_ma[data][0] or 
                    not self.slow_ma[data] or not self.slow_ma[data][0] or
                    not self.crossover[data] or not isinstance(self.crossover[data][0], (int, float))):
                    continue
                
                # Get current position size
                position = self.getposition(data).size
                
                # Trading logic based on crossover
                if self.crossover[data] > 0 and position == 0:
                    # Buy signal: crossover is positive
                    self.buy(data=data, size=self.params.position_size)
                elif self.crossover[data] < 0 and position > 0:
                    # Sell signal: crossover is negative
                    self.sell(data=data, size=position)
            except Exception as e:
                print(f"Error in next() for {data._name}: {e}")
                continue


class AuctionMarket(bt.Strategy):
    """
    Auction Market strategy implementation for Backtrader.
    
    This provides a basic implementation that can be used for testing
    when the actual strategy is not available.
    """
    params = (
        ('volume_period', 20),   # Period for volume moving average
        ('price_period', 10),    # Period for price moving average
        ('position_size', 10)    # Size of position to take
    )
    
    def __init__(self):
        # Calculate warmup period - ensure it's long enough for safe use
        self.warmup_period = max(self.params.volume_period, self.params.price_period) * 3
        
        # Set minimum periods to ensure indicators have enough data
        self.addminperiod(self.warmup_period)
        
        # Use dictionary-based indicator storage for safety
        self.volume_ma = {}
        self.price_ma = {}
        
        # Initialize indicators for data feeds
        for data in self.datas:
            # Volume indicators
            self.volume_ma[data] = bt.indicators.SMA(data.volume, period=self.params.volume_period)
            
            # Price indicators
            self.price_ma[data] = bt.indicators.SMA(data.close, period=self.params.price_period)
        
        # Track portfolio values every day
        self.daily_values = []
        self.val_start = self.broker.getvalue()
        
        # Keep track of positions
        self.positions_info = {}  # ticker -> quantity
        
        # State for trading logic and ensure minimum bars processed
        self.bars_processed = 0
        self.min_bars_required = self.warmup_period
    
    def notify_order(self, order):
        """Track order executions to update positions accurately"""
        if order.status in [order.Completed]:
            # Get the data name
            data_name = order.data._name
            
            # Update position info for this ticker
            if order.isbuy():
                # Add to position
                current_qty = self.positions_info.get(data_name, 0)
                self.positions_info[data_name] = current_qty + order.size
            else:
                # Reduce position
                current_qty = self.positions_info.get(data_name, 0)
                self.positions_info[data_name] = current_qty - order.size
    
    def log_values(self):
        """Log portfolio values for each day"""
        # Get current date
        dt = self.data.datetime.date(0)
        
        # Use broker's getvalue() method which accounts for cash + position values
        value = self.broker.getvalue()
        
        # Store the date and value
        self.daily_values.append({
            'date': dt,
            'value': value,
            'cash': self.broker.get_cash(),
            'positions': self.get_position_values()
        })
    
    def get_position_values(self):
        """Get the current value of all positions"""
        position_values = {}
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size != 0:
                ticker = data._name
                position_values[ticker] = {
                    'size': pos.size,
                    'price': pos.price,
                    'value': pos.size * data.close[0]
                }
        return position_values
    
    def next(self):
        # Increment bars processed counter
        self.bars_processed += 1
        
        # Log portfolio value
        self.log_values()
        
        # Skip trading until we have enough bars for reliable indicator values
        if self.bars_processed < self.min_bars_required:
            return
        
        # Process each data feed
        for data in self.datas:
            try:
                # Skip if not enough data
                if len(data) < self.warmup_period:
                    continue
                
                # Safety check to ensure all indicators have valid values
                if (not self.price_ma[data] or not self.price_ma[data][0] or
                    not self.volume_ma[data] or not self.volume_ma[data][0]):
                    continue
                
                # Current position
                position = self.getposition(data).size
                
                # Simple trading logic based on price and volume
                if position == 0:
                    # Entry condition: price above MA and volume above average
                    if (data.close[0] > self.price_ma[data][0] and 
                        data.volume[0] > self.volume_ma[data][0]):
                        self.buy(data=data, size=self.params.position_size)
                else:
                    # Exit condition: price below MA or volume below average
                    if (data.close[0] < self.price_ma[data][0] or 
                        data.volume[0] < self.volume_ma[data][0] * 0.8):
                        self.sell(data=data, size=position)
            except Exception as e:
                print(f"Error in next() for {data._name}: {e}")
                continue


class MultiPosition(bt.Strategy):
    """
    MultiPosition strategy implementation for Backtrader.
    
    This strategy can hold multiple positions based on different indicators.
    It uses a fast and slow moving average to make entry/exit decisions.
    """
    params = (
        ('fast_period', 10),     # Fast moving average period
        ('slow_period', 30),     # Slow moving average period
        ('max_positions', 3),    # Maximum number of positions to hold
        ('position_size', 10),   # Size of each position
        ('rsi_period', 14),      # RSI period
        ('rsi_overbought', 70),  # RSI overbought level
        ('rsi_oversold', 30)     # RSI oversold level
    )
    
    def __init__(self):
        # Calculate warmup period
        self.warmup_period = max(self.params.slow_period, self.params.rsi_period) + 10
        
        # Initialize indicators
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        
        # Track portfolio values every day
        self.daily_values = []
        self.val_start = self.broker.getvalue()
        
        # Keep track of positions by ticker
        self.positions_info = {}  # ticker -> quantity
        
        # State for trading logic
        self.bars_processed = 0
        self.position_tracker = {}  # Track positions by entry price
    
    def notify_order(self, order):
        """Track order executions to update positions accurately"""
        if order.status in [order.Completed]:
            # Get the data name
            data_name = order.data._name
            
            # Update position info for this ticker
            if order.isbuy():
                # Add to position
                current_qty = self.positions_info.get(data_name, 0)
                self.positions_info[data_name] = current_qty + order.size
            else:
                # Reduce position
                current_qty = self.positions_info.get(data_name, 0)
                self.positions_info[data_name] = current_qty - order.size
    
    def log_values(self):
        """Log portfolio values for each day"""
        # Get current date
        dt = self.data.datetime.date(0)
        
        # Use broker's getvalue() method which accounts for cash + position values
        value = self.broker.getvalue()
        
        # Store the date and value
        self.daily_values.append({
            'date': dt,
            'value': value,
            'cash': self.broker.get_cash(),
            'positions': self.get_position_values()
        })
    
    def get_position_values(self):
        """Get the current value of all positions"""
        position_values = {}
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size != 0:
                ticker = data._name
                position_values[ticker] = {
                    'size': pos.size,
                    'price': pos.price,
                    'value': pos.size * data.close[0]
                }
        return position_values
        
    def next(self):
        # Increment bars processed counter
        self.bars_processed += 1
        
        # Log portfolio value
        self.log_values()
        
        # Skip trading until we have enough bars for reliable indicator values
        if self.bars_processed < self.warmup_period:
            return
        
        # Get current position
        position_size = self.getposition().size
        
        # Close positions if needed
        if position_size > 0:
            # Exit signal based on MA crossover down or RSI overbought
            if self.fast_ma[0] < self.slow_ma[0] or self.rsi[0] > self.params.rsi_overbought:
                self.sell(size=position_size)
                self.position_tracker = {}  # Clear positions dictionary
        
        # Entry signals if we have capacity for more positions
        if len(self.position_tracker) < self.params.max_positions:
            # Entry signal based on MA crossover up and RSI not overbought
            if (self.fast_ma[0] > self.slow_ma[0] and 
                self.rsi[0] < self.params.rsi_overbought):
                
                # Check if we don't already have a position at this price level
                entry_price = self.data.close[0]
                price_key = f"{entry_price:.2f}"  # Convert to string to avoid floating point issues
                if price_key not in self.position_tracker:
                    # Buy with the position size
                    self.buy(size=self.params.position_size)
                    # Record this position
                    self.position_tracker[price_key] = self.params.position_size 