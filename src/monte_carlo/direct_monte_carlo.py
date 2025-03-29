#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct Monte Carlo Simulation Wrapper

This script implements a direct Monte Carlo simulation approach that bypasses
the problematic parts of the walk_forward_monte_carlo framework while preserving
the core functionality of testing strategies against permuted data.
"""

import os
import sys
import json
import time
import random
import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require a display
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
import csv
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import backtrader
import backtrader as bt

# Import strategies from registry
from src.strategies import registry

# Import specific strategies
try:
    from src.strategies.simple_stock_strategy import SimpleStock as SimpleStockStrategy
    from src.strategies.ma_crossover import MACrossover as MACrossoverStrategy
    from src.strategies.multi_position import MultiPosition as MultiPositionStrategy
    from src.strategies.auction_market import AuctionMarket as AuctionMarketStrategy
    
    # Strategy mapping
    STRATEGY_CLASSES = {
        'SimpleStock': SimpleStockStrategy,
        'MACrossover': MACrossoverStrategy,
        'MultiPosition': MultiPositionStrategy,
        'AuctionMarket': AuctionMarketStrategy
    }
except ImportError as e:
    print(f"Warning: Could not import strategy: {e}")
    STRATEGY_CLASSES = {}

# Add PortfolioValue observer
class PortfolioValue(bt.Observer):
    """Observer that tracks portfolio value throughout the backtest"""
    lines = ('value',)
    
    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()

# Define TradeLog analyzer
class TradeLog(bt.Analyzer):
    """Analyzer that logs all trades during a backtest"""
    
    def __init__(self):
        self.log = []
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.log.append({
                'date': bt.num2date(trade.dtclose).strftime('%Y-%m-%d'),
                'type': 'SELL',
                'price': trade.price,
                'size': trade.size,
                'value': trade.price * trade.size,
                'pnl': trade.pnl,
                'commission': trade.commission
            })
        elif trade.justopened:
            self.log.append({
                'date': bt.num2date(trade.dtopen).strftime('%Y-%m-%d'),
                'type': 'BUY',
                'price': trade.price,
                'size': trade.size,
                'value': trade.price * trade.size,
                'pnl': 0.0,
                'commission': trade.commission
            })

# Helper function to save JSON data
def save_to_json(data, filepath):
    """Save data to a JSON file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving JSON data: {e}")
        return False

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


# Add basic AuctionMarket strategy
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


# Add basic MultiPosition strategy
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


# Update STRATEGY_CLASSES with the internal backtrader strategy classes as fallback
if 'SimpleStock' not in STRATEGY_CLASSES:
    STRATEGY_CLASSES['SimpleStock'] = SimpleStock

if 'MACrossover' not in STRATEGY_CLASSES:
    STRATEGY_CLASSES['MACrossover'] = MACrossover

if 'AuctionMarket' not in STRATEGY_CLASSES:
    STRATEGY_CLASSES['AuctionMarket'] = AuctionMarket

if 'MultiPosition' not in STRATEGY_CLASSES:
    STRATEGY_CLASSES['MultiPosition'] = MultiPosition

class DirectMonteCarloTest:
    """
    A direct implementation of Monte Carlo testing for trading strategies.
    
    This class handles data permutation, backtesting, and result analysis
    in a straightforward way to avoid the complex interaction issues in
    the full framework.
    """
    
    def __init__(self, strategy_name, tickers, in_sample_start='2015-01-01', in_sample_end='2019-12-31', 
                 out_sample_start='2020-01-01', out_sample_end='2021-12-31', parameters=None, 
                 output_dir=None, num_permutations=10, enable_in_sample_mc=False):
        """
        Initialize the Monte Carlo test.
        
        Args:
            strategy_name (str): Name of the strategy to test
            tickers (list): List of tickers to backtest
            in_sample_start (str): Start date for in-sample period (YYYY-MM-DD)
            in_sample_end (str): End date for in-sample period (YYYY-MM-DD)
            out_sample_start (str): Start date for out-of-sample period (YYYY-MM-DD)
            out_sample_end (str): End date for out-of-sample period (YYYY-MM-DD)
            parameters (dict): Strategy parameters
            output_dir (str): Directory to save test results
            num_permutations (int): Number of permutations to run
            enable_in_sample_mc (bool): Whether to run Monte Carlo permutations on in-sample data
        """
        self.strategy_name = strategy_name
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        
        # Convert date strings to datetime objects
        self.in_sample_start = pd.to_datetime(in_sample_start)
        self.in_sample_end = pd.to_datetime(in_sample_end)
        self.out_sample_start = pd.to_datetime(out_sample_start)
        self.out_sample_end = pd.to_datetime(out_sample_end)
        
        # Set default parameters based on strategy
        if parameters is None:
            if strategy_name == 'MACrossover':
                self.parameters = {
                    'fast_period': 5,
                    'slow_period': 20,
                    'position_size': 10
                }
            elif strategy_name == 'SimpleStock':
                self.parameters = {
                    'sma_period': 20,
                    'position_size': 10
                }
            else:
                self.parameters = {}
        else:
            self.parameters = parameters.copy()
        
        # Set up output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            self.output_dir = os.path.join("output", "direct_monte_carlo", f"{strategy_name}_{timestamp}")
        else:
            self.output_dir = os.path.abspath(output_dir)
        
        self.num_permutations = num_permutations
        self.enable_in_sample_mc = enable_in_sample_mc
        
        # Create data directory for storing processed data
        self.data_dir = os.path.join(self.output_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Log initialization
        print(f"Initialized Direct Monte Carlo Test for {strategy_name}")
        print(f"Tickers: {tickers}")
        print(f"In-sample period: {in_sample_start} to {in_sample_end}")
        print(f"Out-of-sample period: {out_sample_start} to {out_sample_end}")
        print(f"Number of permutations: {num_permutations}")
        print(f"In-sample Monte Carlo: {'Enabled' if enable_in_sample_mc else 'Disabled'}")
        print(f"Output directory: {self.output_dir}")
        print(f"Using parameters: {self.parameters}")
    
    def _load_stock_data(self):
        """Load stock data for the specified tickers from a CSV file."""
        print("\nLoading stock data...")
        
        try:
            # Get the absolute path of the project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Define paths to look for stock data
            potential_paths = [
                # First check if data is in the data_dir (output directory)
                os.path.join(self.data_dir, 'stock_data.csv'),
                
                # Then check in the standard input location
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input', 'stock_data.csv'),
                
                # Check project root
                os.path.join(project_root, 'input', 'stock_data.csv'),
                
                # Check current directory
                os.path.join(os.getcwd(), 'input', 'stock_data.csv'),
                
                # Check relative to current working directory
                'input/stock_data.csv',
                'stock_data.csv'
            ]
            
            # Find the first path that exists
            data_path = None
            for path in potential_paths:
                if os.path.exists(path):
                    data_path = path
                    break
            
            if data_path is None:
                print("Error: stock_data.csv file not found in any of the following locations:")
                for path in potential_paths:
                    print(f" - {path}")
            return None
        
            # Load the data
            print(f"Loading stock data from {data_path}")
            data = pd.read_csv(data_path)
            
            # Convert Date column to datetime
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            
            print(f"Loaded stock data with {len(data)} rows and {len(data.columns)} columns.")
            
            # Process the data for each ticker
            ticker_data = {}
            for ticker in self.tickers:
                # Extract columns for this ticker
                ticker_cols = [col for col in data.columns if col.startswith(f"{ticker}_") or col == 'Date']
                
                if len(ticker_cols) <= 1:  # Just Date column or empty
                    print(f"Warning: No data found for ticker {ticker}")
                    continue
                
                # Create a DataFrame for this ticker
                df = data[ticker_cols].copy()
                
                # Rename columns to standard OHLCV format
                rename_map = {}
                for col in ticker_cols:
                    if col == 'Date':
                        continue
                    field = col.split('_')[1]  # e.g., AAPL_Close -> Close
                    rename_map[col] = field
                
                df = df.rename(columns=rename_map)
                
                # Ensure OHLCV columns exist
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"Warning: Missing columns for {ticker}: {missing_cols}")
                    # Fill in missing columns with Close price
                    for col in missing_cols:
                        if 'Close' in df.columns:
                            df[col] = df['Close']
                        else:
                            print(f"Error: Cannot fill missing column {col} for {ticker}")
                            continue
                
                # Add adjusted close if it doesn't exist
                if 'Adj Close' not in df.columns and 'Close' in df.columns:
                    df['Adj Close'] = df['Close']
                
                # Create the data directory if it doesn't exist
                os.makedirs(self.data_dir, exist_ok=True)
                
                # Save processed data
                ticker_data[ticker] = df
                
                # Save to CSV for backtrader
                csv_path = os.path.join(self.data_dir, f"{ticker}_data.csv")
                df.to_csv(csv_path, index=False)
                print(f"Saved processed data for {ticker} to {csv_path}")
            
            return ticker_data
            
        except Exception as e:
            print(f"Error loading stock data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _permute_data(self, original_df, permutation_type='returns', permutation_seed=None, permute_period='out_sample'):
        """
        Create a permuted version of the price data.
        
        Args:
            original_df (pd.DataFrame): Original price data with OHLCV columns
            permutation_type (str): Type of permutation - 'returns' or 'blocks'
            permutation_seed (int): Random seed for reproducibility
            permute_period (str): Period to permute - 'in_sample', 'out_sample', or 'both'
            
        Returns:
            pd.DataFrame: Permuted data
        """
        # Set random seed for reproducibility
        if permutation_seed is not None:
            np.random.seed(permutation_seed)
            random.seed(permutation_seed)
        
        # Make a copy to avoid modifying the original
        df = original_df.copy()
        
        # Define in-sample and out-of-sample periods
        is_mask = (df['Date'] >= self.in_sample_start) & (df['Date'] <= self.in_sample_end)
        oos_mask = (df['Date'] >= self.out_sample_start) & (df['Date'] <= self.out_sample_end)
        
        # Determine which period(s) to permute
        periods_to_permute = []
        if permute_period == 'in_sample' or permute_period == 'both':
            periods_to_permute.append(('in_sample', is_mask))
        if permute_period == 'out_sample' or permute_period == 'both':
            periods_to_permute.append(('out_sample', oos_mask))
        
        for period_name, period_mask in periods_to_permute:
            # Get the data for this period to permute
            period_data = df[period_mask].copy()
            
            if len(period_data) == 0:
                print(f"Warning: No {period_name} data to permute")
                continue
        
        # Permute returns for price columns
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
            price_cols = [col for col in price_cols if col in period_data.columns]
        
        if permutation_type == 'returns':
            # Permute by shuffling returns
            for col in price_cols:
                # Calculate returns
                    prices = period_data[col].values
                returns = np.diff(prices) / prices[:-1]
                
                # Shuffle returns
                shuffled_returns = np.random.permutation(returns)
                
                # Reconstruct prices
                new_prices = np.zeros_like(prices)
                new_prices[0] = prices[0]  # Keep first price
                
                for i in range(1, len(new_prices)):
                    new_prices[i] = new_prices[i-1] * (1 + shuffled_returns[i-1])
                
                # Replace prices with permuted version
                    period_data[col] = new_prices
        
        elif permutation_type == 'blocks':
            # Permute by shuffling blocks of data
            for col in price_cols:
                    prices = period_data[col].values
                
                # Create blocks of ~5 days
                block_size = 5
                blocks = []
                
                for i in range(0, len(prices), block_size):
                    end = min(i + block_size, len(prices))
                    blocks.append(prices[i:end])
                
                # Shuffle blocks
                random.shuffle(blocks)
                
                # Reconstruct series
                new_prices = np.concatenate(blocks)
                
                # Ensure same length
                new_prices = new_prices[:len(prices)]
                
                # Replace prices
                    period_data[col] = new_prices
        
        # Also permute volume if available
            if 'Volume' in period_data.columns:
                volumes = period_data['Volume'].values
            permuted_volumes = np.random.permutation(volumes)
                period_data['Volume'] = permuted_volumes
        
            # Replace period data with permuted data
            df.loc[period_mask] = period_data
        
        return df
    
    def _run_single_permutation(self, permutation_index, permute_period='out_sample'):
        """Run a single permutation test.
        
        Args:
            permutation_index (int): The index of this permutation
            permute_period (str): Period to permute - 'in_sample', 'out_sample', or 'both'
            
        Returns:
            dict: The results of this permutation
        """
        print(f"Running permutation {permutation_index} on {permute_period} data...")
        
        # Create output directory for this permutation
        perm_output_dir = os.path.join(self.output_dir, f"permutation_{permute_period}_{permutation_index}")
        os.makedirs(perm_output_dir, exist_ok=True)
        
        # Load the stock data
        ticker_data = self._load_stock_data()
        if ticker_data is None:
            print(f"Failed to load stock data for permutation {permutation_index}")
            return None
        
        # Apply permutation to each ticker's data
        permuted_ticker_data = {}
        for ticker, df in ticker_data.items():
            permuted_df = self._permute_data(df, permutation_seed=permutation_index, permute_period=permute_period)
            permuted_ticker_data[ticker] = permuted_df
            
            # Save permuted data for reference
            permuted_df.to_csv(os.path.join(perm_output_dir, f"permuted_data_{ticker}.csv"), index=False)
        
        # Run backtest on permuted data
        permutation_results = self._run_backtest(
            ticker_data=permuted_ticker_data,
            output_dir=perm_output_dir,
            label=f"permutation_{permute_period}_{permutation_index}"
        )
        
        # Rename trade log for clarity
        trade_log = os.path.join(perm_output_dir, "trade_log.csv")
        if os.path.exists(trade_log):
            new_name = os.path.join(perm_output_dir, f"trade_log_permutation_{permute_period}_{permutation_index}.csv")
            shutil.copy(trade_log, new_name)
        
        return permutation_results

    def run_test(self, n_jobs=None):
        """
        Run the Monte Carlo testing with the specified parameters.
        
        Args:
            n_jobs (int, optional): Number of parallel jobs to run. If None, defaults to all available cores - 1.
            
        Returns:
            dict: Results of the testing
        """
        # Determine the number of cores to use
        available_cores = multiprocessing.cpu_count()
        
        if n_jobs is None:
            # Default to using all cores except one
            n_jobs = max(1, available_cores - 1)
                else:
            # Ensure n_jobs is valid
            n_jobs = min(max(1, n_jobs), available_cores)
        
        print(f"Running Monte Carlo test with {self.num_permutations} permutations using {n_jobs} CPU cores")
        
        # Create output directory for original results
        original_output_dir = os.path.join(self.output_dir, "original")
        os.makedirs(original_output_dir, exist_ok=True)
        
        # Load the stock data
        ticker_data = self._load_stock_data()
        if ticker_data is None:
            print("Failed to load stock data")
            return None
        
        # Save original data for reference
        for ticker, df in ticker_data.items():
            df.to_csv(os.path.join(original_output_dir, f"original_data_{ticker}.csv"), index=False)
        
        # Run the original backtest
        print("Running original backtest...")
        original_results = self._run_backtest(
            ticker_data=ticker_data,
            output_dir=original_output_dir,
            label="original"
        )
        
        if original_results is None:
            print("Original backtest failed")
            return None
        
        # Calculate original metrics
        original_metrics = self._calculate_metrics(original_results)
        
        # Rename original trade log for clarity
        original_trade_log = os.path.join(original_output_dir, "trade_log.csv")
        if os.path.exists(original_trade_log):
            new_name = os.path.join(original_output_dir, "trade_log_original.csv")
            shutil.copy(original_trade_log, new_name)
        
        # Print original metrics
        print("\nOriginal Backtest Metrics:")
        for key, value in original_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Define permutation periods based on configuration
        permutation_periods = []
        
        # Always include out-of-sample
        permutation_periods.append('out_sample')
        
        # Conditionally include in-sample
        if self.enable_in_sample_mc:
            permutation_periods.append('in_sample')
        
        # Results storage
        all_permutation_results = {period: [] for period in permutation_periods}
        all_permutation_metrics = {period: [] for period in permutation_periods}
        
        # Run permutations for each period in parallel
        for period in permutation_periods:
            print(f"\nRunning {period} permutations...")
            
            # Define arguments for each permutation
            permutation_args = [(i, period) for i in range(self.num_permutations)]
            
            if n_jobs > 1:
                # Run in parallel
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [executor.submit(self._run_single_permutation, *args) for args in permutation_args]
                    
                    # Collect results as they complete
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            metrics = self._calculate_metrics(result)
                            all_permutation_results[period].append(result)
                            all_permutation_metrics[period].append(metrics)
        else:
                # Run sequentially
                for args in permutation_args:
                    result = self._run_single_permutation(*args)
                    if result is not None:
                        metrics = self._calculate_metrics(result)
                        all_permutation_results[period].append(result)
                        all_permutation_metrics[period].append(metrics)
        
        # Print permutation results for each period
        for period in permutation_periods:
            print(f"\n{period.replace('_', ' ').title()} Permutation Results:")
            for i, metrics in enumerate(all_permutation_metrics[period]):
                print(f"Permutation {i}:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")
        
        # Calculate significance for each period
        all_analysis_results = {}
        for period in permutation_periods:
            analysis_results = self._analyze_results(original_metrics, all_permutation_metrics[period])
            all_analysis_results[period] = analysis_results
            
            print(f"\n{period.replace('_', ' ').title()} Metric Significance:")
            for key, value in analysis_results.items():
                print(f"{key}: p-value = {value:.4f}")
        
        # If both in-sample and out-of-sample were run, compare them
        if len(permutation_periods) > 1:
            self._compare_periods(all_permutation_metrics, all_analysis_results)
        
        # Return all results
        return {
            "original_metrics": original_metrics,
            "permutation_metrics": all_permutation_metrics,
            "analysis": all_analysis_results
        }

    def _compare_periods(self, all_permutation_metrics, all_analysis_results):
        """
        Compare in-sample and out-of-sample permutation results.
        
        Args:
            all_permutation_metrics (dict): Dictionary of permutation metrics for each period
            all_analysis_results (dict): Dictionary of analysis results for each period
        """
        if 'in_sample' not in all_permutation_metrics or 'out_sample' not in all_permutation_metrics:
            return
        
        print("\nComparison of In-Sample vs Out-of-Sample Permutation Results:")
        print("-------------------------------------------------------------")
        
        # Compare metrics distributions
        for metric in ['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor', 'max_drawdown']:
            in_sample_values = [p.get(metric, 0) for p in all_permutation_metrics['in_sample'] if p.get(metric) is not None]
            out_sample_values = [p.get(metric, 0) for p in all_permutation_metrics['out_sample'] if p.get(metric) is not None]
            
            if not in_sample_values or not out_sample_values:
                continue
            
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  In-Sample Mean: {np.mean(in_sample_values):.4f}, Std: {np.std(in_sample_values):.4f}")
            print(f"  Out-Sample Mean: {np.mean(out_sample_values):.4f}, Std: {np.std(out_sample_values):.4f}")
            
            # Calculate statistical difference (t-test)
            if len(in_sample_values) > 1 and len(out_sample_values) > 1:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(in_sample_values, out_sample_values)
                print(f"  Difference p-value: {p_value:.4f} (significant if < 0.05)")
        
        # Compare p-values
        print("\nP-value Comparison:")
        for metric in ['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor', 'max_drawdown']:
            if metric in all_analysis_results['in_sample'] and metric in all_analysis_results['out_sample']:
                in_sample_p = all_analysis_results['in_sample'][metric]
                out_sample_p = all_analysis_results['out_sample'][metric]
                
                print(f"  {metric.replace('_', ' ').title()}: In-Sample p={in_sample_p:.4f}, Out-Sample p={out_sample_p:.4f}")
                
                # Check for overfitting
                if in_sample_p < 0.05 and out_sample_p > 0.05:
                    print(f"    WARNING: Potential overfitting - strategy is significant in-sample but not out-of-sample")
                elif in_sample_p < 0.05 and out_sample_p < 0.05:
                    print(f"    GOOD: Strategy is significant in both in-sample and out-of-sample periods")
        
        # Save comparison results
        comparison_results = {
            'metric_comparison': {},
            'p_value_comparison': {},
            'overfitting_assessment': {}
        }
        
        for metric in ['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor', 'max_drawdown']:
            in_sample_values = [p.get(metric, 0) for p in all_permutation_metrics['in_sample'] if p.get(metric) is not None]
            out_sample_values = [p.get(metric, 0) for p in all_permutation_metrics['out_sample'] if p.get(metric) is not None]
            
            if not in_sample_values or not out_sample_values:
                continue
            
            comparison_results['metric_comparison'][metric] = {
                'in_sample_mean': float(np.mean(in_sample_values)),
                'in_sample_std': float(np.std(in_sample_values)),
                'out_sample_mean': float(np.mean(out_sample_values)),
                'out_sample_std': float(np.std(out_sample_values))
            }
            
            if metric in all_analysis_results['in_sample'] and metric in all_analysis_results['out_sample']:
                comparison_results['p_value_comparison'][metric] = {
                    'in_sample_p': all_analysis_results['in_sample'][metric],
                    'out_sample_p': all_analysis_results['out_sample'][metric]
                }
                
                # Assess overfitting
                in_sample_p = all_analysis_results['in_sample'][metric]
                out_sample_p = all_analysis_results['out_sample'][metric]
                
                if in_sample_p < 0.05 and out_sample_p > 0.05:
                    overfitting_status = "WARNING: Potential overfitting detected"
                elif in_sample_p < 0.05 and out_sample_p < 0.05:
                    overfitting_status = "GOOD: Strategy is robust across both periods"
                else:
                    overfitting_status = "NEUTRAL: No clear pattern"
                
                comparison_results['overfitting_assessment'][metric] = overfitting_status
        
        # Save comparison results
        comparison_filepath = os.path.join(self.output_dir, "analysis", "period_comparison.json")
        os.makedirs(os.path.dirname(comparison_filepath), exist_ok=True)
        save_to_json(comparison_results, comparison_filepath)
        print(f"\nPeriod comparison results saved to {comparison_filepath}")
        
        # Generate comparison plots
        self._generate_comparison_plots(all_permutation_metrics)

    def _generate_comparison_plots(self, all_permutation_metrics):
        """
        Generate comparison plots between in-sample and out-of-sample results.
        
        Args:
            all_permutation_metrics (dict): Dictionary of permutation metrics for each period
        """
        if 'in_sample' not in all_permutation_metrics or 'out_sample' not in all_permutation_metrics:
            return
            
        try:
            # Create plots directory
            plots_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # For each key metric, create a histogram comparison
            for metric in ['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor', 'max_drawdown']:
                in_sample_values = [p.get(metric, 0) for p in all_permutation_metrics['in_sample'] if p.get(metric) is not None]
                out_sample_values = [p.get(metric, 0) for p in all_permutation_metrics['out_sample'] if p.get(metric) is not None]
                
                if not in_sample_values or not out_sample_values:
                    continue
                
                plt.figure(figsize=(10, 6))
                plt.hist(in_sample_values, alpha=0.5, label='In-Sample', bins=10)
                plt.hist(out_sample_values, alpha=0.5, label='Out-of-Sample', bins=10)
                plt.axvline(np.mean(in_sample_values), color='blue', linestyle='dashed', linewidth=1)
                plt.axvline(np.mean(out_sample_values), color='orange', linestyle='dashed', linewidth=1)
                plt.title(f'Distribution of {metric.replace("_", " ").title()} Across Permutations')
                plt.xlabel(metric.replace('_', ' ').title())
                plt.ylabel('Frequency')
                plt.legend()
                plt.tight_layout()
                
                # Save the plot
                plot_path = os.path.join(plots_dir, f'{metric}_comparison.png')
                plt.savefig(plot_path)
                plt.close()
                
                print(f"Saved comparison plot for {metric} to {plot_path}")
        except Exception as e:
            print(f"Error generating comparison plots: {e}")
            import traceback
            traceback.print_exc()

    def _run_backtest(self, ticker_data, output_dir, label="backtest"):
        """
        Run a backtest with the given ticker data.
        
        Args:
            ticker_data (dict): Dictionary of ticker DataFrames
            output_dir (str): Directory to save results
            label (str): Label for this backtest
            
        Returns:
            dict: Results of the backtest
        """
        try:
            # Create Cerebro instance
            cerebro = bt.Cerebro()
            
            # Set the initial cash
            initial_cash = 100000.0
            cerebro.broker.setcash(initial_cash)
            
            # Set the commission
            cerebro.broker.setcommission(commission=0.001)
            
            # Store all dates for reference
            all_dates = []
            
            # Add data feeds
            for ticker, df in ticker_data.items():
                # Create a copy to avoid modifying the original
                df_copy = df.copy()
                
                # Ensure the Date column is a datetime
                if 'Date' in df_copy.columns:
                    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
                    # Store all unique dates
                    all_dates.extend(df_copy['Date'].dt.strftime('%Y-%m-%d').tolist())
                    # Set Date as index for backtrader
                    df_copy = df_copy.set_index('Date')
            else:
                    # If Date is already the index
                    all_dates.extend(df_copy.index.strftime('%Y-%m-%d').tolist())
                
                # Create a data feed
                data = bt.feeds.PandasData(
                    dataname=df_copy,
                    datetime=None,  # Date is in the index
                    open='Open',
                    high='High',
                    low='Low',
                    close='Close',
                    volume='Volume',
                    openinterest=-1  # Not available
                )
                
                # Add the data feed with a name
                cerebro.adddata(data, name=ticker)
            
            # Get all unique dates in chronological order
            all_dates = sorted(list(set(all_dates)))
            
            # Enable broker value tracking at each step
            cerebro.addobserver(bt.observers.Broker)
            cerebro.addobserver(bt.observers.Value)
            
            # Add observers and analyzers
            cerebro.addobserver(bt.observers.Trades)
            cerebro.addobserver(PortfolioValue)
            
            # Add standard analyzers for key metrics
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
            cerebro.addanalyzer(TradeLog, _name='trade_log')
            
            # Add cash value recorder to Cerebro
            cerebro.addwriter(bt.WriterFile, 
                             out=os.path.join(output_dir, 'cerebro_log.csv'), 
                             csv=True)
            
            # Add the appropriate strategy based on name
            if self.strategy_name == "SimpleStock":
                cerebro.addstrategy(
                    SimpleStock,
                    **self.parameters
                )
            elif self.strategy_name == "MACrossover":
                cerebro.addstrategy(
                    MACrossover,
                    **self.parameters
                )
            elif self.strategy_name == "AuctionMarket":
                cerebro.addstrategy(
                    AuctionMarket,
                    **self.parameters
                )
            elif self.strategy_name == "MultiPosition":
                cerebro.addstrategy(
                    MultiPosition,
                    **self.parameters
                )
            else:
                print(f"Error: Strategy {self.strategy_name} not supported")
                return None
            
            # Run the backtest
            print(f"Running backtest for {label}...")
            results = cerebro.run()
            
            if not results:
                print("Error: No results from backtest")
                return None
            
            strategy = results[0]
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract analyzer results
            trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()
            trade_log = strategy.analyzers.trade_log.log
            
            # Save trade log to CSV
            trade_log_path = os.path.join(output_dir, 'trade_log.csv')
            with open(trade_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'Type', 'Price', 'Size', 'Value', 'PnL', 'Commission'])
                for trade in trade_log:
                    writer.writerow([
                        trade.get('date', ''),
                        trade.get('type', ''),
                        trade.get('price', 0.0),
                        trade.get('size', 0),
                        trade.get('value', 0.0),
                        trade.get('pnl', 0.0),
                        trade.get('commission', 0.0)
                    ])
            
            # Extract portfolio values from observer data
            # Use bt.observers.Value which records portfolio value at each step
            portfolio_values = []
            
            # Method 1: Extract from strategy's daily_values if available (most accurate)
            if hasattr(strategy, 'daily_values') and strategy.daily_values:
                # Convert to dictionary first (date -> value)
                date_to_value = {}
                for entry in strategy.daily_values:
                    date_str = entry['date'].strftime('%Y-%m-%d')
                    date_to_value[date_str] = entry['value']
                
                # Create values for all dates
                for date in all_dates:
                    if date in date_to_value:
                        portfolio_values.append(date_to_value[date])
                    else:
                        # For dates without values, use the last known value
                        last_date = max([d for d in date_to_value.keys() if d <= date], default=None)
                        if last_date:
                            portfolio_values.append(date_to_value[last_date])
                        else:
                            portfolio_values.append(initial_cash)
            
            # Method 2: Extract from the observer values
            elif hasattr(strategy, 'observers') and hasattr(strategy.observers, 'value'):
                value_line = strategy.observers.value.lines.value
                actual_values = []
                actual_dates = []
                
                # Extract all values from the observer
                for i in range(len(value_line)):
                    value = float(value_line[i])
                    # Get corresponding date
                    if i < len(strategy.datas[0].datetime):
                        dt = bt.num2date(strategy.datas[0].datetime[i])
                        date_str = dt.strftime('%Y-%m-%d')
                        actual_dates.append(date_str)
                        actual_values.append(value)
                
                # Create a dictionary for quick lookup
                date_to_value = dict(zip(actual_dates, actual_values))
                
                # Fill in values for all dates
                for date in all_dates:
                    if date in date_to_value:
                        portfolio_values.append(date_to_value[date])
                    else:
                        # For dates without values, use the last known value
                        last_date = max([d for d in date_to_value.keys() if d <= date], default=None)
                        if last_date:
                            portfolio_values.append(date_to_value[last_date])
                        else:
                            portfolio_values.append(initial_cash)
            
            # Method 3: Use the strategy's equity_curve if available
            elif hasattr(strategy, 'equity_curve') and strategy.equity_curve:
                # Convert to dictionary first (date -> value)
                date_to_value = {}
                for entry in strategy.equity_curve:
                    date_str = entry.get('Date', '')
                    value = entry.get('Value', initial_cash)
                    if date_str and value:
                        date_to_value[date_str] = value
                
                # Create values for all dates
                for date in all_dates:
                    if date in date_to_value:
                        portfolio_values.append(date_to_value[date])
                    else:
                        # For dates without values, use the last known value
                        last_date = max([d for d in date_to_value.keys() if d <= date], default=None)
                        if last_date:
                            portfolio_values.append(date_to_value[last_date])
                        else:
                            portfolio_values.append(initial_cash)
            
            # Method 4: If no other method is available, use fallback method
            # This recreates the portfolio value using position data and prices
            else:
                print("Warning: No portfolio value observer data found. Reconstructing values.")
                
                # Create a list of initial cash values
                portfolio_values = [initial_cash] * len(all_dates)
                
                # Update values based on trades
                date_to_index = {date: i for i, date in enumerate(all_dates)}
                cash = initial_cash
                positions = {}  # ticker -> quantity
                
                for trade in sorted(trade_log, key=lambda x: x.get('date', '')):
                    date_str = trade.get('date', '')
                    if date_str and date_str in date_to_index:
                        idx = date_to_index[date_str]
                        
                        # Update cash based on trade
                        ticker = trade.get('ticker', 'unknown')
                        price = trade.get('price', 0.0)
                        size = trade.get('size', 0)
                        commission = trade.get('commission', 0.0)
                        
                        # Update position for this ticker
                        if trade.get('type', '') == 'BUY':
                            positions[ticker] = positions.get(ticker, 0) + size
                            cash -= price * size + commission
                        else:  # SELL
                            positions[ticker] = positions.get(ticker, 0) - size
                            cash += price * size - commission
                        
                        # Update all portfolio values from this point forward
                        for i in range(idx, len(all_dates)):
                            # Calculate positions value
                            positions_value = 0
                            for pos_ticker, pos_size in positions.items():
                                # Need to get price for this ticker on this date
                                pos_date = all_dates[i]
                                ticker_df = ticker_data.get(pos_ticker)
                                if ticker_df is not None:
                                    if 'Date' in ticker_df.columns:
                                        price_row = ticker_df[ticker_df['Date'].dt.strftime('%Y-%m-%d') == pos_date]
                else:
                                        price_row = ticker_df[ticker_df.index.strftime('%Y-%m-%d') == pos_date]
                                    
                                    if not price_row.empty:
                                        pos_price = price_row['Close'].iloc[0]
                                        positions_value += pos_size * pos_price
                            
                            # Update portfolio value
                            portfolio_values[i] = cash + positions_value
            
            # Get final portfolio value (cash + positions)
            final_value = strategy.broker.getvalue()
            
            # Print debug info
            print(f"Portfolio values: {len(portfolio_values)} entries, range: {min(portfolio_values):.2f} to {max(portfolio_values):.2f}")
            
            # Create DataFrame with dates and portfolio values
            portfolio_df = pd.DataFrame({
                'Date': all_dates,
                'PortfolioValue': portfolio_values
            })
            
            # Save to CSV
            portfolio_df.to_csv(os.path.join(output_dir, 'portfolio_values.csv'), index=False)
            
            # Calculate metrics
            total_return = (final_value - initial_cash) / initial_cash
            
            # Get metrics from analyzers if available
            sharpe_ratio = getattr(strategy.analyzers.sharpe, 'get_analysis', lambda: {})().get('sharperatio', 0.0)
            max_drawdown = getattr(strategy.analyzers.drawdown, 'get_analysis', lambda: {})().get('max', {}).get('drawdown', 0.0)
            
            # Create a plot of the portfolio value
            try:
                plt.figure(figsize=(12, 6))
                dates_array = np.array(portfolio_df['Date'])
                values_array = np.array(portfolio_df['PortfolioValue'])
                plt.plot(dates_array, values_array)
                plt.title(f'Portfolio Value - {label}')
                plt.xlabel('Date')
                plt.ylabel('Value ($)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'portfolio_value.png'))
                plt.close()
            except Exception as e:
                print(f"Warning: Failed to create portfolio value plot: {e}")
                import traceback
                traceback.print_exc()
            
            # Return the results
            return {
                'initial_value': initial_cash,
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trade_analysis': trade_analysis,
                'portfolio_values': portfolio_df
            }
        
        except Exception as e:
            print(f"Error running backtest: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_metrics(self, backtest_results):
        """
        Calculate performance metrics from backtest results.
        
        Args:
            backtest_results (dict): Results from the backtest
            
        Returns:
            dict: Performance metrics
        """
        if backtest_results is None:
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        try:
            # Extract portfolio values
            portfolio_values = backtest_results.get('portfolio_values', None)
            
            if portfolio_values is None or len(portfolio_values) == 0:
                return {
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0
                }
            
            # Calculate daily returns
            portfolio_values['Return'] = portfolio_values['PortfolioValue'].pct_change(fill_method=None)
            
            # Calculate metrics
            initial_value = 100000.0  # Initial portfolio value
            final_value = backtest_results.get('final_value', initial_value)
            
            # Total return
            if pd.isna(final_value) or final_value <= 0:
                total_return = 0.0
            else:
                total_return = (final_value - initial_value) / initial_value
            
            # Sharpe ratio (annualized)
            daily_returns = portfolio_values['Return'].dropna()
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # Max drawdown
            portfolio_values['Cummax'] = portfolio_values['PortfolioValue'].cummax()
            portfolio_values['Drawdown'] = (portfolio_values['PortfolioValue'] - portfolio_values['Cummax']) / portfolio_values['Cummax']
            max_drawdown = abs(portfolio_values['Drawdown'].min())
            
            # Win rate and profit factor
            trade_analysis = backtest_results.get('trade_analysis', {})
            
            # Total trades
            total_trades = trade_analysis.get('total', {}).get('total', 0)
            
            # Winning trades
            won_trades = trade_analysis.get('won', {}).get('total', 0)
            
            # Win rate
            win_rate = won_trades / total_trades if total_trades > 0 else 0.0
            
            # Gross profit and loss
            gross_won = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0.0)
            gross_lost = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0.0))
            
            # Profit factor
            profit_factor = gross_won / gross_lost if gross_lost > 0 else 0.0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }

    def _analyze_results(self, original_metrics, permutation_metrics):
        """
        Analyze the statistical significance of the original backtest vs permutation tests.
        
        Args:
            original_metrics (dict): Original backtest metrics
            permutation_metrics (list): List of permutation test metrics
            
        Returns:
            dict: P-values for each metric
        """
        if not permutation_metrics:
            return {metric: 1.0 for metric in original_metrics.keys()}
        
        results = {}
        
        for metric in original_metrics.keys():
            original_value = original_metrics[metric]
            permutation_values = [p[metric] for p in permutation_metrics if metric in p]
            
            if not permutation_values:
                results[metric] = 1.0
                continue
            
            # Calculate p-value (proportion of permutation results >= original)
            if original_value > 0:
                # For positive metrics (higher is better), count permutations >= original
                p_value = sum(1 for p in permutation_values if p >= original_value) / len(permutation_values)
            else:
                # For negative metrics (lower is better), count permutations <= original
                p_value = sum(1 for p in permutation_values if p <= original_value) / len(permutation_values)
            
            results[metric] = p_value
        
        return results

    def generate_plots(self, original_metrics, permutation_metrics):
        """Generate plots for the Monte Carlo test results"""
        # Implementation of generate_plots method
        pass

    def extract_metrics(self, results):
        """Extract performance metrics from backtrader results"""
        # Implementation of extract_metrics method
        pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a Direct Monte Carlo Test")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name (SimpleStock or MACrossover)")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of tickers")
    parser.add_argument("--num_permutations", type=int, default=10, help="Number of permutations to run")
    parser.add_argument("--in_sample_start", type=str, default="2016-01-01", help="In-sample start date")
    parser.add_argument("--in_sample_end", type=str, default="2019-12-31", help="In-sample end date")
    parser.add_argument("--out_sample_start", type=str, default="2020-01-01", help="Out-of-sample start date")
    parser.add_argument("--out_sample_end", type=str, default="2021-12-31", help="Out-of-sample end date")
    parser.add_argument("--num_cores", type=int, default=None, help="Number of CPU cores to use")
    parser.add_argument("--enable_in_sample_mc", action="store_true", help="Enable in-sample Monte Carlo testing")
    
    args = parser.parse_args()
    
    # Set up test
    tickers = args.tickers.split(",")
    
    # Run test
    test = DirectMonteCarloTest(
        strategy_name=args.strategy,
        tickers=tickers,
        in_sample_start=args.in_sample_start,
        in_sample_end=args.in_sample_end,
        out_sample_start=args.out_sample_start,
        out_sample_end=args.out_sample_end,
        num_permutations=args.num_permutations,
        enable_in_sample_mc=args.enable_in_sample_mc
    )
    
    results = test.run_test(n_jobs=args.num_cores)
    
    if results:
        print("\nDirect Monte Carlo Test completed successfully.")
    else:
        print("\nDirect Monte Carlo Test failed.")

# Custom JSON encoder to handle pandas Series and other non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

# Save results to JSON with custom encoder
def save_to_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, cls=CustomJSONEncoder) 