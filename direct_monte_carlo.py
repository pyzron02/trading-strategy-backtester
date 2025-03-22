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
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
import csv

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

# Define strategies directly for backtrader to avoid import issues
class SimpleStock(bt.Strategy):
    """
    A simple strategy that buys when price is above SMA and sells when price is below SMA.
    """
    params = (
        ('sma_period', 20),
        ('position_size', 10),
    )

    def __init__(self):
        # Calculate warmup period
        self.warmup_period = self.params.sma_period + 5
        
        # Use standard Backtrader SMA for stability
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.sma_period)
        
        # Keep track of the equity curve for analysis
        self.equity_curve = []
        
        # Variables to track if we have enough data for trading
        self.min_bars_required = max(self.params.sma_period + 10, 30)  # Ensure sufficient warmup
        self.bars_processed = 0

    def next(self):
        # Increment bars processed counter
        self.bars_processed += 1
        
        # Log portfolio value for the equity curve
        date = self.data.datetime.date(0).isoformat()
        value = self.broker.getvalue()
        self.equity_curve.append({'Date': date, 'Value': value})

        # Skip trading until we have enough bars for indicators to be reliable
        if self.bars_processed < self.min_bars_required:
            return

        # Skip if not enough data
        if len(self.data) < self.warmup_period:
            return
            
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
        # Calculate warmup period
        self.warmup_period = max(self.params.slow_period * 2, 50)
        
        # Use standard SMA indicators
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        
        # Use standard CrossOver indicator
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # Keep track of the equity curve for analysis
        self.equity_curve = []
        
        # Variables to track if we have enough data for trading
        self.min_bars_required = self.warmup_period
        self.bars_processed = 0

    def next(self):
        # Increment bars processed counter
        self.bars_processed += 1
        
        # Log portfolio value for the equity curve
        date = self.data.datetime.date(0).isoformat()
        value = self.broker.getvalue()
        self.equity_curve.append({'Date': date, 'Value': value})
        
        # Skip trading until we have enough bars for reliable indicator values
        if self.bars_processed < self.min_bars_required:
            return
        
        # Skip if not enough data
        if len(self.data) < self.warmup_period:
            return
            
        # Get current position size
        position = self.getposition().size
        
        # Trading logic based on crossover
        if self.crossover > 0 and position == 0:
            # Buy signal: crossover is positive
            self.buy(size=self.params.position_size)
        elif self.crossover < 0 and position > 0:
            # Sell signal: crossover is negative
            self.sell(size=position)

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
        # Calculate warmup period
        self.warmup_period = max(self.params.volume_period, self.params.price_period) + 10
        
        # Volume indicators
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.params.volume_period)
        
        # Price indicators
        self.price_ma = bt.indicators.SMA(self.data.close, period=self.params.price_period)
        
        # Keep track of the equity curve for analysis
        self.equity_curve = []
        
        # State for trading logic
        self.bars_processed = 0
    
    def next(self):
        # Increment bars processed counter
        self.bars_processed += 1
        
        # Log portfolio value for the equity curve
        date = self.data.datetime.date(0).isoformat()
        value = self.broker.getvalue()
        self.equity_curve.append({'Date': date, 'Value': value})
        
        # Skip trading until we have enough bars for reliable indicator values
        if self.bars_processed < self.warmup_period:
            return
        
        # Current position
        position = self.getposition().size
        
        # Simple trading logic based on price and volume
        if position == 0:
            # Entry condition: price above MA and volume above average
            if (self.data.close[0] > self.price_ma[0] and 
                self.data.volume[0] > self.volume_ma[0]):
                self.buy(size=self.params.position_size)
        else:
            # Exit condition: price below MA or volume below average
            if (self.data.close[0] < self.price_ma[0] or 
                self.data.volume[0] < self.volume_ma[0] * 0.8):
                self.sell(size=position)


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
        
        # Keep track of the equity curve for analysis
        self.equity_curve = []
        
        # State for trading logic
        self.bars_processed = 0
        self.position_tracker = {}  # Track positions by entry price
        
    def next(self):
        # Increment bars processed counter
        self.bars_processed += 1
        
        # Log portfolio value for the equity curve
        date = self.data.datetime.date(0).isoformat()
        value = self.broker.getvalue()
        self.equity_curve.append({'Date': date, 'Value': value})
        
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
                 output_dir=None, num_permutations=10):
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
        
        # Create data directory for storing processed data
        self.data_dir = os.path.join(self.output_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Log initialization
        print(f"Initialized Direct Monte Carlo Test for {strategy_name}")
        print(f"Tickers: {tickers}")
        print(f"In-sample period: {in_sample_start} to {in_sample_end}")
        print(f"Out-of-sample period: {out_sample_start} to {out_sample_end}")
        print(f"Number of permutations: {num_permutations}")
        print(f"Output directory: {self.output_dir}")
        print(f"Using parameters: {self.parameters}")
    
    def _load_stock_data(self):
        """Load stock data for the specified tickers."""
        print("\nLoading stock data...")
        
        # Get project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Define path to stock data
        data_path = os.path.join(project_root, 'input', 'stock_data.csv')
        
        # Check if file exists
        if not os.path.exists(data_path):
            print(f"Error: Stock data file not found at {data_path}")
            return None
        
        try:
            # Load the data
            data = pd.read_csv(data_path)
            
            # Convert Date column to datetime
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            
            print(f"Loaded stock data from {data_path} with {len(data)} rows and {len(data.columns)} columns.")
            
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
    
    def _permute_data(self, original_df, permutation_type='returns', permutation_seed=None):
        """
        Create a permuted version of the price data.
        
        Args:
            original_df (pd.DataFrame): Original price data with OHLCV columns
            permutation_type (str): Type of permutation - 'returns' or 'blocks'
            permutation_seed (int): Random seed for reproducibility
            
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
        
        # Get the out-of-sample data to permute
        oos_data = df[oos_mask].copy()
        
        if len(oos_data) == 0:
            print("Warning: No out-of-sample data to permute")
            return df
        
        # Permute returns for price columns
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        price_cols = [col for col in price_cols if col in oos_data.columns]
        
        if permutation_type == 'returns':
            # Permute by shuffling returns
            for col in price_cols:
                # Calculate returns
                prices = oos_data[col].values
                returns = np.diff(prices) / prices[:-1]
                
                # Shuffle returns
                shuffled_returns = np.random.permutation(returns)
                
                # Reconstruct prices
                new_prices = np.zeros_like(prices)
                new_prices[0] = prices[0]  # Keep first price
                
                for i in range(1, len(new_prices)):
                    new_prices[i] = new_prices[i-1] * (1 + shuffled_returns[i-1])
                
                # Replace prices with permuted version
                oos_data[col] = new_prices
        
        elif permutation_type == 'blocks':
            # Permute by shuffling blocks of data
            for col in price_cols:
                prices = oos_data[col].values
                
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
                oos_data[col] = new_prices
        
        # Also permute volume if available
        if 'Volume' in oos_data.columns:
            volumes = oos_data['Volume'].values
            permuted_volumes = np.random.permutation(volumes)
            oos_data['Volume'] = permuted_volumes
        
        # Replace out-of-sample data with permuted data
        df.loc[oos_mask] = oos_data
        
        return df
    
    def _run_backtest(self, data, is_permutation=False, permutation_id=None):
        """
        Run a backtest with the given data.
        
        Args:
            data (pd.DataFrame): Price data for backtesting
            is_permutation (bool): Whether this is a permutation test
            permutation_id (int): ID of the permutation if applicable
            
        Returns:
            dict: Backtest results
        """
        # Create a cerebro instance
        cerebro = bt.Cerebro()
        
        # Set initial cash
        cerebro.broker.setcash(100000.0)
        
        # Add data feed for each ticker
        for ticker, df in data.items():
            # Save data to CSV for backtrader
            csv_path = os.path.join(self.data_dir, f"{ticker}_{'permuted' if is_permutation else 'original'}.csv")
            df.to_csv(csv_path, index=False)
            
            # Create a data feed
            data_feed = bt.feeds.GenericCSVData(
                dataname=csv_path,
                dtformat='%Y-%m-%d',
                datetime=0,  # Date column index
                open=1,
                high=2,
                low=3,
                close=4,
                volume=5,
                openinterest=-1,  # No open interest column
                fromdate=self.in_sample_start.to_pydatetime(),
                todate=self.out_sample_end.to_pydatetime()
            )
            
            # Add data feed
            cerebro.adddata(data_feed, name=ticker)
        
        # Add the strategy
        if self.strategy_name in STRATEGY_CLASSES:
            cerebro.addstrategy(STRATEGY_CLASSES[self.strategy_name], **self.parameters)
        else:
            print(f"Error: Unknown strategy {self.strategy_name}")
            return None
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Run the backtest
        print(f"Running {'permutation' if is_permutation else 'original'} backtest...")
        results = cerebro.run()
        
        if not results:
            print("Error: No results from backtest")
            return None
        
        # Extract results
        strat = results[0]
        
        # Get analyzer results
        sharpe_ratio = strat.analyzers.sharpe.get_analysis()['sharperatio'] if hasattr(strat.analyzers.sharpe, 'get_analysis') else 0.0
        returns = strat.analyzers.returns.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades_analysis = strat.analyzers.trades.get_analysis()
        
        # Calculate key metrics
        total_return = returns['rtot'] if 'rtot' in returns else 0.0
        max_drawdown = drawdown['max']['drawdown'] if 'max' in drawdown else 0.0
        
        # Get trade stats
        total_trades = trades_analysis.total.closed if hasattr(trades_analysis, 'total') else 0
        
        # Calculate win rate
        if hasattr(trades_analysis, 'won') and hasattr(trades_analysis, 'lost'):
            win_trades = trades_analysis.won.total if hasattr(trades_analysis.won, 'total') else 0
            lose_trades = trades_analysis.lost.total if hasattr(trades_analysis.lost, 'total') else 0
            win_rate = win_trades / total_trades if total_trades > 0 else 0.0
        else:
            win_rate = 0.0
            
        # Calculate profit factor
        if hasattr(trades_analysis, 'won') and hasattr(trades_analysis, 'lost'):
            # Handle AutoOrderedDict properly
            if hasattr(trades_analysis.won, 'pnl'):
                if isinstance(trades_analysis.won.pnl, (int, float)):
                    total_won = trades_analysis.won.pnl
                else:
                    # Try to convert the AutoOrderedDict to float
                    try:
                        total_won = float(trades_analysis.won.pnl)
                    except:
                        total_won = 0.0
            else:
                total_won = 0.0
                
            if hasattr(trades_analysis.lost, 'pnl'):
                if isinstance(trades_analysis.lost.pnl, (int, float)):
                    total_lost = abs(trades_analysis.lost.pnl)
                else:
                    # Try to convert the AutoOrderedDict to float
                    try:
                        total_lost = abs(float(trades_analysis.lost.pnl))
                    except:
                        total_lost = 0.0
            else:
                total_lost = 0.0
                
            profit_factor = total_won / total_lost if total_lost > 0 else 0.0
        else:
            profit_factor = 0.0
        
        # Get equity curve
        equity_curve = strat.equity_curve if hasattr(strat, 'equity_curve') else []
        
        # Store results in a dictionary
        result_dict = {
            'sharpe_ratio': float(sharpe_ratio),
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'total_trades': int(total_trades),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'equity_curve': equity_curve,
            'final_value': float(cerebro.broker.getvalue())
        }
        
        # Save the results
        if is_permutation:
            suffix = f"permutation_{permutation_id}"
        else:
            suffix = "original"
        
        results_dir = os.path.join(self.output_dir, suffix)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save trade details to CSV
        self._save_trade_log(trades_analysis, results_dir, suffix)
        
        with open(os.path.join(results_dir, "results.json"), "w") as f:
            # Convert equity curve to list of dicts for JSON serialization
            serializable_results = result_dict.copy()
            serializable_results['equity_curve'] = [
                {'date': str(p.get('Date', '')), 'value': float(p.get('Value', 0.0))} 
                for p in result_dict['equity_curve']
            ]
            json.dump(serializable_results, f, indent=4)
        
        print(f"Backtest results for {suffix}:")
        print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"  Total Return: {total_return:.4f}")
        print(f"  Max Drawdown: {max_drawdown:.4f}")
        print(f"  Win Rate: {win_rate:.4f}")
        print(f"  Profit Factor: {profit_factor:.4f}")
        print(f"  Final Value: ${cerebro.broker.getvalue():.2f}")
        print(f"  Trades logged to: {os.path.join(results_dir, f'trade_log_{suffix}.csv')}")
        
        return result_dict
    
    def _save_trade_log(self, trades_analysis, results_dir, suffix):
        """
        Save detailed trade information to CSV.
        
        Args:
            trades_analysis: The TradeAnalyzer results
            results_dir: Directory to save results
            suffix: Suffix for the file name
        """
        trade_log_path = os.path.join(results_dir, f"trade_log_{suffix}.csv")
        
        # Extract trade information
        trade_data = []
        
        # Check if we have trade details to log
        if hasattr(trades_analysis, 'total') and trades_analysis.total.total > 0:
            # We can't directly access individual trades from TradeAnalyzer
            # Instead, we'll log the summary statistics by trade type
            
            # Header row
            header = ['Type', 'Direction', 'Count', 'PnL Total', 'PnL Avg', 'PnL Max', 'Length Avg', 'Length Max']
            trade_data.append(header)
            
            # Overall statistics
            trade_data.append(['All', 'All', 
                             str(trades_analysis.total.total), 
                             str(getattr(trades_analysis, 'pnl', {}).get('total', 0.0)),
                             str(getattr(trades_analysis, 'pnl', {}).get('average', 0.0)),
                             str(getattr(trades_analysis, 'pnl', {}).get('max', 0.0)),
                             str(getattr(trades_analysis, 'len', {}).get('average', 0)),
                             str(getattr(trades_analysis, 'len', {}).get('max', 0))])
            
            # Winning trades
            if hasattr(trades_analysis, 'won') and trades_analysis.won.total > 0:
                trade_data.append(['Won', 'All', 
                                 str(trades_analysis.won.total), 
                                 str(getattr(trades_analysis.won, 'pnl', {}).get('total', 0.0)),
                                 str(getattr(trades_analysis.won, 'pnl', {}).get('average', 0.0)),
                                 str(getattr(trades_analysis.won, 'pnl', {}).get('max', 0.0)),
                                 str(getattr(trades_analysis.won, 'len', {}).get('average', 0)),
                                 str(getattr(trades_analysis.won, 'len', {}).get('max', 0))])
            
            # Losing trades
            if hasattr(trades_analysis, 'lost') and trades_analysis.lost.total > 0:
                trade_data.append(['Lost', 'All', 
                                 str(trades_analysis.lost.total), 
                                 str(getattr(trades_analysis.lost, 'pnl', {}).get('total', 0.0)),
                                 str(getattr(trades_analysis.lost, 'pnl', {}).get('average', 0.0)),
                                 str(getattr(trades_analysis.lost, 'pnl', {}).get('max', 0.0)),
                                 str(getattr(trades_analysis.lost, 'len', {}).get('average', 0)),
                                 str(getattr(trades_analysis.lost, 'len', {}).get('max', 0))])
                
            # Long trades
            if hasattr(trades_analysis, 'long') and trades_analysis.long.total > 0:
                trade_data.append(['All', 'Long', 
                                 str(trades_analysis.long.total), 
                                 str(getattr(trades_analysis.long, 'pnl', {}).get('total', 0.0)),
                                 str(getattr(trades_analysis.long, 'pnl', {}).get('average', 0.0)),
                                 str(getattr(trades_analysis.long, 'pnl', {}).get('max', 0.0)),
                                 str(getattr(trades_analysis.long, 'len', {}).get('average', 0)),
                                 str(getattr(trades_analysis.long, 'len', {}).get('max', 0))])
                
                # Long winning trades
                if hasattr(trades_analysis.long, 'won') and trades_analysis.long.won.total > 0:
                    trade_data.append(['Won', 'Long', 
                                     str(trades_analysis.long.won.total), 
                                     str(getattr(trades_analysis.long.won, 'pnl', {}).get('total', 0.0)),
                                     str(getattr(trades_analysis.long.won, 'pnl', {}).get('average', 0.0)),
                                     str(getattr(trades_analysis.long.won, 'pnl', {}).get('max', 0.0)),
                                     str(getattr(trades_analysis.long.won, 'len', {}).get('average', 0)),
                                     str(getattr(trades_analysis.long.won, 'len', {}).get('max', 0))])
                
                # Long losing trades
                if hasattr(trades_analysis.long, 'lost') and trades_analysis.long.lost.total > 0:
                    trade_data.append(['Lost', 'Long', 
                                     str(trades_analysis.long.lost.total), 
                                     str(getattr(trades_analysis.long.lost, 'pnl', {}).get('total', 0.0)),
                                     str(getattr(trades_analysis.long.lost, 'pnl', {}).get('average', 0.0)),
                                     str(getattr(trades_analysis.long.lost, 'pnl', {}).get('max', 0.0)),
                                     str(getattr(trades_analysis.long.lost, 'len', {}).get('average', 0)),
                                     str(getattr(trades_analysis.long.lost, 'len', {}).get('max', 0))])
            
            # Short trades
            if hasattr(trades_analysis, 'short') and trades_analysis.short.total > 0:
                trade_data.append(['All', 'Short', 
                                 str(trades_analysis.short.total), 
                                 str(getattr(trades_analysis.short, 'pnl', {}).get('total', 0.0)),
                                 str(getattr(trades_analysis.short, 'pnl', {}).get('average', 0.0)),
                                 str(getattr(trades_analysis.short, 'pnl', {}).get('max', 0.0)),
                                 str(getattr(trades_analysis.short, 'len', {}).get('average', 0)),
                                 str(getattr(trades_analysis.short, 'len', {}).get('max', 0))])
                
                # Short winning trades
                if hasattr(trades_analysis.short, 'won') and trades_analysis.short.won.total > 0:
                    trade_data.append(['Won', 'Short', 
                                     str(trades_analysis.short.won.total), 
                                     str(getattr(trades_analysis.short.won, 'pnl', {}).get('total', 0.0)),
                                     str(getattr(trades_analysis.short.won, 'pnl', {}).get('average', 0.0)),
                                     str(getattr(trades_analysis.short.won, 'pnl', {}).get('max', 0.0)),
                                     str(getattr(trades_analysis.short.won, 'len', {}).get('average', 0)),
                                     str(getattr(trades_analysis.short.won, 'len', {}).get('max', 0))])
                
                # Short losing trades
                if hasattr(trades_analysis.short, 'lost') and trades_analysis.short.lost.total > 0:
                    trade_data.append(['Lost', 'Short', 
                                     str(trades_analysis.short.lost.total), 
                                     str(getattr(trades_analysis.short.lost, 'pnl', {}).get('total', 0.0)),
                                     str(getattr(trades_analysis.short.lost, 'pnl', {}).get('average', 0.0)),
                                     str(getattr(trades_analysis.short.lost, 'pnl', {}).get('max', 0.0)),
                                     str(getattr(trades_analysis.short.lost, 'len', {}).get('average', 0)),
                                     str(getattr(trades_analysis.short.lost, 'len', {}).get('max', 0))])
        else:
            # No trades executed
            header = ['Type', 'Direction', 'Count', 'PnL Total', 'PnL Avg', 'PnL Max', 'Length Avg', 'Length Max']
            trade_data.append(header)
            trade_data.append(['No trades', 'N/A', '0', '0.0', '0.0', '0.0', '0', '0'])
        
        # Write to CSV
        with open(trade_log_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for row in trade_data:
                csv_writer.writerow(row)
        
        return trade_log_path
    
    def _analyze_results(self, original_results, permutation_results):
        """
        Analyze the results of the Monte Carlo test.
        
        Args:
            original_results (dict): Results from original backtest
            permutation_results (list): List of results from permutation tests
            
        Returns:
            dict: Analysis of results
        """
        # Create a results directory
        analysis_dir = os.path.join(self.output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Collect metrics from permutations
        metrics = {
            'sharpe_ratio': [],
            'total_return': [],
            'max_drawdown': [],
            'win_rate': [],
            'profit_factor': []
        }
        
        for perm_result in permutation_results:
            for metric in metrics:
                if metric in perm_result:
                    metrics[metric].append(perm_result[metric])
        
        # Calculate p-values
        p_values = {}
        
        # For positive metrics (higher is better), p-value is proportion of permutations >= original
        for metric in ['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor']:
            original_value = original_results.get(metric, 0.0)
            values = metrics[metric]
            
            if values:
                # Count how many permutations have value >= original
                count = sum(1 for v in values if v >= original_value)
                p_values[metric] = count / len(values)
            else:
                p_values[metric] = None
        
        # For negative metrics (lower is better), p-value is proportion of permutations <= original
        for metric in ['max_drawdown']:
            original_value = original_results.get(metric, 0.0)
            values = metrics[metric]
            
            if values:
                # Count how many permutations have value <= original
                count = sum(1 for v in values if v <= original_value)
                p_values[metric] = count / len(values)
            else:
                p_values[metric] = None
        
        # Calculate statistics for each metric
        stats = {}
        for metric, values in metrics.items():
            if values:
                stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'p_value': p_values.get(metric, None),
                    'original': float(original_results.get(metric, 0.0))
                }
            else:
                stats[metric] = {
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'p_value': None,
                    'original': float(original_results.get(metric, 0.0))
                }
        
        # Save analysis
        with open(os.path.join(analysis_dir, "monte_carlo_analysis.json"), "w") as f:
            json.dump({
                'metrics': stats,
                'original_results': original_results,
                'num_permutations': len(permutation_results)
            }, f, indent=4)
        
        # Generate plots
        self._generate_plots(stats, analysis_dir)
        
        # Return the analysis
        return {
            'metrics': stats,
            'p_values': p_values
        }
    
    def _generate_plots(self, stats, output_dir):
        """
        Generate plots for the Monte Carlo analysis.
        
        Args:
            stats (dict): Statistics for each metric
            output_dir (str): Directory to save plots
        """
        for metric, metric_stats in stats.items():
            if metric_stats['mean'] is None:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Get values to plot
            values = []
            for perm_id in range(self.num_permutations):
                perm_results_path = os.path.join(self.output_dir, f"permutation_{perm_id+1}", "results.json")
                if os.path.exists(perm_results_path):
                    try:
                        with open(perm_results_path, "r") as f:
                            perm_results = json.load(f)
                            if metric in perm_results:
                                values.append(perm_results[metric])
                    except:
                        pass
            
            # Plot histogram of permutation values
            if values:
                plt.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                
                # Add original value as vertical line
                original_value = metric_stats['original']
                plt.axvline(x=original_value, color='red', linestyle='--', 
                            label=f'Original: {original_value:.4f}')
                
                # Add p-value
                p_value = metric_stats['p_value']
                if p_value is not None:
                    plt.text(0.05, 0.95, f'p-value: {p_value:.4f}', transform=plt.gca().transAxes,
                            fontsize=12, verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Format plot
                plt.title(f'Distribution of {metric.replace("_", " ").title()}')
                plt.xlabel(metric.replace("_", " ").title())
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Save plot
                plt.savefig(os.path.join(output_dir, f"{metric}_distribution.png"), dpi=120, bbox_inches='tight')
                plt.close()
    
    def run_test(self):
        """
        Run the Monte Carlo test.
        
        Returns:
            dict: Test results and analysis
        """
        print("\nStarting Direct Monte Carlo Test...\n")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save test parameters
        self.save_test_parameters()
        
        # Load and process stock data
        ticker_data = self._load_stock_data()
        if not ticker_data:
            print("Error: Failed to load stock data")
            return None
        
        # Run original backtest
        print("\nRunning original backtest...")
        original_results = self._run_backtest(ticker_data, is_permutation=False)
        
        if not original_results:
            print("Error: Original backtest failed")
            return None
        
        # Run permutation tests
        print(f"\nRunning {self.num_permutations} permutation tests...")
        permutation_results = []
        
        try:
            # Use tqdm for progress tracking
            for i in tqdm(range(self.num_permutations)):
                permutation_id = i + 1
                permutation_seed = random.randint(10000, 99999) + permutation_id
                
                # Permute the data
                permuted_data = {}
                for ticker, df in ticker_data.items():
                    permuted_df = self._permute_data(df, permutation_type='returns', permutation_seed=permutation_seed)
                    permuted_data[ticker] = permuted_df
                
                # Run backtest on permuted data
                perm_results = self._run_backtest(permuted_data, is_permutation=True, permutation_id=permutation_id)
                
                if perm_results:
                    permutation_results.append(perm_results)
                else:
                    print(f"Warning: Permutation {permutation_id} failed")
                
        except Exception as e:
            print(f"Error during permutation tests: {e}")
            import traceback
            traceback.print_exc()
        
        # Analyze results
        if permutation_results:
            print(f"\nAnalyzing results of {len(permutation_results)} permutation tests...")
            analysis = self._analyze_results(original_results, permutation_results)
            
            # Print p-values
            print("\nMonte Carlo Test Results:")
            print("-" * 50)
            
            if 'p_values' in analysis:
                for metric, p_value in analysis['p_values'].items():
                    if p_value is not None:
                        significance = "Significant" if p_value < 0.05 else "Not significant"
                        print(f"{metric.replace('_', ' ').title()}: p-value = {p_value:.4f} ({significance})")
            
            print("\nTest completed successfully!")
            return {
                'original_results': original_results,
                'permutation_results': permutation_results,
                'analysis': analysis,
                'output_dir': self.output_dir
            }
        else:
            print("Error: No valid permutation results")
            return None

    def save_result_metrics(self, metrics, prefix=''):
        """Save metrics to a JSON file"""
        filename = f"{prefix}metrics.json" if prefix else "metrics.json"
        filepath = os.path.join(self.output_dir, filename)
        save_to_json(metrics, filepath)
        print(f"Metrics saved to {filepath}")
        
    def save_analysis_results(self, analysis_results):
        """Save analysis results to a JSON file"""
        filepath = os.path.join(self.output_dir, "analysis", "monte_carlo_analysis.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_to_json(analysis_results, filepath)
        print(f"Analysis results saved to {filepath}")
        
    def save_test_parameters(self):
        """Save test parameters to a JSON file"""
        parameters = {
            'strategy_name': self.strategy_name,
            'tickers': self.tickers,
            'in_sample_start': self.in_sample_start,
            'in_sample_end': self.in_sample_end,
            'out_sample_start': self.out_sample_start,
            'out_sample_end': self.out_sample_end,
            'parameters': self.parameters,
            'num_permutations': self.num_permutations
        }
        filepath = os.path.join(self.output_dir, "test_parameters.json")
        save_to_json(parameters, filepath)
        print(f"Test parameters saved to {filepath}")

    def run_backtest_with_data(self, data, output_subdir, label="backtest"):
        """Run a backtest with the given data"""
        try:
            # Create output subdirectory
            output_dir = os.path.join(self.output_dir, output_subdir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Get data for the in-sample period
            in_sample_data = data[(data.index >= self.in_sample_start) & 
                                 (data.index <= self.in_sample_end)]
            
            if len(in_sample_data) == 0:
                print(f"Error: No data for in-sample period {self.in_sample_start} to {self.in_sample_end}")
                return None
            
            # Create cerebro instance
            cerebro = self.initialize_cerebro(in_sample_data, output_dir)
            
            # Run the backtest
            print(f"Running {label}...")
            results = cerebro.run()
            
            if not results:
                print(f"Error: No results from {label}")
                return None
            
            # Extract performance metrics
            metrics = self.extract_metrics(results[0])
            
            # Save metrics
            self.save_result_metrics(metrics, f"{output_subdir}_")
            
            # Print results
            print(f"Backtest results for {label}:")
            for metric, value in metrics.items():
                if metric in ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'profit_factor']:
                    print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            print(f"  Final Value: ${results[0].broker.getvalue():.2f}")
            
            return metrics
        except Exception as e:
            print(f"Error running {label}: {e}")
            traceback.print_exc()
            return None

    def analyze_results(self, original_metrics, permutation_metrics):
        """Analyze the results of the monte carlo test"""
        # Calculate mean and standard deviation for each metric
        analysis = {
            'metrics': {}
        }
        
        for metric in ['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor', 'max_drawdown']:
            # Extract values from permutation metrics
            values = [p.get(metric, 0) for p in permutation_metrics if p is not None and p.get(metric) is not None]
            
            if not values:
                continue
                
            # Calculate summary statistics
            analysis['metrics'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p_value': None,
                'original': original_metrics.get(metric, None)
            }
            
            # Calculate p-value
            original_value = original_metrics.get(metric, None)
            if original_value is not None:
                if metric == 'max_drawdown':
                    # For max_drawdown, lower is better
                    p_value = np.mean([1 if x <= original_value else 0 for x in values])
                else:
                    # For other metrics, higher is better
                    p_value = np.mean([1 if x >= original_value else 0 for x in values])
                analysis['metrics'][metric]['p_value'] = p_value
        
        # Extract p-values for easy reference
        analysis['p_values'] = {
            metric: analysis['metrics'][metric]['p_value'] 
            for metric in analysis['metrics']
            if 'p_value' in analysis['metrics'][metric]
        }
        
        # Print summary of results
        print("\nMonte Carlo Test Results:")
        print("--------------------------------------------------")
        for metric, info in analysis['metrics'].items():
            if 'p_value' in info and info['p_value'] is not None:
                significant = "Significant" if info['p_value'] < 0.05 else "Not significant"
                print(f"{metric.replace('_', ' ').title()}: p-value = {info['p_value']:.4f} ({significant})")
        
        # Save analysis results
        self.save_analysis_results(analysis)
        
        # Generate plots if we have enough data
        if len(permutation_metrics) > 1:
            self.generate_plots(original_metrics, permutation_metrics)
        
        return analysis

    def initialize_cerebro(self, data, output_dir):
        """Initialize a backtrader cerebro instance with the strategy"""
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0)
        
        # Add the data
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)
        
        # Add the strategy
        if self.strategy_name not in STRATEGY_CLASSES:
            print(f"Error: Unknown strategy {self.strategy_name}")
            return None
            
        strategy_class = STRATEGY_CLASSES[self.strategy_name]
        cerebro.addstrategy(strategy_class, **self.parameters)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        
        # Add observers
        cerebro.addobserver(bt.observers.Broker)
        cerebro.addobserver(bt.observers.Trades)
        cerebro.addobserver(bt.observers.BuySell)
        
        # Configure output directory for this run
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        return cerebro


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
        num_permutations=args.num_permutations
    )
    
    results = test.run_test()
    
    if results:
        print(f"\nDirect Monte Carlo Test completed successfully.")
        print(f"Results and analysis saved to: {results['output_dir']}")
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