#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trade-Based Monte Carlo Simulation for Out-of-Sample Testing

This module implements Monte Carlo simulation by permuting stock data and running backtests
using the run_backtest.py engine. This approach tests the robustness of trading strategies
by analyzing the distribution of performance metrics across multiple permuted datasets.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
import sys
import warnings
import shutil
import time

# Add the src directory to the path to enable importing from strategies
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import strategy registry
from strategies import registry

# Import statistics and visualization tools
from scipy import stats

# Import run_backtest engine
from engine.run_backtest import run_backtest, run_parallel_backtests


# Custom JSON encoder to handle NumPy types and other non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif hasattr(obj, 'dtype'):  # Handle other NumPy types
            return obj.item()
        return super(CustomJSONEncoder, self).default(obj)


class TradeBasedMonteCarloTest:
    """
    Implements Monte Carlo simulation by permuting stock data and running backtests.
    
    This class leverages the run_backtest.py engine to run backtests on the permuted data,
    then analyzes the distribution of performance metrics to assess strategy robustness.
    """
    
    def __init__(
        self,
        strategy_name: str,
        parameters: Dict[str, Any],
        tickers: List[str],
        input_dir: str = "input",
        output_dir: str = None,
        num_simulations: int = 1000,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        data_format: str = "standard",
        seed: int = None,
        verbose: bool = False,
        num_workers: int = None,
        keep_permuted_data: bool = False
    ):
        """
        Initialize the Trade-Based Monte Carlo Test.
        
        Args:
            strategy_name (str): Name of the strategy to test
            parameters (dict): Strategy parameters (optimized)
            tickers (list): List of ticker symbols
            input_dir (str): Directory containing input data
            output_dir (str): Directory to save output
            num_simulations (int): Number of Monte Carlo simulations
            initial_capital (float): Initial capital for the backtest
            commission (float): Commission rate for trades
            data_format (str): Format of input data ('standard', 'custom', etc.)
            seed (int): Random seed for reproducibility
            verbose (bool): Whether to print verbose output
            num_workers (int): Number of CPU cores to use for parallel processing.
                              If None, uses all cores except one.
            keep_permuted_data (bool): Whether to save the permuted stock data files after simulation.
        """
        self.strategy_name = strategy_name
        self.parameters = parameters
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.input_dir = input_dir
        self.commission = commission
        self.data_format = data_format
        self.original_stock_csv = None
        self.num_workers = num_workers
        self.keep_permuted_data = keep_permuted_data
        
        # Create output directory if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join("output", f"{self.strategy_name}_trade_monte_carlo_{timestamp}")
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create directory for permuted data
        self.permuted_data_dir = os.path.join(self.output_dir, "permuted_data")
        os.makedirs(self.permuted_data_dir, exist_ok=True)
        
        self.num_simulations = num_simulations
        self.initial_capital = initial_capital
        self.verbose = verbose
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Store original performance metrics
        self.original_metrics = None
        
        # Store Monte Carlo simulation results
        self.simulated_metrics = []
        
        # Placeholders for analysis results
        self.analysis_results = {}
        
        # Get strategy info from registry if available
        self.strategy_info = {}
        try:
            self.strategy_info = registry.get_strategy_info(self.strategy_name) or {}
        except:
            pass
        
        # Print initialization information
        if self.verbose:
            print(f"Initialized Trade-Based Monte Carlo Test for {strategy_name} strategy")
            print(f"Parameters: {parameters}")
            print(f"Tickers: {tickers}")
            print(f"Number of simulations: {num_simulations}")
            if num_workers is not None:
                print(f"Using {num_workers} CPU cores for parallel processing")
            else:
                print(f"Using automatic parallel processing settings")
        
        # Always print the keep_permuted_data setting to help debug
        print(f"Keep permuted data: {self.keep_permuted_data} (This will {'save' if self.keep_permuted_data else 'delete'} permuted data files)")
        if self.keep_permuted_data:
            print(f"Permuted data will be saved in: {self.permuted_data_dir}")
        else:
            print("Permuted data files will be deleted after simulation")
    
    @classmethod
    def from_trade_log(cls, trade_log, initial_capital=100000.0, n_iterations=1000, output_dir=None, 
                     strategy_name=None, parameters=None, tickers=None, verbose=False, keep_permuted_data=False):
        """
        Alternative constructor that creates a Monte Carlo simulation from a trade log.
        
        Args:
            trade_log (list or DataFrame): List of trade records from a backtest
            initial_capital (float): Initial capital for the simulation
            n_iterations (int): Number of Monte Carlo iterations
            output_dir (str): Directory to save simulation results
            strategy_name (str, optional): Name of the strategy, will be extracted from trade log if not provided
            parameters (dict, optional): Strategy parameters
            tickers (list, optional): List of ticker symbols
            verbose (bool): Whether to print detailed output
            keep_permuted_data (bool): Whether to keep permuted data files
            
        Returns:
            TradeBasedMonteCarloTest: Instance configured for trade log simulations
        """
        # Extract strategy name from the trade log if not provided
        if strategy_name is None:
            strategy_name = "Unknown"
            if isinstance(trade_log, pd.DataFrame) and 'strategy' in trade_log.columns and len(trade_log) > 0:
                strategy_name = trade_log['strategy'].iloc[0]
        
        # Use provided tickers or default
        if tickers is None:
            tickers = ["SPY"]  # Default ticker, not used for trade log simulation
        
        # Use provided parameters or empty dict
        if parameters is None:
            parameters = {}
            
        # Create a basic instance with minimal required information
        instance = cls(
            strategy_name=strategy_name,
            parameters=parameters,
            tickers=tickers,
            output_dir=output_dir,
            num_simulations=n_iterations,
            initial_capital=initial_capital,
            verbose=verbose,
            keep_permuted_data=keep_permuted_data
        )
        
        # Store the trade log directly
        instance.trade_log = trade_log
        
        # Set simulation mode to trade-based (rather than bootstrapping stock data)
        instance.simulation_mode = "trade_log"
        
        return instance
    
    def run(self, progress_callback=None):
        """
        Run the Monte Carlo simulation.
        
        Args:
            progress_callback: Optional callback function to update progress
            
        Returns:
            dict: Dictionary of simulation results
        """
        if hasattr(self, 'simulation_mode') and self.simulation_mode == "trade_log":
            return self._run_trade_log_simulation(progress_callback)
        else:
            # Call the original test method for stock data permutation
            return self.run_test(None)  # No need for out_of_sample_start in this case
    
    def _run_trade_log_simulation(self, progress_callback=None):
        """
        Run Monte Carlo simulation using the trade log.
        
        Args:
            progress_callback: Optional callback function to update progress
            
        Returns:
            dict: Dictionary of simulation results
        """
        if not hasattr(self, 'trade_log') or self.trade_log is None:
            raise ValueError("No trade log available for simulation")
        
        print(f"Running trade-based Monte Carlo simulation with {self.num_simulations} iterations")
        
        # Convert trade log to DataFrame if it's not already
        if not isinstance(self.trade_log, pd.DataFrame):
            trade_log_df = pd.DataFrame(self.trade_log)
        else:
            trade_log_df = self.trade_log.copy()
        
        # Extract key statistics from trade log
        trade_stats = self._extract_trade_statistics(trade_log_df)
        
        # Store original metrics
        self.original_metrics = {
            'total_return': trade_stats.get('total_return', 0),
            'sharpe_ratio': trade_stats.get('sharpe_ratio', 0),
            'max_drawdown': trade_stats.get('max_drawdown', 0),
            'win_rate': trade_stats.get('win_rate', 0)
        }
        
        # Run simulations
        all_results = []
        for i in range(self.num_simulations):
            if progress_callback:
                progress_callback(i, self.num_simulations)
            
            # Generate simulated trade outcomes by bootstrapping
            sim_results = self._simulate_trades(trade_log_df)
            all_results.append(sim_results)
        
        # Calculate summary statistics
        self.simulated_metrics = all_results
        
        # Calculate summary stats
        metrics_keys = all_results[0].keys()
        summary = {}
        
        for key in metrics_keys:
            values = [result[key] for result in all_results]
            summary[f"median_{key}"] = np.median(values)
            summary[f"mean_{key}"] = np.mean(values)
            summary[f"std_{key}"] = np.std(values)
            summary[f"min_{key}"] = np.min(values)
            summary[f"max_{key}"] = np.max(values)
        
        # Save summary to output directory
        results_file = os.path.join(self.output_dir, "monte_carlo_results.json")
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=4, cls=CustomJSONEncoder)
        
        print(f"Monte Carlo simulation complete. Results saved to {results_file}")
        return summary
    
    def _extract_trade_statistics(self, trade_log_df):
        """Extract key statistics from the trade log"""
        stats = {}
        
        # Basic counts
        trades_count = len(trade_log_df)
        if trades_count == 0:
            return stats
        
        # Calculate P&L if it exists
        if 'pnl' in trade_log_df.columns:
            # Win rate
            winning_trades = trade_log_df[trade_log_df['pnl'] > 0]
            stats['win_rate'] = len(winning_trades) / trades_count if trades_count > 0 else 0
            
            # Profit factor
            total_gain = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            losing_trades = trade_log_df[trade_log_df['pnl'] <= 0]
            total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            stats['profit_factor'] = total_gain / total_loss if total_loss > 0 else float('inf')
            
            # Average win/loss
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            stats['avg_win_loss_ratio'] = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            # Total return
            stats['total_return'] = trade_log_df['pnl'].sum() / self.initial_capital
            
            # Calculate equity curve if datetime is available
            if 'datetime' in trade_log_df.columns:
                # Sort by datetime
                sorted_trades = trade_log_df.sort_values('datetime')
                
                # Create cumulative equity curve
                equity = np.cumsum(sorted_trades['pnl'].values)
                equity = self.initial_capital + equity
                
                # Calculate maximum drawdown
                peak = np.maximum.accumulate(equity)
                drawdown = (equity - peak) / peak
                stats['max_drawdown'] = abs(min(drawdown)) if len(drawdown) > 0 else 0
                
                # Calculate Sharpe ratio if we have enough trades
                if len(equity) > 1:
                    returns = np.diff(equity) / equity[:-1]
                    annualized_return = (equity[-1] / equity[0]) ** (252 / len(equity)) - 1
                    annualized_volatility = np.std(returns) * np.sqrt(252)
                    stats['sharpe_ratio'] = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        return stats
    
    def _simulate_trades(self, trade_log_df):
        """
        Simulate trades by bootstrapping from the trade log
        
        Args:
            trade_log_df: DataFrame containing trade records
            
        Returns:
            dict: Simulated performance metrics
        """
        # Extract PnL from trades
        if 'pnl' not in trade_log_df.columns:
            raise ValueError("Trade log must contain 'pnl' column for simulation")
        
        # Get number of trades
        n_trades = len(trade_log_df)
        
        # Bootstrap trades (sample with replacement)
        np.random.seed(int(time.time() * 1000) % 100000)  # Use current time for seeding
        sampled_indices = np.random.choice(n_trades, n_trades, replace=True)
        sampled_trades = trade_log_df.iloc[sampled_indices]
        
        # Calculate equity curve
        equity = np.cumsum(sampled_trades['pnl'].values)
        equity = self.initial_capital + equity
        
        # Calculate metrics
        metrics = {}
        
        # Total return
        metrics['final_equity'] = equity[-1] if len(equity) > 0 else self.initial_capital
        metrics['total_return'] = (equity[-1] - self.initial_capital) / self.initial_capital if len(equity) > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        metrics['max_drawdown'] = abs(min(drawdown)) if len(drawdown) > 0 else 0
        
        # Sharpe ratio
        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            annualized_return = (equity[-1] / equity[0]) ** (252 / len(equity)) - 1
            annualized_volatility = np.std(returns) * np.sqrt(252)
            metrics['sharpe_ratio'] = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        else:
            metrics['sharpe_ratio'] = 0
        
        # Win rate
        if 'pnl' in sampled_trades.columns:
            winning_trades = sampled_trades[sampled_trades['pnl'] > 0]
            metrics['win_rate'] = len(winning_trades) / len(sampled_trades) if len(sampled_trades) > 0 else 0
        
        return metrics
    
    def calculate_confidence_intervals(self, confidence=0.95):
        """
        Calculate confidence intervals for the Monte Carlo simulation results.
        
        Args:
            confidence: Confidence level (default: 0.95 for 95% confidence)
            
        Returns:
            dict: Dictionary of confidence intervals for each metric
        """
        if not self.simulated_metrics:
            return {}
        
        # Get all available metrics
        metrics = self.simulated_metrics[0].keys()
        
        # Calculate confidence intervals
        intervals = {}
        for metric in metrics:
            values = [result[metric] for result in self.simulated_metrics]
            lower_percentile = (1 - confidence) / 2 * 100
            upper_percentile = (1 - (1 - confidence) / 2) * 100
            lower = np.percentile(values, lower_percentile)
            upper = np.percentile(values, upper_percentile)
            intervals[metric] = (lower, upper)
        
        return intervals
    
    def _find_stock_data_csv(self):
        """
        Find the stock_data.csv file in the input directory.
        
        Returns:
            str: Path to the stock_data.csv file
        """
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Define potential paths
        potential_paths = [
            os.path.join(self.input_dir, "stock_data.csv"),
            os.path.join(project_root, "input", "stock_data.csv"),
            os.path.join(project_root, "data", "stock_data.csv"),
            os.path.join("input", "stock_data.csv")
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                if self.verbose:
                    print(f"Found stock data at: {path}")
                return path
        
        raise FileNotFoundError(f"Could not find stock_data.csv in any of the expected locations: {potential_paths}")
    
    def _permute_stock_data(self, stock_data_df, permutation_id):
        """
        Create a permuted version of the stock data using returns-based bootstrap approach.
        
        This method implements a returns-based bootstrap Monte Carlo simulation:
        1. Sorts data by date to ensure chronological order
        2. Extracts closing prices for each ticker
        3. Calculates daily returns for each ticker
        4. Bootstraps (randomly samples with replacement) from historical returns
        5. Reconstructs price series from bootstrapped returns
        
        Args:
            stock_data_df (DataFrame): Original stock data
            permutation_id (int): ID of the permutation
            
        Returns:
            DataFrame: Permuted stock data with new price series
        """
        # Make a copy of the original data
        permuted_df = stock_data_df.copy()
        
        # Sort by date to ensure chronological order
        if 'Date' in permuted_df.columns:
            permuted_df['Date'] = pd.to_datetime(permuted_df['Date'])
            permuted_df = permuted_df.sort_values('Date').reset_index(drop=True)
        
        # Keep the Date column intact
        dates = permuted_df['Date'].copy()
        
        # Set random seed for reproducibility but different for each permutation
        np.random.seed(permutation_id + 42)
        
        # Process each ticker separately
        for ticker in self.tickers:
            # Find all columns for this ticker
            ticker_columns = [col for col in permuted_df.columns if col.startswith(f"{ticker}_")]
            
            # Skip if no columns found for this ticker
            if not ticker_columns:
                if self.verbose:
                    print(f"No data columns found for ticker {ticker}")
                continue
            
            # Find the closing price column
            close_col = f"{ticker}_Close"
            if close_col not in ticker_columns:
                # Try to find any price column if Close isn't available
                price_cols = [col for col in ticker_columns if any(x in col.lower() for x in ['close', 'price', 'adj'])]
                if price_cols:
                    close_col = price_cols[0]
                else:
                    if self.verbose:
                        print(f"No closing price column found for ticker {ticker}, skipping")
                    continue
            
            # Extract closing prices
            prices = permuted_df[close_col].values
            
            # Calculate daily returns (skip first day which will be NaN)
            returns = np.zeros(len(prices))
            returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
            
            # Filter out any invalid returns (NaN, inf)
            valid_returns = returns[~np.isnan(returns) & ~np.isinf(returns) & (returns != 0)]
            
            # Skip if not enough valid returns
            if len(valid_returns) < 10:  # Require at least 10 valid returns
                if self.verbose:
                    print(f"Not enough valid returns for ticker {ticker}")
                continue
            
            # Bootstrap returns - randomly sample with replacement
            np.random.seed(permutation_id + hash(ticker) % 10000)  # Ensure reproducibility but different for each ticker
            sampled_returns = np.random.choice(valid_returns, size=len(prices)-1, replace=True)
            
            # Reconstruct a new price series from bootstrapped returns
            new_prices = np.zeros(len(prices))
            new_prices[0] = prices[0]  # Start with the original first price
            
            # Generate the rest of the series using the bootstrapped returns
            for i in range(1, len(prices)):
                new_prices[i] = new_prices[i-1] * (1 + sampled_returns[i-1])
            
            # Update the closing price column with the new prices
            permuted_df[close_col] = new_prices
            
            # Define price columns to update
            price_cols = {
                'high': next((col for col in ticker_columns if col.lower().endswith('_high') or 'high' in col.lower()), None),
                'low': next((col for col in ticker_columns if col.lower().endswith('_low') or 'low' in col.lower()), None),
                'open': next((col for col in ticker_columns if col.lower().endswith('_open') or 'open' in col.lower()), None)
            }
            
            # Calculate historical ratios and percentiles if columns exist
            ratio_percentiles = {}
            
            for price_type, col in price_cols.items():
                if col is not None:
                    # Calculate the original ratio with close price
                    orig_df = stock_data_df.copy()
                    ratio = orig_df[col] / orig_df[close_col].replace(0, np.nan)
                    
                    # Remove NaN and infinite values
                    valid_ratios = ratio.dropna().replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(valid_ratios) >= 10:  # Ensure sufficient data
                        # Calculate 5th and 95th percentiles
                        p05 = np.percentile(valid_ratios, 5)
                        p95 = np.percentile(valid_ratios, 95)
                        
                        # Add some constraints based on price type
                        if price_type == 'high':
                            # High should be >= Close
                            p05 = max(p05, 1.0)
                        elif price_type == 'low':
                            # Low should be <= Close
                            p95 = min(p95, 1.0)
                        
                        ratio_percentiles[price_type] = {
                            'p05': p05,
                            'p95': p95,
                            'column': col
                        }
                    else:
                        if self.verbose:
                            print(f"Not enough valid ratios for {ticker} {price_type}")
            
            # Apply randomized ratios to create new OHLC values
            for price_type, percentiles in ratio_percentiles.items():
                # Get column name and percentile boundaries
                col = percentiles['column']
                p05 = percentiles['p05']
                p95 = percentiles['p95']
                
                # Generate random ratios within the percentile bounds for each day
                random_ratios = np.random.uniform(p05, p95, size=len(new_prices))
                
                # Apply ratios to Close prices
                if price_type == 'high':
                    # Ensure High is always >= Close
                    permuted_df[col] = new_prices * np.maximum(random_ratios, 1.0)
                elif price_type == 'low':
                    # Ensure Low is always <= Close
                    permuted_df[col] = new_prices * np.minimum(random_ratios, 1.0)
                else:  # Open
                    permuted_df[col] = new_prices * random_ratios
                    
                # Apply additional constraints to maintain OHLC relationship
                if price_type == 'open':
                    # Ensure Open is between High and Low
                    high_col = price_cols.get('high')
                    low_col = price_cols.get('low')
                    if high_col is not None and low_col is not None:
                        # Constrain Open to be within High and Low
                        permuted_df[col] = np.minimum(permuted_df[high_col], 
                                                    np.maximum(permuted_df[low_col], permuted_df[col]))
            
            # Handle volume columns separately if they exist
            volume_cols = [col for col in ticker_columns if 'volume' in col.lower()]
            for vol_col in volume_cols:
                # Keep the original volume data but shuffle it in blocks
                # to maintain some autocorrelation in volume
                vol_data = permuted_df[vol_col].values
                block_size = min(5, len(vol_data) // 10)  # Small blocks to preserve some volume patterns
                
                if block_size > 1:
                    blocks = []
                    for i in range(0, len(vol_data), block_size):
                        end_idx = min(i + block_size, len(vol_data))
                        blocks.append(vol_data[i:end_idx])
                    
                    np.random.shuffle(blocks)
                    new_vol_data = np.concatenate(blocks)
                    
                    # Ensure the length matches
                    if len(new_vol_data) > len(vol_data):
                        new_vol_data = new_vol_data[:len(vol_data)]
                    elif len(new_vol_data) < len(vol_data):
                        # Repeat last block if needed
                        padding = np.tile(new_vol_data[-block_size:], 
                                         (len(vol_data) - len(new_vol_data) + block_size - 1) // block_size)
                        new_vol_data = np.concatenate([new_vol_data, padding[:len(vol_data) - len(new_vol_data)]])
                    
                    permuted_df[vol_col] = new_vol_data
            
            if self.verbose and permutation_id == 0:
                print(f"Created bootstrap simulation for {ticker} with {len(valid_returns)} unique returns")
                # Print ratio statistics for the first permutation
                for price_type, percentiles in ratio_percentiles.items():
                    print(f"  {ticker} {price_type.title()} ratio range: {percentiles['p05']:.4f} to {percentiles['p95']:.4f}")
        
        # Ensure the Date column is preserved
        permuted_df['Date'] = dates
        
        # Print some summary statistics for the first permutation
        if self.verbose and permutation_id == 0:
            for ticker in self.tickers:
                close_col = f"{ticker}_Close"
                if close_col in permuted_df.columns:
                    orig_prices = stock_data_df[close_col]
                    new_prices = permuted_df[close_col]
                    print(f"Ticker {ticker} original vs permuted price statistics:")
                    print(f"  Original: mean={orig_prices.mean():.2f}, std={orig_prices.std():.2f}, min={orig_prices.min():.2f}, max={orig_prices.max():.2f}")
                    print(f"  Permuted: mean={new_prices.mean():.2f}, std={new_prices.std():.2f}, min={new_prices.min():.2f}, max={new_prices.max():.2f}")
        
        return permuted_df
    
    def run_original_backtest(self, out_of_sample_start: str) -> Dict:
        """
        Run the original backtest using the run_backtest engine.
        
        Args:
            out_of_sample_start (str): Start date for out-of-sample period (YYYY-MM-DD)
            
        Returns:
            dict: Results from the original backtest
        """
        if self.verbose:
            print(f"Running original backtest from {out_of_sample_start}")
        
        # Find the stock data CSV file
        self.original_stock_csv = self._find_stock_data_csv()
        
        # Create output directory for original test
        original_output_dir = os.path.join(self.output_dir, "original")
        os.makedirs(original_output_dir, exist_ok=True)
        
        # Run the backtest using the run_backtest engine
        original_results = run_backtest(
            output_dir=original_output_dir,
            strategy_name=self.strategy_name,
            tickers=self.tickers,
            parameters=self.parameters,
            start_date=out_of_sample_start,
            end_date=None,  # Use all available data after start_date
            stock_csv=self.original_stock_csv,
            plot=False,
            warmup_period=50  # Use a standard warmup period
        )
        
        if not original_results:
            raise ValueError("Original backtest failed to produce results")
        
        # Extract key metrics into a standardized format
        metrics = {
            'initial_value': original_results.get('initial_value', self.initial_capital),
            'final_value': original_results.get('final_value', 0),
            'total_return': original_results.get('total_return', 0),
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0
        }
        
        # Extract Sharpe ratio and max drawdown from metrics if available
        if 'metrics' in original_results:
            backtest_metrics = original_results.get('metrics', {})
            
            if isinstance(backtest_metrics, dict):
                # Get sharpe_ratio
                if 'sharpe_ratio' in backtest_metrics:
                    metrics['sharpe_ratio'] = backtest_metrics['sharpe_ratio']
                
                # Get max_drawdown
                if 'max_drawdown' in backtest_metrics:
                    metrics['max_drawdown'] = backtest_metrics['max_drawdown']
                    
                # Try to get from trade_analysis nested dict
                trade_analysis = backtest_metrics.get('trade_analysis', {})
                if isinstance(trade_analysis, dict):
                    if 'win_rate' in trade_analysis:
                        metrics['win_rate'] = trade_analysis['win_rate']
                    if 'profit_factor' in trade_analysis:
                        metrics['profit_factor'] = trade_analysis['profit_factor']
        
        # Calculate metrics from equity curve if still missing
        if (metrics['sharpe_ratio'] == 0 or metrics['max_drawdown'] == 0) and 'equity_curve' in original_results:
            equity_metrics = self._calculate_metrics_from_equity_curve(original_results['equity_curve'])
            
            # Update metrics if they are missing
            if metrics['sharpe_ratio'] == 0 and 'sharpe_ratio' in equity_metrics:
                metrics['sharpe_ratio'] = equity_metrics['sharpe_ratio']
            
            if metrics['max_drawdown'] == 0 and 'max_drawdown' in equity_metrics:
                metrics['max_drawdown'] = equity_metrics['max_drawdown']
        
        # Extract win rate and profit factor from trades if available
        trades = original_results.get('trades', [])
        if trades and metrics['win_rate'] == 0:
            won_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            lost_trades = sum(1 for trade in trades if trade.get('pnl', 0) < 0)
            total_trades = len(trades)
            
            metrics['total_trades'] = total_trades
            metrics['win_rate'] = won_trades / total_trades if total_trades > 0 else 0
            
            gross_won = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
            gross_lost = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
            metrics['profit_factor'] = gross_won / gross_lost if gross_lost > 0 else 0
        
        # Save metrics to JSON
        metrics_path = os.path.join(original_output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4, cls=CustomJSONEncoder)
        
        if self.verbose:
            print(f"Original backtest completed with {metrics['total_trades']} trades")
            print(f"Total return: {metrics['total_return']:.2f}%")
            print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"Win rate: {metrics['win_rate']:.2f}")
            print(f"Profit factor: {metrics['profit_factor']:.2f}")
    
        return metrics
    
    def run_monte_carlo_simulations(self, out_of_sample_start: str) -> List[Dict]:
        """
        Run Monte Carlo simulations by permuting stock data and running backtests.
        
        Args:
            out_of_sample_start (str): Start date for out-of-sample period
            
        Returns:
            list: List of performance metrics dictionaries for each simulation
        """
        if not self.original_stock_csv:
            raise ValueError("Original stock data CSV not found. Run original backtest first.")
        
        # Always print the keep_permuted_data value at the start of this method
        print(f"DEBUG: run_monte_carlo_simulations called with keep_permuted_data={self.keep_permuted_data}")
        print(f"DEBUG: permuted_data_dir path: {self.permuted_data_dir}")
        
        # Load the original stock data
        original_data = pd.read_csv(self.original_stock_csv)
        
        if self.verbose:
            print(f"Running {self.num_simulations} Monte Carlo simulations")
        
        # Save original data for reference if keeping permuted data
        if self.keep_permuted_data:
            original_copy_path = os.path.join(self.permuted_data_dir, "original_stock_data.csv")
            original_data.to_csv(original_copy_path, index=False)
            if self.verbose:
                print(f"Saved copy of original stock data to {original_copy_path}")
            
            # Verify the directory exists and the file was created
            if os.path.exists(original_copy_path):
                print(f"DEBUG: Successfully created original stock data copy at {original_copy_path}")
            else:
                print(f"ERROR: Failed to create original stock data copy at {original_copy_path}")
                print(f"DEBUG: Directory exists: {os.path.exists(self.permuted_data_dir)}")
                print(f"DEBUG: Directory contents: {os.listdir(self.permuted_data_dir) if os.path.exists(self.permuted_data_dir) else 'N/A'}")
        
        # Create permuted datasets
        permuted_csvs = []
        permutation_info = []
        
        for i in range(self.num_simulations):
            # Create a permuted version of the stock data
            permuted_data = self._permute_stock_data(original_data, i)
            
            # Record information about this permutation
            info = {
                "permutation_id": i,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tickers": self.tickers,
                "file_path": f"permuted_data_{i}.csv",
                "num_rows": len(permuted_data),
                "permutation_type": "block_permutation",
                "description": "Block permutation with random block sizes (5-20 days)"
            }
            permutation_info.append(info)
            
            # Save the permuted data to a CSV file
            permuted_csv = os.path.join(self.permuted_data_dir, f"permuted_data_{i}.csv")
            permuted_data.to_csv(permuted_csv, index=False)
            permuted_csvs.append(permuted_csv)
        
        # Save metadata about permutations
        if self.keep_permuted_data:
            permutation_info_path = os.path.join(self.permuted_data_dir, "permutation_metadata.json")
            with open(permutation_info_path, 'w') as f:
                json.dump(permutation_info, f, indent=4, cls=CustomJSONEncoder)
            
            # Create a README file explaining the permutation process
            readme_path = os.path.join(self.permuted_data_dir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(f"# Monte Carlo Permuted Stock Data\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Strategy: {self.strategy_name}\n")
                f.write(f"Tickers: {', '.join(self.tickers)}\n")
                f.write(f"Number of permutations: {self.num_simulations}\n\n")
                f.write("## Permutation Method\n\n")
                f.write("Each permutation uses a block permutation approach where:\n")
                f.write("1. The original stock data is divided into random-sized blocks (5-20 days)\n")
                f.write("2. These blocks are shuffled randomly\n")
                f.write("3. The blocks are reassembled to create a permuted version of the data\n")
                f.write("4. The Date column is preserved to maintain the time series structure\n\n")
                f.write("## File Structure\n\n")
                f.write("- `original_stock_data.csv`: The original unmodified stock data\n")
                f.write("- `permuted_data_X.csv`: Permuted stock data for permutation X\n")
                f.write("- `permutation_metadata.json`: Metadata about each permutation\n")
            
            if self.verbose:
                print(f"Created permutation documentation in {self.permuted_data_dir}")
        
        # Prepare backtest configurations for parallel execution
        backtest_configs = []
        for i, permuted_csv in enumerate(permuted_csvs):
            # Create output directory for this simulation
            sim_output_dir = os.path.join(self.output_dir, f"simulation_{i}")
            os.makedirs(sim_output_dir, exist_ok=True)
            
            # Create configuration for this backtest
            config = {
                'output_dir': sim_output_dir,
                'strategy_name': self.strategy_name,
                'tickers': self.tickers,
                'parameters': self.parameters,
                'start_date': out_of_sample_start,
                'end_date': None,  # Use all available data after start_date
                'stock_csv': permuted_csv,
                'plot': False,
                'warmup_period': 50  # Use a standard warmup period
            }
            backtest_configs.append(config)
        
        # Determine the number of workers - use class variable num_workers if defined, otherwise use default
        num_workers = getattr(self, 'num_workers', None)
        
        # Run backtests in parallel
        if self.verbose:
            print(f"Running Monte Carlo simulations using parallel processing")
        
        # Run backtests in parallel
        sim_results_list = run_parallel_backtests(backtest_configs, num_workers)
        
        # Process results
        simulated_metrics = []
        for i, sim_results in enumerate(sim_results_list):
            if sim_results and not isinstance(sim_results, dict) or sim_results.get('error', None) is None:
                # Extract key metrics into a standardized format
                metrics = {
                    'simulation_id': i,
                    'initial_value': sim_results.get('initial_value', self.initial_capital),
                    'final_value': sim_results.get('final_value', 0),
                    'total_return': sim_results.get('total_return', 0),
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'total_trades': 0
                }
                
                # Add reference to permuted data file if keeping them
                if self.keep_permuted_data:
                    metrics['permuted_data_file'] = f"permuted_data_{i}.csv"
                
                # Extract Sharpe ratio and max drawdown from metrics if available
                if 'metrics' in sim_results:
                    backtest_metrics = sim_results.get('metrics', {})
                    
                    if isinstance(backtest_metrics, dict):
                        # Get sharpe_ratio
                        if 'sharpe_ratio' in backtest_metrics:
                            metrics['sharpe_ratio'] = backtest_metrics['sharpe_ratio']
                        
                        # Get max_drawdown
                        if 'max_drawdown' in backtest_metrics:
                            metrics['max_drawdown'] = backtest_metrics['max_drawdown']
                        
                        # Try to get from trade_analysis nested dict
                        trade_analysis = backtest_metrics.get('trade_analysis', {})
                        if isinstance(trade_analysis, dict):
                            if 'win_rate' in trade_analysis:
                                metrics['win_rate'] = trade_analysis['win_rate']
                            if 'profit_factor' in trade_analysis:
                                metrics['profit_factor'] = trade_analysis['profit_factor']
                
                # Calculate metrics from equity curve if still missing
                if (metrics['sharpe_ratio'] == 0 or metrics['max_drawdown'] == 0) and 'equity_curve' in sim_results:
                    equity_metrics = self._calculate_metrics_from_equity_curve(sim_results['equity_curve'])
                    
                    # Update metrics if they are missing
                    if metrics['sharpe_ratio'] == 0 and 'sharpe_ratio' in equity_metrics:
                        metrics['sharpe_ratio'] = equity_metrics['sharpe_ratio']
                    
                    if metrics['max_drawdown'] == 0 and 'max_drawdown' in equity_metrics:
                        metrics['max_drawdown'] = equity_metrics['max_drawdown']
                
                # Extract win rate and profit factor from trades if available
                trades = sim_results.get('trades', [])
                if trades and metrics['win_rate'] == 0:
                    won_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
                    lost_trades = sum(1 for trade in trades if trade.get('pnl', 0) < 0)
                    total_trades = len(trades)
                    
                    metrics['total_trades'] = total_trades
                    metrics['win_rate'] = won_trades / total_trades if total_trades > 0 else 0
                    
                    gross_won = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
                    gross_lost = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
                    metrics['profit_factor'] = gross_won / gross_lost if gross_lost > 0 else 0
                
                simulated_metrics.append(metrics)
            else:
                print(f"Warning: Simulation {i} failed to produce results")
        
        # If we're keeping permuted data, make sure to preserve it by copying files if needed
        if self.keep_permuted_data:
            # Check if any permuted data files exist in the permuted_data directory
            files_in_dir = os.listdir(self.permuted_data_dir)
            csv_files = [f for f in files_in_dir if f.endswith('.csv') and f.startswith('permuted_data_')]
            
            print(f"DEBUG: After simulations, found {len(csv_files)} permuted CSV files in {self.permuted_data_dir}")
            
            if len(csv_files) < self.num_simulations:
                # Some files are missing, which could happen if run_backtest is using the files directly
                # and not making copies. In this case, make copies of permuted data files.
                print(f"DEBUG: Expected {self.num_simulations} files but found {len(csv_files)}. Attempting to copy missing files...")
                
            for i in range(self.num_simulations):
                    src_file = permuted_csvs[i]
                    dst_file = os.path.join(self.permuted_data_dir, f"permuted_data_{i}.csv")
                    
                    # Only copy if the destination doesn't exist
                    if not os.path.exists(dst_file) and os.path.exists(src_file):
                        try:
                            shutil.copy2(src_file, dst_file)
                            print(f"DEBUG: Copied file from {src_file} to {dst_file}")
                        except Exception as e:
                            print(f"ERROR copying permuted data file: {e}")
                    elif not os.path.exists(src_file):
                        print(f"ERROR: Source file {src_file} does not exist for copying")
                    elif os.path.exists(dst_file):
                        print(f"DEBUG: Destination file {dst_file} already exists, no need to copy")
            
            # Verify final state of permuted_data directory
            files_after = os.listdir(self.permuted_data_dir)
            csv_files_after = [f for f in files_after if f.endswith('.csv') and f.startswith('permuted_data_')]
            print(f"DEBUG: Final count of permuted CSV files in directory: {len(csv_files_after)} out of {self.num_simulations} expected")
            if len(csv_files_after) < self.num_simulations:
                print(f"WARNING: Some permuted data files could not be preserved ({len(csv_files_after)} out of {self.num_simulations})")
                # List the first few missing indices for debugging
                existing_indices = set([int(f.split('_')[2].split('.')[0]) for f in csv_files_after])
                missing_indices = [i for i in range(self.num_simulations) if i not in existing_indices]
                print(f"DEBUG: Sample of missing indices: {missing_indices[:5] if missing_indices else 'None'}")
            
            if self.verbose:
                print(f"Final number of files in permuted data directory: {len(files_after)}")
        else:
            # Clean up permuted CSV files if not keeping them
            for csv_file in permuted_csvs:
                if os.path.exists(csv_file):
                    try:
                        os.remove(csv_file)
                    except Exception as e:
                        print(f"Warning: Could not delete permuted file {csv_file}: {e}")
            if self.verbose:
                print(f"Deleted {len(permuted_csvs)} permuted stock data files to save disk space")
            
            # Also remove the permuted_data directory if it's empty
            try:
                if os.path.exists(self.permuted_data_dir) and not os.listdir(self.permuted_data_dir):
                    os.rmdir(self.permuted_data_dir)
                    if self.verbose:
                        print(f"Removed empty permuted_data directory")
            except Exception as e:
                print(f"Warning: Could not remove permuted_data directory: {e}")
        
        # Save all simulation metrics to CSV
        if simulated_metrics:
            sim_metrics_df = pd.DataFrame(simulated_metrics)
            sim_metrics_path = os.path.join(self.output_dir, 'simulation_metrics.csv')
            sim_metrics_df.to_csv(sim_metrics_path, index=False)
            
            if self.verbose:
                print(f"Completed {len(simulated_metrics)} Monte Carlo simulations")
                print(f"Saved simulation metrics to {sim_metrics_path}")
        else:
            print("Warning: No simulation metrics were collected")
        
        return simulated_metrics
    
    def analyze_results(self, original_metrics: Dict, simulated_metrics: List[Dict]) -> Dict:
        """
        Analyze the distribution of Monte Carlo simulation results.
        
        Args:
            original_metrics (dict): Performance metrics from the original backtest
            simulated_metrics (list): List of performance metrics from simulations
            
        Returns:
            dict: Analysis results
        """
        if self.verbose:
            print("Analyzing Monte Carlo simulation results")
            print(f"Original metrics: {original_metrics}")
            print(f"Number of simulation metrics: {len(simulated_metrics)}")
            # Print sample of first simulation metrics
            if simulated_metrics:
                print(f"First simulation metrics: {simulated_metrics[0]}")
        
        # Convert to DataFrame for easier analysis
        sim_df = pd.DataFrame(simulated_metrics)
        
        # Calculate statistics for each metric
        metrics_to_analyze = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        
        analysis_results = {}
        for metric in metrics_to_analyze:
            # Skip if metric is not available
            if metric not in sim_df.columns:
                if self.verbose:
                    print(f"Warning: Metric '{metric}' not found in simulation results")
                continue
                
            # Get values for this metric
            sim_values = sim_df[metric].values
            original_value = original_metrics.get(metric, 0)
            
            if self.verbose:
                print(f"\nAnalyzing {metric}:")
                print(f"  Original value: {original_value}")
                print(f"  Simulation values min: {np.min(sim_values)}, max: {np.max(sim_values)}")
                print(f"  Simulation values mean: {np.mean(sim_values)}, std: {np.std(sim_values)}")
            
            # Calculate statistics
            stats_dict = {
                'mean': np.mean(sim_values),
                'median': np.median(sim_values),
                'std': np.std(sim_values),
                'min': np.min(sim_values),
                'max': np.max(sim_values),
                'p5': np.percentile(sim_values, 5),
                'p25': np.percentile(sim_values, 25),
                'p75': np.percentile(sim_values, 75),
                'p95': np.percentile(sim_values, 95),
                'original': original_value
            }
            
            # Calculate p-value (two-tailed)
            # For metrics where higher is better (all except max_drawdown)
            if metric != 'max_drawdown':
                p_value = np.mean(sim_values >= original_value)
            else:
                p_value = np.mean(sim_values <= original_value)
            
            stats_dict['p_value'] = p_value
            
            # Add to results
            analysis_results[metric] = stats_dict
        
        # Save analysis results to JSON
        analysis_path = os.path.join(self.output_dir, 'analysis_results.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=4, cls=CustomJSONEncoder)
        
        if self.verbose:
            print(f"Saved analysis results to {analysis_path}")
            
            # Print summary
            print("\nAnalysis Summary:")
            for metric, stats in analysis_results.items():
                print(f"\n{metric.replace('_', ' ').title()}:")
                print(f"  Original: {stats['original']:.4f}")
                print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Range: [{stats['p5']:.4f}, {stats['p95']:.4f}] (90% confidence)")
                print(f"  P-value: {stats['p_value']:.4f}")
        
        return analysis_results
    
    def create_visualizations(self, original_metrics: Dict, simulated_metrics: List[Dict]) -> None:
        """
        Create visualizations of the Monte Carlo simulation results.
        
        Args:
            original_metrics (dict): Performance metrics from the original backtest
            simulated_metrics (list): List of performance metrics from simulations
        """
        if self.verbose:
            print("Creating visualizations")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Convert to DataFrame for easier plotting
        sim_df = pd.DataFrame(simulated_metrics)
        
        # Set up plotting style
        try:
            # Try newer seaborn style first
            plt.style.use('seaborn-darkgrid')
        except:
            try:
                # If that fails, try the default seaborn style
                plt.style.use('seaborn')
            except:
                # If all else fails, use the default style
                pass
                
        metrics_to_plot = {
            'total_return': 'Total Return',
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown': 'Maximum Drawdown',
            'win_rate': 'Win Rate',
            'profit_factor': 'Profit Factor'
        }
        
        # Create distribution plots for each metric
        for metric, title in metrics_to_plot.items():
            # Skip if metric is not available
            if metric not in sim_df.columns:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Get values and original value - convert to numpy array
            sim_values = np.array(sim_df[metric].values)
            original_value = float(original_metrics.get(metric, 0))
            
            # Create histogram plot using plain matplotlib instead of seaborn
            # This avoids the multi-dimensional indexing issue
            n, bins, patches = plt.hist(sim_values, bins=15, alpha=0.6, density=True)
            
            # Calculate simple statistics for better visualization
            mean_value = float(np.mean(sim_values))
            median_value = float(np.median(sim_values))
            std_value = float(np.std(sim_values))
            
            # Add vertical lines for original value, mean, and percentiles
            plt.axvline(original_value, color='red', linestyle='--', linewidth=2, 
                        label=f'Original: {original_value:.4f}')
            
            plt.axvline(mean_value, color='green', linestyle='-', linewidth=2, 
                        label=f'Mean: {mean_value:.4f}')
            
            plt.axvline(median_value, color='blue', linestyle='-.', linewidth=2, 
                        label=f'Median: {median_value:.4f}')
            
            p5 = float(np.percentile(sim_values, 5))
            p95 = float(np.percentile(sim_values, 95))
            plt.axvline(p5, color='orange', linestyle=':', linewidth=2, 
                        label=f'5th Percentile: {p5:.4f}')
            plt.axvline(p95, color='orange', linestyle=':', linewidth=2, 
                        label=f'95th Percentile: {p95:.4f}')
            
            # Calculate p-value
            if metric != 'max_drawdown':
                p_value = float(np.mean(sim_values >= original_value))
            else:
                p_value = float(np.mean(sim_values <= original_value))
            
            # Add title and labels with enhanced statistical information
            plt.title(f'{title} Distribution - Monte Carlo Simulation\n'
                     f'Mean: {mean_value:.4f}, Std: {std_value:.4f}, P-value: {p_value:.4f}', fontsize=14)
            plt.xlabel(title, fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(viz_dir, f'{metric}_distribution.png')
            plt.savefig(plot_path)
            plt.close()
        
        if self.verbose:
            print(f"Saved distribution plots to {viz_dir}")
    
    def create_equity_curve_comparison(self) -> None:
        """
        Create a comparison plot of equity curves from all permutations alongside the original.
        
        This method loads the equity curve data from the original backtest and all permutations,
        then plots them together. The original equity curve is highlighted in red.
        
        Features:
        - Original equity curve in bold red
        - Permutation equity curves in light blue
        - Shaded confidence interval (5th to 95th percentile)
        - Mean performance line
        - Profit/loss zones highlighted
        - Statistical summary in the plot
        """
        if self.verbose:
            print("Creating equity curve comparison plot")
        
        # Create visualizations directory if it doesn't exist
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load original equity curve
        original_equity_path = os.path.join(self.output_dir, 'original', 'equity_curve.csv')
        if not os.path.exists(original_equity_path):
            print(f"Warning: Original equity curve file not found at {original_equity_path}")
            return
        
        try:
            original_equity = pd.read_csv(original_equity_path)
            original_equity['Date'] = pd.to_datetime(original_equity['Date'])
            original_equity.set_index('Date', inplace=True)
            
            # Create figure
            plt.figure(figsize=(14, 10))
            
            # Collect all equity curves for percentile calculations
            all_equity_curves = []
            common_dates = None
            
            # First pass: collect all equity curves with valid data
            for i in range(self.num_simulations):
                sim_equity_path = os.path.join(self.output_dir, f'simulation_{i}', 'equity_curve.csv')
                if os.path.exists(sim_equity_path):
                    try:
                        sim_equity = pd.read_csv(sim_equity_path)
                        sim_equity['Date'] = pd.to_datetime(sim_equity['Date'])
                        sim_equity.set_index('Date', inplace=True)
                        
                        # Only use equity curves that have the same date range
                        if common_dates is None:
                            common_dates = set(sim_equity.index)
                        else:
                            common_dates = common_dates.intersection(set(sim_equity.index))
                            
                        all_equity_curves.append(sim_equity)
                    except Exception as e:
                        if self.verbose:
                            print(f"Error loading simulation {i} equity curve: {e}")
            
            # If we have common dates and curves, create the percentile envelope
            if common_dates and all_equity_curves:
                common_dates = sorted(list(common_dates))
                
                # Ensure the original equity curve has these dates too
                common_dates = [date for date in common_dates if date in original_equity.index]
                
                # Create a DataFrame to hold all values for each date
                values_by_date = {date: [] for date in common_dates}
                
                # Collect values for each date across all simulations
                for curve in all_equity_curves:
                    for date in common_dates:
                        if date in curve.index:
                            values_by_date[date].append(curve.loc[date, 'Value'])
                
                # Calculate percentiles for each date
                percentiles = {}
                for date in common_dates:
                    if values_by_date[date]:  # Ensure we have values
                        values = np.array(values_by_date[date])
                        percentiles[date] = {
                            'p5': np.percentile(values, 5),
                            'p25': np.percentile(values, 25),
                            'p50': np.percentile(values, 50),
                            'p75': np.percentile(values, 75),
                            'p95': np.percentile(values, 95),
                            'mean': np.mean(values)
                        }
                
                # Create DataFrames for percentile lines
                p5_line = pd.Series({date: percentiles[date]['p5'] for date in common_dates}, name='p5')
                p25_line = pd.Series({date: percentiles[date]['p25'] for date in common_dates}, name='p25')
                p50_line = pd.Series({date: percentiles[date]['p50'] for date in common_dates}, name='p50')
                p75_line = pd.Series({date: percentiles[date]['p75'] for date in common_dates}, name='p75')
                p95_line = pd.Series({date: percentiles[date]['p95'] for date in common_dates}, name='p95')
                mean_line = pd.Series({date: percentiles[date]['mean'] for date in common_dates}, name='mean')
                
                # Convert common_dates and Series to numpy arrays for plotting
                dates_array = np.array(common_dates)
                p5_array = p5_line.values
                p25_array = p25_line.values
                p50_array = p50_line.values
                p75_array = p75_line.values
                p95_array = p95_line.values
                mean_array = mean_line.values
                
                # Plot confidence interval as shaded region
                plt.fill_between(dates_array, p5_array, p95_array, color='lightblue', alpha=0.3, 
                                label='90% Confidence Interval')
                plt.fill_between(dates_array, p25_array, p75_array, color='skyblue', alpha=0.3, 
                                label='50% Confidence Interval')
                
                # Plot median and mean lines
                plt.plot(dates_array, p50_array, color='blue', linestyle='-', linewidth=1.0, 
                        label='Median Performance')
                plt.plot(dates_array, mean_array, color='green', linestyle='-', linewidth=1.5, 
                        label='Mean Performance')
                
                # Highlight profit/loss zones
                initial_value = self.initial_capital
                plt.axhline(y=initial_value, color='darkgray', linestyle='-', linewidth=1.0, 
                           label='Initial Capital')
                
                # Add a light red zone for values below initial capital
                plt.axhspan(0, initial_value, color='red', alpha=0.05)
                # Add a light green zone for values above initial capital
                plt.axhspan(initial_value, max(p95_array) * 1.1, color='green', alpha=0.05)
            
            # Convert original equity curve to numpy arrays
            original_dates = original_equity.index.to_numpy()
            original_values = original_equity['Value'].to_numpy()
            
            # Now plot individual permutation equity curves with lower alpha
            for i in range(min(50, self.num_simulations)):  # Limit to 50 to avoid overcrowding
                sim_equity_path = os.path.join(self.output_dir, f'simulation_{i}', 'equity_curve.csv')
                if os.path.exists(sim_equity_path):
                    try:
                        sim_equity = pd.read_csv(sim_equity_path)
                        sim_equity['Date'] = pd.to_datetime(sim_equity['Date'])
                        sim_equity.set_index('Date', inplace=True)
                        
                        # Convert to numpy arrays for plotting
                        sim_dates = sim_equity.index.to_numpy()
                        sim_values = sim_equity['Value'].to_numpy()
                        
                        # Plot permutation in light blue with low alpha
                        plt.plot(sim_dates, sim_values, color='lightblue', alpha=0.1, linewidth=0.5)
                    except Exception as e:
                        pass
            
            # Plot original equity curve in red with thicker line
            plt.plot(original_dates, original_values, color='red', linewidth=2.5, 
                     label='Original Backtest')
            
            # Collect final statistics
            all_final_values = []
            for curve in all_equity_curves:
                if not curve.empty:
                    all_final_values.append(curve['Value'].iloc[-1])
            
            # Calculate statistics for display
            if all_final_values:
                final_values_array = np.array(all_final_values)
                mean_final = np.mean(final_values_array)
                std_final = np.std(final_values_array)
                p5_final = np.percentile(final_values_array, 5)
                p95_final = np.percentile(final_values_array, 95)
                
                # Get original final value
                original_final = original_equity['Value'].iloc[-1] if not original_equity.empty else 0
                
                # Calculate p-value (what percentage of simulations performed better than original)
                p_value = np.mean(final_values_array >= original_final)
                
                # Add stats box with key metrics
                stats_text = (
                    f"Final Value Statistics:\n"
                    f"Original: ${original_final:,.2f}\n"
                    f"Mean: ${mean_final:,.2f}\n"
                    f"Std Dev: ${std_final:,.2f}\n"
                    f"90% Range: [${p5_final:,.2f}, ${p95_final:,.2f}]\n"
                    f"P-value: {p_value:.4f} ({'Significant' if p_value < 0.1 else 'Not Significant'})"
                )
                
                # Position the text box in the upper left corner
                plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                            verticalalignment='top', fontsize=11)
            
            # Add labels and title with more details
            plt.title(f'Equity Curve Comparison - {self.strategy_name}\n'
                     f'Original vs {len(all_equity_curves)} Monte Carlo Permutations', fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Portfolio Value ($)', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Improve legend
            plt.legend(loc='lower right', fontsize=11, framealpha=0.8)
            
            # Format y-axis as currency
            from matplotlib.ticker import FuncFormatter
            def currency_formatter(x, pos):
                return f'${x:,.0f}'
            plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
            
            # Make background white
            plt.gca().set_facecolor('white')
            
            # Adjust margins to fit everything
            plt.tight_layout()
            plt.subplots_adjust(right=0.95, left=0.1)
            
            # Save plot as both PNG and PDF for high-quality prints
            equity_plot_path_png = os.path.join(viz_dir, 'equity_curve_comparison.png')
            equity_plot_path_pdf = os.path.join(viz_dir, 'equity_curve_comparison.pdf')
            plt.savefig(equity_plot_path_png, dpi=150)
            plt.savefig(equity_plot_path_pdf)
            plt.close()
                    
            if self.verbose:
                print(f"Equity curve comparison saved to {equity_plot_path_png} and {equity_plot_path_pdf}")
                
        except Exception as e:
            print(f"Error creating equity curve comparison: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_metrics_from_equity_curve(self, equity_curve_data):
        """
        Calculate key performance metrics from equity curve data
        
        Args:
            equity_curve_data: List of dicts or DataFrame with equity curve data
            
        Returns:
            dict: Dictionary with calculated metrics
        """
        metrics = {}
        
        try:
            # Convert to DataFrame if not already
            if not isinstance(equity_curve_data, pd.DataFrame):
                equity_curve = pd.DataFrame(equity_curve_data)
                if 'Date' in equity_curve.columns:
                    equity_curve['Date'] = pd.to_datetime(equity_curve['Date'])
                    equity_curve.set_index('Date', inplace=True)
            else:
                equity_curve = equity_curve_data.copy()
            
            if 'Value' in equity_curve.columns:
                # Calculate daily returns
                equity_curve['Daily_Return'] = equity_curve['Value'].pct_change()
                
                # Calculate Sharpe ratio (annualized)
                daily_returns = equity_curve['Daily_Return'].dropna()
                if len(daily_returns) > 0:
                    mean_return = daily_returns.mean()
                    std_return = daily_returns.std()
                    if std_return > 0:
                        sharpe = np.sqrt(252) * mean_return / std_return
                        metrics['sharpe_ratio'] = sharpe
                    else:
                        metrics['sharpe_ratio'] = 0 if mean_return >= 0 else -999
                else:
                    metrics['sharpe_ratio'] = 0
                
                # Calculate max drawdown
                values = equity_curve['Value'].values
                peak = np.maximum.accumulate(values)
                drawdowns = 1 - values / peak
                max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
                metrics['max_drawdown'] = max_dd * 100  # Convert to percentage
                
                # Calculate total return
                if len(values) > 0:
                    initial_value = values[0]
                    final_value = values[-1]
                    if initial_value > 0:
                        total_return = (final_value / initial_value - 1) * 100  # Convert to percentage
                        metrics['total_return'] = total_return
                    else:
                        metrics['total_return'] = 0
                else:
                    metrics['total_return'] = 0
            else:
                metrics['sharpe_ratio'] = 0
                metrics['max_drawdown'] = 0
                metrics['total_return'] = 0
        except Exception as e:
            if self.verbose:
                print(f"Error calculating metrics from equity curve: {e}")
            metrics['sharpe_ratio'] = 0
            metrics['max_drawdown'] = 0
            metrics['total_return'] = 0
        
        return metrics
    
    def create_permuted_price_comparison(self) -> None:
        """
        Create a plot comparing the permuted stock price series with the original data.
        
        This method generates a visualization showing:
        - The original stock price series in red
        - Multiple permuted price series in light blue
        - Summary statistics and date ranges
        
        The plot helps visualize the range of price paths generated by the bootstrap simulation.
        """
        if self.verbose:
            print("Creating permuted price series comparison plot")
            
        # Create visualizations directory if it doesn't exist
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Check if permuted data exists
        if not os.path.exists(self.permuted_data_dir) or not os.listdir(self.permuted_data_dir):
            print(f"Warning: No permuted data found in {self.permuted_data_dir}")
            return
        
        # Load original stock data
        if not self.original_stock_csv or not os.path.exists(self.original_stock_csv):
            print(f"Warning: Original stock data CSV not found at {self.original_stock_csv}")
            return
                
        # Find original stock data copy (it might be in the permuted_data directory)
        original_data_path = os.path.join(self.permuted_data_dir, "original_stock_data.csv")
        if not os.path.exists(original_data_path):
            original_data_path = self.original_stock_csv
        
        try:
            # Load original data
            original_data = pd.read_csv(original_data_path)
            original_data['Date'] = pd.to_datetime(original_data['Date'])
            
            # Create separate plots for each ticker
            for ticker in self.tickers:
                # Find the closing price column
                close_col = f"{ticker}_Close"
                if close_col not in original_data.columns:
                    price_cols = [col for col in original_data.columns if col.startswith(f"{ticker}_") and 
                                 any(x in col.lower() for x in ['close', 'price', 'adj'])]
                    if not price_cols:
                        print(f"Warning: No closing price column found for ticker {ticker}, skipping")
                    continue
                    close_col = price_cols[0]
                
                # Create figure for this ticker
                plt.figure(figsize=(14, 8))
                
                # First, collect all permuted data files
                permuted_files = sorted([f for f in os.listdir(self.permuted_data_dir) 
                                        if f.startswith('permuted_data_') and f.endswith('.csv')])
                
                # Limit to a reasonable number of series to plot
                max_series_to_plot = min(30, len(permuted_files))
                file_sample = permuted_files[:max_series_to_plot]
                
                # Load and plot each permuted series
                permuted_series = []
                valid_series_count = 0
                
                for file_name in file_sample:
                    file_path = os.path.join(self.permuted_data_dir, file_name)
                    try:
                        perm_data = pd.read_csv(file_path)
                        perm_data['Date'] = pd.to_datetime(perm_data['Date'])
                        
                        if close_col in perm_data.columns:
                            # Convert pandas Series to NumPy arrays before plotting
                            dates_array = perm_data['Date'].to_numpy()
                            price_array = perm_data[close_col].to_numpy()
                            
                            # Plot permuted price series in light blue with low alpha
                            plt.plot(dates_array, price_array, 
                                    color='skyblue', alpha=0.15, linewidth=0.8)
                            permuted_series.append(price_array)
                            valid_series_count += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"Error loading permuted file {file_name}: {e}")
                
                # Convert original data to NumPy arrays before plotting
                original_dates = original_data['Date'].to_numpy()
                original_prices = original_data[close_col].to_numpy()
                
                # Plot original price series in red with thick line
                plt.plot(original_dates, original_prices, 
                        color='red', linewidth=2.5, label='Original Data')
                
                # Calculate summary statistics if we have permuted series
                if permuted_series:
                    # Convert to numpy arrays of equal length
                    min_length = min(len(series) for series in permuted_series)
                    aligned_series = np.array([series[:min_length] for series in permuted_series])
                    
                    # Calculate statistics across permutations
                    mean_series = np.mean(aligned_series, axis=0)
                    std_series = np.std(aligned_series, axis=0)
                    
                    # Truncate original data to same length
                    orig_series = original_prices[:min_length]
                    
                    # Calculate correlation between original and mean
                    try:
                        correlation = np.corrcoef(orig_series, mean_series)[0, 1]
                    except:
                        correlation = np.nan
                    
                    # Add statistics to the plot - use truncated dates array
                    trunc_dates = original_dates[:min_length]
                    plt.plot(trunc_dates, mean_series, 
                            color='blue', linewidth=1.5, label='Mean of Permutations')
                    
                    # Add shaded area for standard deviation
                    plt.fill_between(
                        trunc_dates,
                        mean_series - std_series,
                        mean_series + std_series,
                        color='blue', alpha=0.2, label='±1 Std Dev'
                    )
                    
                    # Create stats text box
                    stats_text = (
                        f"Permutation Statistics:\n"
                        f"Number of permutations: {valid_series_count}\n"
                        f"Correlation with original: {correlation:.4f}\n"
                        f"Mean final price: ${mean_series[-1]:.2f}\n"
                        f"Std Dev of final price: ${std_series[-1]:.2f}\n"
                        f"Original final price: ${orig_series[-1]:.2f}"
                    )
                    
                    # Position the text box in the upper left corner
                    plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                                verticalalignment='top', fontsize=10)
                
                # Add title and labels
                title = f"{ticker} Price Series Comparison - Original vs. Permuted"
                subtitle = f"Returns-based Bootstrap Monte Carlo Simulation"
                plt.title(f"{title}\n{subtitle}", fontsize=16)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel(f'{ticker} Price ($)', fontsize=12)
                
                # Format y-axis as currency
                plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('${x:.2f}'))
                
                # Add grid and legend
                plt.grid(True, alpha=0.3)
                plt.legend(loc='upper left', fontsize=10)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save the plot
                plot_path = os.path.join(viz_dir, f'{ticker}_permuted_price_comparison.png')
                plt.savefig(plot_path, dpi=150)
                if self.verbose:
                    print(f"Saved permuted price comparison for {ticker} to {plot_path}")
                plt.close()
                
        except Exception as e:
            print(f"Error creating permuted price comparison: {e}")
            import traceback
            traceback.print_exc()
    
    def run_test(self, out_of_sample_start: str) -> Dict:
        """
        Run the complete out-of-sample Monte Carlo test.
        
        Args:
            out_of_sample_start (str): Start date for out-of-sample period (YYYY-MM-DD)
            
        Returns:
            dict: Analysis results
        """
        try:
            # Step 0: Validate strategy existence
            strategy_class = registry.get_strategy_class(self.strategy_name)
            if not strategy_class:
                raise ValueError(f"Strategy '{self.strategy_name}' not found in registry. " 
                                 f"Available strategies: {registry.list_strategies()}")
            
            # Step 1: Run original backtest
            if self.verbose:
                print(f"Running original backtest with {self.strategy_name} strategy")
            
            original_metrics = self.run_original_backtest(out_of_sample_start)
            self.original_metrics = original_metrics
            
            # Step 2: Run Monte Carlo simulations
            if self.verbose:
                print(f"Running {self.num_simulations} Monte Carlo simulations")
            
            simulated_metrics = self.run_monte_carlo_simulations(out_of_sample_start)
            self.simulated_metrics = simulated_metrics
            
            # Step 3: Analyze results
            if self.verbose:
                print("Analyzing Monte Carlo simulation results")
            
            analysis_results = self.analyze_results(original_metrics, simulated_metrics)
            self.analysis_results = analysis_results
            
            # Step 4: Create visualizations (wrapped in try-except to not fail the whole test)
            try:
                if self.verbose:
                    print("Creating visualizations")
                self.create_visualizations(original_metrics, simulated_metrics)
                self.create_equity_curve_comparison()
                self.create_permuted_price_comparison()
                if self.verbose:
                    print(f"Visualizations saved to {os.path.join(self.output_dir, 'visualizations')}")
            except Exception as viz_err:
                print(f"Warning: Visualization creation failed - {viz_err}")
                print("This does not affect the analysis results, only the plots.")
            
            # Print summary of test results
            if self.verbose:
                print("\nTrade-Based Monte Carlo Test Summary:")
                print(f"Strategy: {self.strategy_name}")
                print(f"Out-of-sample period from: {out_of_sample_start}")
                print(f"Monte Carlo simulations: {self.num_simulations}")
                print(f"Results saved to: {self.output_dir}")
                
                # Print p-values interpretation
                print("\nP-values for performance metrics:")
                for metric, stats in analysis_results.items():
                    p_value = stats['p_value']
                    significance = "Significant" if p_value < 0.1 else "Not significant"
                    print(f"  {metric.replace('_', ' ').title()}: {p_value:.4f} ({significance})")
            
            # Save a summary file with key results
            summary = {
                'strategy': self.strategy_name,
                'parameters': self.parameters,
                'tickers': self.tickers,
                'out_of_sample_start': out_of_sample_start,
                'num_simulations': self.num_simulations,
                'original_metrics': original_metrics,
                'analysis_summary': {
                    k: {
                        'mean': v['mean'],
                        'std': v['std'],
                        'p_value': v['p_value'],
                        'significance': 'Significant' if v['p_value'] < 0.1 else 'Not significant'
                    } for k, v in analysis_results.items()
                }
            }
            
            summary_path = os.path.join(self.output_dir, 'test_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4, cls=CustomJSONEncoder)
            
            return analysis_results
            
        except Exception as e:
            print(f"Error running trade-based Monte Carlo test: {e}")
            
            # Add diagnostic information
            print("\nDiagnostic Information:")
            print(f"Strategy: {self.strategy_name}")
            print(f"Tickers: {self.tickers}")
            print(f"Input directory: {self.input_dir}")
            print(f"Out-of-sample start date: {out_of_sample_start}")
            
            # Check if data files exist
            print("\nChecking for data files:")
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            potential_paths = [
                os.path.join(self.input_dir, "stock_data.csv"),
                os.path.join(project_root, "input", "stock_data.csv")
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    print(f"  ✓ {path} exists")
                else:
                    print(f"  ✗ {path} not found")
            
            # Check available strategies
            try:
                strategies = registry.list_strategies()
                print(f"\nAvailable strategies: {strategies}")
            except:
                print("\nCould not retrieve list of available strategies")
            
            import traceback
            traceback.print_exc()
            return {} 