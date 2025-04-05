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
from engine.run_backtest import run_backtest


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
        verbose: bool = False
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
        """
        self.strategy_name = strategy_name
        self.parameters = parameters
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.input_dir = input_dir
        self.commission = commission
        self.data_format = data_format
        self.original_stock_csv = None
        
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
        Create a permuted version of the stock data for Monte Carlo simulation.
        
        Args:
            stock_data_df (DataFrame): Original stock data
            permutation_id (int): ID of the permutation
            
        Returns:
            DataFrame: Permuted stock data
        """
        # Make a copy of the original data
        permuted_df = stock_data_df.copy()
        
        # Keep the Date column as is
        dates = permuted_df['Date'].copy()
        
        # For each ticker, permute the data
        for ticker in self.tickers:
            # Get columns for this ticker
            ticker_columns = [col for col in permuted_df.columns if col.startswith(f"{ticker}_")]
            
            if ticker_columns:
                # Extract the data for this ticker
                ticker_data = permuted_df[ticker_columns].copy()
                
                # Create random blocks of 5-20 days (to maintain some serial correlation)
                block_size = random.randint(5, 20)
                
                # Calculate number of blocks
                num_rows = len(ticker_data)
                num_blocks = num_rows // block_size
                if num_blocks < 2:
                    # If not enough data for blocks, skip permutation for this ticker
                    warnings.warn(f"Not enough data for ticker {ticker} to perform block permutation")
                    continue
                
                # Reshape data into blocks
                blocks = []
                for i in range(num_blocks):
                    start_idx = i * block_size
                    end_idx = min((i + 1) * block_size, num_rows)
                    blocks.append(ticker_data.iloc[start_idx:end_idx])
                
                # Shuffle the blocks
                random.shuffle(blocks)
                
                # Reassemble the permuted data
                permuted_ticker_data = pd.concat(blocks)
                
                # If there's a remainder, add it at the end
                if num_blocks * block_size < num_rows:
                    remainder = ticker_data.iloc[num_blocks * block_size:]
                    permuted_ticker_data = pd.concat([permuted_ticker_data, remainder])
                
                # Ensure the permuted data has the same length as the original
                permuted_ticker_data = permuted_ticker_data.iloc[:num_rows]
                
                # Replace the ticker columns in the permuted dataframe
                for col in ticker_columns:
                    permuted_df[col] = permuted_ticker_data[col].values
        
        # Restore the original Date column
        permuted_df['Date'] = dates
        
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
            'sharpe_ratio': original_results.get('sharpe_ratio', 0),
            'max_drawdown': original_results.get('max_drawdown', 0),
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0
        }
        
        # Extract win rate and profit factor from trades if available
        trades = original_results.get('trades', [])
        if trades:
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
        
        # Load the original stock data
        original_data = pd.read_csv(self.original_stock_csv)
        
        if self.verbose:
            print(f"Running {self.num_simulations} Monte Carlo simulations")
        
        # Run simulations with progress bar
        simulated_metrics = []
        for i in tqdm(range(self.num_simulations), desc="Monte Carlo Sims", disable=not self.verbose):
            # Create a permuted version of the stock data
            permuted_data = self._permute_stock_data(original_data, i)
            
            # Save the permuted data to a temporary CSV file
            permuted_csv = os.path.join(self.permuted_data_dir, f"permuted_data_{i}.csv")
            permuted_data.to_csv(permuted_csv, index=False)
            
            # Create output directory for this simulation
            sim_output_dir = os.path.join(self.output_dir, f"simulation_{i}")
            os.makedirs(sim_output_dir, exist_ok=True)
            
            try:
                # Run backtest with the permuted data
                sim_results = run_backtest(
                    output_dir=sim_output_dir,
                    strategy_name=self.strategy_name,
                    tickers=self.tickers,
                    parameters=self.parameters,
                    start_date=out_of_sample_start,
                    end_date=None,  # Use all available data after start_date
                    stock_csv=permuted_csv,
                    plot=False,
                    warmup_period=50  # Use a standard warmup period
                )
                
                if sim_results:
                    # Extract key metrics into a standardized format
                    metrics = {
                        'simulation_id': i,
                        'initial_value': sim_results.get('initial_value', self.initial_capital),
                        'final_value': sim_results.get('final_value', 0),
                        'total_return': sim_results.get('total_return', 0),
                        'sharpe_ratio': sim_results.get('sharpe_ratio', 0),
                        'max_drawdown': sim_results.get('max_drawdown', 0),
                        'win_rate': 0,
                        'profit_factor': 0,
                        'total_trades': 0
                    }
                    
                    # Extract win rate and profit factor from trades if available
                    trades = sim_results.get('trades', [])
                    if trades:
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
            except Exception as e:
                print(f"Error in simulation {i}: {e}")
            
            # Optional: Remove the permuted CSV to save disk space
            if os.path.exists(permuted_csv):
                os.remove(permuted_csv)
        
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
        
        # Convert to DataFrame for easier analysis
        sim_df = pd.DataFrame(simulated_metrics)
        
        # Calculate statistics for each metric
        metrics_to_analyze = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        
        analysis_results = {}
        for metric in metrics_to_analyze:
            # Skip if metric is not available
            if metric not in sim_df.columns:
                continue
                
            # Get values for this metric
            sim_values = sim_df[metric].values
            original_value = original_metrics.get(metric, 0)
            
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
                
                # Plot confidence interval as shaded region
                plt.fill_between(common_dates, p5_line, p95_line, color='lightblue', alpha=0.3, 
                                label='90% Confidence Interval')
                plt.fill_between(common_dates, p25_line, p75_line, color='skyblue', alpha=0.3, 
                                label='50% Confidence Interval')
                
                # Plot median and mean lines
                plt.plot(common_dates, p50_line, color='blue', linestyle='-', linewidth=1.0, 
                        label='Median Performance')
                plt.plot(common_dates, mean_line, color='green', linestyle='-', linewidth=1.5, 
                        label='Mean Performance')
                
                # Highlight profit/loss zones
                initial_value = self.initial_capital
                plt.axhline(y=initial_value, color='darkgray', linestyle='-', linewidth=1.0, 
                           label='Initial Capital')
                
                # Add a light red zone for values below initial capital
                plt.axhspan(0, initial_value, color='red', alpha=0.05)
                # Add a light green zone for values above initial capital
                plt.axhspan(initial_value, max(p95_line) * 1.1, color='green', alpha=0.05)
            
            # Now plot individual permutation equity curves with lower alpha
            for i in range(min(50, self.num_simulations)):  # Limit to 50 to avoid overcrowding
                sim_equity_path = os.path.join(self.output_dir, f'simulation_{i}', 'equity_curve.csv')
                if os.path.exists(sim_equity_path):
                    try:
                        sim_equity = pd.read_csv(sim_equity_path)
                        sim_equity['Date'] = pd.to_datetime(sim_equity['Date'])
                        sim_equity.set_index('Date', inplace=True)
                        
                        # Plot permutation in light blue with low alpha
                        plt.plot(sim_equity.index, sim_equity['Value'], color='lightblue', alpha=0.1, linewidth=0.5)
                    except Exception as e:
                        pass
            
            # Plot original equity curve in red with thicker line
            plt.plot(original_equity.index, original_equity['Value'], color='red', linewidth=2.5, 
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