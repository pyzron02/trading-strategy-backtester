#!/usr/bin/env python3
# in_sample_excellence.py - Optimize strategy parameters on historical data

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from datetime import datetime
from tqdm import tqdm
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add the current directory to the path
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from engine.run_backtest import run_backtest
from engine.logging_system import logger

# Function to execute a single backtest (used for parallel processing)
def _run_single_backtest(args):
    """
    Run a single backtest with the given parameters.
    
    Args:
        args (tuple): Tuple containing (strategy_name, tickers, params, start_date, 
                      end_date, param_dir, warmup_period, i)
                      
    Returns:
        tuple: (i, results, params) - index, backtest results, and parameters
    """
    strategy_name, tickers, params, start_date, end_date, param_dir, warmup_period, i = args
    
    # Create directory if it doesn't exist
    os.makedirs(param_dir, exist_ok=True)
    
    # Save parameters to file with improved formatting
    params_file = os.path.join(param_dir, "parameters.txt")
    with open(params_file, 'w') as f:
        f.write(f"Parameter Set {i} for {strategy_name}\n")
        f.write(f"Testing period: {start_date} to {end_date}\n\n")
        f.write("Parameters:\n")
        # Format the parameters in a clear, consistent way
        for param, value in params.items():
            f.write(f"  {param}: {value}\n")
        
        # Add timestamp for tracking
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Run backtest with these parameters
        results = run_backtest(
            strategy_name=strategy_name,
            tickers=tickers,
            parameters=params,
            start_date=start_date,
            end_date=end_date,
            output_dir=param_dir,
            warmup_period=warmup_period
        )
        
        # Save results to a file
        results_file = os.path.join(param_dir, "backtest_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Update parameters.txt with the results if they're available
        if results:
            try:
                # Extract key metrics for the parameter file
                with open(params_file, 'a') as f:
                    f.write("\nResults:\n")
                    if 'metrics' in results:
                        for metric_name, metric_value in results['metrics'].items():
                            f.write(f"  {metric_name}: {metric_value}\n")
                    
                    if 'total_return' in results:
                        f.write(f"  total_return: {results['total_return']}\n")
                    
                    if 'final_value' in results and 'initial_value' in results:
                        initial = results['initial_value']
                        final = results['final_value']
                        pct_return = ((final - initial) / initial) * 100 if initial > 0 else 0
                        f.write(f"  return_pct: {pct_return:.2f}%\n")
            except Exception as e:
                print(f"Warning: Could not update parameters.txt with results: {e}")
        
        return (i, results, params)
    except Exception as e:
        # Log the error to the parameters file
        with open(params_file, 'a') as f:
            f.write(f"\nERROR: {e}\n")
        
        print(f"Error in backtest {i} with params {params}: {e}")
        return (i, None, params)

class InSampleExcellence:
    """
    Optimize strategy parameters on historical data to achieve strong performance metrics.
    This class performs a grid search over parameter combinations to find the best performing
    strategy configuration based on specified metrics.
    """
    
    def __init__(self, strategy_name, tickers=None, start_date='2015-01-01', end_date='2019-12-31',
                 output_dir='output/in_sample_excellence', parameter_grid=None, random_seed=42):
        """
        Initialize the InSampleExcellence test.
        
        Args:
            strategy_name (str): Name of the strategy to test
            tickers (list): List of ticker symbols to test
            start_date (str): Start date for the test period (YYYY-MM-DD)
            end_date (str): End date for the test period (YYYY-MM-DD)
            output_dir (str): Directory to save test results
            parameter_grid (dict): Grid of parameters to optimize
            random_seed (int): Random seed for reproducibility
        """
        self.strategy_name = strategy_name
        self.tickers = tickers if tickers is not None else ['SPY']
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.parameter_grid = parameter_grid
        self.random_seed = random_seed
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # Initialize results
        self.results = []
        self.best_result = None
        
        # Log file
        self.log_file = os.path.join(self.output_dir, f"{self.strategy_name}_optimization_log.txt")
        with open(self.log_file, 'w') as f:
            f.write(f"Parameter Optimization Log for {self.strategy_name}\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tickers: {self.tickers}\n")
            f.write(f"Period: {self.start_date} to {self.end_date}\n\n")
        
        # Define parameter grid based on strategy
        self.param_grid = self._get_param_grid(strategy_name)
    
    def _get_param_grid(self, strategy_name):
        """
        Get the parameter grid for the specified strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            
        Returns:
            dict: Parameter grid for the strategy
        """
        # If parameter_grid is provided, use it
        if self.parameter_grid is not None:
            return self.parameter_grid
            
        # Otherwise, use default parameter grids based on strategy name
        if strategy_name == 'SimpleStock':
            return {
                'sma_period': [10, 20, 50, 100, 200],
                'position_size': [10, 20, 50, 100]
            }
        elif strategy_name == 'MultiPositionStrategy':
            return {
                'sma_period': [10, 20, 50, 100],
                'position_size': [10, 20, 50],
                'max_positions': [3, 5, 10]
            }
        elif strategy_name == 'AuctionMarketStrategy':
            return {
                'volume_threshold': [1.5, 2.0, 2.5, 3.0],
                'price_threshold': [0.5, 1.0, 1.5, 2.0],
                'position_size': [10, 20, 50]
            }
        else:
            # Default parameter grid
            return {
                'param1': [1, 2, 3],
                'param2': [10, 20, 30]
            }
    
    def _generate_parameter_combinations(self, param_grid, max_combinations=100):
        """
        Generate parameter combinations for grid search, limiting to max_combinations.
        
        Args:
            param_grid (dict): Parameter grid with parameter names as keys and lists of values as values.
            max_combinations (int): Maximum number of combinations to generate.
            
        Returns:
            list: List of parameter dictionaries.
        """
        # Calculate all possible combinations
        keys = param_grid.keys()
        values = param_grid.values()
        all_combinations = list(itertools.product(*values))
        
        # If there are too many combinations, sample randomly
        if len(all_combinations) > max_combinations:
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            combinations = [all_combinations[i] for i in indices]
        else:
            combinations = all_combinations
        
        # Convert to list of dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def run_optimization(self, metric='sharpe_ratio', max_combinations=100, n_jobs=None):
        """
        Run the parameter optimization process.
        
        Args:
            metric (str): Metric to optimize for (e.g., 'sharpe_ratio', 'profit_factor', 'total_return')
            max_combinations (int): Maximum number of parameter combinations to test
            n_jobs (int): Number of parallel jobs. If None, will use all available CPU cores.
            
        Returns:
            dict: Results of the optimization process
        """
        # Determine number of parallel jobs to use
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
        n_jobs = max(1, min(n_jobs, multiprocessing.cpu_count()))  # Ensure valid range
        
        print(f"Starting parameter optimization for {self.strategy_name}")
        print(f"Optimizing for {metric}")
        print(f"Testing up to {max_combinations} parameter combinations")
        print(f"Using {n_jobs} CPU cores for parallel processing")
        
        # Get parameter grid
        param_grid = self.param_grid
        print(f"Parameter grid: {param_grid}")
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(param_grid, max_combinations)
        print(f"Testing {len(param_combinations)} parameter combinations")
        
        # Create a timestamp for this test run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_dir = os.path.join(self.output_dir, f"test_{timestamp}")
        os.makedirs(test_dir, exist_ok=True)
        
        # Prepare arguments for parallel processing
        backtest_args = []
        for i, params in enumerate(param_combinations):
            # Create a unique directory for this parameter set
            param_dir = os.path.join(test_dir, f"params_{i}")
            
            # Calculate warmup period based on strategy and parameters
            warmup_period = 60  # Default
            if self.strategy_name == 'MACrossover':
                if params and 'slow_period' in params:
                    warmup_period = params['slow_period'] * 2
                else:
                    warmup_period = 60  # Default is twice the default slow period (30)
            
            # Prepare args tuple
            args = (self.strategy_name, self.tickers, params, self.start_date, 
                   self.end_date, param_dir, warmup_period, i)
            backtest_args.append(args)
        
        # Run backtests in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all jobs
            future_to_idx = {executor.submit(_run_single_backtest, args): i 
                            for i, args in enumerate(backtest_args)}
            
            # Process results as they complete
            with tqdm(total=len(param_combinations), desc="Testing parameters") as pbar:
                for future in as_completed(future_to_idx):
                    i, results, params = future.result()
                    pbar.update(1)
                    
                    if results is not None:
                        # Calculate metrics from the results dictionary directly
                        metrics = self._extract_metrics_from_results(results)
                        
                        # Add parameters and metrics to results
                        result = {
                            'parameters': params,
                            **metrics
                        }
                        self.results.append(result)
                        
                        # Log results
                        with open(self.log_file, 'a') as f:
                            f.write(f"Parameters {i}:\n")
                            for param, value in params.items():
                                f.write(f"  {param}: {value}\n")
                            f.write(f"  {metric}: {metrics.get(metric, 'N/A')}\n\n")
        
        # Find best parameters based on the specified metric
        if self.results:
            # Sort results by the specified metric (higher is better)
            sorted_results = sorted(self.results, key=lambda x: x.get(metric, 0), reverse=True)
            best_result = sorted_results[0]
            
            print(f"\nBest parameters found:")
            for param, value in best_result['parameters'].items():
                print(f"  {param}: {value}")
            print(f"Best {metric}: {best_result.get(metric, 'N/A')}")
            
            # Extract metrics dictionary for best result
            best_metrics = {k: v for k, v in best_result.items() if k != 'parameters'}
            
            # Save best parameters with separated metrics dictionary
            results = self._save_best_parameters(best_result['parameters'], best_metrics, param_combinations, metric)
            
            # Plot parameter importance
            self._plot_parameter_importance(metric)
            
            # Return best parameters and metrics
            return {
                'best_parameters': best_result['parameters'],
                'best_metrics': best_metrics,
                'best_sharpe': best_result.get('sharpe_ratio', 0),
                'best_profit_factor': best_result.get('profit_factor', 0),
                'best_total_return': best_result.get('total_return', 0),
                'all_results': self.results
            }
        else:
            print("No valid results were found during optimization")
            return {
                'best_parameters': {},
                'best_metrics': {},
                'best_sharpe': 0,
                'best_profit_factor': 0,
                'best_total_return': 0,
                'all_results': []
            }
    
    def _calculate_metrics(self, results_path):
        """
        Calculate performance metrics from backtest results.
        
        Args:
            results_path (str): Path to the backtest results directory.
            
        Returns:
            dict: Dictionary of performance metrics.
        """
        # Load equity curve
        equity_curve_path = os.path.join(results_path, 'equity_curve.csv')
        if not os.path.exists(equity_curve_path):
            return {}
        
        equity_curve = pd.read_csv(equity_curve_path)
        equity_curve['Date'] = pd.to_datetime(equity_curve['Date'])
        equity_curve.set_index('Date', inplace=True)
        
        # Load trade log
        trade_log_path = os.path.join(results_path, 'trade_log.csv')
        if not os.path.exists(trade_log_path):
            return {}
        
        trade_log = pd.read_csv(trade_log_path)
        
        # Calculate metrics
        metrics = {}
        
        # Calculate returns
        equity_curve['Return'] = equity_curve['Value'].pct_change()
        
        # Total return
        initial_value = equity_curve['Value'].iloc[0]
        final_value = equity_curve['Value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        metrics['total_return'] = total_return
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        metrics['annualized_return'] = annualized_return
        
        # Volatility
        daily_volatility = equity_curve['Return'].std()
        annualized_volatility = daily_volatility * (252 ** 0.5)
        metrics['volatility'] = annualized_volatility
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # Maximum drawdown
        equity_curve['Peak'] = equity_curve['Value'].cummax()
        equity_curve['Drawdown'] = (equity_curve['Value'] - equity_curve['Peak']) / equity_curve['Peak']
        max_drawdown = abs(equity_curve['Drawdown'].min())
        metrics['max_drawdown'] = max_drawdown
        
        # Calculate trade metrics
        if not trade_log.empty:
            # Filter to closed trades
            closed_trades = trade_log[trade_log['type'] == 'close']
            
            # Win rate
            winning_trades = closed_trades[closed_trades['pnl'] > 0]
            win_rate = len(winning_trades) / len(closed_trades) if len(closed_trades) > 0 else 0
            metrics['win_rate'] = win_rate
            
            # Profit factor
            gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
            losing_trades = closed_trades[closed_trades['pnl'] <= 0]
            gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            metrics['profit_factor'] = profit_factor
            
            # Average win/loss ratio
            avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if not losing_trades.empty else 0
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
            metrics['win_loss_ratio'] = win_loss_ratio
            
            # Number of trades
            metrics['num_trades'] = len(closed_trades)
        
        return metrics
    
    def _save_best_parameters(self, best_params, best_metrics, parameter_combinations, metric_name):
        """
        Save the best parameters to a file and return them for further use.
        """
        # Get the best parameter set info
        best_params_dict = best_params.copy()
        
        # If not a dict (e.g. a parameter set object), convert
        if not isinstance(best_params_dict, dict):
            best_params_dict = best_params_dict.to_dict()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save to pickle file
        best_params_pickle = os.path.join(self.output_dir, 'best_parameters.pkl')
        with open(best_params_pickle, 'wb') as f:
            pickle.dump(best_params_dict, f)
        
        # Save to human-readable text file
        best_params_txt = os.path.join(self.output_dir, 'best_parameters.txt')
        with open(best_params_txt, 'w') as f:
            f.write(f"Strategy: {self.strategy_name}\n")
            f.write(f"Optimization Metric: {metric_name}\n")
            f.write(f"In-sample Period: {self.start_date} to {self.end_date}\n\n")
            f.write("Parameters:\n")
            for param, value in best_params_dict.items():
                f.write(f"  {param}: {value}\n")
            
            f.write("\nPerformance Metrics:\n")
            for metric, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric}: {value:.4f}\n")
                else:
                    f.write(f"  {metric}: {value}\n")

        # Save to standard parameters.txt file that tools expect
        params_txt = os.path.join(self.output_dir, 'parameters.txt')
        with open(params_txt, 'w') as f:
            # Header
            f.write(f"# {self.strategy_name} - Optimized Parameters\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Optimization Metric: {metric_name}\n")
            f.write(f"# In-sample Period: {self.start_date} to {self.end_date}\n\n")
            
            # Parameters in key:value format
            for param, value in best_params_dict.items():
                f.write(f"{param}: {value}\n")
            
            # Performance metrics (with simple formatting for numeric values)
            f.write("\n# Performance Metrics\n")
            for metric, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{metric}: {value:.6f}\n")
                else:
                    f.write(f"{metric}: {value}\n")
        
        logger.info('in_sample', f"Best parameters saved to {best_params_txt} and {params_txt}")
        
        # Create dictionary of results
        results = {
            'best_parameters': best_params_dict,
            'best_metrics': best_metrics,
            'all_results': parameter_combinations,
            'metric_name': metric_name
        }
        
        return results
    
    def _plot_parameter_importance(self, metric):
        """
        Plot the importance of each parameter on the specified metric.
        
        Args:
            metric (str): Metric to analyze
        """
        if not self.results:
            return
        
        # Create a figure with subplots for each parameter
        param_names = list(self.results[0]['parameters'].keys())
        n_params = len(param_names)
        
        if n_params == 0:
            return
        
        # Create a figure with subplots
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))
        if n_params == 1:
            axes = [axes]  # Make axes iterable if there's only one subplot
        
        # For each parameter, plot its effect on the metric
        for i, param in enumerate(param_names):
            # Extract parameter values and corresponding metric values
            param_values = []
            metric_values = []
            
            for result in self.results:
                param_values.append(result['parameters'][param])
                metric_values.append(result.get(metric, 0))
            
            # Create a DataFrame for easier plotting
            df = pd.DataFrame({
                'param_value': param_values,
                'metric_value': metric_values
            })
            
            # Group by parameter value and calculate mean metric value
            grouped = df.groupby('param_value')['metric_value'].mean().reset_index()
            
            # Plot
            x_values = range(len(grouped))  # Create numeric x positions
            axes[i].bar(x_values, grouped['metric_value'])
            axes[i].set_title(f'Effect of {param} on {metric}')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel(metric)
            
            # Set both x-ticks and x-tick labels
            axes[i].set_xticks(x_values)  # Set tick positions
            
            # Rotate x-axis labels if there are many values
            if len(grouped) > 5:
                axes[i].set_xticklabels(grouped['param_value'].astype(str), rotation=45)
            else:
                axes[i].set_xticklabels(grouped['param_value'].astype(str))
        
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.output_dir, f'parameter_importance_{metric}.png')
        plt.savefig(fig_path)
        plt.close()
        
        print(f"Parameter importance plot saved to {fig_path}")

    def _extract_metrics_from_results(self, results):
        """
        Extract performance metrics directly from the results dictionary.
        
        Args:
            results (dict): Dictionary of backtest results.
            
        Returns:
            dict: Dictionary of performance metrics.
        """
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = results.get('total_return', 0)
        metrics['alpha'] = results.get('alpha', 0)
        metrics['benchmark_return'] = results.get('benchmark_return', 0)
        
        # Calculate Sharpe ratio if we have equity curve data
        equity_curve = results.get('equity_curve', [])
        if equity_curve:
            # Convert to DataFrame
            equity_df = pd.DataFrame(equity_curve)
            if 'Date' in equity_df.columns and 'Value' in equity_df.columns:
                equity_df['Date'] = pd.to_datetime(equity_df['Date'])
                equity_df.set_index('Date', inplace=True)
                
                # Calculate daily returns
                equity_df['Return'] = equity_df['Value'].pct_change().dropna()
                
                # Calculate Sharpe ratio (annualized)
                if len(equity_df) > 1 and equity_df['Return'].std() > 0:
                    sharpe = equity_df['Return'].mean() / equity_df['Return'].std() * np.sqrt(252)
                    metrics['sharpe_ratio'] = sharpe
                else:
                    metrics['sharpe_ratio'] = 0
        
        # Add other metrics if available
        if hasattr(results, 'get'):
            for key in ['max_drawdown', 'win_rate', 'profit_factor']:
                if key in results:
                    metrics[key] = results[key]
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters on historical data.")
    parser.add_argument('--strategy_name', type=str, default='SimpleStock',
                        help="Name of the strategy to optimize (e.g., SimpleStock, MultiPosition, AuctionMarket)")
    parser.add_argument('--tickers', type=str, default=None,
                        help="Comma-separated list of ticker symbols (e.g., MSFT,AAPL)")
    parser.add_argument('--start_date', type=str, default='2015-01-01',
                        help="Start date for in-sample period (YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, default='2019-12-31',
                        help="End date for in-sample period (YYYY-MM-DD)")
    parser.add_argument('--metric', type=str, default='profit_factor',
                        help="Metric to optimize (profit_factor, sharpe_ratio, total_return, win_rate)")
    parser.add_argument('--max_combinations', type=int, default=100,
                        help="Maximum number of parameter combinations to test")
    
    args = parser.parse_args()
    
    # Parse tickers if provided
    tickers = None
    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    
    # Run optimization
    optimizer = InSampleExcellence(
        strategy_name=args.strategy_name,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    optimizer.run_optimization(
        metric=args.metric,
        max_combinations=args.max_combinations
    )

if __name__ == "__main__":
    main() 