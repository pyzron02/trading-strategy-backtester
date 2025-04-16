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
                      end_date, param_dir, warmup_period, initial_capital, commission, 
                      data_dir, i)
                      
    Returns:
        tuple: (i, results, params) - index, backtest results, and parameters
    """
    strategy_name, tickers, params, start_date, end_date, param_dir, warmup_period, initial_capital, commission, data_dir, i = args
    
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
            initial_capital=initial_capital,
            commission=commission,
            data_dir=data_dir,
            plot=False  # Don't generate plots for individual parameter trials
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
                 output_dir='output/in_sample_excellence', parameter_grid=None, param_grid_file=None,
                 n_trials=100, optimization_metric='sharpe_ratio', random_seed=42, initial_capital=100000.0,
                 commission=0.001, data_dir="input", max_combinations=None, verbose=False, plot=False):
        """
        Initialize the InSampleExcellence test.
        
        Args:
            strategy_name (str): Name of the strategy to test
            tickers (list): List of ticker symbols to test
            start_date (str): Start date for the test period (YYYY-MM-DD)
            end_date (str): End date for the test period (YYYY-MM-DD)
            output_dir (str): Directory to save test results
            parameter_grid (dict): Grid of parameters to optimize
            param_grid_file (str): Path to a JSON file with parameter grid definition
            n_trials (int): Number of parameter combinations to try
            optimization_metric (str): Metric to optimize (e.g., 'sharpe_ratio')
            random_seed (int): Random seed for reproducibility
        """
        self.strategy_name = strategy_name
        self.tickers = tickers if tickers is not None else ['SPY']
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.parameter_grid = parameter_grid
        self.param_grid_file = param_grid_file
        self.n_trials = n_trials
        self.optimization_metric = optimization_metric
        self.random_seed = random_seed
        self.initial_capital = initial_capital
        self.commission = commission
        self.data_dir = data_dir
        self.max_combinations = max_combinations
        self.verbose = verbose
        self.plot = plot  # Whether to generate plots
        
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
        
        # If param_grid_file is provided, load from file
        if self.param_grid_file is not None and os.path.exists(self.param_grid_file):
            try:
                with open(self.param_grid_file, 'r') as f:
                    param_grid = json.load(f)
                # Log that we're using parameter grid from file
                with open(self.log_file, 'a') as f:
                    f.write(f"Using parameter grid from file: {self.param_grid_file}\n")
                    f.write(f"Parameter grid: {param_grid}\n\n")
                return param_grid
            except Exception as e:
                # Log error and fall back to default grid
                with open(self.log_file, 'a') as f:
                    f.write(f"Error loading parameter grid from file: {e}\n")
                    f.write("Falling back to default parameter grid.\n\n")
            
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
        Run optimization process and find the best parameter combination.
        
        Args:
            metric (str): Metric to optimize (e.g., 'sharpe_ratio', 'profit_factor')
            max_combinations (int): Maximum number of parameter combinations to test
            n_jobs (int): Number of parallel processes to use
            
        Returns:
            tuple: (best_parameters, trials_dataframe)
        """
        self.optimization_metric = metric
        parameter_combinations = self._generate_parameter_combinations(self.param_grid, max_combinations)
        
        # If parameter combinations is empty, return early
        if not parameter_combinations:
            with open(self.log_file, 'a') as f:
                f.write("No parameter combinations could be generated.\n")
            return None, pd.DataFrame()
        
        # Create parameter directories
        param_dirs = []
        for i in range(len(parameter_combinations)):
            param_dir = os.path.join(self.output_dir, f"params_{i}")
            os.makedirs(param_dir, exist_ok=True)
            param_dirs.append(param_dir)
        
        # Prepare arguments for parallel execution
        args_list = []
        for i, (params, param_dir) in enumerate(zip(parameter_combinations, param_dirs)):
            args_list.append((
                self.strategy_name, 
                self.tickers, 
                params, 
                self.start_date, 
                self.end_date, 
                param_dir,
                None,  # warmup_period
                getattr(self, 'initial_capital', 100000.0),
                getattr(self, 'commission', 0.001),
                getattr(self, 'data_dir', 'input'),
                i
            ))
        
        # Determine number of processes to use
        if n_jobs is None:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)
        n_jobs = min(n_jobs, len(args_list))
        
        # Log progress information
        with open(self.log_file, 'a') as f:
            f.write(f"Running optimization with {len(parameter_combinations)} parameter combinations\n")
            f.write(f"Using {n_jobs} parallel processes\n")
            f.write("Starting parallel backtests...\n\n")
        
        # Clear previous results
        self.results = []
        
        # Run backtests in parallel with progress bar
        with tqdm(total=len(args_list), desc="Backtesting") as pbar:
            try:
                # Use process pool for parallel execution
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    # Submit all jobs
                    future_to_idx = {executor.submit(_run_single_backtest, args): i for i, args in enumerate(args_list)}
                    
                    # Process results as they complete
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            i, result, params = future.result()
                            if result:
                                self.results.append({
                                    'parameters': params,
                                    'metrics': result.get('metrics', {}),
                                    'total_return': result.get('total_return', 0),
                                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                                    'max_drawdown': result.get('max_drawdown', 0),
                                    'equity_curve': result.get('equity_curve', None)
                                })
                            else:
                                self.results.append({
                                    'parameters': params,
                                    'metrics': {},
                                    'total_return': 0,
                                    'sharpe_ratio': 0,
                                    'max_drawdown': 0,
                                    'equity_curve': None
                                })
                        except Exception as e:
                            with open(self.log_file, 'a') as f:
                                f.write(f"Error in backtest {idx}: {str(e)}\n")
                        pbar.update(1)
            except Exception as e:
                with open(self.log_file, 'a') as f:
                    f.write(f"Error in parallel execution: {str(e)}\n")
        
        # Calculate metrics from results
        metrics = self._calculate_metrics(self.output_dir)
        
        # Get best parameter combination based on the selected metric
        best_params = None
        best_value = None
        trials_data = []
        
        # Determine whether this metric should be maximized or minimized
        is_maximize = True
        if metric.startswith('max_drawdown'):
            is_maximize = False  # For drawdown, lower is better
        
        # Extract parameters and metrics for each trial
        for i, result in enumerate(self.results):
            if not result:
                continue
                
            trial_data = {}
            
            # Add parameters with prefix
            if 'parameters' in result:
                for key, value in result['parameters'].items():
                    trial_data[f'param_{key}'] = value
            
            # Add metrics
            if 'metrics' in result:
                trial_data.update(result['metrics'])
            
            # Add other result fields
            for key in ['total_return', 'sharpe_ratio', 'max_drawdown']:
                if key in result:
                    trial_data[key] = result[key]
            
            # Add trial number
            trial_data['trial_number'] = i
            
            # Extract the metric value we're optimizing for
            current_value = None
            
            # First look in metrics
            if 'metrics' in result and metric in result['metrics']:
                current_value = result['metrics'][metric]
            # Then look directly in result
            elif metric in result:
                current_value = result[metric]
            
            # Update best parameters if this is better
            if current_value is not None and np.isfinite(current_value):
                if best_value is None or (is_maximize and current_value > best_value) or (not is_maximize and current_value < best_value):
                    best_value = current_value
                    best_params = result.get('parameters', {})
            
            trials_data.append(trial_data)
        
        # Create DataFrame for trials
        trials_df = pd.DataFrame(trials_data) if trials_data else pd.DataFrame()
        
        # Save metrics to file
        self._save_best_parameters(
            best_params,
            {metric: best_value} if best_value is not None else {},
            parameter_combinations,
            metric
        )
        
        # Log completion
        with open(self.log_file, 'a') as f:
            f.write(f"Optimization complete.\n")
            if best_params:
                f.write(f"Best parameters: {best_params}\n")
                f.write(f"Best {metric}: {best_value}\n")
            else:
                f.write("No valid results found.\n")
        
        # Plot parameter importance if we have enough data
        if not trials_df.empty and len(trials_df) > 5:
            try:
                self._plot_parameter_importance(metric)
            except Exception as e:
                with open(self.log_file, 'a') as f:
                    f.write(f"Error creating parameter importance plot: {str(e)}\n")
        
        return best_params, trials_df
    
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
        # Skip plotting if plot is disabled
        if not self.plot:
            return
            
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

    def run(self):
        """
        Run the optimization process.
        
        Returns:
            tuple: (best_parameters, trials_dataframe)
        """
        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations(self.param_grid, max_combinations=self.n_trials)
        
        # Log information about the optimization process
        with open(self.log_file, 'a') as f:
            f.write(f"Running optimization with {len(parameter_combinations)} parameter combinations\n")
            f.write(f"Optimization metric: {self.optimization_metric}\n\n")
        
        # Run the optimization
        best_params, trials_df = self.run_optimization(
            metric=self.optimization_metric,
            max_combinations=self.n_trials
        )
        
        # Check if we have valid results
        if best_params is None or trials_df is None or trials_df.empty:
            print("Warning: No valid parameter combinations found during optimization")
            
            # Try to salvage results if we have any parameter combinations
            if parameter_combinations and len(parameter_combinations) > 0:
                # Use the first parameter combination as a fallback
                best_params = parameter_combinations[0]
                print(f"Using the first parameter combination as fallback: {best_params}")
                
                # Create a minimal trials dataframe if needed
                if trials_df is None or trials_df.empty:
                    if hasattr(self, 'results') and self.results:
                        # Try to construct from existing results
                        trials_df = pd.DataFrame(self.results)
                    else:
                        # Create a minimal dataframe with just parameters
                        trials_data = []
                        for i, params in enumerate(parameter_combinations):
                            param_data = {f'param_{k}': v for k, v in params.items()}
                            param_data['trial_number'] = i
                            param_data[self.optimization_metric] = 0.0  # Default metric value
                            trials_data.append(param_data)
                        trials_df = pd.DataFrame(trials_data)
        
        # Save trial data to CSV
        if trials_df is not None and not trials_df.empty:
            trials_csv = os.path.join(self.output_dir, f"{self.strategy_name}_trials.csv")
            trials_df.to_csv(trials_csv, index=False)
            print(f"Trial data saved to {trials_csv}")
        
        return best_params, trials_df

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