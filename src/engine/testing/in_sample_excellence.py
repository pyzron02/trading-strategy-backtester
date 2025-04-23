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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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
        
        # Initialize logger
        self.logger = logger.get_logger(__name__)
        
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
        self.param_grid = self._get_param_grid()
    
    def _get_param_grid(self):
        """
        Get the parameter grid for the strategy.
        
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
        strategy_name = self.strategy_name
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
        elif strategy_name == 'AuctionMarketStrategy' or strategy_name == 'AuctionMarket':
            return {
                'param_preset': ['default', 'aggressive', 'conservative'],
                'value_area': [0.7, 0.75, 0.8],
                'position_size': [50, 100, 200],
                'risk_percent': [0.01, 0.02, 0.05],
                'atr_period': [14, 20, 30]
            }
        else:
            # Default parameter grid
            return {
                'param1': [1, 2, 3],
                'param2': [10, 20, 30]
            }
    
    def _generate_parameter_combinations(self, param_grid, max_combinations=None):
        """
        Generate parameter combinations for grid search, limiting to max_combinations.
        
        Args:
            param_grid (dict): Parameter grid with parameter names as keys and lists of values as values.
            max_combinations (int): Maximum number of combinations to generate. If None, uses self.max_combinations.
            
        Returns:
            list: List of parameter dictionaries.
        """
        if not param_grid:
            self.logger.error("Parameter grid is empty, cannot generate combinations")
            return []
            
        # Use class max_combinations if none provided
        if max_combinations is None:
            max_combinations = self.max_combinations if self.max_combinations is not None else self.n_trials
            self.logger.debug(f"Using max_combinations: {max_combinations}")
            
        try:
            # Calculate all possible combinations
            keys = list(param_grid.keys())
            values = list(param_grid.values())
            
            # Log parameter grid information
            self.logger.info(f"Parameter grid contains {len(keys)} parameters")
            for k, v in param_grid.items():
                self.logger.debug(f"Parameter {k}: {len(v)} values - {v}")
            
            # Calculate total number of combinations
            total_combinations = 1
            for v in values:
                total_combinations *= len(v)
                
            self.logger.info(f"Total possible combinations: {total_combinations}")
            
            # If total is less than max, use all combinations
            if total_combinations <= max_combinations:
                self.logger.info(f"Using all {total_combinations} combinations (less than max: {max_combinations})")
                all_combinations = list(itertools.product(*values))
                combinations = all_combinations
            else:
                # If there are too many combinations, sample randomly
                self.logger.info(f"Sampling {max_combinations} from {total_combinations} possible combinations")
                
                # Set random seed for reproducibility
                np.random.seed(self.random_seed)
                
                # Generate all combinations if reasonable, otherwise sample strategically
                if total_combinations > 1000000:  # If too many combinations to generate all
                    self.logger.warning(f"Too many combinations to enumerate ({total_combinations}). Using strategic sampling.")
                    combinations = []
                    for _ in range(max_combinations):
                        # Generate a random parameter combination by selecting one value from each parameter list
                        combo = tuple(np.random.choice(param_values) for param_values in values)
                        combinations.append(combo)
                else:
                    # Generate all combinations and then sample
                    all_combinations = list(itertools.product(*values))
                    indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
                    combinations = [all_combinations[i] for i in indices]
            
            # Convert to list of dictionaries
            param_combinations = []
            for combo in combinations:
                param_dict = dict(zip(keys, combo))
                param_combinations.append(param_dict)
            
            # Log a sample of generated combinations
            sample_size = min(5, len(param_combinations))
            sample_combinations = param_combinations[:sample_size]
            self.logger.debug(f"Sample of generated combinations: {sample_combinations}")
            
            return param_combinations
            
        except Exception as e:
            self.logger.error(f"Error generating parameter combinations: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return []
    
    def run_optimization(self, metric_name="sharpe_ratio"):
        """
        Run the parameter optimization process using grid search.
        
        Args:
            metric_name (str): The metric to optimize (default: sharpe_ratio)
            
        Returns:
            dict: Optimization results containing all parameter combinations and their metrics
        """
        self.logger.info(f"Starting parameter optimization for {self.strategy_name} using {metric_name}")
        
        # Get parameter grid for the strategy
        param_grid = self._get_param_grid()
        if not param_grid:
            self.logger.error("Parameter grid is empty. Cannot run optimization.")
            # Create an error file to indicate the optimization failed
            self._create_error_log("Parameter grid is empty. Cannot run optimization.")
            return {"error": "Parameter grid is empty", "results": pd.DataFrame()}
            
        # Generate all parameter combinations
        parameter_combinations = self._generate_parameter_combinations(param_grid)
        if not parameter_combinations:
            self.logger.error("No parameter combinations generated. Cannot run optimization.")
            self._create_error_log("No parameter combinations generated.")
            return {"error": "No parameter combinations generated", "results": pd.DataFrame()}
            
        total_combinations = len(parameter_combinations)
        self.logger.info(f"Generated {total_combinations} parameter combinations for optimization")
        
        # Prepare results storage
        results = []
        successful_combinations = 0
        
        # Create progress bar if verbose mode is enabled
        if self.verbose:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_combinations, desc="Optimizing")
            except ImportError:
                self.logger.info("tqdm not installed, progress bar disabled")
                pbar = None
        else:
            pbar = None
        
        # Run backtest for each parameter combination
        for i, params in enumerate(parameter_combinations):
            try:
                self.logger.debug(f"Testing combination {i+1}/{total_combinations}: {params}")
                
                # Run backtest with current parameters
                backtest_result = self._run_single_backtest(params)
                
                if backtest_result:
                    # Extract metrics from backtest results
                    metrics = self._extract_metrics_from_results(backtest_result)
                    
                    # Add parameter values to metrics dictionary with param_ prefix
                    for param_name, param_value in params.items():
                        metrics[f'param_{param_name}'] = param_value
                    
                    # Make sure we have at least one key metric
                    has_key_metric = False
                    for key_metric in ['total_return', 'sharpe_ratio', 'calmar_ratio']:
                        if key_metric in metrics:
                            has_key_metric = True
                            break
                    
                    if not has_key_metric:
                        # If no key metrics, add a default one to prevent failures
                        self.logger.warning(f"No key metrics found for combination {i+1}, adding default metric")
                        metrics['total_return'] = 0.0
                    
                    # Add to results list
                    results.append(metrics)
                    successful_combinations += 1
                    
                    if self.verbose:
                        if metric_name in metrics:
                            self.logger.debug(f"Combination {i+1} {metric_name}: {metrics.get(metric_name, 'N/A')}")
                        else:
                            self.logger.debug(f"Combination {i+1} metrics: {list(metrics.keys())}")
                else:
                    self.logger.warning(f"No valid results for combination {i+1}")
                    
            except Exception as e:
                self.logger.error(f"Error testing combination {i+1}: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
                
            finally:
                # Update progress bar
                if pbar:
                    pbar.update(1)
                    
        # Close progress bar
        if pbar:
            pbar.close()
            
        self.logger.info(f"Completed {successful_combinations}/{total_combinations} parameter combinations")
        
        # Convert results to DataFrame for easier processing
        if not results:
            self.logger.warning("No successful combinations found during optimization")
            self._create_error_log("No successful combinations found during optimization.")
            return {"error": "No successful combinations", "results": pd.DataFrame(), "successful": 0, "total": total_combinations}
            
        results_df = pd.DataFrame(results)
        
        # Ensure the metric column exists
        if metric_name not in results_df.columns:
            self.logger.error(f"Metric '{metric_name}' not found in results. Available metrics: {list(results_df.columns)}")
            
            # Try to use an alternative metric
            alternative_metrics = ['total_return', 'sharpe_ratio', 'calmar_ratio', 'profit_factor']
            for alt_metric in alternative_metrics:
                if alt_metric in results_df.columns:
                    self.logger.info(f"Using alternative metric '{alt_metric}' instead of '{metric_name}'")
                    metric_name = alt_metric
                    break
            else:
                # If still no suitable metric, use the first numeric column
                numeric_cols = results_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0 and not all(col.startswith('param_') for col in numeric_cols):
                    # Find first numeric column that is not a parameter
                    for col in numeric_cols:
                        if not col.startswith('param_'):
                            metric_name = col
                            self.logger.info(f"No standard metrics found. Using '{metric_name}' as fallback")
                            break
                else:
                    self.logger.error("No usable metric columns found")
                    self._create_error_log("No usable metric columns found in optimization results.")
                    return {"error": "No metric columns", "results": results_df, "successful": successful_combinations, "total": total_combinations}
        
        # Save best parameters
        best_params = self._save_best_parameters(results_df, metric_name)
        
        # Check if best_params is empty
        if not best_params:
            self.logger.error("Failed to find or save best parameters")
            self._create_error_log("Failed to find or save best parameters.")
            return {"error": "Failed to save best parameters", "results": results_df, "successful": successful_combinations, "total": total_combinations}
        
        # Create parameter importance plots
        if self.plot:
            self._plot_parameter_importance(results_df, metric_name)
        
        return {
            "results": results_df, 
            "successful": successful_combinations, 
            "total": total_combinations,
            "best_params": best_params,
            "metric": metric_name
        }
        
    def _create_error_log(self, error_message):
        """Create an error log file to indicate optimization failure."""
        error_file = os.path.join(self.output_dir, "optimization_error.txt")
        try:
            with open(error_file, 'w') as f:
                f.write(f"Optimization Error: {error_message}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Strategy: {self.strategy_name}\n")
                f.write(f"Period: {self.start_date} to {self.end_date}\n")
            self.logger.info(f"Error log saved to {error_file}")
        except Exception as e:
            self.logger.error(f"Failed to create error log: {str(e)}")
    
    def _calculate_metrics(self, results, desired_metric=None):
        """
        Calculate performance metrics from backtest results.
        
        Args:
            results (dict): Dictionary containing backtest results
            desired_metric (str): Optional specific metric to return
            
        Returns:
            dict: Dictionary of calculated metrics or specific metric value if desired_metric is provided
        """
        metrics = {}
        
        try:
            # Check if results contains the necessary data
            if not results or not isinstance(results, dict):
                self.logger.warning("Invalid results format for metric calculation")
                return metrics
                
            # Extract portfolio dataframe if available
            portfolio = None
            if 'portfolio' in results and isinstance(results['portfolio'], pd.DataFrame):
                portfolio = results['portfolio']
            elif 'portfolio_history' in results and isinstance(results['portfolio_history'], pd.DataFrame):
                portfolio = results['portfolio_history']
            else:
                self.logger.warning("No portfolio data found in results")
                return metrics
                
            # Ensure portfolio has required columns
            required_cols = ['equity']
            if not all(col in portfolio.columns for col in required_cols):
                self.logger.warning(f"Portfolio missing required columns: {required_cols}")
                return metrics
                
            # Extract equity curve
            equity = portfolio['equity'].values
            returns = np.diff(equity) / equity[:-1]
            
            # Skip metrics calculation if we don't have enough data
            if len(equity) < 2:
                self.logger.warning("Not enough data points to calculate metrics")
                return metrics
                
            # Calculate basic metrics
            initial_equity = equity[0]
            final_equity = equity[-1]
            total_return = (final_equity / initial_equity) - 1
            
            metrics['initial_equity'] = initial_equity
            metrics['final_equity'] = final_equity
            metrics['total_return'] = total_return
            
            # Calculate additional metrics if we have enough data
            if len(returns) > 0:
                # Annualized metrics (assuming daily data)
                trading_days = 252
                years = len(returns) / trading_days
                
                if years > 0:
                    # Annual return
                    annual_return = (1 + total_return) ** (1 / years) - 1
                    metrics['annual_return'] = annual_return
                    
                    # Volatility
                    daily_std = np.std(returns)
                    annual_volatility = daily_std * np.sqrt(trading_days)
                    metrics['volatility'] = annual_volatility
                    
                    # Sharpe ratio (assuming risk-free rate of 0)
                    if annual_volatility > 0:
                        sharpe_ratio = annual_return / annual_volatility
                        metrics['sharpe_ratio'] = sharpe_ratio
                    
                # Maximum drawdown
                peak = np.maximum.accumulate(equity)
                drawdown = (equity - peak) / peak
                max_drawdown = abs(np.min(drawdown))
                metrics['max_drawdown'] = max_drawdown
                
                # Calmar ratio
                if max_drawdown > 0 and years > 0:
                    calmar_ratio = annual_return / max_drawdown
                    metrics['calmar_ratio'] = calmar_ratio
                
                # Win rate if trades are available
                if 'trades' in results and isinstance(results['trades'], pd.DataFrame) and len(results['trades']) > 0:
                    trades_df = results['trades']
                    if 'profit' in trades_df.columns:
                        profitable_trades = (trades_df['profit'] > 0).sum()
                        total_trades = len(trades_df)
                        if total_trades > 0:
                            win_rate = profitable_trades / total_trades
                            metrics['win_rate'] = win_rate
                            
                            # Average profit per trade
                            avg_profit = trades_df['profit'].mean()
                            metrics['avg_profit'] = avg_profit
                            
                            # Profit factor
                            gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum()
                            gross_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
                            
                            if gross_loss > 0:
                                profit_factor = gross_profit / gross_loss
                                metrics['profit_factor'] = profit_factor
            
            # Return specific metric if requested
            if desired_metric is not None:
                return metrics.get(desired_metric)
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return metrics
    
    def _calculate_annualized_return(self, equity_curve):
        """Calculate annualized return from equity curve."""
        try:
            # Calculate total return
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            
            # Calculate years
            start_date = equity_curve.index[0]
            end_date = equity_curve.index[-1]
            days = (end_date - start_date).days
            years = days / 365.25
            
            # Avoid division by zero
            if years <= 0:
                return total_return
                
            # Calculate annualized return
            return (1 + total_return) ** (1 / years) - 1
        except Exception:
            return 0
    
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown from equity curve."""
        try:
            # Calculate running maximum
            running_max = equity_curve.cummax()
            
            # Calculate drawdown
            drawdown = (equity_curve / running_max) - 1
            
            # Return maximum drawdown (will be negative)
            return abs(drawdown.min())
        except Exception:
            return 0
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """Calculate Sharpe ratio from returns."""
        try:
            # Annualize mean and std
            mean_return = returns.mean() * 252
            std_return = returns.std() * (252 ** 0.5)
            
            # Avoid division by zero
            if std_return == 0:
                return 0
                
            # Calculate Sharpe ratio
            return (mean_return - risk_free_rate) / std_return
        except Exception:
            return 0
    
    def _calculate_sortino_ratio(self, returns, risk_free_rate=0.0):
        """Calculate Sortino ratio from returns."""
        try:
            # Annualize mean and downside std
            mean_return = returns.mean() * 252
            
            # Only consider negative returns for downside risk
            negative_returns = returns[returns < 0]
            
            # If no negative returns, return a high value
            if len(negative_returns) == 0:
                return 100  # arbitrary high value
                
            downside_std = negative_returns.std() * (252 ** 0.5)
            
            # Avoid division by zero
            if downside_std == 0:
                return 0
                
            # Calculate Sortino ratio
            return (mean_return - risk_free_rate) / downside_std
        except Exception:
            return 0
    
    def _calculate_calmar_ratio(self, annualized_return, max_drawdown):
        """Calculate Calmar ratio."""
        try:
            # Avoid division by zero
            if max_drawdown == 0:
                return 0 if annualized_return <= 0 else 100  # arbitrary high value
                
            # Calculate Calmar ratio
            return annualized_return / max_drawdown
        except Exception:
            return 0
    
    def _calculate_win_rate(self, results):
        """Calculate win rate from trades."""
        try:
            trades = results.get('trades', [])
            
            # If no trades, return 0
            if not trades:
                return 0
                
            # Count profitable trades
            profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
            
            # Calculate win rate
            return profitable_trades / len(trades)
        except Exception:
            return 0
    
    def _calculate_profit_factor(self, results):
        """Calculate profit factor from trades."""
        try:
            trades = results.get('trades', [])
            
            # If no trades, return 0
            if not trades:
                return 0
                
            # Calculate gross profit and gross loss
            gross_profit = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
            gross_loss = sum(abs(trade.get('profit', 0)) for trade in trades if trade.get('profit', 0) < 0)
            
            # Avoid division by zero
            if gross_loss == 0:
                return 0 if gross_profit <= 0 else 100  # arbitrary high value
                
            # Calculate profit factor
            return gross_profit / gross_loss
        except Exception:
            return 0
    
    def _calculate_recovery_factor(self, total_return, max_drawdown):
        """Calculate recovery factor."""
        try:
            # Avoid division by zero
            if max_drawdown == 0:
                return 0 if total_return <= 0 else 100  # arbitrary high value
                
            # Calculate recovery factor
            return total_return / max_drawdown
        except Exception:
            return 0
    
    def _save_best_parameters(self, results_df, metric_name):
        """
        Save the best parameters from optimization results.
        
        Args:
            results_df (pd.DataFrame): DataFrame containing optimization results
            metric_name (str): The metric used for determining the best parameters
        
        Returns:
            dict: The best parameters
        """
        if not isinstance(results_df, pd.DataFrame):
            self.logger.warning(f"Results is not a DataFrame, it's a {type(results_df)}")
            return {}
            
        if results_df.empty:
            self.logger.warning("No results to save best parameters from")
            return {}
            
        try:
            # Check if the metric exists in the DataFrame
            if metric_name not in results_df.columns:
                self.logger.error(f"Metric '{metric_name}' not found in results DataFrame")
                available_metrics = list(results_df.columns)
                self.logger.info(f"Available metrics: {available_metrics}")
                
                # Use a default metric if available
                default_metrics = ['sharpe_ratio', 'total_return', 'annualized_return']
                for default_metric in default_metrics:
                    if default_metric in results_df.columns:
                        self.logger.info(f"Using '{default_metric}' as fallback metric")
                        metric_name = default_metric
                        break
                else:
                    self.logger.error("No suitable fallback metric found")
                    # Use the first numeric column as a last resort
                    numeric_cols = results_df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        metric_name = numeric_cols[0]
                        self.logger.info(f"Using '{metric_name}' as last resort metric")
                    else:
                        self.logger.error("No numeric columns found, cannot determine best parameters")
                        return {}
            
            # Copy the DataFrame to avoid modifying the original
            df = results_df.copy()
            
            # For metrics where lower is better (e.g., max_drawdown, volatility), invert the values
            invert_metrics = ['max_drawdown', 'volatility', 'max_drawdown_pct', 'ulcer_index']
            if metric_name in invert_metrics:
                self.logger.info(f"Inverting values for metric '{metric_name}' (lower is better)")
                # Avoid division by zero
                df[metric_name] = df[metric_name].apply(lambda x: -x if x != 0 else 0)
            
            # Sort by the metric (descending) to get the best parameters
            sorted_df = df.sort_values(by=metric_name, ascending=False)
            
            # Check if we actually have any valid rows after sorting
            if len(sorted_df) == 0:
                self.logger.error("No valid rows found after sorting by metric")
                return {}
            
            # Get the best row
            best_row = sorted_df.iloc[0]
            
            # Extract parameter columns
            # Step 1: Try columns with param_ prefix
            param_columns = [col for col in df.columns if col.startswith('param_')]
            
            # Step 2: If no param_ columns found, try to infer parameter columns
            if not param_columns:
                self.logger.warning("No columns with 'param_' prefix found. Attempting to infer parameter columns.")
                
                # Common known metric names to exclude (these are definitely not parameters)
                known_metric_names = set(['sharpe_ratio', 'total_return', 'annualized_return', 
                               'max_drawdown', 'win_rate', 'profit_factor', 'sortino_ratio',
                               'calmar_ratio', 'volatility', 'num_trades', 'avg_trade_pnl',
                               'profit_loss_ratio', 'avg_trade_duration', 'max_drawdown_pct',
                               'ulcer_index', 'recovery_factor', 'expectancy', 'gain_to_pain',
                               'omega_ratio', 'information_ratio', 'treynor_ratio'])
                
                # First check for columns that have both numeric and non-numeric values
                # (parameters often have consistent types across rows)
                param_candidates = set()
                
                # Add columns that have string values (likely strategy parameters)
                string_cols = df.select_dtypes(include=['object']).columns
                param_candidates.update(string_cols)
                
                # Add integer columns that have a small number of unique values (likely parameters)
                for col in df.select_dtypes(include=['int', 'int32', 'int64']).columns:
                    if col not in known_metric_names and df[col].nunique() < 50:  # Arbitrary threshold
                        param_candidates.add(col)
                
                # Add float columns with few unique values (likely parameters, not metrics)
                for col in df.select_dtypes(include=['float', 'float32', 'float64']).columns:
                    if col not in known_metric_names and df[col].nunique() < 30:  # Parameters usually have fewer unique values
                        param_candidates.add(col)
                
                # Remove columns that are likely performance metrics
                param_columns = [col for col in param_candidates if col not in known_metric_names]
                
                # If still no parameters found, use columns that don't match known metric names
                if not param_columns:
                    all_cols = set(df.columns)
                    param_columns = list(all_cols - known_metric_names)
                
                self.logger.info(f"Inferred parameter columns: {param_columns}")
            
            # Create a dictionary of best parameters
            best_params = {}
            for col in param_columns:
                # Extract the parameter name (remove 'param_' prefix if it exists)
                param_name = col[6:] if col.startswith('param_') else col
                param_value = best_row[col]
                
                # Convert NumPy types to native Python types
                if isinstance(param_value, (np.integer, np.int32, np.int64)):
                    param_value = int(param_value)
                elif isinstance(param_value, (np.float32, np.float64)):
                    param_value = float(param_value)
                elif isinstance(param_value, np.bool_):
                    param_value = bool(param_value)
                
                best_params[param_name] = param_value
            
            # If we still don't have any parameters, try one last approach - look at column names
            if not best_params:
                self.logger.warning("Still no parameters identified. Attempting heuristic approach.")
                # Look for columns that might represent strategy parameters based on name patterns
                potential_param_names = ['period', 'size', 'threshold', 'window', 'lookback', 
                                        'length', 'multiplier', 'factor', 'level', 'ratio', 
                                        'stop', 'target', 'limit', 'fast', 'slow', 'signal']
                
                for col in df.columns:
                    # Check if any of the potential parameter names appear in the column name
                    if any(param_name in col.lower() for param_name in potential_param_names):
                        best_params[col] = best_row[col]
            
            # Ensure we don't include the main metric in parameters
            if metric_name in best_params:
                del best_params[metric_name]
            
            # If we still have no parameters, issue a clear error
            if not best_params:
                self.logger.error("Failed to identify any parameter columns in the results dataframe")
                error_message = "Could not identify parameter columns in optimization results"
                self._create_error_log(error_message)
                return {}
            
            # Create a dictionary of best metrics
            all_cols = set(df.columns)
            param_cols_set = set(param_columns)
            metric_cols = all_cols - param_cols_set
            best_metrics = {col: best_row[col] for col in metric_cols if col in best_row}
            
            # Log the best parameters and their performance
            self.logger.info(f"Best parameters found for metric '{metric_name}':")
            for param, value in best_params.items():
                self.logger.info(f"  {param}: {value}")
            
            self.logger.info(f"Performance of best parameters:")
            for metric, value in best_metrics.items():
                self.logger.info(f"  {metric}: {value}")
            
            # Create summary information
            summary_info = {
                "strategy": self.strategy_name,
                "optimization_metric": metric_name,
                "best_value": float(best_row[metric_name]) if metric_name in best_row else None,
                "parameters": best_params,
                "metrics": best_metrics,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save the best parameters to a JSON file
            import json
            
            # Create the output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save the best parameters
            best_params_file = os.path.join(self.output_dir, f"best_params_{self.strategy_name}.json")
            with open(best_params_file, 'w') as f:
                json.dump(best_params, f, indent=4)
            
            # Also save a more comprehensive summary file
            summary_file = os.path.join(self.output_dir, f"optimization_summary_{self.strategy_name}.json")
            with open(summary_file, 'w') as f:
                json.dump(summary_info, f, indent=4)
            
            self.logger.info(f"Best parameters saved to {best_params_file}")
            self.logger.info(f"Optimization summary saved to {summary_file}")
            
            # Save a summary of all results
            results_file = os.path.join(self.output_dir, f"optimization_results_{self.strategy_name}.csv")
            results_df.to_csv(results_file, index=False)
            self.logger.info(f"All optimization results saved to {results_file}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error saving best parameters: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {}
    
    def _plot_parameter_importance(self, results_df, metric_name):
        """
        Plot the importance of each parameter on the optimization metric.
        
        Args:
            results_df (pd.DataFrame): DataFrame containing optimization results
            metric_name (str): The metric used for optimization
        """
        if not isinstance(results_df, pd.DataFrame):
            self.logger.warning(f"Results is not a DataFrame, it's a {type(results_df)}")
            return
            
        if results_df.empty:
            self.logger.warning("Results DataFrame is empty, cannot plot parameter importance")
            return
            
        # Check if the metric exists in the DataFrame
        if metric_name not in results_df.columns:
            self.logger.warning(f"Metric '{metric_name}' not found in columns, cannot plot parameter importance")
            return
            
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create output directory for plots
            plots_dir = os.path.join(self.output_dir, "parameter_plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Get parameter columns (those with 'param_' prefix)
            param_columns = [col for col in results_df.columns if col.startswith('param_')]
            
            if not param_columns:
                self.logger.warning("No parameter columns found, cannot plot parameter importance")
                return
                
            # Determine if we should maximize or minimize the metric
            minimize_metrics = ['max_drawdown', 'volatility']
            ascending = metric_name in minimize_metrics
            
            # Create a figure for the summary of top N best combinations
            plt.figure(figsize=(12, 8))
            
            # Sort by the metric and get top N combinations
            top_n = min(20, len(results_df))
            try:
                top_results = results_df.sort_values(by=metric_name, ascending=ascending).head(top_n)
                
                # Create bar chart of top results
                ax = top_results[metric_name].plot(kind='bar', color='skyblue')
                plt.title(f'Top {top_n} Parameter Combinations by {metric_name}')
                plt.ylabel(metric_name)
                plt.xlabel('Combination Index')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'top_{top_n}_combinations.png'))
                plt.close()
            except Exception as e:
                self.logger.warning(f"Error creating top combinations plot: {str(e)}")
                plt.close()
            
            # For each parameter, create a box plot showing its impact on the metric
            for param in param_columns:
                param_name = param[6:]  # Remove 'param_' prefix
                plt.figure(figsize=(10, 6))
                
                try:
                    # Check if parameter is numeric or categorical
                    if pd.api.types.is_numeric_dtype(results_df[param]):
                        # For numeric parameters, create scatter plot
                        plt.scatter(results_df[param], results_df[metric_name], alpha=0.6)
                        plt.title(f'Impact of {param_name} on {metric_name}')
                        plt.xlabel(param_name)
                        plt.ylabel(metric_name)
                        
                        # Add trend line if there are enough data points
                        if len(results_df) > 5:
                            x_values = np.array(results_df[param].values.astype(float))
                            y_values = np.array(results_df[metric_name].values.astype(float))
                            z = np.polyfit(x_values, y_values, 1)
                            p = np.poly1d(z)
                            plt.plot(x_values, p(x_values), "r--", alpha=0.8)
                    else:
                        # For categorical parameters, create box plot
                        sns.boxplot(x=param, y=metric_name, data=results_df)
                        plt.title(f'Impact of {param_name} on {metric_name}')
                        plt.xlabel(param_name)
                        plt.ylabel(metric_name)
                        plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f'param_impact_{param_name}.png'))
                    plt.close()
                except Exception as e:
                    self.logger.warning(f"Error creating impact plot for {param_name}: {str(e)}")
                    plt.close()
            
            # Create heatmap of parameter correlations with the metric
            try:
                # First, select only numeric columns
                numeric_cols = [col for col in results_df.columns 
                               if pd.api.types.is_numeric_dtype(results_df[col])]
                
                if len(numeric_cols) > 1:  # Need at least 2 numeric columns for correlation
                    plt.figure(figsize=(12, 10))
                    correlation = results_df[numeric_cols].corr()
                    mask = np.triu(np.ones_like(correlation, dtype=bool))
                    sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', 
                               vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
                    plt.title('Parameter Correlation Matrix')
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, 'parameter_correlation.png'))
                    plt.close()
            except Exception as e:
                self.logger.warning(f"Error creating correlation heatmap: {str(e)}")
                plt.close()
            
            # Create parallel coordinates plot for top combinations
            try:
                # Make sure param_columns is not empty and top_results has rows
                has_params = len(param_columns) > 1
                has_results = isinstance(top_results, pd.DataFrame) and len(top_results) > 0
                
                if has_params and has_results:
                    plt.figure(figsize=(15, 8))
                    pd.plotting.parallel_coordinates(
                        top_results[param_columns + [metric_name]].reset_index(), 
                        metric_name, 
                        colormap='viridis'
                    )
                    plt.title(f'Parallel Coordinates Plot for Top {top_n} Combinations')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, 'parallel_coordinates.png'))
                    plt.close()
            except Exception as e:
                self.logger.warning(f"Error creating parallel coordinates plot: {str(e)}")
                plt.close()
            
            self.logger.info(f"Parameter importance plots saved to {plots_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating parameter importance plots: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _extract_metrics_from_results(self, results):
        """
        Extract specific metrics from backtest results.
        
        Args:
            results (dict): The backtest results
            
        Returns:
            dict: The extracted metrics
        """
        extracted_metrics = {}
        
        if not results or not isinstance(results, dict):
            self.logger.error("No valid results to extract metrics from")
            return extracted_metrics
            
        try:
            # Extract basic metrics directly from results
            for metric in ['total_return', 'sharpe_ratio', 'calmar_ratio']:
                if metric in results:
                    extracted_metrics[metric] = results[metric]
                elif metric == 'sharpe_ratio' and 'sharpe_ratio' not in results:
                    # Calculate Sharpe ratio if not provided directly
                    returns = results.get('returns', None)
                    if returns is not None and len(returns) > 0:
                        try:
                            sharpe = self._calculate_sharpe_ratio(returns)
                            extracted_metrics['sharpe_ratio'] = sharpe
                        except Exception as e:
                            self.logger.warning(f"Could not calculate Sharpe ratio: {str(e)}")
                elif metric == 'volatility' and 'volatility' not in results:
                    # Calculate volatility if not provided directly
                    returns = results.get('returns', None)
                    if returns is not None and len(returns) > 0:
                        try:
                            vol = self._calculate_volatility(returns)
                            extracted_metrics['volatility'] = vol
                        except Exception as e:
                            self.logger.warning(f"Could not calculate volatility: {str(e)}")
                elif metric == 'calmar_ratio' and 'calmar_ratio' not in results:
                    # Calculate Calmar ratio if not provided directly
                    returns = results.get('returns', None)
                    if returns is not None and len(returns) > 0:
                        try:
                            calmar = self._calculate_calmar_ratio(returns)
                            extracted_metrics['calmar_ratio'] = calmar
                        except Exception as e:
                            self.logger.warning(f"Could not calculate Calmar ratio: {str(e)}")
                            
            # Extract from metrics dictionary if available and original extraction didn't succeed
            if 'metrics' in results and isinstance(results['metrics'], dict):
                metrics_dict = results['metrics']
                for metric in ['total_return', 'sharpe_ratio', 'calmar_ratio']:
                    if metric not in extracted_metrics and metric in metrics_dict:
                        extracted_metrics[metric] = metrics_dict[metric]
            
            # Extract from stats dictionary if available and original extraction didn't succeed
            if 'stats' in results and isinstance(results['stats'], dict):
                stats_dict = results['stats']
                for metric in ['total_return', 'sharpe_ratio', 'calmar_ratio']:
                    if metric not in extracted_metrics and metric in stats_dict:
                        extracted_metrics[metric] = stats_dict[metric]
                        
            # Calculate average trade duration
            if 'trades' in results and isinstance(results['trades'], pd.DataFrame) and not results['trades'].empty:
                try:
                    trades_df = results['trades']
                    trades_df['duration'] = (pd.to_datetime(trades_df['exit_time']) - 
                                           pd.to_datetime(trades_df['entry_time']))
                    avg_duration = trades_df['duration'].mean()
                    if pd.notna(avg_duration):
                        extracted_metrics['avg_trade_duration'] = avg_duration.total_seconds() / 86400  # in days
                except Exception as e:
                    self.logger.warning(f"Could not calculate average trade duration: {str(e)}")
            
            # Default metrics if none were extracted
            if not extracted_metrics:
                self.logger.warning("No metrics could be extracted from results, adding default metrics")
                
                # Check if there are any returns or equity values
                returns = results.get('returns', None)
                equity = results.get('equity', None)
                
                if returns is not None and len(returns) > 0:
                    # Add some default metrics based on returns
                    extracted_metrics['total_return'] = float(returns.iloc[-1] if isinstance(returns, pd.Series) else returns[-1])
                elif equity is not None and len(equity) > 0:
                    # Calculate return from equity
                    start_equity = equity[0] if isinstance(equity, list) else equity.iloc[0]
                    end_equity = equity[-1] if isinstance(equity, list) else equity.iloc[-1]
                    if start_equity > 0:
                        extracted_metrics['total_return'] = (end_equity / start_equity) - 1.0
            
            # Convert numpy and pandas types to Python native types
            for key, value in list(extracted_metrics.items()):
                if isinstance(value, (np.integer, np.floating)):
                    extracted_metrics[key] = float(value)
                elif isinstance(value, pd.Series):
                    extracted_metrics[key] = float(value.iloc[-1])
                elif isinstance(value, pd.Timestamp):
                    extracted_metrics[key] = value.isoformat()
                elif isinstance(value, datetime):
                    extracted_metrics[key] = value.isoformat()
            
            return extracted_metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return extracted_metrics

    def _run_single_backtest(self, params):
        """
        Run a single backtest with the given parameters.
        
        Args:
            params (dict): Dictionary of parameters for the backtest
            
        Returns:
            dict: Results of the backtest
        """
        try:
            # Create a parameter directory for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            param_dir = os.path.join(self.output_dir, f"param_set_{timestamp}")
            os.makedirs(param_dir, exist_ok=True)
            
            # Run backtest with these parameters
            self.logger.debug(f"Running backtest with parameters: {params}")
            
            results = run_backtest(
                strategy_name=self.strategy_name,
                tickers=self.tickers,
                parameters=params,
                start_date=self.start_date,
                end_date=self.end_date,
                output_dir=param_dir,
                initial_capital=self.initial_capital,
                commission=self.commission,
                data_dir=self.data_dir,
                plot=False  # Don't generate plots for individual parameter trials
            )
            
            # Save parameters to file
            params_file = os.path.join(param_dir, "parameters.txt")
            with open(params_file, 'w') as f:
                f.write(f"Parameter Set for {self.strategy_name}\n")
                f.write(f"Testing period: {self.start_date} to {self.end_date}\n\n")
                f.write("Parameters:\n")
                for param, value in params.items():
                    f.write(f"  {param}: {value}\n")
                f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Log basic results
            if results:
                if 'metrics' in results and isinstance(results['metrics'], dict):
                    for key, value in results['metrics'].items():
                        self.logger.debug(f"Metric {key}: {value}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in backtest with params {params}: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def run(self, metric_name='sharpe_ratio', ascending=False):
        """
        Run the optimization process to find the best strategy parameters.
        
        Args:
            metric_name (str): The metric to optimize for (default: 'sharpe_ratio')
            ascending (bool): Whether to sort the metric in ascending order (default: False)
                              Set to False for metrics like sharpe_ratio, returns, where higher is better
                              Set to True for metrics like drawdown, where lower is better
        
        Returns:
            dict: The best parameters and their metrics
        """
        self.logger.info(f"Starting optimization for {self.strategy_name}")
        self.logger.info(f"Optimizing for metric: {metric_name}")
        
        # Get parameter grid
        param_grid = self._get_param_grid()
        if not param_grid:
            self.logger.error("Failed to get parameter grid, aborting optimization")
            self._create_error_log("Failed to get parameter grid")
            return {"error": "Failed to get parameter grid", "parameters": {}}
        
        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations(param_grid)
        if not parameter_combinations:
            self.logger.error("No parameter combinations generated, aborting optimization")
            self._create_error_log("No parameter combinations generated")
            return {"error": "No parameter combinations generated", "parameters": {}}
            
        self.logger.info(f"Generated {len(parameter_combinations)} parameter combinations")
        
        # Create results directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Run optimization
        optimization_results = self.run_optimization(metric_name)
        
        # Check if optimization returned an error
        if "error" in optimization_results:
            error_msg = optimization_results.get('error', 'Unknown error')
            self.logger.error(f"Optimization failed: {error_msg}")
            
            # Write summary file with error information
            summary_file = os.path.join(self.output_dir, "optimization_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Optimization for {self.strategy_name} failed\n")
                f.write(f"Error: {error_msg}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Metric: {metric_name}\n")
                f.write(f"Number of combinations attempted: {optimization_results.get('total', 0)}\n")
                f.write(f"Number of successful combinations: {optimization_results.get('successful', 0)}\n")
            
            return {"error": error_msg, "parameters": {}}
            
        # Get results dataframe
        results_df = optimization_results.get('results')
        
        # If no results dataframe or it's empty, return error
        if results_df is None or not isinstance(results_df, pd.DataFrame) or results_df.empty:
            error_msg = "Optimization produced no valid results"
            self.logger.error(error_msg)
            self._create_error_log(error_msg)
            return {"error": error_msg, "parameters": {}}
        
        # Use the metric name that was actually used (might have been changed in run_optimization)
        used_metric_name = optimization_results.get('metric', metric_name)
        if used_metric_name != metric_name:
            self.logger.info(f"Using metric '{used_metric_name}' instead of requested '{metric_name}'")
            metric_name = used_metric_name
        
        # Check if best_params are already in the optimization results
        if "best_params" in optimization_results and optimization_results["best_params"]:
            best_params = optimization_results["best_params"]
            self.logger.info(f"Using best parameters from optimization results: {best_params}")
            
            # Extract metrics for the best parameters if available
            best_metrics = {}
            if "metrics" in optimization_results:
                best_metrics = optimization_results["metrics"]
            
            # Create a summary file
            summary_file = os.path.join(self.output_dir, "optimization_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Optimization for {self.strategy_name} complete\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Metric: {metric_name}\n")
                f.write(f"Number of combinations tested: {optimization_results.get('total', 0)}\n")
                f.write(f"Number of successful combinations: {optimization_results.get('successful', 0)}\n\n")
                f.write("Best Parameters:\n")
                for param, value in best_params.items():
                    f.write(f"  {param}: {value}\n")
                
                if best_metrics:
                    f.write("\nBest Metrics:\n")
                    for metric, value in best_metrics.items():
                        f.write(f"  {metric}: {value}\n")
            
            # Return best parameters and metrics
            return {
                "parameters": best_params,
                "metrics": best_metrics,
                "all_results": results_df
            }
        
        # If best_params not in optimization_results, extract from DataFrame
        try:
            # Verify metric exists in DataFrame
            if metric_name not in results_df.columns:
                self.logger.warning(f"Metric '{metric_name}' not in results, checking for alternatives")
                
                # Try common alternative metrics
                alternative_metrics = ['sharpe_ratio', 'total_return', 'profit_factor', 'calmar_ratio']
                for alt_metric in alternative_metrics:
                    if alt_metric in results_df.columns:
                        self.logger.info(f"Using alternative metric '{alt_metric}'")
                        metric_name = alt_metric
                        break
                else:
                    # If no standard metrics found, use first numeric non-parameter column
                    numeric_cols = [col for col in results_df.columns 
                                   if pd.api.types.is_numeric_dtype(results_df[col]) 
                                   and not col.startswith('param_')]
                    
                    if numeric_cols:
                        metric_name = numeric_cols[0]
                        self.logger.info(f"Using fallback metric '{metric_name}'")
                    else:
                        error_msg = "No usable metric columns found in results"
                        self.logger.error(error_msg)
                        self._create_error_log(error_msg)
                        return {"error": error_msg, "parameters": {}}
            
            # Sort by metric value
            # For metrics where lower is better (e.g., max_drawdown), use ascending=True
            invert_metrics = ['max_drawdown', 'volatility', 'max_drawdown_pct', 'ulcer_index']
            is_ascending = ascending
            if metric_name in invert_metrics:
                is_ascending = not ascending
                self.logger.info(f"Inverting sort order for metric '{metric_name}'")
            
            # Sort the dataframe
            sorted_df = results_df.sort_values(by=metric_name, ascending=is_ascending)
            
            # Extract parameter columns
            param_columns = [col for col in sorted_df.columns if col.startswith('param_')]
            
            # If no param columns found, try to infer
            if not param_columns:
                self.logger.warning("No 'param_' columns found, trying to infer parameter columns")
                # Exclude common metric columns
                common_metrics = ['sharpe_ratio', 'total_return', 'calmar_ratio', 'volatility', 
                                 'max_drawdown', 'win_rate', 'profit_factor']
                param_columns = [col for col in sorted_df.columns if col not in common_metrics]
                self.logger.info(f"Inferred parameter columns: {param_columns}")
            
            # Extract best parameters
            if len(sorted_df) > 0 and param_columns:
                best_row = sorted_df.iloc[0]
                best_params = {}
                
                for col in param_columns:
                    # Remove 'param_' prefix if present
                    param_name = col[6:] if col.startswith('param_') else col
                    best_params[param_name] = best_row[col]
                
                # Extract metrics for the best parameters
                best_metrics = {}
                metric_columns = [col for col in sorted_df.columns if col not in param_columns]
                
                for col in metric_columns:
                    best_metrics[col] = best_row[col]
                
                # Log best parameters
                self.logger.info(f"Best parameters found: {best_params}")
                self.logger.info(f"Best {metric_name}: {best_row.get(metric_name, 'N/A')}")
                
                # Create a summary file
                summary_file = os.path.join(self.output_dir, "optimization_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(f"Optimization for {self.strategy_name} complete\n")
                    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Metric: {metric_name}\n")
                    f.write(f"Number of combinations tested: {len(parameter_combinations)}\n")
                    f.write(f"Number of successful combinations: {len(sorted_df)}\n\n")
                    f.write("Best Parameters:\n")
                    for param, value in best_params.items():
                        f.write(f"  {param}: {value}\n")
                    
                    if best_metrics:
                        f.write("\nBest Metrics:\n")
                        for metric, value in best_metrics.items():
                            if not metric.startswith('param_'):
                                f.write(f"  {metric}: {value}\n")
                
                # Save as JSON for easier programmatic access
                best_result = {
                    "strategy": self.strategy_name,
                    "parameters": best_params,
                    "metrics": best_metrics,
                    "optimization_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Save to JSON file
                import json
                best_params_file = os.path.join(self.output_dir, "best_parameters.json")
                with open(best_params_file, 'w') as f:
                    json.dump(best_result, f, indent=4)
                
                return {
                    "parameters": best_params,
                    "metrics": best_metrics,
                    "all_results": sorted_df
                }
            else:
                error_msg = "Could not determine best parameters from results"
                self.logger.error(error_msg)
                self._create_error_log(error_msg)
                return {"error": error_msg, "parameters": {}}
        
        except Exception as e:
            error_msg = f"Error finding best parameters: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.debug(traceback.format_exc())
            self._create_error_log(error_msg)
            return {"error": error_msg, "parameters": {}}

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
    
    optimizer.run(
        metric=args.metric,
        max_combinations=args.max_combinations
    )

if __name__ == "__main__":
    main() 