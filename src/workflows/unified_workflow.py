#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified workflow for backtest, optimization, and walk-forward validation.
"""
import os
import sys
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import logging
import time
import traceback
import multiprocessing

# Add the parent directory to the path so we can import from engine
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
project_root = os.path.dirname(src_dir)  # Go up to project root
if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import components
from engine.parameter_management import ParameterManager
from engine.logging_system import logger, log_execution_time
from engine.testing import WalkForwardTest
from engine.testing.in_sample_excellence import InSampleExcellence
from strategies import registry
from engine.run_backtest import run_backtest

# Import the Monte Carlo implementations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.monte_carlo.trade_based_monte_carlo import TradeBasedMonteCarloTest

# Add imports from runners
sys.path.append(os.path.join(src_dir, 'runners'))
from run_simple_backtest import run_strategy_backtest
from run_walk_forward_test import run_walk_forward_test

# Custom JSON encoder to handle pandas Timestamp objects and NumPy types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif hasattr(obj, 'dtype'):  # Handle NumPy types
            return obj.item()
        return super().default(obj)

def convert_timestamps_in_dict(obj):
    """Recursively convert any Timestamp keys in dictionaries to strings."""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # Convert key if it's a Timestamp
            if isinstance(k, pd.Timestamp):
                k = k.strftime('%Y-%m-%d %H:%M:%S')
            # Recursively convert value
            new_obj[k] = convert_timestamps_in_dict(v)
        return new_obj
    elif isinstance(obj, list):
        return [convert_timestamps_in_dict(item) for item in obj]
    else:
        return obj

@log_execution_time('workflow')
def run_simple_workflow(strategy_name, tickers=None, start_date=None, end_date=None, 
                       param_file=None, output_dir=None, detailed_analysis=False,
                       verbose=False):
    """
    Run a simple backtest workflow for a strategy.
    
    This is a convenience wrapper around run_strategy_backtest.
    
    Args:
        strategy_name (str): Name of the strategy to backtest
        tickers (list): List of ticker symbols
        start_date (str): Start date for backtesting (YYYY-MM-DD)
        end_date (str): End date for backtesting (YYYY-MM-DD)
        param_file (str): Path to parameter file
        output_dir (str): Directory to save output files
        detailed_analysis (bool): Whether to run detailed performance analysis
        verbose (bool): Whether to print detailed information
    
    Returns:
        dict: Backtest results
    """
    # Load parameters from file if provided
    parameters = None
    if param_file:
        param_manager = ParameterManager()
        parameters = param_manager.load_parameter_file(param_file)
    
    # Set output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(os.path.dirname(src_dir), 'output', f"{strategy_name}_simple_{timestamp}")
    
    # Run the backtest
    return run_strategy_backtest(
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters,
        output_dir=output_dir,
        detailed_analysis=detailed_analysis,
        verbose=verbose
    )

@log_execution_time('workflow')
def run_monte_carlo_safely(strategy_name, tickers=None, start_date=None, end_date=None, 
                           num_simulations=10, parameters=None, best_params=None, 
                           output_dir=None, num_workers=None, verbose=False, seed=42,
                           keep_permuted_data=False):
    """
    Run monte carlo tests safely by handling errors gracefully.
    Uses the TradeBasedMonteCarloTest implementation.
    
    Args:
        strategy_name (str): Name of the strategy to test
        tickers (list): List of ticker symbols to include
        start_date (str): Start date for historical data
        end_date (str): End date for historical data
        num_simulations (int): Number of Monte Carlo simulations to run
        parameters (dict): Parameter ranges for the strategy optimization
        best_params (dict): Best parameters to use for the strategy
        output_dir (str): Directory to save results
        num_workers (int): Number of parallel workers to use for Monte Carlo simulations
        verbose (bool): Whether to print verbose output
        seed (int): Random seed for reproducibility
        keep_permuted_data (bool): Whether to save the permuted stock data files
    
    Returns:
        dict: Results of the Monte Carlo testing
    """
    # Convert any date parameters to strings if they are pandas Timestamps
    if start_date is not None and isinstance(start_date, pd.Timestamp):
        start_date = start_date.strftime('%Y-%m-%d')
    
    if end_date is not None and isinstance(end_date, pd.Timestamp):
        end_date = end_date.strftime('%Y-%m-%d')
    
    # Handle strategy parameters
    if best_params is None:
        if strategy_name == "MACrossover":
            best_params = {'fast_period': 5, 'slow_period': 20, 'position_size': 10}
        elif strategy_name == "SimpleStock":
            best_params = {'sma_period': 20, 'position_size': 10}
        else:
            best_params = {}
    
    # Set default dates if not provided
    if start_date is None:
        start_date = "2015-01-01"
    
    if end_date is None:
        end_date = "2021-12-31"
    
    # Calculate in-sample and out-of-sample date ranges
    full_range = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    in_sample_days = int(full_range.days * 0.8)
    in_sample_end = (pd.to_datetime(start_date) + timedelta(days=in_sample_days)).strftime('%Y-%m-%d')
    out_sample_start = (pd.to_datetime(in_sample_end) + timedelta(days=1)).strftime('%Y-%m-%d')
    
    if verbose:
        print(f"Date ranges: In-sample: {start_date} to {in_sample_end}, Out-of-sample: {out_sample_start} to {end_date}")
    
    # Set a default output directory if None
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(os.path.dirname(src_dir), 'output', f"{strategy_name}_monte_carlo_{timestamp}")
        if verbose:
            print(f"Using default output directory: {output_dir}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the number of workers to use
    if num_workers is None:
        # Use all cores except one by default
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    if verbose:
        print(f"Using {num_workers} CPU cores for Monte Carlo simulations")
    
    try:
        if verbose:
            print("Using Trade-Based Monte Carlo implementation")
            print(f"Running {num_simulations} simulations with resampled trade returns")
            if num_workers is not None:
                print(f"Using {num_workers} CPU cores for parallel processing")
        
        # Initialize the Trade-Based Monte Carlo test
        monte_carlo_test = TradeBasedMonteCarloTest(
            strategy_name=strategy_name,
            parameters=best_params,
            tickers=tickers,
            input_dir=os.path.join(project_root, 'input'),
            output_dir=output_dir,
            num_simulations=num_simulations,
            seed=seed,  # Use provided seed for reproducibility
            verbose=verbose,
            num_workers=num_workers,
            keep_permuted_data=keep_permuted_data
        )
        
        # Run the test with out-of-sample start date
        results = monte_carlo_test.run_test(out_sample_start)
        
        if results:
            # Trade log information
            trade_log_path = os.path.join(output_dir, 'original_trade_log.csv')
            trade_log_exists = os.path.exists(trade_log_path)
            
            # Create a trade log summary structure for compatibility
            trade_log_summary = {
                'original': trade_log_path if trade_log_exists else None,
                'permutations': []
            }
            
            # Save the trade log summary
            with open(os.path.join(output_dir, 'trade_log_summary.json'), 'w') as f:
                json.dump(trade_log_summary, f, indent=4, cls=CustomJSONEncoder)
            
            if verbose:
                print(f"Trade logs summary saved to: {os.path.join(output_dir, 'trade_log_summary.json')}")
                if trade_log_exists:
                    print(f"Original strategy trade log: {trade_log_path}")
            
            return {
                'success': True,
                'results': {
                    'analysis': {
                        'p_values': {k: v.get('p_value', None) for k, v in results.items()}
                    },
                    'metrics': results
                },
                'parameters': best_params,
                'dates': {
                    'in_sample_start': start_date,
                    'in_sample_end': in_sample_end,
                    'out_sample_start': out_sample_start,
                    'out_sample_end': end_date
                },
                'trade_logs': trade_log_summary
            }
        else:
            return {
                'success': False,
                'error': 'Trade-Based Monte Carlo test failed to produce results',
                'parameters': best_params,
                'dates': {
                    'in_sample_start': start_date,
                    'in_sample_end': in_sample_end,
                    'out_sample_start': out_sample_start,
                    'out_sample_end': end_date
                }
            }
        
    except Exception as e:
        error_msg = f"Error during Monte Carlo testing: {e}"
        print(error_msg)
        traceback_str = ""
        try:
            traceback_str = traceback.format_exc()
            print(traceback_str)
        except:
            pass
        
        # Create an error log file in the output directory
        try:
            with open(os.path.join(output_dir, "monte_carlo_error.log"), "w") as f:
                f.write(f"Error: {error_msg}\n\n")
                f.write(f"Traceback:\n{traceback_str}")
        except:
            pass
            
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback_str,
            'parameters': best_params,
            'dates': {
                'in_sample_start': start_date,
                'in_sample_end': in_sample_end,
                'out_sample_start': out_sample_start,
                'out_sample_end': end_date
            }
        }

@log_execution_time('workflow')
def run_trade_monte_carlo_safely(strategy_name, tickers=None, start_date=None, end_date=None, 
                            out_of_sample_start=None, num_simulations=1000, parameters=None, 
                            best_params=None, output_dir=None, verbose=False, seed=None,
                            num_workers=None, keep_permuted_data=False):
    """
    Run trade-based Monte Carlo tests safely by handling errors gracefully.
    Uses the TradeBasedMonteCarloTest implementation which resamples trade returns.
    
    Args:
        strategy_name (str): Name of the strategy to test
        tickers (list): List of ticker symbols to include
        start_date (str): Start date for historical data
        end_date (str): End date for historical data
        out_of_sample_start (str): Start date for out-of-sample period
        num_simulations (int): Number of Monte Carlo simulations to run
        parameters (dict): Parameter ranges for the strategy optimization
        best_params (dict): Best parameters to use for the strategy
        output_dir (str): Directory to save results
        verbose (bool): Whether to print verbose output
        seed (int): Random seed for reproducibility
        num_workers (int): Number of parallel workers to use for Monte Carlo simulations
        keep_permuted_data (bool): Whether to save the permuted stock data files
    
    Returns:
        dict: Results of the trade-based Monte Carlo testing
    """
    # Convert any date parameters to strings if they are pandas Timestamps
    if start_date is not None and isinstance(start_date, pd.Timestamp):
        start_date = start_date.strftime('%Y-%m-%d')
    
    if end_date is not None and isinstance(end_date, pd.Timestamp):
        end_date = end_date.strftime('%Y-%m-%d')
        
    if out_of_sample_start is not None and isinstance(out_of_sample_start, pd.Timestamp):
        out_of_sample_start = out_of_sample_start.strftime('%Y-%m-%d')
    
    # Handle strategy parameters
    if best_params is None:
        if strategy_name == "MACrossover":
            best_params = {'fast_period': 5, 'slow_period': 20, 'position_size': 10}
        elif strategy_name == "SimpleStock":
            best_params = {'sma_period': 20, 'position_size': 10}
        else:
            best_params = {}
    
    # Set default dates if not provided
    if start_date is None:
        start_date = "2015-01-01"
    
    if end_date is None:
        end_date = "2021-12-31"
    
    # Calculate out-of-sample start date if not provided
    if out_of_sample_start is None:
        full_range = pd.to_datetime(end_date) - pd.to_datetime(start_date)
        in_sample_days = int(full_range.days * 0.7)  # 70% for in-sample by default
        out_of_sample_start = (pd.to_datetime(start_date) + timedelta(days=in_sample_days)).strftime('%Y-%m-%d')
    
    if verbose:
        print(f"Using out-of-sample start date: {out_of_sample_start}")
        print(f"Full date range: {start_date} to {end_date}")
    
    # Set a default output directory if None
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(os.path.dirname(src_dir), 'output', f"{strategy_name}_trade_monte_carlo_{timestamp}")
        if verbose:
            print(f"Using default output directory: {output_dir}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if verbose:
            print("Using Trade-Based Monte Carlo implementation")
            print(f"Running {num_simulations} simulations with resampled trade returns")
            if num_workers is not None:
                print(f"Using {num_workers} CPU cores for parallel processing")
        
        # Initialize the Trade-Based Monte Carlo test
        monte_carlo_test = TradeBasedMonteCarloTest(
            strategy_name=strategy_name,
            parameters=best_params,
            tickers=tickers,
            input_dir=os.path.join(project_root, 'input'),
            output_dir=output_dir,
            num_simulations=num_simulations,
            seed=seed,
            verbose=verbose,
            num_workers=num_workers,
            keep_permuted_data=keep_permuted_data
        )
        
        # Run the test with out-of-sample start date
        results = monte_carlo_test.run_test(out_of_sample_start)
        
        if results:
            # Trade log information
            trade_log_path = os.path.join(output_dir, 'original_trade_log.csv')
            trade_log_exists = os.path.exists(trade_log_path)
            
            if verbose:
                print(f"Analysis results summary:")
                for metric, stats in results.items():
                    print(f"  {metric}: Original={stats.get('original', 'N/A'):.4f}, "
                          f"Mean={stats.get('mean', 'N/A'):.4f}, "
                          f"p-value={stats.get('p_value', 'N/A'):.4f}")
            
            return {
                'success': True,
                'results': results,
                'parameters': best_params,
                'dates': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'out_of_sample_start': out_of_sample_start
                },
                'trade_log': {
                    'original': trade_log_path if trade_log_exists else None
                }
            }
        else:
            return {
                'success': False,
                'error': 'Trade-Based Monte Carlo test failed to produce results',
                'parameters': best_params,
                'dates': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'out_of_sample_start': out_of_sample_start
                }
            }
        
    except Exception as e:
        error_msg = f"Error during Trade-Based Monte Carlo testing: {e}"
        print(error_msg)
        traceback_str = ""
        try:
            traceback_str = traceback.format_exc()
            print(traceback_str)
        except:
            pass
        
        # Create an error log file in the output directory
        try:
            with open(os.path.join(output_dir, "trade_monte_carlo_error.log"), "w") as f:
                f.write(f"Error: {error_msg}\n\n")
                f.write(f"Traceback:\n{traceback_str}")
        except:
            pass
            
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback_str,
            'parameters': best_params,
            'dates': {
                'start_date': start_date,
                'end_date': end_date,
                'out_of_sample_start': out_of_sample_start
            }
        }

@log_execution_time('workflow')
def run_complete_workflow(strategy_name, tickers=None, start_date=None, end_date=None, 
                         param_file=None, num_workers=None, output_dir=None, 
                         in_sample_ratio=0.7, num_simulations=0, verbose=False, seed=42):
    """
    Run a complete workflow with in-sample optimization and walk-forward testing.
    
    Args:
        strategy_name (str): Name of the strategy to test
        tickers (list): List of ticker symbols
        start_date (str): Start date for the entire test period (YYYY-MM-DD)
        end_date (str): End date for the entire test period (YYYY-MM-DD)
        param_file (str): Path to parameter file for optimization
        num_workers (int): Number of parallel workers for optimization (CPU cores to use)
        output_dir (str): Directory to save output files
        in_sample_ratio (float): Ratio of data to use for in-sample period
        num_simulations (int): Number of simulations for Monte Carlo testing
        verbose (bool): Whether to print detailed information
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Workflow results
    """
    # Set defaults
    if tickers is None:
        tickers = ['AAPL']
    
    if start_date is None:
        start_date = '2015-01-01'
    
    if end_date is None:
        end_date = '2021-12-31'
    
    # Set up logging output based on verbosity
    if verbose:
        logger.set_level('INFO')
    else:
        logger.set_level('WARNING')
    
    # Set output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(os.path.dirname(src_dir), 'output', f"{strategy_name}_complete_{timestamp}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: In-sample optimization
    logger.info('workflow', f"Starting complete workflow for {strategy_name}")
    logger.info('workflow', f"Tickers: {tickers}")
    logger.info('workflow', f"Period: {start_date} to {end_date}")
    logger.info('workflow', f"Output directory: {output_dir}")
    
    # Create subdirectories for different phases
    optimization_dir = os.path.join(output_dir, 'optimization')
    walk_forward_dir = os.path.join(output_dir, 'walk_forward')
    monte_carlo_dir = os.path.join(output_dir, 'monte_carlo')
    
    os.makedirs(optimization_dir, exist_ok=True)
    os.makedirs(walk_forward_dir, exist_ok=True)
    os.makedirs(monte_carlo_dir, exist_ok=True)
    
    # Load parameter grid
    param_manager = ParameterManager()
    param_grid = param_manager.load_parameter_grid(strategy_name, param_file)
    logger.info('workflow', f"Created parameter grid: {param_grid}")
    
    # Calculate in-sample and out-sample dates based on in_sample_ratio
    date_range = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
    in_sample_days = int(date_range * in_sample_ratio)
    in_sample_end_date = (datetime.strptime(start_date, '%Y-%m-%d') + pd.Timedelta(days=in_sample_days)).strftime('%Y-%m-%d')
    out_sample_start_date = (datetime.strptime(in_sample_end_date, '%Y-%m-%d') + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    logger.info('workflow', f"In-sample period: {start_date} to {in_sample_end_date}")
    logger.info('workflow', f"Out-of-sample period: {out_sample_start_date} to {end_date}")
    
    # Step 1: In-sample optimization
    logger.info('workflow', "Starting in-sample optimization...")
    optimizer = InSampleExcellence(
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=in_sample_end_date,
        parameter_grid=param_grid,
        output_dir=optimization_dir
    )
    
    # If num_workers is provided, use it for parallelization
    # Otherwise use a default of 100 max combinations
    max_combinations = 100  # Default value
    if num_workers is not None:
        logger.info('workflow', f"Using {num_workers} CPU cores for parallel optimization")
    else:
        num_workers = max(1, multiprocessing.cpu_count() - 1)  # Use all cores except one by default
        logger.info('workflow', f"Using {num_workers} of {multiprocessing.cpu_count()} available CPU cores for parallel optimization")
    
    optimization_results = optimizer.run_optimization(
        metric='sharpe_ratio', 
        max_combinations=max_combinations,
        n_jobs=num_workers
    )
    best_params = optimization_results.get('best_parameters', {})
    
    # Step 2: Walk-Forward Testing
    logger.info('workflow', "Starting walk-forward testing...")
    walk_forward = WalkForwardTest(
        strategy_name=strategy_name,
        tickers=tickers,
        in_sample_start=start_date,
        in_sample_end=in_sample_end_date,
        out_sample_start=out_sample_start_date,
        out_sample_end=end_date,
        parameters=best_params,
        output_dir=walk_forward_dir
    )
    
    walk_forward_results = walk_forward.run_test()
    walk_forward_metrics = walk_forward_results.get('comparison', {})
    logger.info('workflow', f"Walk-forward testing completed")
    
    # Step 3: Monte Carlo Testing (if requested)
    monte_carlo_results = None
    monte_carlo_success = False
    if num_simulations > 0:
        logger.info('workflow', f"Starting Trade-Based Monte Carlo testing with {num_simulations} simulations using {num_workers} workers...")
        
        # Use the Trade-Based Monte Carlo wrapper function
        monte_carlo_result = run_monte_carlo_safely(
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            num_simulations=num_simulations,
            parameters=None,
            best_params=best_params,
            output_dir=monte_carlo_dir,
            num_workers=num_workers,
            verbose=verbose,
            seed=seed
        )
        
        monte_carlo_success = monte_carlo_result.get("success", False)
        monte_carlo_results = monte_carlo_result.get("results", None)
        
        if not monte_carlo_success:
            logger.error('workflow', f"Error during Trade-Based Monte Carlo testing: {monte_carlo_result.get('error', 'Unknown error')}")
            logger.warning('workflow', f"Monte Carlo testing results may be incomplete")
            # Store the error in the results
            monte_carlo_results = {
                "error": monte_carlo_result.get('error', 'Unknown error'),
                "traceback": monte_carlo_result.get('traceback', '')
            }
    
    # Create a summary file
    summary_file = os.path.join(output_dir, f"{strategy_name}_workflow_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Strategy: {strategy_name}\n")
        f.write(f"Tickers: {tickers}\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        f.write(f"In-sample period: {start_date} to {in_sample_end_date}\n")
        f.write(f"Out-of-sample period: {out_sample_start_date} to {end_date}\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Walk-forward testing completed: {walk_forward_results is not None}\n")
        f.write(f"Monte Carlo testing completed: {monte_carlo_success}\n")
    
    # Compile overall results
    overall_results = {
        'strategy_name': strategy_name,
        'tickers': tickers,
        'period': {'start': start_date, 'end': end_date},
        'in_sample_period': {'start': start_date, 'end': in_sample_end_date},
        'out_sample_period': {'start': out_sample_start_date, 'end': end_date},
        'best_parameters': best_params,
        'optimization_results': optimization_results,
        'walk_forward_metrics': walk_forward_metrics,
    }
    
    if monte_carlo_results:
        overall_results['monte_carlo_results'] = monte_carlo_results
    
    # Convert any Timestamp keys to strings
    overall_results = convert_timestamps_in_dict(overall_results)
    
    # Save overall results to JSON
    results_file = os.path.join(output_dir, f"{strategy_name}_workflow_results.json")
    with open(results_file, 'w') as f:
        json.dump(overall_results, f, indent=4, cls=CustomJSONEncoder)
    
    logger.info('workflow', f"Complete workflow finished. Results saved to {output_dir}")
    
    if verbose:
        print(f"\nWorkflow completed successfully for {strategy_name}")
        print(f"Best parameters: {best_params}")
        print(f"Results saved to {output_dir}")
    
    return overall_results

def main():
    """Main entry point for the unified workflow."""
    parser = argparse.ArgumentParser(description='Backtest a trading strategy')
    
    # Add general arguments
    parser.add_argument('--strategy', type=str, required=True, 
                        help='Strategy name to test')
    parser.add_argument('--tickers', type=str, nargs='+', 
                        help='Ticker symbols to test')
    parser.add_argument('--start-date', type=str, default='2015-01-01', 
                        help='Start date for the test period (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2021-12-31', 
                        help='End date for the test period (YYYY-MM-DD)')
    parser.add_argument('--out-of-sample-start', type=str,
                        help='Start date for out-of-sample testing (YYYY-MM-DD)')
    parser.add_argument('--param-file', type=str,
                        help='Path to a parameter file to use for the backtest')
    parser.add_argument('--num-simulations', type=int, default=1000,
                        help='Number of simulations to use for Monte Carlo testing')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save output files')
    parser.add_argument('--in-sample-ratio', type=float, default=0.7,
                        help='Ratio of data to use for in-sample period')
    parser.add_argument('--verbose', action='store_true',
                        help='Print additional output')
    parser.add_argument('--num-cores', '--num-workers', type=int, dest='num_cores',
                        help='Number of cores to use for parallel processing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--workflow-type', type=str, default='simple',
                        choices=['simple', 'monte-carlo', 'trade-monte-carlo', 'complete', 'walk-forward'],
                        help='Type of workflow to run')
    parser.add_argument('--keep-permuted-data', action='store_true',
                        help='Save the permuted stock data files for analysis')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Set the verbose output
    if args.verbose:
        logger.set_level('INFO')
    else:
        logger.set_level('WARNING')
    
    # Run the appropriate workflow based on the type
    results = None
    
    # Set the parallel CPU cores if provided
    num_workers = args.num_cores if args.num_cores else None
    
    if args.workflow_type == 'simple':
        # Run a simple backtest
        print(f"Running simple backtest for {args.strategy}...")
        results = run_simple_workflow(
            strategy_name=args.strategy,
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            param_file=args.param_file,
            output_dir=args.output_dir,
            detailed_analysis=args.verbose,
            verbose=args.verbose
        )
    
    elif args.workflow_type == 'monte-carlo':
        # Run the Monte Carlo test
        print(f"Running Trade-Based Monte Carlo test for {args.strategy}...")
        
        # Use params from file if provided
        parameters = None
        if args.param_file:
            try:
                param_manager = ParameterManager()
                parameters = param_manager.load_parameters(args.strategy, args.param_file)
            except Exception as e:
                print(f"Error loading parameters from file: {e}")
                print("Using default parameters.")
        
        results = run_monte_carlo_safely(
            strategy_name=args.strategy, 
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            num_simulations=args.num_simulations,
            parameters=parameters,
            output_dir=args.output_dir,
            num_workers=num_workers,
            verbose=args.verbose,
            seed=args.seed,
            keep_permuted_data=args.keep_permuted_data
        )
        
        if results.get("success", False):
            print("Monte Carlo testing completed successfully!")
            monte_carlo_results = results.get("results", {})
            
            # Save results to file
            if args.output_dir:
                json_path = os.path.join(args.output_dir, f"{args.strategy}_monte_carlo_results.json")
                with open(json_path, 'w') as f:
                    json.dump(monte_carlo_results, f, indent=4, cls=CustomJSONEncoder)
                print(f"Results saved to {json_path}")
                
            # Print p-values
            p_values = monte_carlo_results.get("p_values", {})
            if p_values:
                print("\nMetric p-values (lower is better, less than 0.05 is significant):")
                for metric, p_value in p_values.items():
                    significance = "SIGNIFICANT" if p_value < 0.05 else "not significant"
                    print(f"  - {metric}: {p_value:.4f} ({significance})")
        else:
            print(f"Error during Monte Carlo testing: {results.get('error', 'Unknown error')}")
    
    elif args.workflow_type == 'trade-monte-carlo':
        # Run the Trade-Based Monte Carlo test
        print(f"Running Trade-Based Monte Carlo test for {args.strategy}...")
        
        # Use params from file if provided
        parameters = None
        best_params = None
        if args.param_file:
            try:
                param_manager = ParameterManager()
                parameters = param_manager.load_parameters(args.strategy, args.param_file)
                best_params = parameters
            except Exception as e:
                print(f"Error loading parameters from file: {e}")
                print("Using default parameters.")
        
        results = run_trade_monte_carlo_safely(
            strategy_name=args.strategy, 
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            out_of_sample_start=args.out_of_sample_start,
            num_simulations=args.num_simulations,
            parameters=parameters,
            best_params=best_params,
            output_dir=args.output_dir,
            verbose=args.verbose,
            seed=args.seed,
            num_workers=num_workers,
            keep_permuted_data=args.keep_permuted_data
        )
        
        if results.get("success", False):
            print("Trade-Based Monte Carlo testing completed successfully!")
            monte_carlo_results = results.get("results", {})
            
            # Save results to file
            if args.output_dir:
                json_path = os.path.join(args.output_dir, f"{args.strategy}_trade_monte_carlo_results.json")
                with open(json_path, 'w') as f:
                    json.dump(monte_carlo_results, f, indent=4, cls=CustomJSONEncoder)
                print(f"Results saved to {json_path}")
                
            # Print summary
            analysis = monte_carlo_results.get("analysis", {})
            if analysis:
                print("\nMonte Carlo Analysis Summary:")
                for metric, value in analysis.items():
                    print(f"  - {metric}: {value}")
        else:
            print(f"Error during Trade-Based Monte Carlo testing: {results.get('error', 'Unknown error')}")
    
    elif args.workflow_type == 'complete':
        # Run the complete workflow
        print(f"Running complete workflow for {args.strategy}...")
        results = run_complete_workflow(
            strategy_name=args.strategy,
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            param_file=args.param_file,
            num_workers=num_workers,
            output_dir=args.output_dir,
            in_sample_ratio=args.in_sample_ratio,
            num_simulations=args.num_simulations,
            verbose=args.verbose,
            seed=args.seed
        )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 