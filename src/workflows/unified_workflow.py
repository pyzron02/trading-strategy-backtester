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

# Import the direct Monte Carlo implementation
sys.path.append(project_root)
from direct_monte_carlo import DirectMonteCarloTest

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

def run_monte_carlo_safely(strategy_name, tickers=None, start_date=None, end_date=None, 
                           num_permutations=10, parameters=None, best_params=None, 
                           output_dir=None, verbose=False):
    """
    Run monte carlo tests safely by handling errors gracefully.
    Uses the DirectMonteCarloTest implementation.
    
    Args:
        strategy_name (str): Name of the strategy to test
        tickers (list): List of ticker symbols to include
        start_date (str): Start date for historical data
        end_date (str): End date for historical data
        num_permutations (int): Number of permutations to run
        parameters (dict): Parameter ranges for the strategy optimization
        best_params (dict): Best parameters to use for the strategy
        output_dir (str): Directory to save results
        verbose (bool): Whether to print verbose output
    
    Returns:
        dict: Results of the permutation testing
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
    
    try:
        if verbose:
            print("Using Direct Monte Carlo implementation")
            print(f"Trade logs will be generated for each permutation in subdirectories of {output_dir}")
        
        # Initialize the Direct Monte Carlo test
        monte_carlo_test = DirectMonteCarloTest(
            strategy_name=strategy_name,
            tickers=tickers,
            in_sample_start=start_date,
            in_sample_end=in_sample_end,
            out_sample_start=out_sample_start,
            out_sample_end=end_date,
            parameters=best_params,
            output_dir=output_dir,
            num_permutations=num_permutations
        )
        
        # Run the test
        results = monte_carlo_test.run_test()
        
        if results:
            # Create a summary of trade logs
            trade_log_summary = {
                'original': os.path.join(output_dir, 'original', 'trade_log_original.csv'),
                'permutations': []
            }
            
            # Add permutation trade logs
            for i in range(num_permutations):
                perm_log_path = os.path.join(output_dir, f'permutation_{i}', f'trade_log_permutation_{i}.csv')
                if os.path.exists(perm_log_path):
                    trade_log_summary['permutations'].append(perm_log_path)
            
            # Save the trade log summary
            with open(os.path.join(output_dir, 'trade_log_summary.json'), 'w') as f:
                json.dump(trade_log_summary, f, indent=4, cls=CustomJSONEncoder)
            
            if verbose:
                print(f"Trade logs summary saved to: {os.path.join(output_dir, 'trade_log_summary.json')}")
                print(f"Original strategy trade log: {trade_log_summary['original']}")
                print(f"Generated {len(trade_log_summary['permutations'])} permutation trade logs")
            
            return {
                'success': True,
                'results': results,
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
                'error': 'Direct Monte Carlo test failed to produce results',
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
            import traceback
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
def run_complete_workflow(strategy_name, tickers=None, start_date=None, end_date=None, 
                         param_file=None, num_workers=None, output_dir=None, 
                         in_sample_ratio=0.7, num_permutations=0, verbose=False):
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
        num_permutations (int): Number of permutations for Monte Carlo testing
        verbose (bool): Whether to print detailed information
    
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
        import multiprocessing
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
    if num_permutations > 0:
        logger.info('workflow', f"Starting Monte Carlo testing with {num_permutations} permutations...")
        
        # Use the safe Monte Carlo wrapper function
        monte_carlo_result = run_monte_carlo_safely(
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            num_permutations=num_permutations,
            parameters=None,
            best_params=best_params,
            output_dir=monte_carlo_dir,
            verbose=verbose
        )
        
        monte_carlo_success = monte_carlo_result.get("success", False)
        monte_carlo_results = monte_carlo_result.get("results", None)
        
        if not monte_carlo_success:
            logger.error('workflow', f"Error during Monte Carlo testing: {monte_carlo_result.get('error', 'Unknown error')}")
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
    """Main function to parse command line arguments and run the appropriate workflow."""
    parser = argparse.ArgumentParser(description='Unified workflow runner for trading strategies')
    
    # Common arguments
    parser.add_argument('--workflow-type', '-w', default='simple', 
                        choices=['simple', 'complete', 'walk-forward', 'monte-carlo'],
                        help='Type of workflow to run')
    parser.add_argument('--strategy', '-s', required=True, 
                        help='Name of the strategy to test')
    parser.add_argument('--tickers', '-t', nargs='+', default=['AAPL'], 
                        help='List of ticker symbols')
    parser.add_argument('--start-date', default='2015-01-01', 
                        help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2021-12-31', 
                        help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default=None, 
                        help='Directory to save output files')
    parser.add_argument('--param-file', default=None, 
                        help='Path to parameter file for optimization')
    parser.add_argument('--verbose', action='store_true', 
                        help='Print verbose output')
    parser.add_argument('--num-cores', type=int, default=None,
                        help='Number of CPU cores to use for parallel processing')
    
    # Workflow-specific arguments
    parser.add_argument('--in-sample-ratio', type=float, default=0.7, 
                        help='Ratio of data to use for in-sample period')
    parser.add_argument('--num-permutations', type=int, default=10, 
                        help='Number of permutations for Monte Carlo testing')
    parser.add_argument('--detailed-analysis', action='store_true', 
                        help='Run detailed performance analysis')
    
    args = parser.parse_args()
    
    # Print available strategies if in verbose mode
    if args.verbose:
        registered_strategies = registry.get_registered_strategies()
        strategy_names = [strategy['name'] for strategy in registered_strategies]
        print(f"Available strategies: {strategy_names}")
        print(f"\nRunning {args.workflow_type} workflow for {args.strategy}")
    
    # Run the appropriate workflow
    if args.workflow_type == 'simple':
        # Run the simple workflow
        run_simple_workflow(
            strategy_name=args.strategy,
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            param_file=args.param_file,
            output_dir=args.output_dir,
            detailed_analysis=args.detailed_analysis,
            verbose=args.verbose
        )
    elif args.workflow_type == 'monte-carlo':
        # Run the Monte Carlo test
        print(f"Running Monte Carlo test for {args.strategy}...")
        
        # Convert permutations to integer
        num_permutations = int(args.num_permutations) if args.num_permutations else 10
        
        # Use params from file if provided
        parameters = None
        if args.param_file:
            try:
                param_manager = ParameterManager()
                parameters = param_manager.load_parameter_file(args.param_file)
                if args.verbose:
                    print(f"Loaded parameters from {args.param_file}: {parameters}")
            except Exception as e:
                print(f"Error loading parameters from {args.param_file}: {e}")
                print("Using default parameters instead.")
        
        # Run the Monte Carlo test safely
        results = run_monte_carlo_safely(
            strategy_name=args.strategy,
            tickers=args.tickers if isinstance(args.tickers, list) else (args.tickers.split(',') if args.tickers else ['AAPL']),
            start_date=args.start_date,
            end_date=args.end_date,
            num_permutations=num_permutations,
            parameters=parameters,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        if not results or not results.get('success', False):
            error_msg = results.get('error', 'Unknown error') if results else 'No results returned'
            print(f"Monte Carlo testing failed: {error_msg}")
        else:
            print("Monte Carlo testing completed successfully.")
            
            # Save detailed results
            if args.output_dir:
                results_file = os.path.join(args.output_dir, "monte_carlo_results.json")
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=4, cls=CustomJSONEncoder)
                print(f"Results saved to {results_file}")
            
            # Summary of p-values
            if (results and 'results' in results and results['results'] and 
                'analysis' in results['results'] and 'p_values' in results['results']['analysis']):
                p_values = results['results']['analysis']['p_values']
                print("\nP-values for performance metrics:")
                for metric, p_value in p_values.items():
                    if p_value is not None:
                        significance = "Significant" if p_value < 0.05 else "Not significant"
                        print(f"  {metric.replace('_', ' ').title()}: {p_value:.4f} ({significance})")
            else:
                print("No p-values available in the results.")
            
            # Display trade log information
            if 'trade_logs' in results:
                trade_logs = results['trade_logs']
                print("\nTrade logs generated:")
                print(f"  Original strategy: {os.path.basename(trade_logs['original'])}")
                print(f"  Number of permutation logs: {len(trade_logs['permutations'])}")
                if args.verbose and trade_logs['permutations']:
                    print("  Permutation trade logs:")
                    for i, log_path in enumerate(trade_logs['permutations'][:5]):  # Show first 5 for brevity
                        print(f"    - Permutation {i}: {os.path.basename(log_path)}")
                    if len(trade_logs['permutations']) > 5:
                        print(f"    - ... and {len(trade_logs['permutations']) - 5} more")
                if args.output_dir:
                    print(f"\nTrade logs summary saved to: {os.path.join(args.output_dir, 'trade_log_summary.json')}")
                else:
                    print("\nTrade logs summary saved to the output directory")
    elif args.workflow_type == 'complete':
        # Run the complete workflow with optimization and Monte Carlo
        run_complete_workflow(
            strategy_name=args.strategy,
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            param_file=args.param_file,
            num_workers=args.num_cores,
            output_dir=args.output_dir,
            in_sample_ratio=args.in_sample_ratio,
            num_permutations=args.num_permutations,
            verbose=args.verbose
        )
    elif args.workflow_type == 'walk-forward':
        # Calculate dates based on in-sample ratio
        date_range = (datetime.strptime(args.end_date, '%Y-%m-%d') - datetime.strptime(args.start_date, '%Y-%m-%d')).days
        in_sample_days = int(date_range * args.in_sample_ratio)
        in_sample_end = (datetime.strptime(args.start_date, '%Y-%m-%d') + pd.Timedelta(days=in_sample_days)).strftime('%Y-%m-%d')
        out_sample_start = (datetime.strptime(in_sample_end, '%Y-%m-%d') + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        run_walk_forward_test(
            strategy_name=args.strategy,
            tickers=args.tickers,
            in_sample_start=args.start_date,
            in_sample_end=in_sample_end,
            out_sample_start=out_sample_start,
            out_sample_end=args.end_date,
            param_file=args.param_file,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 