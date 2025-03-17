#!/usr/bin/env python3
# unified_workflow.py - Centralized entry point for all trading strategy backtester workflows

import os
import sys
import argparse
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import logging
import time

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import components
from engine.parameter_management import ParameterManager
from engine.logging_system import logger, log_execution_time
from engine.testing.in_sample_excellence import InSampleExcellence
from engine.testing.walk_forward_test import WalkForwardTest
from engine.testing.walk_forward_monte_carlo import WalkForwardMonteCarloTest
from engine.evaluate_performance import evaluate_performance
from strategies import registry
from engine.run_backtest import run_backtest

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

@log_execution_time
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

@log_execution_time
def run_complete_workflow(strategy_name, tickers=None, start_date=None, end_date=None, 
                         param_file=None, num_workers=None, output_dir=None, 
                         in_sample_ratio=0.7, num_permutations=10, verbose=False):
    """
    Run a complete workflow with in-sample optimization and walk-forward testing.
    
    Args:
        strategy_name (str): Name of the strategy to test
        tickers (list): List of ticker symbols
        start_date (str): Start date for the entire test period (YYYY-MM-DD)
        end_date (str): End date for the entire test period (YYYY-MM-DD)
        param_file (str): Path to parameter file for optimization
        num_workers (int): Number of parallel workers for optimization
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
        output_dir=optimization_dir,
        num_workers=num_workers
    )
    
    optimization_results = optimizer.run_optimization(target_metric='sharpe_ratio')
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
    if num_permutations > 0:
        logger.info('workflow', f"Starting Monte Carlo testing with {num_permutations} permutations...")
        monte_carlo = WalkForwardMonteCarloTest(
            strategy_name=strategy_name,
            tickers=tickers,
            in_sample_start=start_date,
            in_sample_end=in_sample_end_date,
            out_sample_start=out_sample_start_date,
            out_sample_end=end_date,
            parameters=best_params,
            output_dir=monte_carlo_dir
        )
        
        monte_carlo_results = monte_carlo.run_test(num_permutations=num_permutations)
        logger.info('workflow', f"Monte Carlo testing completed")
    
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
                        choices=['simple', 'complete', 'walk-forward'],
                        help='Type of workflow to run')
    parser.add_argument('--strategy', '-s', required=True, 
                        help='Name of the strategy to test')
    parser.add_argument('--tickers', '-t', nargs='+', default=['AAPL'], 
                        help='List of ticker symbols')
    parser.add_argument('--start-date', default='2015-01-01', 
                        help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2021-12-31', 
                        help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--param-file', 
                        help='Path to parameter file')
    parser.add_argument('--output-dir', 
                        help='Directory to save output files')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Print detailed information')
    
    # Workflow-specific arguments
    parser.add_argument('--in-sample-ratio', type=float, default=0.7, 
                        help='Ratio of data to use for in-sample period')
    parser.add_argument('--num-permutations', type=int, default=10, 
                        help='Number of permutations for Monte Carlo testing')
    parser.add_argument('--num-workers', type=int, 
                        help='Number of parallel workers for optimization')
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
    elif args.workflow_type == 'complete':
        run_complete_workflow(
            strategy_name=args.strategy,
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            param_file=args.param_file,
            num_workers=args.num_workers,
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