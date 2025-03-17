#!/usr/bin/env python3
# run_walk_forward_test.py - Unified walk-forward test runner for all strategies

import os
import sys
import argparse
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import components
from engine.testing.walk_forward_test import WalkForwardTest
from engine.parameter_management import ParameterManager
from engine.logging_system import logger
from strategies import registry

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

def run_walk_forward_test(strategy_name, tickers=None, in_sample_start=None, in_sample_end=None,
                        out_sample_start=None, out_sample_end=None, parameters=None, 
                        param_file=None, output_dir=None, in_sample_ratio=0.7, 
                        save_results=True, verbose=False):
    """
    Run a walk-forward test for any strategy.
    
    Args:
        strategy_name (str): Name of the strategy
        tickers (list): List of ticker symbols
        in_sample_start (str): Start date for in-sample period (YYYY-MM-DD)
        in_sample_end (str): End date for in-sample period (YYYY-MM-DD)
        out_sample_start (str): Start date for out-of-sample period (YYYY-MM-DD)
        out_sample_end (str): End date for out-of-sample period (YYYY-MM-DD)
        parameters (dict): Strategy parameters to use
        param_file (str): Path to parameter file (alternative to parameters)
        output_dir (str): Directory to save results
        in_sample_ratio (float): Ratio of data to use for in-sample period if dates not specified
        save_results (bool): Whether to save results to files
        verbose (bool): Whether to print detailed information
    
    Returns:
        dict: Walk-forward test results
    """
    # Set defaults for dates if not provided
    full_start = in_sample_start or '2015-01-01'
    full_end = out_sample_end or '2021-12-31'
    
    # If in-sample and out-of-sample dates are not fully specified, calculate them
    if in_sample_end is None or out_sample_start is None:
        date_range = (datetime.strptime(full_end, '%Y-%m-%d') - datetime.strptime(full_start, '%Y-%m-%d')).days
        in_sample_days = int(date_range * in_sample_ratio)
        
        # Calculate the in-sample end date
        if in_sample_end is None:
            in_sample_end = (datetime.strptime(full_start, '%Y-%m-%d') + pd.Timedelta(days=in_sample_days)).strftime('%Y-%m-%d')
        
        # Calculate the out-of-sample start date
        if out_sample_start is None:
            out_sample_start = (datetime.strptime(in_sample_end, '%Y-%m-%d') + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Set defaults for tickers if not provided
    if tickers is None:
        tickers = ['AAPL']
    
    # Get strategy class
    strategy_class = registry.get_strategy_class(strategy_name)
    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy_name}' not found")
    
    # Handle parameters
    if parameters is None:
        if param_file:
            # Load parameters from file
            param_manager = ParameterManager()
            parameters = param_manager.load_parameter_file(param_file)
        elif hasattr(strategy_class, 'get_default_parameters'):
            # Use default parameters from strategy class
            parameters = strategy_class.get_default_parameters()
        else:
            # Fallback to empty parameters
            parameters = {}
    
    if verbose:
        print(f"\nRunning Walk-Forward Test for {strategy_name}")
        print(f"Tickers: {tickers}")
        print(f"In-sample period: {in_sample_start} to {in_sample_end}")
        print(f"Out-of-sample period: {out_sample_start} to {out_sample_end}")
        print(f"Parameters: {parameters}")
    
    # Set output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(os.path.dirname(src_dir), 'output', f"{strategy_name}_wf_{timestamp}")
    
    # Ensure output directory exists
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize walk-forward test
    walk_forward = WalkForwardTest(
        strategy_name=strategy_name,
        tickers=tickers,
        in_sample_start=in_sample_start,
        in_sample_end=in_sample_end,
        out_sample_start=out_sample_start,
        out_sample_end=out_sample_end,
        parameters=parameters,
        output_dir=output_dir
    )
    
    # Run test
    results = walk_forward.run_test()
    
    # Save results to JSON
    if save_results:
        # Make results JSON serializable
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, (pd.Series, pd.DataFrame)):
                        serializable_results[key][k] = v.to_dict()
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        results_file = os.path.join(output_dir, 'walk_forward_results.json')
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=4, cls=CustomJSONEncoder)
        
        if verbose:
            print(f"\nWalk-Forward Test completed. Results saved to {output_dir}")
    
    return results

def main():
    """Run walk-forward test from command line."""
    parser = argparse.ArgumentParser(description='Run a walk-forward test for a trading strategy')
    
    parser.add_argument('--strategy', required=True, help='Name of the strategy to test')
    parser.add_argument('--tickers', nargs='+', help='List of ticker symbols')
    parser.add_argument('--in-sample-start', help='Start date for in-sample period (YYYY-MM-DD)')
    parser.add_argument('--in-sample-end', help='End date for in-sample period (YYYY-MM-DD)')
    parser.add_argument('--out-sample-start', help='Start date for out-of-sample period (YYYY-MM-DD)')
    parser.add_argument('--out-sample-end', help='End date for out-of-sample period (YYYY-MM-DD)')
    parser.add_argument('--in-sample-ratio', type=float, default=0.7,
                        help='Ratio of data to use for in-sample period if dates not specified')
    parser.add_argument('--param-file', help='Path to parameter file')
    parser.add_argument('--output-dir', help='Directory to save results')
    parser.add_argument('--log-level', default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    
    args = parser.parse_args()
    
    # Set logging level
    logger.set_level(args.log_level)
    
    # Print available strategies if in verbose mode
    if args.verbose:
        registered_strategies = registry.get_registered_strategies()
        strategy_names = [strategy['name'] for strategy in registered_strategies]
        print(f"Available strategies: {strategy_names}")
    
    # Run walk-forward test
    run_walk_forward_test(
        strategy_name=args.strategy,
        tickers=args.tickers,
        in_sample_start=args.in_sample_start,
        in_sample_end=args.in_sample_end,
        out_sample_start=args.out_sample_start,
        out_sample_end=args.out_sample_end,
        param_file=args.param_file,
        output_dir=args.output_dir,
        in_sample_ratio=args.in_sample_ratio,
        verbose=args.verbose
    )
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 