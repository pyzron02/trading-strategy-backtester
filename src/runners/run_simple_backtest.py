#!/usr/bin/env python3
# run_simple_backtest.py - Unified backtest runner for all strategies

import os
import sys
import argparse
from datetime import datetime
import json
import pandas as pd

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import components from the streamlined system
from engine.data_management import DataManager
from engine.logging_system import logger
from engine.run_backtest import run_backtest
from engine.parameter_management import ParameterManager
from engine.evaluate_performance import evaluate_performance
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

def run_strategy_backtest(strategy_name, tickers=None, start_date=None, end_date=None, 
                         parameters=None, output_dir=None, detailed_analysis=False,
                         save_results=True, verbose=False):
    """
    Run a backtest for any strategy with specified parameters.
    
    Args:
        strategy_name (str): Name of the strategy to backtest
        tickers (list): List of ticker symbols
        start_date (str): Start date for backtesting (YYYY-MM-DD)
        end_date (str): End date for backtesting (YYYY-MM-DD)
        parameters (dict): Strategy parameters
        output_dir (str): Directory to save output files
        detailed_analysis (bool): Whether to run detailed performance analysis
        save_results (bool): Whether to save results to files
        verbose (bool): Whether to print detailed information during execution
    
    Returns:
        dict: Backtest results
    """
    if tickers is None:
        tickers = ['AAPL', 'MSFT']
    
    if start_date is None:
        start_date = '2020-01-01'
    
    if end_date is None:
        end_date = '2021-12-31'
    
    # Get strategy class
    strategy_class = registry.get_strategy_class(strategy_name)
    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy_name}' not found")
    
    # If parameters not provided, use defaults from strategy class
    if parameters is None:
        if hasattr(strategy_class, 'get_default_parameters'):
            parameters = strategy_class.get_default_parameters()
        else:
            parameters = {}
    
    if verbose:
        print(f"Strategy: {strategy_name}")
        print(f"Tickers: {tickers}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Parameters: {parameters}")
        print("\nStarting backtest...\n")
    
    # Set output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(os.path.dirname(src_dir), 'output', f"{strategy_name}_{timestamp}")
    
    # Ensure the output directory exists
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run backtest
    result = run_backtest(
        output_dir=output_dir,
        strategy_name=strategy_name,
        tickers=tickers,
        parameters=parameters,
        start_date=start_date,
        end_date=end_date
    )
    
    if verbose:
        print("\nBacktest Results:")
        print(f"Initial Value: ${result['initial_value']:.2f}")
        print(f"Final Value: ${result['final_value']:.2f}")
        print(f"Total Return: {result['total_return']:.2f}%")
        print(f"Benchmark Return: {result['benchmark_return']:.2f}%")
        print(f"Alpha: {result['alpha']:.2f}%")
    
    # Save results to JSON
    if save_results:
        # Convert any pandas Series or DataFrame objects to native Python types
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, pd.Series):
                serializable_result[key] = value.to_dict()
            elif isinstance(value, pd.DataFrame):
                serializable_result[key] = value.to_dict(orient='records')
            else:
                serializable_result[key] = value
        
        # Recursively convert any Timestamp keys to strings
        serializable_result = convert_timestamps_in_dict(serializable_result)
        
        results_file = os.path.join(output_dir, f"{strategy_name}_results.json")
        with open(results_file, 'w') as f:
            json.dump(serializable_result, f, indent=4, cls=CustomJSONEncoder)
        
        if verbose:
            print(f"\nBacktest completed. Results saved to {output_dir}")
    
    # Run detailed performance analysis if requested
    if detailed_analysis and save_results:
        if verbose:
            print("\nRunning detailed performance analysis...")
        
        backtest_results_path = os.path.join(output_dir, "backtest_results.pkl")
        if os.path.exists(backtest_results_path):
            evaluate_performance(backtest_results_path)
            if verbose:
                print(f"Detailed analysis completed. Visualizations saved to {output_dir}")
        else:
            # Try to find the backtest results file in subdirectories
            found_file = None
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file == 'backtest_results.pkl':
                        found_file = os.path.join(root, file)
                        break
                if found_file:
                    break
            
            if found_file:
                evaluate_performance(found_file)
                if verbose:
                    print(f"Detailed analysis completed. Visualizations saved to {os.path.dirname(found_file)}")
            elif verbose:
                print(f"Warning: Could not find backtest results file in {output_dir}")
    
    return result

def run_ma_crossover_backtest(tickers=None, start_date=None, end_date=None, 
                             fast_period=None, slow_period=None, output_dir=None,
                             detailed_analysis=False, verbose=False):
    """
    Specialized function to run a backtest for the MACrossover strategy.
    This is a convenience wrapper around run_strategy_backtest for backward compatibility.
    """
    # Set default parameters
    parameters = {}
    if fast_period is not None:
        parameters['fast_period'] = fast_period
    if slow_period is not None:
        parameters['slow_period'] = slow_period
    
    # Run the backtest using the generic function
    return run_strategy_backtest(
        strategy_name='MACrossover',
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters,
        output_dir=output_dir,
        detailed_analysis=detailed_analysis,
        verbose=verbose
    )

def main():
    """Run a backtest for a strategy from the command line."""
    parser = argparse.ArgumentParser(description='Run a backtest for any trading strategy')
    
    parser.add_argument('--strategy', '-s', required=True, help='Name of the strategy to backtest')
    parser.add_argument('--tickers', '-t', nargs='+', default=['AAPL', 'MSFT'], help='List of ticker symbols')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2021-12-31', help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--output-dir', help='Directory to save output files')
    parser.add_argument('--param-file', help='JSON file with parameters')
    parser.add_argument('--fast-period', type=int, help='Fast period for MA Crossover strategy')
    parser.add_argument('--slow-period', type=int, help='Slow period for MA Crossover strategy')
    parser.add_argument('--detailed-analysis', action='store_true', help='Run detailed performance analysis')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information during execution')
    
    args = parser.parse_args()
    
    # Set logging level
    logger.set_level(args.log_level)
    
    # Print available strategies if in verbose mode
    if args.verbose:
        registered_strategies = registry.get_registered_strategies()
        strategy_names = [strategy['name'] for strategy in registered_strategies]
        print(f"Available strategies: {strategy_names}")
    
    # Get parameters
    parameters = None
    if args.param_file:
        # Load parameters from file
        param_manager = ParameterManager()
        parameters = param_manager.load_parameter_grid(args.strategy, args.param_file)
    elif args.strategy == 'MACrossover' and (args.fast_period is not None or args.slow_period is not None):
        # Special handling for MA Crossover strategy
        parameters = {}
        if args.fast_period is not None:
            parameters['fast_period'] = args.fast_period
        if args.slow_period is not None:
            parameters['slow_period'] = args.slow_period
    
    # Run the backtest
    run_strategy_backtest(
        strategy_name=args.strategy,
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        parameters=parameters,
        output_dir=args.output_dir,
        detailed_analysis=args.detailed_analysis,
        verbose=args.verbose
    )
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 