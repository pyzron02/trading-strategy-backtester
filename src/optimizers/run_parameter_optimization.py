#!/usr/bin/env python3
# run_parameter_optimization.py - Optimize parameters for a strategy

import os
import sys
import argparse
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up to src directory
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import components
from engine.parameter_management import ParameterManager
from engine.logging_system import logger
from engine.run_backtest import run_backtest
from engine.evaluate_performance import evaluate_performance
from strategies import registry

def run_parameter_optimization(strategy_name, tickers=None, start_date=None, end_date=None, 
                              param_file=None, num_combinations=10, output_dir=None, 
                              metric='sharpe_ratio', verbose=False, detailed_analysis=False):
    """
    Run parameter optimization for a strategy.
    
    Args:
        strategy_name (str): Name of the strategy
        tickers (list): List of tickers
        start_date (str): Start date
        end_date (str): End date
        param_file (str): Path to parameter file
        num_combinations (int): Number of parameter combinations to test
        output_dir (str): Output directory
        metric (str): Metric to optimize for
        verbose (bool): Whether to print verbose output
        detailed_analysis (bool): Whether to run detailed performance analysis
        
    Returns:
        dict: Results of the optimization
    """
    # Set default values
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    if start_date is None:
        start_date = '2020-01-01'
    
    if end_date is None:
        end_date = '2021-12-31'
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(current_dir), 'output')
    
    # Create timestamp directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f"{strategy_name}_optimization_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up logging
    logger.set_level('INFO')
    logger.info('optimization', f"Starting parameter optimization for {strategy_name}")
    logger.info('optimization', f"Tickers: {tickers}")
    logger.info('optimization', f"Period: {start_date} to {end_date}")
    logger.info('optimization', f"Output directory: {run_dir}")
    
    # Load parameter grid
    param_manager = ParameterManager()
    if param_file:
        param_grid = param_manager.load_parameter_grid(strategy_name, param_file)
        logger.info('optimization', f"Loaded parameter grid from {param_file}")
    else:
        # Get default parameters and create a simple grid
        default_params = param_manager.get_default_parameters(strategy_name)
        param_grid = {}
        
        # Create a simple grid for each parameter
        for param_name, param_value in default_params.items():
            if isinstance(param_value, int):
                param_grid[param_name] = [
                    max(1, int(param_value * 0.5)),
                    param_value,
                    int(param_value * 1.5)
                ]
            elif isinstance(param_value, float):
                param_grid[param_name] = [
                    max(0.1, param_value * 0.5),
                    param_value,
                    param_value * 1.5
                ]
            elif isinstance(param_value, bool):
                param_grid[param_name] = [True, False]
            else:
                # For other types, just use the default value
                param_grid[param_name] = [param_value]
        
        logger.info('optimization', f"Created parameter grid: {param_grid}")
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Generate all combinations
    all_combinations = list(itertools.product(*param_values))
    
    # Limit to num_combinations
    if len(all_combinations) > num_combinations:
        import random
        random.shuffle(all_combinations)
        combinations = all_combinations[:num_combinations]
    else:
        combinations = all_combinations
    
    logger.info('optimization', f"Testing {len(combinations)} parameter combinations")
    
    # Run backtest for each parameter combination
    results = []
    
    for i, combo in enumerate(tqdm(combinations, desc="Testing parameters")):
        # Create parameter dictionary
        params = {}
        for j, param_name in enumerate(param_names):
            params[param_name] = combo[j]
        
        # Create a directory for this parameter set
        param_dir = os.path.join(run_dir, f"params_{i}")
        os.makedirs(param_dir, exist_ok=True)
        
        # Save parameters to file
        params_file = os.path.join(param_dir, "parameters.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=4)
        
        # Run backtest
        result = run_backtest(
            output_dir=param_dir,
            strategy_name=strategy_name,
            tickers=tickers,
            parameters=params
        )
        
        # Add parameters to result
        result['parameters'] = params
        
        # Add to results list
        results.append(result)
    
    # Find best result based on metric
    if metric == 'total_return':
        best_result = max(results, key=lambda x: x['total_return'])
    elif metric == 'sharpe_ratio':
        # Calculate Sharpe ratio for each result
        for result in results:
            if 'equity_curve' in result and result['equity_curve']:
                equity_df = pd.DataFrame(result['equity_curve'])
                equity_df['Return'] = equity_df['Value'].pct_change()
                sharpe = equity_df['Return'].mean() / equity_df['Return'].std() * (252 ** 0.5) if equity_df['Return'].std() > 0 else 0
                result['sharpe_ratio'] = sharpe
            else:
                result['sharpe_ratio'] = 0
        
        best_result = max(results, key=lambda x: x['sharpe_ratio'])
    elif metric == 'alpha':
        best_result = max(results, key=lambda x: x['alpha'])
    else:
        best_result = max(results, key=lambda x: x['total_return'])
    
    # Save best parameters
    best_params_file = os.path.join(run_dir, f"{strategy_name}_best_params.json")
    with open(best_params_file, 'w') as f:
        json.dump(best_result['parameters'], f, indent=4)
    
    # Create summary report
    summary = []
    summary.append("=" * 80)
    summary.append(f"{strategy_name} Parameter Optimization Report")
    summary.append("=" * 80)
    summary.append("")
    
    summary.append(f"Tickers: {tickers}")
    summary.append(f"Period: {start_date} to {end_date}")
    summary.append(f"Metric: {metric}")
    summary.append(f"Tested {len(combinations)} parameter combinations")
    summary.append("")
    
    summary.append("Best Parameters:")
    for param, value in best_result['parameters'].items():
        summary.append(f"  - {param}: {value}")
    summary.append("")
    
    summary.append("Performance Metrics:")
    summary.append(f"  - Initial Value: ${best_result['initial_value']:.2f}")
    summary.append(f"  - Final Value: ${best_result['final_value']:.2f}")
    summary.append(f"  - Total Return: {best_result['total_return']:.2f}%")
    summary.append(f"  - Benchmark Return: {best_result['benchmark_return']:.2f}%")
    summary.append(f"  - Alpha: {best_result['alpha']:.2f}%")
    if 'sharpe_ratio' in best_result:
        summary.append(f"  - Sharpe Ratio: {best_result['sharpe_ratio']:.4f}")
    summary.append("")
    
    summary.append("All Tested Parameters:")
    for i, result in enumerate(results):
        summary.append(f"  {i+1}. Parameters: {result['parameters']}")
        summary.append(f"     Total Return: {result['total_return']:.2f}%")
        if 'sharpe_ratio' in result:
            summary.append(f"     Sharpe Ratio: {result['sharpe_ratio']:.4f}")
        summary.append(f"     Alpha: {result['alpha']:.2f}%")
        summary.append("")
    
    summary.append("=" * 80)
    
    # Write summary to file
    summary_file = os.path.join(run_dir, f"{strategy_name}_optimization_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("\n".join(summary))
    
    # Plot parameter impact on performance
    for param_name in param_names:
        # Skip if only one value
        if len(set([result['parameters'][param_name] for result in results])) <= 1:
            continue
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        
        # Extract parameter values and returns
        param_values = [result['parameters'][param_name] for result in results]
        returns = [result['total_return'] for result in results]
        
        # Sort by parameter value
        sorted_data = sorted(zip(param_values, returns))
        sorted_values, sorted_returns = zip(*sorted_data)
        
        # Plot
        plt.plot(sorted_values, sorted_returns, 'o-')
        plt.title(f"Impact of {param_name} on Total Return")
        plt.xlabel(param_name)
        plt.ylabel("Total Return (%)")
        plt.grid(True)
        
        # Save plot
        plot_file = os.path.join(run_dir, f"{strategy_name}_{param_name}_impact.png")
        plt.savefig(plot_file)
        plt.close()
    
    logger.info('optimization', f"Parameter optimization completed")
    logger.info('optimization', f"Best parameters: {best_result['parameters']}")
    logger.info('optimization', f"Best {metric}: {best_result.get(metric, best_result['total_return'])}")
    logger.info('optimization', f"Summary saved to {summary_file}")
    
    # Run detailed performance analysis on the best result if requested
    if detailed_analysis:
        logger.info('optimization', "Running detailed performance analysis on best parameter set...")
        best_param_index = results.index(best_result)
        best_param_dir = os.path.join(run_dir, f"params_{best_param_index}")
        backtest_results_path = os.path.join(best_param_dir, "backtest_results.pkl")
        
        if os.path.exists(backtest_results_path):
            evaluate_performance(backtest_results_path)
            logger.info('optimization', f"Detailed analysis completed. Visualizations saved to {best_param_dir}")
        else:
            # Try to find the backtest results file in subdirectories
            found_file = None
            for root, dirs, files in os.walk(best_param_dir):
                for file in files:
                    if file == 'backtest_results.pkl':
                        found_file = os.path.join(root, file)
                        break
                if found_file:
                    break
            
            if found_file:
                evaluate_performance(found_file)
                logger.info('optimization', f"Detailed analysis completed. Visualizations saved to {os.path.dirname(found_file)}")
            else:
                logger.warning('optimization', f"Could not find backtest results file for the best parameter set")
    
    return {
        'strategy_name': strategy_name,
        'tickers': tickers,
        'start_date': start_date,
        'end_date': end_date,
        'best_parameters': best_result['parameters'],
        'best_result': best_result,
        'all_results': results,
        'summary_file': summary_file,
        'output_dir': run_dir
    }

def main():
    """Main function to parse arguments and run the optimization."""
    parser = argparse.ArgumentParser(description='Optimize parameters for a strategy')
    
    parser.add_argument('--strategy', '-s', required=True, help='Name of the strategy to optimize')
    parser.add_argument('--tickers', '-t', nargs='+', default=['AAPL', 'MSFT'], help='List of ticker symbols')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2021-12-31', help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--param-file', help='JSON file with parameter grid')
    parser.add_argument('--num-combinations', '-n', type=int, default=10, help='Number of parameter combinations to test')
    parser.add_argument('--output-dir', help='Directory to save output files')
    parser.add_argument('--metric', '-m', default='sharpe_ratio', choices=['total_return', 'sharpe_ratio', 'alpha'],
                        help='Metric to optimize for')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    parser.add_argument('--detailed-analysis', '-d', action='store_true', help='Run detailed performance analysis on best result')
    
    args = parser.parse_args()
    
    # Run the optimization
    result = run_parameter_optimization(
        strategy_name=args.strategy,
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        param_file=args.param_file,
        num_combinations=args.num_combinations,
        output_dir=args.output_dir,
        metric=args.metric,
        verbose=args.verbose,
        detailed_analysis=args.detailed_analysis
    )
    
    # Print the summary
    with open(result['summary_file'], 'r') as f:
        print(f.read())
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 