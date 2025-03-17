#!/usr/bin/env python3
# run_ma_crossover.py - Example script to run the MA Crossover strategy

import os
import sys
import json

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from run_master import run_workflow
from engine.parameter_management import ParameterManager
from strategies import registry

def main():
    """Run the MA Crossover strategy with the master workflow."""
    print("\n" + "="*80)
    print("Running MA Crossover Strategy Backtest")
    print("="*80 + "\n")
    
    # Define parameters
    strategy_name = "MACrossover"
    tickers = ['AAPL', 'MSFT']  # Removed GOOGL as it's not in the data
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    
    # Verify the strategy is registered
    strategy_names = registry.get_strategy_names()
    print(f"Available strategies: {strategy_names}")
    
    if strategy_name not in strategy_names:
        print(f"Error: Strategy '{strategy_name}' not found in registry")
        return 1
    
    # Initialize ParameterManager
    param_manager = ParameterManager()
    param_manager.define_parameters_from_registry(registry)
    
    # Verify parameters are defined
    try:
        default_params = param_manager.get_default_parameters(strategy_name)
        print(f"Default parameters for {strategy_name}: {default_params}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Load parameter ranges from JSON file
    param_file = os.path.join(current_dir, 'ma_crossover_params.json')
    with open(param_file, 'r') as f:
        param_ranges = json.load(f)
    
    print(f"Strategy: {strategy_name}")
    print(f"Tickers: {tickers}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Parameter Ranges: {param_ranges}")
    print("\nStarting workflow...\n")
    
    # Run the workflow
    result = run_workflow(
        strategy_name=strategy_name,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        param_ranges=param_ranges,
        max_workers=4  # Adjust based on your CPU cores
    )
    
    # Print the report
    print("\nBacktest Report:")
    print(result['report'])
    
    print("\n" + "="*80)
    print("MA Crossover Strategy Backtest Complete")
    print("="*80 + "\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 