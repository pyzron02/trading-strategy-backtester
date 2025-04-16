#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to run Trade-Based Monte Carlo simulations for trading strategy validation.

This script uses the TradeBasedMonteCarloTest class to:
1. Run a trading strategy on historical data to generate original performance metrics
2. Create permutations of the historical data to generate Monte Carlo simulations
3. Analyze the distribution of performance metrics across simulations
4. Generate visualizations to evaluate strategy robustness
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the TradeBasedMonteCarloTest class
from src.monte_carlo.trade_based_monte_carlo import TradeBasedMonteCarloTest

# Import strategy registry
from src.strategies.registry import registry


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Trade-Based Monte Carlo simulations')
    
    # Strategy and tickers
    parser.add_argument('--strategy', type=str, required=True,
                        help='Strategy name (e.g., SimpleStock, MACrossover)')
    parser.add_argument('--tickers', type=str, required=True,
                        help='Comma-separated list of ticker symbols (e.g., AAPL,MSFT)')
    
    # Data and time periods
    parser.add_argument('--input-dir', type=str, default='input',
                        help='Directory containing input data')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-01-01',
                        help='End date for backtest (YYYY-MM-DD)')
    
    # Simulation parameters
    parser.add_argument('--num-simulations', type=int, default=100,
                        help='Number of Monte Carlo simulations')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                        help='Initial capital for backtest')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate for trades')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of workers for parallel processing')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
    parser.add_argument('--keep-permuted-data', action='store_true',
                        help='Keep permuted data files after simulation')
                        
    # Parameters for the strategy (as JSON string or file path)
    parser.add_argument('--parameters', type=str, default="{}",
                        help='JSON string or file path for strategy parameters')
    
    return parser.parse_args()


def main():
    """Run the Trade-Based Monte Carlo simulation."""
    # Parse command line arguments
    args = parse_args()
    
    # Parse tickers
    tickers = args.tickers.split(',')
    
    # Parse parameters
    try:
        if os.path.isfile(args.parameters):
            with open(args.parameters, 'r') as f:
                parameters = json.load(f)
        else:
            parameters = json.loads(args.parameters)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error parsing parameters: {e}")
        return 1
    
    # Check if strategy exists
    if not registry.strategy_exists(args.strategy):
        print(f"Error: Strategy '{args.strategy}' not found")
        print("Available strategies:")
        for strategy in registry.get_all_strategy_names():
            print(f"  - {strategy}")
        return 1
    
    # Create output directory if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", f"{args.strategy}_monte_carlo_{timestamp}")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config = {
        'strategy': args.strategy,
        'tickers': tickers,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'num_simulations': args.num_simulations,
        'initial_capital': args.initial_capital,
        'commission': args.commission,
        'parameters': parameters,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, 'monte_carlo_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create the TradeBasedMonteCarloTest instance
    monte_carlo_test = TradeBasedMonteCarloTest(
        strategy_name=args.strategy,
        parameters=parameters,
        tickers=tickers,
        input_dir=args.input_dir,
        output_dir=output_dir,
        num_simulations=args.num_simulations,
        initial_capital=args.initial_capital,
        commission=args.commission,
        seed=args.seed,
        verbose=args.verbose,
        num_workers=args.num_workers,
        keep_permuted_data=args.keep_permuted_data
    )
    
    # Run the test
    results = monte_carlo_test.run_test(out_of_sample_start=args.start_date)
    
    # Check if results were generated
    if not results:
        print("Error: No results generated from Monte Carlo test")
        return 1
    
    # Print completion message
    print(f"\nTrade-Based Monte Carlo analysis completed successfully!")
    print(f"Results saved to: {monte_carlo_test.output_dir}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for metric, stats in results.items():
        if isinstance(stats, dict) and 'original' in stats and 'mean' in stats:
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Original: {stats['original']:.4f}")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            if 'p_value' in stats:
                print(f"  P-value: {stats['p_value']:.4f}")
                if stats['p_value'] < 0.05:
                    if metric != 'max_drawdown' and stats['original'] > stats['mean']:
                        print("  Strategy significantly outperforms random simulations (p<0.05)")
                    elif metric == 'max_drawdown' and stats['original'] < stats['mean']:
                        print("  Strategy significantly outperforms random simulations (p<0.05)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 